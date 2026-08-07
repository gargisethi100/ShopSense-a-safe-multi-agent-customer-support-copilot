"""Retrieval over the policy docs: find the right sections for a question.

THE JOB
    Input:  a customer-ish question ("how long do I have to send it back?")
    Output: the few policy SECTIONS most likely to contain the answer,
            each carrying its citation metadata ([RET-1], returns.md).

    That's all retrieval is: a ranking problem. Score every chunk against
    the query, return the top handful. The policy agent then pastes those
    chunks into Claude's prompt and answers FROM them, citing the ids.

WHY THE `Retriever` PROTOCOL (the seam)
    A Protocol is Python's way of saying "anything with THIS method shape
    counts". The policy tool will depend on the Protocol, not on BM25.
    The day we want semantic search (embeddings), we write a new class
    with the same .search() signature and change ONE constructor call.
    Same trick as get_llm(): depend on the interface, swap the engine.

WHY BM25 AND NOT EMBEDDINGS (recap from requirements.txt)
    BM25 is keyword ranking - the algorithm behind classic search engines.
    Embeddings understand synonyms better, but need either a second vendor
    or a ~1GB model download. For THREE documents where users mostly use
    the policy's own vocabulary ("refund", "return", "warranty", "track"),
    BM25 is 95% of the value at 0.1% of the complexity budget.

HOW BM25 THINKS (the 60-second version)
    A chunk scores high for a query word when:
      1. the word appears often IN THAT CHUNK        (term frequency)
      2. the word is RARE across all chunks          (rarity = signal;
         "the" is everywhere so it's worth ~nothing, "warranty" appears
         in few chunks so it's worth a lot)
      3. the chunk isn't just long                   (length normalization -
         long chunks match everything a little; that's penalised)
    Add the scores of all query words -> the chunk's score. No AI, no
    network, pure arithmetic over word counts. Fast and predictable.

Run directly to build the index and try real queries:

    python -m rag.retriever
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


# ---------------------------------------------------------------------------
# The unit of retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One policy section + everything needed to CITE it.

    The metadata is not decoration: section_id is what the agent puts in
    its answer, and source is how a human finds the file to verify. A
    retrieval result you can't cite is a rumor.
    """

    section_id: str  # "RET-1"  - the citation anchor
    title: str       # "Return window"
    source: str      # "returns.md"
    text: str        # the full section, heading included

    def label(self) -> str:
        return f"[{self.section_id}] {self.title} ({self.source})"


class Retriever(Protocol):
    """The seam. Everything downstream depends on this shape only."""

    def search(self, query: str, k: int = 3) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Chunking: one chunk per policy section
# ---------------------------------------------------------------------------

# Matches our heading convention exactly: '## [RET-1] Return window'.
# The docs were WRITTEN for this line - stable ids, self-contained sections.
# Content-first design means the parser can be one regex.
_HEADING = re.compile(r"^## \[([A-Z]+-\d+)\] (.+)$", re.MULTILINE)


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Split every .md file into per-section chunks.

    WHY SECTIONS ARE THE CHUNK SIZE: chunk too small (sentences) and you
    retrieve context-free fragments; chunk too big (whole files) and the
    prompt fills with irrelevant text that drowns the answer. A policy
    section is the natural 'one complete thought' unit - which is exactly
    why we wrote the docs as self-contained sections.
    """
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        matches = list(_HEADING.finditer(body))
        for i, m in enumerate(matches):
            # A section runs from its heading to the next heading (or EOF).
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            chunks.append(
                Chunk(
                    section_id=m.group(1),
                    title=m.group(2).strip(),
                    source=path.name,
                    text=body[m.start():end].strip(),
                )
            )
    if not chunks:
        raise RuntimeError(
            f"No policy sections found under {docs_dir}. Expected .md files "
            "with '## [ID-N] Title' headings (see docs/returns.md)."
        )
    # Duplicate ids would make citations ambiguous - fail loudly, now.
    ids = [c.section_id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuntimeError(f"Duplicate section ids across docs: {sorted(dupes)}")
    return chunks


# Words that carry no topical signal. Removing them looks optional and is
# NOT - see the bug note in _tokenize below.
_STOPWORDS = frozenset("""
a an the this that these those and or but if then so of in on at to for from
by with without within as is are was were be been being do does did doing
have has had having i you he she it we they me my your our their there here
what which who whom how when where why can could may might shall should will
would must not no nor only own same than too very s t just also about into
over under again further once
""".split())


def _stem(word: str) -> str:
    """Crude suffix stripper so 'items' and 'item' count as one word.

    BUG THIS FIXES (found live by the policy agent, 2026-08): asked "how
    long do I have to return an ITEM", BM25 scored RET-1 ("...ITEMS may be
    returned within 30 days...") low, because 'item' and 'items' are
    different strings. The agent then correctly refused to state a return
    window it could not see - a RETRIEVAL failure wearing a generation
    failure's clothes.

    Deliberately not a real stemmer (no NLTK dependency for 17 documents).
    It does not have to be linguistically right - it has to be CONSISTENT,
    because both the query and the documents pass through it. 'process' and
    'processing' both collapsing to 'proces' is fine; they still match.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ss"):
        pass                                    # business, address: keep
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s"):
        word = word[:-1]                        # items -> item
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]                        # shipping -> shipp
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]                        # returned -> return
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]                        # shipp -> ship
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]                        # damage/damaged -> damag
    return word


def _tokenize(text: str) -> list[str]:
    """Text -> comparable word list. BM25 compares TOKENS, not strings.

    Three steps, and the last two were added after a real miss:
      1. lowercase + split on non-alphanumerics ('Delivered,' -> 'delivered')
      2. DROP STOPWORDS. BM25's rarity heuristic assumes a big corpus: a
         word in few documents looks like signal. With 17 sections, 'do'
         and 'have' appear in only two or three - so BM25 rated them
         HIGHLY informative and let an irrelevant chunk outrank the right
         one on filler words alone.
      3. STEM, so singular/plural and tense variants match (see _stem).
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# The BM25 implementation of the Retriever protocol
# ---------------------------------------------------------------------------


# How many times the section TITLE is repeated in the indexed text.
# A title is a human's summary of what the section is ABOUT, so a title
# match is stronger evidence than a body match - but BM25 has no notion of
# fields, only a bag of words. Repeating the title is the standard poor
# man's field boost: it raises the title's term frequency without needing
# a different ranking library.
_TITLE_BOOST = 3

# "...within the [RET-1] window" - our policy docs cross-reference each
# other by section id. That is a hand-authored relevance signal sitting
# right there in the corpus, so we follow it (see _expand_citations).
_XREF = re.compile(r"\[([A-Z]+-\d+)\]")


class BM25Retriever:
    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.chunks = load_chunks(docs_dir)
        self._by_id = {c.section_id: c for c in self.chunks}
        # The index: every chunk pre-tokenized once, at startup. Queries
        # then only pay for scoring - no re-reading files per question.
        self._bm25 = BM25Okapi(
            [_tokenize(f"{(c.title + ' ') * _TITLE_BOOST}{c.text}") for c in self.chunks]
        )

    def _expand_citations(self, hits: list[Chunk], limit: int) -> list[Chunk]:
        """Pull in sections that the hits explicitly point at.

        WHY THIS EXISTS (the honest version): BM25 matches WORDS, and a
        customer's words often aren't the policy's. Asked "how long do I
        have to return an item", it ranks [RET-2] "What can and cannot be
        returned" highly - and RET-2's own text says "within the [RET-1]
        window". The document is telling us where the answer is. Following
        that link costs nothing and needs no model.

        This is a cheap stand-in for semantic search, and it only works
        because the corpus was WRITTEN with cross-references. Where the
        docs don't link, the vocabulary gap remains - which is exactly the
        evidence for the embeddings upgrade, now backed by measurement
        instead of assumption.
        """
        seen = {c.section_id for c in hits}
        extra: list[Chunk] = []
        for c in hits:
            for ref in _XREF.findall(c.text):
                if ref not in seen and ref in self._by_id and len(extra) < limit:
                    seen.add(ref)
                    extra.append(self._by_id[ref])
        return extra

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        """Top-k chunks for the query, best first.

        WHY k=3: we paste results into a prompt. One chunk risks missing
        (the right answer is sometimes split across sections - e.g. refund
        timing [RET-5] vs eligibility [RET-3]); ten chunks buries the
        answer in noise and burns tokens. Three is the demo sweet spot;
        it's a parameter precisely so evals can tune it later.
        """
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # score <= 0 means 'shares no meaningful vocabulary with the query'.
        # Returning such chunks anyway would hand the model authoritative-
        # looking-but-irrelevant text - worse than returning nothing,
        # because the model TRUSTS what we paste into its prompt.
        hits = [self.chunks[i] for i in ranked[:k] if scores[i] > 0]
        # Cross-referenced sections ride along, capped so a chain of links
        # can't quietly balloon the prompt.
        return hits + self._expand_citations(hits, limit=2)

    def search_scored(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        """Same, with scores visible - for debugging and the smoke test.

        Cross-referenced sections appear with score 0.0: they were not
        matched, they were FOLLOWED. Keeping that visible stops a debugging
        session from crediting the ranker for a link's work.
        """
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [(scores[i], self.chunks[i]) for i in ranked[:k] if scores[i] > 0]
        expanded = self._expand_citations([c for _, c in hits], limit=2)
        return hits + [(0.0, c) for c in expanded]


@lru_cache(maxsize=1)
def get_retriever() -> BM25Retriever:
    """Lazy singleton, same pattern as the db pools: built on first use,
    shared by everyone after. This is also the ONE line to edit when the
    engine changes - callers only ever see the Retriever protocol."""
    return BM25Retriever()


# ---------------------------------------------------------------------------
# Smoke test - no database, no LLM, no network. Just files and math.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    r = get_retriever()
    print(f"index built: {len(r.chunks)} sections from "
          f"{len(set(c.source for c in r.chunks))} files\n")

    queries = [
        # phrased like customers, not like the docs - the realistic test
        "how long do I have to send something back?",
        "is shipping free?",
        "my headphones stopped working after two months",
        "can I still cancel my order?",
        "package hasn't moved in a week",
        # and one that SHOULD come back empty:
        "what is your favorite color?",
    ]
    for q in queries:
        print(f"query: {q}")
        hits = r.search_scored(q, k=3)
        if not hits:
            print("  (no relevant sections - correctly returned nothing)")
        for score, c in hits:
            first_line = c.text.splitlines()[0]
            print(f"  {score:5.2f}  {c.label():48}  {first_line[:40]}")
        print()
