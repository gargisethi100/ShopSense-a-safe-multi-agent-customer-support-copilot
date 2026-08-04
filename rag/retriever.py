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


def _tokenize(text: str) -> list[str]:
    """Text -> lowercase word list. BM25 compares TOKENS, not strings.

    Deliberately dumb: lowercase, split on anything non-alphanumeric.
    'Delivered,' and 'delivered' must count as the same word - that's all
    we need. (Real engines add stemming so 'refunds' matches 'refund';
    at our corpus size the plural usually appears anyway. Simplest thing
    that works, seam left for better.)
    """
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# The BM25 implementation of the Retriever protocol
# ---------------------------------------------------------------------------


class BM25Retriever:
    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.chunks = load_chunks(docs_dir)
        # The index: every chunk pre-tokenized once, at startup. Queries
        # then only pay for scoring - no re-reading files per question.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

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
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]

    def search_scored(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        """Same, with scores visible - for debugging and the smoke test."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(scores[i], self.chunks[i]) for i in ranked[:k] if scores[i] > 0]


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
