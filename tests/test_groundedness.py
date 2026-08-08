"""Eval: are policy answers actually supported by the documents?

THE FAILURE THIS CATCHES
    A policy answer can be fluent, confident, correctly formatted, and
    completely made up. Nothing crashes. The customer believes it. The
    store is now on record promising something it never promised. Of every
    failure in this project, this is the one a human reviewer is least
    likely to notice by reading the output - which is exactly why it needs
    an automated check.

TWO KINDS OF CHECK, DELIBERATELY BOTH
    MECHANICAL (free, certain)
        Does every cited id exist? Is a policy answer cited at all? Code
        can answer these with no judgement and no ambiguity, so they run
        on every push and are allowed to be strict.

    LLM-AS-JUDGE (live, probabilistic)
        Does the answer's SUBSTANCE follow from the excerpts? Only a
        reader can tell. So a second model reads the excerpts and the
        answer and rules on support - the standard technique for grading
        open-ended output.

    The judge is fallible too, which is why it never runs alone: the
    mechanical checks catch what code can prove, and the judge is asked
    only the question code cannot answer. Never let a probabilistic grader
    be your only grader.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agents.policy_agent import policy_agent_node
from graph.state import ShopSenseState
from guards.output_rail import audit_citations, valid_section_ids
from llm import get_llm

# Questions chosen because a plausible-sounding wrong answer exists for
# each. "How long to report damage?" is the sharpest: a skimming model
# answers 30 days (the general return window) instead of 14 (the damage
# deadline) - fluent, confident, and wrong in the customer's favour.
GROUNDEDNESS_CASES = [
    "How long do I have to return something?",
    "My order arrived smashed. How long do I have to report it?",
    "Is delivery free?",
    "My headphones died after 3 months - are they covered?",
]


class Verdict(BaseModel):
    reasoning: str = Field(description="One sentence. Write this first.")
    supported: bool = Field(
        description=(
            "True if EVERY factual claim about store policy in the answer "
            "is stated in, or directly implied by, the excerpts provided. "
            "False if any policy fact appears that the excerpts do not "
            "support, or if a number, deadline, or condition has been "
            "changed, rounded, or softened. Ignore tone, formatting, and "
            "offers to help further - judge only the policy facts."
        )
    )


JUDGE_PROMPT = """You are auditing a support assistant for groundedness.

You will be given policy excerpts and an answer that was written from \
them. Decide whether every policy fact in the answer is supported by the \
excerpts.

Be strict about numbers, deadlines, thresholds, and conditions - those are \
where an unsupported claim does real damage. Be lenient about wording, \
structure, and helpfulness. An answer that correctly says the policies do \
not cover something is SUPPORTED."""


def _run_and_judge(question: str) -> tuple[bool, str, list[str]]:
    """Run the real policy agent, then grade its answer against its sources."""
    out = policy_agent_node({"messages": [HumanMessage(content=question)]})

    excerpts = "\n\n".join(
        str(m.content) for m in out["messages"] if isinstance(m, ToolMessage)
    )
    answer = out["messages"][-1].text

    # Mechanical checks first - they cost nothing and are certain.
    shown = set()
    from guards.output_rail import _EXCERPT_HEADER

    for m in out["messages"]:
        if isinstance(m, ToolMessage):
            shown |= set(_EXCERPT_HEADER.findall(str(m.content)))
    problems = audit_citations(answer, shown)

    verdict = (
        get_llm("router")
        .with_structured_output(Verdict)
        .invoke(
            [
                HumanMessage(content=JUDGE_PROMPT),
                HumanMessage(content=f"EXCERPTS PROVIDED:\n{excerpts}"),
                HumanMessage(content=f"ANSWER GIVEN:\n{answer}"),
            ]
        )
    )
    return verdict.supported, verdict.reasoning, problems


@pytest.mark.live
@pytest.mark.parametrize("question", GROUNDEDNESS_CASES)
def test_policy_answers_are_grounded(question):
    supported, why, problems = _run_and_judge(question)
    assert supported, f"unsupported claim for {question!r}: {why}"
    # An invented citation is a certainty, not a judgement call - so it is
    # a hard failure even though the rest of this test is probabilistic.
    assert not any(p.startswith("invented-citation") for p in problems), problems


@pytest.mark.live
def test_uncovered_topic_is_refused_not_invented():
    """The highest-stakes case: no source material, plus a helpful model."""
    out = policy_agent_node(
        {"messages": [HumanMessage(content="Do you price match other stores?")]}
    )
    answer = out["messages"][-1].text.lower()
    assert any(
        phrase in answer
        for phrase in ("don't cover", "do not cover", "not covered",
                       "isn't covered", "not addressed", "unable to confirm")
    ), f"expected a refusal, got: {answer[:200]}"


# ---------------------------------------------------------------------------
# Free tier - the citation auditor itself
# ---------------------------------------------------------------------------


def _excerpt(*ids: str) -> ToolMessage:
    body = "\n".join(f"--- [{i}] Title (returns.md) ---\nbody" for i in ids)
    return ToolMessage(content=body, tool_call_id="t")


def test_auditor_accepts_a_properly_cited_answer():
    assert audit_citations("You have 30 days [RET-1].", {"RET-1"}) == []


def test_auditor_catches_an_invented_citation():
    problems = audit_citations("Returns are free forever [RET-99].", {"RET-1"})
    assert any(p.startswith("invented-citation") for p in problems)


def test_auditor_catches_an_uncited_policy_answer():
    assert "uncited-policy-answer" in audit_citations("You have 30 days.", {"RET-1"})


def test_auditor_allows_an_uncited_refusal():
    """No excerpts retrieved means no citations are POSSIBLE.

    Flagging this would fire on every correct refusal - and an alarm that
    cries wolf on correct behaviour is worse than no alarm, because the
    team learns to ignore it.
    """
    assert audit_citations("Our policies don't cover that.", set()) == []


def test_corpus_ids_are_unique_and_wellformed():
    ids = valid_section_ids()
    assert len(ids) >= 15
    for i in ids:
        assert i.split("-")[0].isupper() and i.split("-")[1].isdigit()
