"""Eval: given a question, does the agent reach for the right tool?

WHAT THIS SUITE IS REALLY TESTING
    The five-part docstrings from Phase 2. Nobody hard-codes "email means
    find_customer" anywhere in this project - the model infers it from the
    tool descriptions. So these tests are the DOCSTRINGS' test suite, and
    a failure here usually means a description needs a clearer USE WHEN or
    a sharper DO NOT USE line, not that the model is broken.

    That is also why a passing suite is load-bearing: it is the only thing
    standing between "I tidied up a docstring" and a support bot that
    calls the wrong tool for a week.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.order_agent import SYSTEM_PROMPT as ORDER_PROMPT
from agents.order_agent import TOOLS as ORDER_TOOLS
from llm import get_llm

# (question, tool the FIRST call should use)
# Only the first call is checked: later calls depend on what earlier ones
# returned, so asserting a whole sequence would test the database's
# contents as much as the model's judgement.
TOOL_CASES = [
    ("Where is order ord_1003?", "get_order_status"),
    ("What's the status of ord_1002?", "get_order_status"),
    ("Hi, it's dana@example.com - can you look up my account?", "find_customer"),
    ("My email is sam@example.com, what have I ordered?", "find_customer"),
    ("Order ord_1003 is broken, please refund it", "request_refund"),
]

MAX_MISSES = 1


def _first_tool_call(question: str) -> str | None:
    llm = get_llm("agent").bind_tools(ORDER_TOOLS)
    reply: AIMessage = llm.invoke(
        [SystemMessage(content=ORDER_PROMPT), HumanMessage(content=question)]
    )
    return reply.tool_calls[0]["name"] if reply.tool_calls else None


@pytest.mark.live
def test_tool_selection_table():
    misses = []
    for question, expected in TOOL_CASES:
        got = _first_tool_call(question)
        if got != expected:
            misses.append(f"{question!r}: expected {expected}, got {got}")

    assert len(misses) <= MAX_MISSES, (
        f"{len(misses)}/{len(TOOL_CASES)} wrong tools (limit {MAX_MISSES}):\n  "
        + "\n  ".join(misses)
    )


@pytest.mark.live
def test_no_tool_call_without_an_identifier():
    """Asking beats guessing.

    With neither an order id nor an email, the correct move is a question,
    not a speculative lookup. A tool call here would mean the model
    invented an identifier - the failure mode the arg schemas and the
    'do not guess' instruction exist to prevent.
    """
    assert _first_tool_call("Where is my order?") is None


# ---------------------------------------------------------------------------
# Free tier - retrieval quality, no model involved
# ---------------------------------------------------------------------------

# (query, section id that MUST appear in the results)
# Phrased as customers phrase things, not as the documents phrase things -
# testing with the docs' own vocabulary would only prove that string
# matching works.
RETRIEVAL_CASES = [
    ("how long do I have to return an item", "RET-1"),
    ("my order arrived smashed, how long to report it?", "RET-3"),
    ("is delivery free?", "SHP-2"),
    ("my headphones died after 3 months", "WAR-1"),
    ("can I still cancel my order?", "RET-6"),
]


@pytest.mark.parametrize("query,required", RETRIEVAL_CASES)
def test_retrieval_finds_the_right_section(retriever, query, required):
    ids = [c.section_id for c in retriever.search(query, k=3)]
    assert required in ids, f"{query!r} -> {ids}, missing {required}"


def test_off_topic_query_returns_nothing(retriever):
    """Empty beats irrelevant.

    The model TRUSTS whatever we paste into its prompt, so handing it the
    least-bad chunk for an unanswerable question is worse than handing it
    nothing: it invites a confident answer built on unrelated policy.
    """
    assert retriever.search("do you price match with other stores?") == []


def test_every_chunk_can_be_cited(retriever):
    """A retrieval result without a usable citation is a rumour."""
    for c in retriever.chunks:
        assert c.section_id and c.source and c.title
        assert c.text.startswith(f"## [{c.section_id}]")
