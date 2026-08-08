"""Eval: does each question reach the right specialist?

WHY THIS IS THE FIRST EVAL WE WROTE
    Routing is the highest-leverage failure in a multi-agent system. If a
    policy question reaches the order agent, the customer gets "I can't
    find that order" for a question about the return window - a confusing
    answer produced by two components that are each working perfectly.
    And routing is decided by a PROMPT, which means an innocent wording
    change can silently break it. That combination - high impact, easy to
    break, invisible when broken - is precisely what an eval is for.

THE TABLE IS THE TEST
    ROUTING_CASES below is a specification anyone can read and extend. Add
    a row when you find a misroute in production; the fix is then provably
    a fix, and provably stays fixed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import ShopSenseState
from graph.supervisor import MAX_HOPS, RoutingDecision, supervisor_node

# (question, expected_route, prior_conversation)
ROUTING_CASES = [
    # --- order questions ---------------------------------------------
    ("Where is my order ord_1003?", "order_agent", []),
    ("Has my package shipped yet?", "order_agent", []),
    ("It's dana@example.com - what did I buy last month?", "order_agent", []),
    ("My headphones arrived cracked, I want my money back.", "order_agent", []),
    # --- policy questions --------------------------------------------
    ("How long do I have to return something?", "policy_agent", []),
    ("Is shipping free?", "policy_agent", []),
    ("Does my warranty cover a cracked screen?", "policy_agent", []),
    ("Can I return a final sale item?", "policy_agent", []),
    # --- neither ------------------------------------------------------
    ("hi there!", "FINISH", []),
    ("thanks, that's all", "FINISH", []),
    # --- the case that catches loop-y prompts -------------------------
    # The work is already done and visible in the transcript. A supervisor
    # that routes again here would loop until the hop cap, burning money
    # to re-derive an answer it already has.
    (
        "thanks!",
        "FINISH",
        [
            HumanMessage(content="where is ord_1003?"),
            AIMessage(content="Order ord_1003 was delivered on Jul 23."),
        ],
    ),
]

# Allow one miss out of eleven. Routing is a judgement call at the edges
# ("my headphones arrived cracked" is arguably both), and a suite that
# demands perfection from a probabilistic system gets disabled the first
# time it goes red for a defensible answer. One miss is noise; two is a
# regression, and this threshold says so out loud.
MAX_MISSES = 1


@pytest.mark.live
def test_routing_table():
    misses = []
    for question, expected, prior in ROUTING_CASES:
        state: ShopSenseState = {"messages": [*prior, HumanMessage(content=question)]}
        got = supervisor_node(state)["route"]
        if got != expected:
            misses.append(f"{question!r}: expected {expected}, got {got}")

    assert len(misses) <= MAX_MISSES, (
        f"{len(misses)}/{len(ROUTING_CASES)} misroutes (limit {MAX_MISSES}):\n  "
        + "\n  ".join(misses)
    )


@pytest.mark.live
def test_refund_request_reaches_the_order_agent():
    """Singled out because misrouting THIS one has consequences.

    A refund question that lands on the policy agent gets a lecture about
    the returns policy instead of an actual refund - and no human is ever
    asked to approve anything, because the path to the gate runs through
    the order agent.
    """
    state: ShopSenseState = {
        "messages": [HumanMessage(content="Please refund order ord_1003, it broke.")]
    }
    assert supervisor_node(state)["route"] == "order_agent"


# ---------------------------------------------------------------------------
# Free tier - no model needed
# ---------------------------------------------------------------------------


def test_hop_cap_forces_finish_without_calling_the_model():
    """The runaway-loop guard, checked without spending anything.

    Note what this asserts: that the cap fires BEFORE any LLM call. A cap
    that spends money to discover it should stop is only half a cap.
    """
    state: ShopSenseState = {
        "messages": [HumanMessage(content="hello")],
        "usage": [
            {"node": "supervisor", "model": "m", "input_tokens": 1,
             "output_tokens": 1, "cost_usd": 0.0}
            for _ in range(MAX_HOPS)
        ],
    }
    out = supervisor_node(state)
    assert out["route"] == "FINISH"
    assert "usage" not in out, "the guard must fire before any model call"


def test_routing_schema_rejects_invented_routes():
    """Structured output is a CONSTRAINT, not a suggestion.

    If the model answers 'refund_agent', validation fails loudly rather
    than the graph silently falling through to a default edge.
    """
    from pydantic import ValidationError

    RoutingDecision(reasoning="ok", next="order_agent")  # valid
    with pytest.raises(ValidationError):
        RoutingDecision(reasoning="ok", next="refund_agent")
