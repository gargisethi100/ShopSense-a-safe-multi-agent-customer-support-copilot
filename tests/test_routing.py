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
from graph.supervisor import (
    MAX_HOPS,
    RoutingDecision,
    route_from_state,
    supervisor_node,
)

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

# The clarifying question, which has no customer message of its own: the
# agent has just asked for an email and the turn should END until they
# answer. Routing to a specialist here used to send Bedrock a conversation
# ending in an assistant turn, which it refuses - a 500 in the customer's
# face on a path the system is DESIGNED to take (see
# test_tool_selection.py's "no tool call without an identifier").
# agents/common.py makes that unable to crash; this keeps it from
# happening at all.
CLARIFYING_QUESTION = [
    HumanMessage(content="Where is my order?"),
    AIMessage(content="What's the email address on the order?"),
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


@pytest.mark.live
def test_a_question_to_the_customer_ends_the_turn():
    """Regression: asking for an email then routing again produced a 500.

    This case cannot go in ROUTING_CASES because it has no new customer
    message - the whole point is that the last word is the AGENT'S. The
    right answer is FINISH: nobody can make progress until the customer
    replies, and routing to the order agent only asks the same question a
    second time (which is exactly what the live transcript showed).
    """
    state: ShopSenseState = {"messages": list(CLARIFYING_QUESTION)}
    assert supervisor_node(state)["route"] == "FINISH"


@pytest.mark.live
@pytest.mark.parametrize(
    "greeting", ["hi", "hello there", "thanks!", "what can you help me with?"]
)
def test_conversational_turns_get_an_answer(greeting):
    """Regression: 'hi' produced complete silence in the UI.

    FINISH was overloaded - it meant both "a specialist already answered"
    and "no specialist is needed" - so a greeting matched the second and
    nobody replied. Found by a human typing the most obvious first message
    there is, which is exactly the path developers never test.

    The assertion is on the ROUTE, so it needs no model call beyond the
    supervisor's own, and it fails loudly if the branch is ever removed.
    """
    state: ShopSenseState = {"messages": [HumanMessage(content=greeting)]}
    supervisor_node(state)  # sets state["route"] via its return value
    state["route"] = supervisor_node(state)["route"]
    assert route_from_state(state) == "direct_reply", (
        f"{greeting!r} would end the turn with no answer at all"
    )


# ---------------------------------------------------------------------------
# Free tier - no model needed
# ---------------------------------------------------------------------------


def test_finish_with_an_existing_answer_goes_straight_to_the_rail():
    """The other half of the split: don't re-answer what's answered."""
    state: ShopSenseState = {
        "route": "FINISH",
        "messages": [
            HumanMessage(content="where is ord_1003?"),
            AIMessage(content="It was delivered on Jul 23."),
        ],
    }
    assert route_from_state(state) == "FINISH"


def test_finish_with_no_answer_routes_to_direct_reply():
    state: ShopSenseState = {
        "route": "FINISH",
        "messages": [HumanMessage(content="hi")],
    }
    assert route_from_state(state) == "direct_reply"


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


def test_a_decided_refund_is_always_relayed_by_an_agent():
    """Regression: the approval note was shown to the customer verbatim.

    refund_approval writes a note to the TEAM ("APPROVED, confirm this to
    the customer"). That edge used to go to the supervisor, which was free
    to answer FINISH - and did, because the note names an amount and an
    order and reads like a finished reply. The customer got the internal
    text, reference id and all. Rewording the note did not fix it; the
    router made the same call. Only a fixed edge did.

    Asserted on the compiled graph, so it needs no model and no database -
    build_graph(checkpointer=None) is a complete, if forgetful, system.
    """
    from graph.build import build_graph

    edges = {
        (e.source, e.target) for e in build_graph(checkpointer=None).get_graph().edges
    }
    assert ("refund_approval", "order_agent") in edges, (
        "a human's decision about money must be explained by an agent, "
        "not left to the supervisor's discretion"
    )
    assert ("refund_approval", "supervisor") not in edges


def test_routing_schema_rejects_invented_routes():
    """Structured output is a CONSTRAINT, not a suggestion.

    If the model answers 'refund_agent', validation fails loudly rather
    than the graph silently falling through to a default edge.
    """
    from pydantic import ValidationError

    RoutingDecision(reasoning="ok", next="order_agent")  # valid
    with pytest.raises(ValidationError):
        RoutingDecision(reasoning="ok", next="refund_agent")
