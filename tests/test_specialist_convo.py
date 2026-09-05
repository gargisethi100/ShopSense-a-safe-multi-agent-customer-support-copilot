"""Eval: a specialist never sends the model a conversation it will refuse.

THE BUG THIS PINS
    Bedrock's Converse API rejects a request whose last message is from the
    assistant - "This model does not support assistant message prefill" -
    and the graph reaches that state on two ordinary paths: the order agent
    asking for an email and being routed back to, and the approval node
    handing a decided refund to a specialist to relay. Both produced a 500
    in the customer's face.

WHY IT IS IN THE FREE TIER
    specialist_convo is a pure function of state, so this whole file runs
    with no database, no Bedrock, and no API key - which means it gates
    EVERY push rather than only the ones with credentials. The rule it
    guards is a message-ordering invariant, and an invariant you can check
    without spending money should never be checked by spending money.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agents.common import specialist_convo
from graph.state import ShopSenseState

PROMPT = "You are a specialist."


def test_a_trailing_assistant_message_gets_answered():
    """The regression itself: the agent's own question, routed back to it."""
    state: ShopSenseState = {
        "messages": [
            HumanMessage(content="Where is my order?"),
            AIMessage(content="What's the email on the order?"),
        ]
    }
    convo = specialist_convo(state, PROMPT)

    assert not isinstance(convo[-1], AIMessage), (
        "Converse refuses a conversation ending in an assistant turn"
    )
    assert isinstance(convo[-1], HumanMessage)


def test_a_decided_refund_gets_answered():
    """The other route in: refund_approval hands back an AIMessage.

    Both branches of refund_approval_node return one - the confirmation, or
    the 'DECLINED by X, explain this plainly' instruction - and the money
    has ALREADY moved by then. A 500 here means the row exists and the
    customer was told nothing.
    """
    state: ShopSenseState = {
        "messages": [
            HumanMessage(content="refund ord_1003 please"),
            AIMessage(content="Refund ref_abc123 was DECLINED by dana. Explain this."),
        ]
    }
    assert not isinstance(specialist_convo(state, PROMPT)[-1], AIMessage)


def test_a_trailing_customer_message_is_left_alone():
    """The common case must not grow a message it does not need."""
    state: ShopSenseState = {"messages": [HumanMessage(content="Where is ord_1003?")]}
    convo = specialist_convo(state, PROMPT)

    assert convo[-1].content == "Where is ord_1003?"
    assert len(convo) == 2, "system prompt + the customer, and nothing else"


def test_a_trailing_tool_message_is_left_alone():
    """A tool result is already a user turn to Converse - no help needed."""
    state: ShopSenseState = {
        "messages": [
            HumanMessage(content="Where is ord_1003?"),
            AIMessage(content="", tool_calls=[
                {"name": "get_order_status", "args": {"order_id": "ord_1003"},
                 "id": "call_1"}
            ]),
            ToolMessage(content="status: delivered", tool_call_id="call_1"),
        ]
    }
    assert isinstance(specialist_convo(state, PROMPT)[-1], ToolMessage)


def test_the_memory_preamble_still_comes_before_the_transcript():
    """The splice order is load-bearing: context first, conversation after."""
    state: ShopSenseState = {
        "profile_summary": "Prefers email contact.",
        "summary": "Asked about ord_1003 earlier.",
        "messages": [HumanMessage(content="and the hoodie?")],
    }
    convo = specialist_convo(state, PROMPT)

    assert convo[0].content == PROMPT
    assert isinstance(convo[1], SystemMessage)
    assert "Prefers email contact." in convo[1].content
    assert convo[2].content == "and the hoodie?"


def test_state_messages_are_never_mutated():
    """The continuation turn is OUR working copy, not the transcript.

    If it leaked into state it would be checkpointed, replayed on every
    later turn, and eventually shown to the customer as something they
    said. Nodes return changes; they do not edit what they were given.
    """
    messages = [
        HumanMessage(content="Where is my order?"),
        AIMessage(content="What's the email on the order?"),
    ]
    state: ShopSenseState = {"messages": messages}

    convo = specialist_convo(state, PROMPT)

    assert len(messages) == 2, "the caller's list must be untouched"
    assert len(convo) == 4, "system + 2 transcript + the continuation turn"


# ---------------------------------------------------------------------------
# The other half: what the CUSTOMER is allowed to see
# ---------------------------------------------------------------------------


def test_internal_notes_are_marked_so_the_transcript_can_hide_them():
    """Regression: the refund gate's note was rendered to the customer.

    "APPROVED, confirm this to the customer" is an AIMessage with no tool
    calls, so nothing about its shape distinguishes it from a reply - and
    it appeared in the chat, reference id and approver's name and all,
    directly above the sentence written from it. The name is what
    api/main.py's transcript filters on, so this asserts the marking that
    the filter depends on.
    """
    from graph.state import INTERNAL

    note = AIMessage(content="Refund APPROVED for order ord_1003.", name=INTERNAL)
    reply = AIMessage(content="Great news - your refund has been approved.")

    # Exactly the predicate api/main.py uses to build the transcript.
    def customer_visible(m):
        return not m.tool_calls and m.name != INTERNAL

    assert not customer_visible(note), "internal notes must never be rendered"
    assert customer_visible(reply)


def test_a_specialist_still_sees_the_internal_note():
    """The flip side: hiding a note from the customer must not hide it from
    the agent, which needs it to know what the human decided."""
    from graph.state import INTERNAL

    note = AIMessage(content="Refund APPROVED for order ord_1003.", name=INTERNAL)
    state: ShopSenseState = {
        "messages": [HumanMessage(content="refund please"), note]
    }
    convo = specialist_convo(state, PROMPT)

    assert note in convo, "the agent cannot relay a decision it cannot see"
    assert not isinstance(convo[-1], AIMessage)
