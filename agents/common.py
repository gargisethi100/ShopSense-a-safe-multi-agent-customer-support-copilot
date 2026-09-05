"""What every specialist sends to the model, built once.

WHY THIS FILE EXISTS AT ALL
    order_agent and policy_agent opened with the same four lines, character
    for character: a system prompt, the memory preamble, the transcript.
    Duplication that small is easy to live with right up to the moment it
    needs a fix - and then you fix one copy, ship, and discover the other
    one months later through a customer.

    That is exactly what happened here, so the construction now lives in
    one place and both specialists call it.

THE BUG IT CLOSES
    Bedrock's Converse API refuses a request whose last message is from the
    assistant:

        ValidationException: This model does not support assistant message
        prefill. The conversation must end with a user message.

    Our graph reaches that state on purpose, twice over:

      * THE CLARIFYING QUESTION. The order agent is told to ask rather than
        guess when it has no order id ("what's the email on the order?").
        It reports back, the supervisor routes to it again, and now the
        transcript ends with the agent's own question.

      * A DECIDED REFUND. refund_approval_node returns an AIMessage on both
        branches - the approval confirmation, or "DECLINED by X, explain
        this to the customer plainly" - and hands back to the supervisor.
        Routing that to a specialist to relay puts an assistant message
        last again.

    Neither is a misuse of the system; both are the design working. So the
    guarantee belongs here rather than in a prompt asking the router to
    please avoid it: a provider's message-ordering rule should never be
    able to 500 a customer's conversation.

WHY APPENDING IS SAFE
    The returned list is a LOCAL working copy. Nodes return only their new
    messages as state changes, so the continuation turn below never enters
    state["messages"], never reaches the checkpointer, and never shows up
    in the transcript the customer reads. This is the same trick
    summary_preamble already uses for the memory context - spliced in at
    call time, never stored.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.memory import summary_preamble
from graph.state import ShopSenseState

# Addressed to the specialist, not the customer. It follows either the
# agent's own last message or an instruction from the approval node, and in
# both cases the right next move is the same: say the thing, to the person
# waiting.
#
# THE SUPERSEDES CLAUSE IS NOT PADDING. A refund conversation carries two
# instructions that flatly contradict each other, both true when written:
#   1. from the refund tool - "the request is submitted for review, do NOT
#      say the money is on its way"
#   2. from the approval node, later - "APPROVED, confirm it is going
#      through"
# Without a rule about which wins, the model followed (1), because it is
# emphatic and shouts in capitals. Customers were told their refund was
# still pending several seconds after a human had approved and paid it.
CONTINUE = (
    "Continue from the message immediately above and write the reply the "
    "customer should now see. That message is the most recent state of "
    "this case and supersedes any earlier note in the conversation - where "
    "they disagree, the later one is true. Do not repeat a question the "
    "customer has already answered."
)


def specialist_convo(state: ShopSenseState, system_prompt: str) -> list:
    """The conversation as a specialist sends it to the model.

    Returns a fresh list every call, so the caller is free to append tool
    results to it through the ReAct loop without touching graph state.
    """
    convo: list = [
        SystemMessage(content=system_prompt),
        # Context, not conversation: the rolling summary and the customer's
        # profile. Spliced in here so it never gets summarised into itself.
        *summary_preamble(state),
        *(state.get("messages") or []),
    ]

    # The guarantee. A ToolMessage needs no help - Converse already carries
    # tool results as a user turn - so an AIMessage is the only last message
    # that has to be answered before we may speak again.
    if isinstance(convo[-1], AIMessage):
        convo.append(HumanMessage(content=CONTINUE))

    return convo
