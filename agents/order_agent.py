"""The order specialist: Claude + the order tools, looping until it can answer.

THE REACT LOOP (reason -> act -> observe -> repeat) - what "an agent" IS
    A model cannot run code. So a specialist is a LOOP that we run:

        1. send: conversation + tool catalogue        -> Claude
        2. Claude replies with either
             (a) TEXT              -> it's done; that's the answer, exit
             (b) TOOL CALLS        -> it wants data
        3. we execute those tools ourselves (real SQL against Postgres)
        4. we append the results as ToolMessages and go back to step 1

    Each pass costs one LLM call. A typical order question takes 2-3
    passes: ask -> call get_order_status -> read result -> write answer.
    That loop is the entire mechanism behind the word "agentic".

WHY THIS IS A NODE, NOT A CHATBOT
    The node owns the loop and returns only STATE CHANGES: the new
    messages, and one usage record per LLM call. The graph decides what
    happens next (back to the supervisor). Keeping the loop inside one
    node means the graph stays small and the loop stays testable.

THE THREE SAFETY BOUNDS
    * MAX_TOOL_ROUNDS - a model that keeps calling tools would bill
      forever. Bounded, and the bound is reported honestly, not hidden.
    * tools are the READ-ONLY set. The refund tool is bound here too, but
      it cannot write: it stops at the human-approval seam (Phase 5).
    * tool exceptions never escape - tools/order_tools.py returns errors
      as advice strings, so a database blip degrades the answer instead of
      crashing the conversation.

Run directly (needs a seeded database + Bedrock):

    python -m agents.order_agent
"""

from __future__ import annotations

import re

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import get_settings
from graph.memory import summary_preamble
from graph.state import ShopSenseState
from llm import get_llm, usage_from
from tools.order_tools import ORDER_TOOLS
from tools.refund_tool import REFUND_TOOLS, RefundRequest

# Every tool this specialist may hold. Notice what is absent: the policy
# search. A specialist that could do everything would be the single big
# agent we deliberately split up.
TOOLS = [*ORDER_TOOLS, *REFUND_TOOLS]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# One question should never need more than a handful of lookups. Three is
# generous for the real chains (email -> orders -> details) and cheap when
# a model gets stuck in a lookup rut.
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """You are the order specialist for an online store's support \
team. You have live database access to customers, orders, and delivery status, \
and you can submit refund requests for human approval.

How to work:
- Look things up. Never state an order status, date, tracking number, or \
amount that did not come from a tool result in this conversation.
- If the customer has not identified themselves and you need their account, \
ask for the email on the order - do not guess.
- Prefer one precise lookup over several speculative ones.

How to answer:
- Write to the customer directly, warmly and briefly. No preamble, no \
restating their question back to them.
- Give the specifics you found (status, date, tracking) - those are what the \
customer actually wants.
- If a tool reports something is not found or not allowed, say so plainly and \
give the customer the next step it suggests. Never invent a workaround.
- Refunds are not final until a human approves them. Say the request is \
submitted for approval; never tell a customer their money is on the way.
- You handle orders only. If the question is about POLICY (what the rules \
are - return windows, warranty coverage, shipping costs), say you'll hand it \
to the policy specialist rather than answering from memory."""


def order_agent_node(state: ShopSenseState) -> dict:
    """Run the ReAct loop until Claude writes an answer (or we cap it)."""
    settings = get_settings()
    llm = get_llm("agent").bind_tools(TOOLS)

    # The conversation as the model sees it: our instructions + the real
    # transcript. `state["messages"]` is never mutated - we build a local
    # working list and return only what's new.
    # summary_preamble injects the rolling summary and the customer's
    # profile as context. It is spliced in HERE rather than stored in
    # state["messages"], so it never becomes part of the transcript it
    # describes (and never gets summarised into itself).
    convo = [
        SystemMessage(content=SYSTEM_PROMPT),
        *summary_preamble(state),
        *(state.get("messages") or []),
    ]
    new_messages: list = []
    usage_records: list[dict] = []
    flags: list[str] = []
    pending_refund: dict | None = None
    customer_id: str | None = state.get("customer_id")

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        reply: AIMessage = llm.invoke(convo)

        u = usage_from(reply, settings.model_agent)
        usage_records.append(
            {
                "node": "order_agent",
                "model": settings.model_agent,
                "input_tokens": u.total_input,
                "output_tokens": u.output,
                "cost_usd": u.cost,
            }
        )
        convo.append(reply)
        new_messages.append(reply)

        # No tool calls -> this reply IS the answer. Exit.
        if not reply.tool_calls:
            break

        # Out of rounds but the model still wants tools: stop asking and
        # tell it so, in-band. It gets one final turn (the loop's extra
        # iteration) to answer from what it already has - which is a much
        # better customer experience than an empty response.
        if round_no == MAX_TOOL_ROUNDS:
            flags.append(f"order_agent: tool round cap ({MAX_TOOL_ROUNDS}) hit")
            stop = ToolMessage(
                content=(
                    "TOOL BUDGET EXHAUSTED for this turn. Answer the customer "
                    "with what you already know, and say plainly if something "
                    "could not be checked."
                ),
                tool_call_id=reply.tool_calls[0]["id"],
            )
            convo.append(stop)
            new_messages.append(stop)
            continue

        # Execute every requested tool. Claude may ask for several at once
        # (e.g. two order lookups); each result must come back tagged with
        # the id of the call it answers, or the model cannot match them up.
        for call in reply.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                # Defensive: the model named a tool we don't have. Tell it
                # so rather than crashing - it can correct itself.
                result = (
                    f"UNKNOWN TOOL '{call['name']}'. Available: "
                    f"{', '.join(TOOLS_BY_NAME)}."
                )
                flags.append(f"order_agent: hallucinated tool '{call['name']}'")
            else:
                # .invoke() runs Pydantic validation first, so a malformed
                # argument surfaces as a correctable error, not an exception
                # deep in psycopg.
                try:
                    result = tool.invoke(call["args"])
                except RuntimeError as e:
                    # OUR OWN config guards (e.g. a missing DB url). Not the
                    # model's fault and not retryable by rewording - say so,
                    # or the model wastes turns "fixing" correct arguments.
                    result = (
                        f"SYSTEM NOT CONFIGURED: {e} This is an operator "
                        "problem, not an argument problem. Do not retry this "
                        "tool. Apologise and offer human support."
                    )
                    flags.append(f"order_agent: config error on {call['name']}")
                except Exception as e:  # validation errors, mostly
                    result = (
                        f"TOOL ARGUMENT ERROR: {e}. Re-read the tool's "
                        "description and try again with corrected arguments."
                    )
                    flags.append(f"order_agent: bad args for {call['name']}")

            # A RefundRequest object (rather than a string) means the refund
            # tool VALIDATED a refund and is handing us a proposal. Park it
            # in state; the graph will route to the human-approval node
            # instead of returning to the supervisor.
            #
            # model_dump(mode="json") matters: the payload is checkpointed
            # to Postgres as JSON, and a raw Decimal amount is not JSON
            # serialisable. Pydantic renders it as a string here and parses
            # it back to Decimal on the way out - exact cents, intact.
            if isinstance(result, RefundRequest):
                pending_refund = result.model_dump(mode="json")
                result = (
                    f"REFUND PREPARED (id {result.refund_id}): "
                    f"${result.amount_usd:.2f} for order {result.order_id}. "
                    "It is now waiting for a human to approve it. Tell the "
                    "customer the request is submitted for review - do NOT "
                    "say the money is on its way."
                )

            # Capture the customer's identity the moment it is established.
            # This is what lets Phase 6's memory load their profile and
            # save it afterwards - and it is read from the TOOL RESULT (a
            # database fact), never from what the customer claimed.
            if call["name"] == "find_customer" and not customer_id:
                if m := re.search(r"customer_id: (cust_\d+)", str(result)):
                    customer_id = m.group(1)

            msg = ToolMessage(content=str(result), tool_call_id=call["id"])
            convo.append(msg)
            new_messages.append(msg)

        # Stop looping once a refund is parked: the human decides next, and
        # anything the model says now would pre-empt that decision.
        if pending_refund:
            break

    return {
        "messages": new_messages,
        "usage": usage_records,
        **({"gate_flags": flags} if flags else {}),
        **({"pending_refund": pending_refund} if pending_refund else {}),
        **({"customer_id": customer_id} if customer_id else {}),
    }


# ---------------------------------------------------------------------------
# Smoke test: real questions, real database, real Bedrock.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import psycopg

    from config import enable_utf8_console

    enable_utf8_console()  # the model's answers may contain emoji

    QUESTIONS = [
        "Where is my order ord_1003?",
        "Hi, it's dana@example.com - what have I ordered recently?",
        "Order ord_1004 arrived broken, I'd like a refund please.",
        "What's the status of ord_9999?",   # not found -> graceful handling
    ]

    print(f"agent model: {get_settings().model_agent}\n")
    total = 0.0
    for q in QUESTIONS:
        print(f"=== {q}")
        state: ShopSenseState = {"messages": [HumanMessage(content=q)]}
        try:
            out = order_agent_node(state)
        except (psycopg.Error, RuntimeError) as e:
            print(f"  DATABASE NOT READY: {type(e).__name__}: {str(e)[:120]}")
            print("  -> run the Neon setup + `python seed.py` first.\n")
            break

        # Show the loop: which tools ran, then the final answer.
        for m in out["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                for c in m.tool_calls:
                    print(f"  tool  : {c['name']}({c['args']})")
            elif isinstance(m, ToolMessage):
                first = str(m.content).splitlines()[0]
                print(f"  result: {first[:76]}")
        answer = out["messages"][-1]
        print(f"  ANSWER: {answer.text}\n")

        calls = len(out["usage"])
        cost = sum(r["cost_usd"] for r in out["usage"])
        total += cost
        print(f"  ({calls} LLM calls, ${cost:.4f})\n")

    print(f"total spend this run: ${total:.4f}")
