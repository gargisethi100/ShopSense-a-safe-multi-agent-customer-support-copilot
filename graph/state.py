"""The graph's shared state - the one object every node reads and writes.

WHAT A "GRAPH" IS (the 60-second version, because it starts here)
    From Phase 4 on, the app is a flowchart: boxes called NODES (input
    gate, supervisor, order agent, policy agent, output rail) connected by
    arrows called EDGES. LangGraph runs the flowchart.

    Nodes do NOT call each other. Each node is a plain function with one
    job: read the state, do its work, return the FIELDS IT CHANGED. The
    framework merges those changes into the state and hands the updated
    state to the next node. That indirection is what buys us:
      * checkpointing - the state is a plain dict, so it can be saved to
        Postgres after every step and reloaded on the next message
      * interrupt() - a run can be frozen and resumed, because "where we
        are" is data, not a Python call stack
      * testability - a node is a pure-ish function you can call with a
        hand-made state and assert on the result

THE ONE RULE FOR STATE DESIGN
    Put in state ONLY what more than one node needs, or what must survive
    a restart. Everything else is a local variable. State is saved to the
    database on every step, so a junk field is junk you pay to store, sync,
    and reason about forever.

Run directly to see a state evolve exactly the way the graph will:

    python -m graph.state
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Where the supervisor can send the conversation next. FINISH means "the
# answer is ready; stop looping". Keeping this as a named type (not loose
# strings) means the supervisor, the edges, and the routing evals in Phase
# 10 all agree on the vocabulary - a typo becomes a type error, not a
# mysterious dead end.
Route = Literal["order_agent", "policy_agent", "FINISH"]


class UsageRecord(TypedDict):
    """One LLM call's cost, recorded as it happens.

    Built from llm.usage_from() at each call site. We store the small flat
    numbers rather than the CallUsage object because state gets serialised
    to JSON in Postgres - plain data survives that trip, objects don't.
    """

    node: str          # who made the call ("supervisor", "order_agent", ...)
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ShopSenseState(TypedDict, total=False):
    """The conversation, as data.

    `total=False` means every key is optional - nodes return only what they
    changed, and LangGraph merges. A node that only routes returns
    {"route": ...} and nothing else.
    """

    # --- the conversation ---------------------------------------------
    # THE ONLY FIELD WITH SPECIAL MERGE BEHAVIOUR, and the most important
    # idea in this file. By default, a returned field REPLACES the old
    # value. For messages that would be a disaster: the order agent
    # returning its one reply would erase the whole conversation.
    # Annotated[..., add_messages] tells LangGraph to APPEND instead - and
    # add_messages is smarter than a plain list concat: it de-duplicates by
    # message id, so a node that returns a message the state already has
    # doesn't double it. This is called a REDUCER: a function that decides
    # how new values combine with old ones.
    messages: Annotated[list[AnyMessage], add_messages]

    # --- who we're talking to -----------------------------------------
    # Set once the customer identifies themselves (find_customer). Phase 6
    # uses it to load and save the per-customer profile; the refund tool
    # will use it to check the order actually belongs to this person.
    customer_id: str | None

    # Long-lived facts recalled from user_profiles at session start
    # ("prefers email"). Phase 6 writes it; the agents read it.
    profile_summary: str | None

    # --- routing -------------------------------------------------------
    # The supervisor's decision for THIS hop. Deliberately overwritten each
    # time (no reducer): it is a signpost, not a history. The history is
    # already in `messages`, and in LangSmith.
    route: Route | None

    # --- memory (Phase 6) ----------------------------------------------
    # Rolling summary of older turns, once the conversation outgrows the
    # threshold. Replaced (not appended) on each summarisation - a summary
    # of a summary is the point.
    summary: str | None

    # --- accounting (Phase 8) ------------------------------------------
    # Appended to on every LLM call so the UI can show
    # "6 calls / 9,412 tokens / ~$0.02" for THIS conversation.
    # Note the reducer: `operator.add` on lists means concatenate. We spell
    # it as a lambda-free import below for readability.
    usage: Annotated[list[UsageRecord], lambda old, new: (old or []) + (new or [])]

    # --- guardrails (Phase 7) ------------------------------------------
    # Set by the input gate when it blocks or masks something, so the
    # output rail and the trace log can see WHY a turn went the way it did.
    gate_flags: Annotated[list[str], lambda old, new: (old or []) + (new or [])]

    # True when the input gate refused the message outright. Replaced (not
    # appended) every turn: it describes THIS message, not the history.
    gate_blocked: bool | None

    # --- human-in-the-loop (Phase 5) -----------------------------------
    # The pending RefundRequest, as a plain dict, while the graph is
    # paused at interrupt(). The UI renders this into the approval panel.
    # None whenever no approval is outstanding.
    pending_refund: dict | None


# ---------------------------------------------------------------------------
# Helpers - small, so that nodes stay about their own logic.
# ---------------------------------------------------------------------------


def initial_state(user_message: str, *, customer_id: str | None = None) -> ShopSenseState:
    """The state a brand-new conversation starts from.

    Only the keys we actually know. Everything else is absent, which under
    `total=False` is the same as "not set yet" - the graph fills them in.
    """
    from langchain_core.messages import HumanMessage

    state: ShopSenseState = {"messages": [HumanMessage(content=user_message)]}
    if customer_id:
        state["customer_id"] = customer_id
    return state


def usage_totals(state: ShopSenseState) -> tuple[int, int, int, float]:
    """(calls, input_tokens, output_tokens, dollars) for the cost footer."""
    records = state.get("usage") or []
    return (
        len(records),
        sum(r["input_tokens"] for r in records),
        sum(r["output_tokens"] for r in records),
        sum(r["cost_usd"] for r in records),
    )


def format_cost_footer(state: ShopSenseState) -> str:
    """The line the README promises: '6 LLM calls - 9,412 tokens - ~$0.02'."""
    calls, tin, tout, usd = usage_totals(state)
    return f"{calls} LLM calls - {tin + tout:,} tokens - ~${usd:.4f}"


# ---------------------------------------------------------------------------
# Smoke test: prove the merge rules by hand, before any node exists.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_core.messages import AIMessage

    # LangGraph applies reducers internally; we call them directly here so
    # you can SEE the two behaviours side by side.
    from langgraph.graph.message import add_messages as _add

    print("=== messages: APPEND (reducer) ===")
    state = initial_state("where is order ord_1003?", customer_id="cust_001")
    print(f"  after user turn      : {len(state['messages'])} message(s)")
    merged = _add(state["messages"], [AIMessage(content="Let me check that.")])
    print(f"  after agent replies  : {len(merged)} message(s)  <- appended, not replaced")
    merged2 = _add(merged, [AIMessage(content="Let me check that.", id=merged[-1].id)])
    print(f"  same message re-sent : {len(merged2)} message(s)  <- de-duplicated by id")

    print("\n=== route: REPLACE (no reducer) ===")
    state["route"] = "order_agent"
    print(f"  supervisor decides   : {state['route']}")
    state["route"] = "FINISH"
    print(f"  next hop overwrites  : {state['route']}  <- a signpost, not a history")

    print("\n=== usage: APPEND, and the footer it produces ===")
    state["usage"] = [
        {"node": "supervisor", "model": "haiku", "input_tokens": 420,
         "output_tokens": 3, "cost_usd": 0.00044},
        {"node": "order_agent", "model": "sonnet", "input_tokens": 1310,
         "output_tokens": 96, "cost_usd": 0.00537},
        {"node": "order_agent", "model": "sonnet", "input_tokens": 1502,
         "output_tokens": 141, "cost_usd": 0.00662},
    ]
    print(f"  totals               : {usage_totals(state)}")
    print(f"  footer               : {format_cost_footer(state)}")

    print("\n=== what the checkpointer will store ===")
    for key in ("messages", "customer_id", "route", "usage", "summary",
                "pending_refund", "gate_flags", "profile_summary"):
        present = key in state
        print(f"  {key:16} {'set' if present else 'not set (absent = not yet known)'}")
