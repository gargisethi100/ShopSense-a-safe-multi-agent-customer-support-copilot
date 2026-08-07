"""The human gate: freeze the run, show the refund, resume on a decision.

WHAT interrupt() ACTUALLY DOES
    Calling interrupt(payload) inside a node stops the graph. Not "waits" -
    STOPS. The state (including the payload) is written to Postgres by the
    checkpointer, .invoke() returns to the caller carrying the payload, and
    the Python process is free to exit. Hours later, a different process
    can call:

        graph.invoke(Command(resume=decision), config)   # same thread_id

    ...and execution continues. That is only possible because "where we
    are" is checkpointed data rather than a Python call stack - the
    property we bought back in graph/state.py.

THE GOTCHA THAT SHAPED THIS FILE: RESUME REPLAYS THE WHOLE NODE
    On resume, LangGraph re-runs the interrupted node FROM THE TOP. The
    interrupt() call itself returns the resume value the second time
    through, but everything BEFORE it happens again.

    So a node containing an interrupt must be cheap and safe to re-run.
    That is exactly why the refund tool no longer calls interrupt() from
    inside order_agent_node's reasoning loop: replaying that node would
    re-issue every LLM call in it - paying twice for the same thinking,
    on every approval.

    This node holds no LLM calls and no writes before the pause. Replaying
    it costs nothing. THE RULE: put interrupts in their own small node.

WHAT THE HUMAN SEES
    The payload is the RefundRequest we built in Phase 2 - customer, item,
    amount, reason, and the refund id that will end up in the table. An
    approver should never have to look anything up to decide.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from graph.state import ShopSenseState
from tools.refund_tool import RefundRequest, execute_refund


def refund_approval_node(state: ShopSenseState) -> dict:
    """Pause for a human, then either write the refund or decline it."""
    pending = state.get("pending_refund")
    if not pending:
        # Defensive: the router should never send us here empty-handed.
        return {"pending_refund": None}

    # ==== THE PAUSE ========================================================
    # Everything above this line re-runs on resume. Everything below it runs
    # only once, with `decision` filled in by whoever approved.
    decision = interrupt(
        {
            "type": "refund_approval",
            "prompt": (
                f"Approve a ${pending['amount_usd']} refund to "
                f"{pending['customer_name']} for order {pending['order_id']}?"
            ),
            "refund": pending,
        }
    )
    # =======================================================================

    approver = (decision or {}).get("approved_by") or "unknown"

    if not (decision or {}).get("approved"):
        note = (decision or {}).get("note") or "no reason given"
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"The refund request for order {pending['order_id']} was "
                        f"DECLINED by {approver} ({note}). No money has moved. "
                        "Explain this to the customer plainly and offer to "
                        "escalate to human support if they disagree."
                    )
                )
            ],
            "pending_refund": None,
            "gate_flags": [f"refund {pending['refund_id']} declined by {approver}"],
        }

    # Approved. This is the only line in the entire project that writes
    # money - and it cannot be reached without a human's name in hand.
    result = execute_refund(RefundRequest(**pending), approved_by=approver)
    return {
        "messages": [AIMessage(content=result)],
        "pending_refund": None,
        "gate_flags": [f"refund {pending['refund_id']} approved by {approver}"],
    }


def route_after_order_agent(state: ShopSenseState) -> str:
    """Conditional edge: does a refund need approving before we continue?"""
    return "refund_approval" if state.get("pending_refund") else "supervisor"
