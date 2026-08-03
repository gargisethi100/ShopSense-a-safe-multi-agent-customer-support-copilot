"""The refund tool - the ONLY code path in the project that writes.

WHY THIS TOOL IS SHAPED DIFFERENTLY FROM THE READ TOOLS
    Reads are reversible: a wrong SELECT costs nothing. A refund moves
    money, and moved money does not un-move. So this tool is split into
    three stages with a HUMAN GATE between the last two:

        1. prepare_refund()   validate everything, build an exact proposal
                              (which order, how much, why) - READS ONLY
        2.       ||           <- THE PAUSE. In Phase 5, LangGraph's
                 ||              interrupt() freezes the whole run here and
                 ||              shows the proposal to a human in the UI.
        3. execute_refund()   one INSERT, carrying the approver's name -
                              runs ONLY after a human clicked Approve.

    Until Phase 5 wires the pause, the tool stops after stage 1 and says
    so. THE STUB NEVER WRITES. A placeholder that "temporarily" auto-
    approves is exactly how scary defaults ship to production.

DECISIONS THAT KEEP MONEY SAFE (each maps to a line below)
    * The model NEVER chooses the amount. Refund = the order's total_usd,
      read from the database. An LLM-supplied number is a suggestion from
      an entity that talks to strangers; records beat suggestions.
    * The refund_id is generated BEFORE the pause, so the id the human
      approves, the id in the trace, and the id in the table are provably
      the same thing.
    * Validation reads use the READ-ONLY pool. Only the final INSERT (and
      the 'already refunded?' check - see roles.sql) touches the writer.
    * Every failure is RETURNED as words the model can relay or act on,
      never raised (same rule as order_tools.py).

Run directly for a full dress rehearsal against the seeded db:

    python -m tools.refund_tool
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import psycopg
from langchain_core.tools import tool
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from db.pool import get_ro_pool, get_writer_pool

_DB_ERROR = (
    "DATABASE ERROR: the refund system is temporarily unreachable. Do not "
    "retry more than once. Apologise and promise a follow-up."
)


class RefundRequest(BaseModel):
    """The exact proposal a human will approve or reject.

    Everything the approver needs to decide is IN this object - they should
    never have to go look something up. It doubles as the interrupt()
    payload in Phase 5 and renders directly in the approval panel.
    """

    refund_id: str
    order_id: str
    customer_id: str
    customer_name: str
    product_name: str
    amount_usd: Decimal
    reason: str


class RequestRefundArgs(BaseModel):
    order_id: str = Field(
        pattern=r"^ord_\d+$",
        description="The order to refund, format 'ord_' + digits, e.g. 'ord_1003'.",
    )
    reason: str = Field(
        min_length=5,
        max_length=300,
        description=(
            "The customer's reason in their own words, e.g. 'arrived with a "
            "cracked ear cup'. Shown verbatim to the human approver - do not "
            "editorialise or diagnose."
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1: validate + build the proposal (reads only, no side effects)
# ---------------------------------------------------------------------------


def prepare_refund(order_id: str, reason: str) -> RefundRequest | str:
    """All the ways to say no, before anyone is asked to say yes.

    Returns a RefundRequest when the refund is legitimate, or a plain
    string explaining the problem (which the tool passes to the model).
    Ordered cheapest-check-first; each rejection names the next move.
    """
    try:
        with get_ro_pool().connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            row = cur.execute(
                "SELECT o.id, o.customer_id, o.status, o.total_usd,"
                "       p.name AS product_name, c.name AS customer_name"
                " FROM orders o"
                " JOIN products p ON p.id = o.product_id"
                " JOIN customers c ON c.id = o.customer_id"
                " WHERE o.id = %s",
                (order_id,),
            ).fetchone()
    except psycopg.Error:
        return _DB_ERROR

    if row is None:
        return (
            f"NO ORDER FOUND with id '{order_id}'. Re-check the id with the "
            "customer or use list_recent_orders to locate the right order."
        )

    # Status policy: only DELIVERED orders are refundable through this tool.
    # Each other status gets its own message because each has a different
    # correct next step for the agent.
    if row["status"] == "processing":
        return (
            f"Order {order_id} is still processing - it has not been charged "
            "a delivery yet. A refund does not apply; the customer may want "
            "a cancellation, which this assistant cannot do - offer to have "
            "support cancel it."
        )
    if row["status"] == "shipped":
        return (
            f"Order {order_id} is still in transit. Refunds are issued after "
            "delivery. Ask the customer to wait for delivery, or offer to "
            "flag the shipment with support if it seems lost."
        )
    if row["status"] == "cancelled":
        return (
            f"Order {order_id} was cancelled. If the customer believes they "
            "were still charged, escalate to human support - do not issue a "
            "refund for a cancelled order."
        )

    # Double-refund guard. Note the WRITER pool: refunds history is invisible
    # to agent_ro by design, and refund_writer holds the only SELECT grant on
    # it (added to roles.sql for exactly this check).
    try:
        with get_writer_pool().connection() as conn:
            dup = conn.execute(
                "SELECT 1 FROM refunds WHERE order_id = %s", (order_id,)
            ).fetchone()
    except psycopg.errors.InsufficientPrivilege:
        # Developer-facing: the grant is missing, not the data.
        return (
            "CONFIGURATION ERROR: refund_writer cannot read the refunds "
            "table. Re-run seed.py (or roles.sql) to apply current grants."
        )
    except psycopg.Error:
        return _DB_ERROR

    if dup is not None:
        return (
            f"Order {order_id} has ALREADY been refunded. Do not refund it "
            "again. If the customer says the money never arrived, escalate "
            "to human support with the order id."
        )

    return RefundRequest(
        # Generated NOW, pre-approval, so UI, trace, and table share one id.
        refund_id=f"ref_{uuid4().hex[:12]}",
        order_id=row["id"],
        customer_id=row["customer_id"],
        customer_name=row["customer_name"],
        product_name=row["product_name"],
        amount_usd=row["total_usd"],  # the DATABASE's number, never the model's
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Stage 3: the one INSERT. Called only with a human's name in hand.
# ---------------------------------------------------------------------------


def execute_refund(request: RefundRequest, approved_by: str) -> str:
    """Record the approved refund. approved_by is the human's identity -
    the NOT NULL column in schema.sql exists to force this parameter."""
    try:
        with get_writer_pool().connection() as conn:
            conn.execute(
                "INSERT INTO refunds (id, order_id, amount_usd, reason, approved_by)"
                " VALUES (%s, %s, %s, %s, %s)",
                (
                    request.refund_id,
                    request.order_id,
                    request.amount_usd,
                    request.reason,
                    approved_by,
                ),
            )
            # with-block exit commits (writer pool is transactional).
    except psycopg.Error:
        return _DB_ERROR
    return (
        f"REFUND RECORDED: {request.refund_id} - ${request.amount_usd:.2f} for "
        f"order {request.order_id}, approved by {approved_by}. Tell the "
        "customer the refund is confirmed and typically lands in 5-10 "
        "business days."
    )


# ---------------------------------------------------------------------------
# The tool the agent sees
# ---------------------------------------------------------------------------


@tool("request_refund", args_schema=RequestRefundArgs)
def request_refund(order_id: str, reason: str) -> str:
    """Request a refund for a delivered order. A HUMAN must approve it.

    WHAT: Validates that the order is refundable and submits a refund
    proposal (full order amount) for human approval. This tool does NOT
    move money by itself.

    USE WHEN: The customer of a DELIVERED order wants their money back
    (damaged item, wrong item, not as described) and you have confirmed
    the order id belongs to them.

    DO NOT USE: For orders still processing or in transit (check with
    get_order_status first), to cancel an order (not supported - offer
    support escalation), or a second time for the same order.

    PREREQUISITES: The order id (verify via get_order_status), and the
    customer's stated reason.

    RETURNS: Either a rejection with the reason and the correct next step,
    or confirmation that the refund is awaiting human approval - tell the
    customer approval is pending, and do NOT promise the money is sent.
    """
    prepared = prepare_refund(order_id, reason)
    if isinstance(prepared, str):
        return prepared

    # =======================================================================
    # PHASE 5 SEAM - THE HUMAN GATE GOES EXACTLY HERE.
    #
    # When the graph exists, this becomes:
    #
    #     decision = interrupt(prepared.model_dump())   # freezes the run
    #     if decision["approved"]:
    #         return execute_refund(prepared, decision["approved_by"])
    #     return f"Refund DECLINED by {decision['approved_by']}: ..."
    #
    # Until then the tool stops here, honestly. It does NOT auto-approve:
    # a stub that "temporarily" writes is a production incident on layaway.
    # =======================================================================
    return (
        f"REFUND PREPARED (id {prepared.refund_id}) for order "
        f"{prepared.order_id}: ${prepared.amount_usd:.2f} to "
        f"{prepared.customer_name} - reason: {prepared.reason}. "
        "STATUS: awaiting human approval. The approval flow is not wired "
        "yet (arrives in Phase 5); no money has moved."
    )


REFUND_TOOLS = [request_refund]


# ---------------------------------------------------------------------------
# Smoke test: every rejection branch, then a REAL write (cleaned up after).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== rejection branches ===")
    for label, oid in [
        ("missing order", "ord_9999"),
        ("still processing", "ord_1004"),
        ("still in transit", "ord_1002"),
        ("cancelled", "ord_1005"),
    ]:
        print(f"\n-- {label} ({oid})")
        print(request_refund.invoke({"order_id": oid, "reason": "item arrived damaged"}))

    print("\n=== the tool via the agent path (stub - must NOT write) ===")
    print(request_refund.invoke(
        {"order_id": "ord_1003", "reason": "left ear cup crackles at low volume"}
    ))

    print("\n=== full pipeline rehearsal (prepare -> 'approve' -> execute) ===")
    prepared = prepare_refund("ord_1003", "left ear cup crackles at low volume")
    assert isinstance(prepared, RefundRequest), prepared
    print(f"prepared : {prepared.refund_id} ${prepared.amount_usd:.2f}")
    print(execute_refund(prepared, approved_by="smoke-test-human"))

    print("\n-- double-refund guard must now fire for ord_1003:")
    print(prepare_refund("ord_1003", "second attempt"))

    # Cleanup so the fixture stays pristine. Only ADMIN can delete a refund -
    # neither runtime role can - which is itself the design working. This is
    # a test-only act; production refund corrections are compensating rows.
    from db.pool import admin_connection

    with admin_connection() as conn:
        conn.execute("DELETE FROM refunds WHERE id = %s", (prepared.refund_id,))
    print(f"\ncleanup: {prepared.refund_id} deleted via admin (test-only power).")
