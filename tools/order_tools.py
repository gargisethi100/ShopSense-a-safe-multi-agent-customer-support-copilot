"""The order agent's toolbelt: three read-only tools over the store data.

WHAT A "TOOL" ACTUALLY IS
    A plain Python function, plus a DESCRIPTION OF ITSELF that gets sent to
    the model with every request. Claude never runs Python; it emits
    "call get_order_status with order_id='ord_1003'", OUR code runs the
    function, and the return value is handed back as text for Claude's
    next turn. So a tool is two prompts wrapped around a capability:

        docstring  = the prompt that decides IF and WHEN the model calls it
        return str = the prompt the model reads AFTER, to compose its answer

    Write both for a reader who has zero context and cannot ask questions.

THE FIVE-PART DOCSTRING (used by every tool below)
    WHAT           one sentence, what the capability is
    USE WHEN       positive triggers, phrased like the user requests
    DO NOT USE     the confusable cases, each pointing at the RIGHT sibling
                   tool - this line is what prevents wrong-tool selection
    PREREQUISITES  what the model must already have (and how to get it)
    RETURNS        the exact shapes of success AND every edge case, so the
                   model is never surprised by what comes back

RULES THIS FILE FOLLOWS
    * Read-only pool ONLY (db.pool.get_ro_pool -> role agent_ro). Even a
      bug here cannot write: the keycard says no.
    * Every SQL statement is parameterized (%s). These argument values are
      produced by an LLM that talks to strangers - values must never be
      glued into SQL text.
    * NO EXCEPTION ESCAPES A TOOL. A raised exception crashes the agent
      loop mid-conversation; a returned sentence becomes information the
      model can act on ("not found - try X instead"). Errors here are
      RETURNED as recovery advice, never raised.
    * Results are labeled text, not JSON. Models read "status: shipped"
      more reliably than nested braces, and labels survive truncation.

Run directly to exercise every tool + edge case against the seeded db:

    python -m tools.order_tools
"""

from __future__ import annotations

import psycopg
from langchain_core.tools import tool
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from db.pool import get_ro_pool

# One shared sentence for the "database is down" case: honest, and it gives
# the model a next move instead of leaving it to improvise excuses.
_DB_ERROR = (
    "DATABASE ERROR: the order system is temporarily unreachable. "
    "Do not retry more than once. Apologise to the customer and suggest "
    "they try again in a few minutes."
)


# ---------------------------------------------------------------------------
# Argument schemas (Pydantic)
#
# These are not bureaucracy - they are the third prompt. The field types,
# descriptions, and patterns below are ALL sent to the model, and validation
# runs BEFORE our function: a malformed argument is bounced back to the
# model as a correction ("string should match pattern...") instead of
# reaching the database as garbage.
# ---------------------------------------------------------------------------


class FindCustomerArgs(BaseModel):
    email: str = Field(
        description=(
            "The customer's full email address exactly as they gave it, "
            "e.g. 'dana@example.com'. Case does not matter."
        ),
        min_length=3,
    )


class GetOrderStatusArgs(BaseModel):
    order_id: str = Field(
        # The pattern teaches the model the id format AND rejects anything
        # else before it touches SQL. 'ORD 1003', 'my order' etc. bounce
        # back as validation errors the model can self-correct from.
        pattern=r"^ord_\d+$",
        description="The order id, format 'ord_' + digits, e.g. 'ord_1003'.",
    )


class ListRecentOrdersArgs(BaseModel):
    customer_id: str = Field(
        pattern=r"^cust_\d+$",
        description=(
            "The customer id, format 'cust_' + digits, e.g. 'cust_001'. "
            "Get it from find_customer if you only have an email."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many of the most recent orders to return (1-20).",
    )


# ---------------------------------------------------------------------------
# Formatting helpers - the "second prompt" factory
# ---------------------------------------------------------------------------


def _fmt_order(row: dict) -> str:
    """One order as labeled lines. NULLs become words, never 'None'.

    A raw None leaks a programming concept into the conversation; the model
    might even echo it to the customer. 'not yet shipped' is an answer.
    """
    tracking = row["tracking_no"] or "not yet assigned"
    delivered = (
        row["delivered_at"].strftime("%b %d, %Y")
        if row["delivered_at"]
        else "not yet delivered"
    )
    return (
        f"order_id: {row['id']}\n"
        f"  product: {row['product_name']} (qty {row['quantity']})\n"
        f"  total: ${row['total_usd']:.2f}\n"
        f"  status: {row['status']}\n"
        f"  tracking: {tracking}\n"
        f"  ordered: {row['ordered_at'].strftime('%b %d, %Y')}\n"
        f"  delivered: {delivered}"
    )


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


@tool("find_customer", args_schema=FindCustomerArgs)
def find_customer(email: str) -> str:
    """Look up a customer account by email address.

    WHAT: Returns the customer's id, name, and contact details.

    USE WHEN: The customer has identified themselves by email and you need
    their customer_id to look anything else up. This is usually the FIRST
    tool of a conversation.

    DO NOT USE: To look up orders (use list_recent_orders once you have the
    customer_id) or a specific order's status (use get_order_status).

    PREREQUISITES: None. This is the entry point.

    RETURNS: 'customer_id: ... / name: ... / email: ...' on success.
    'NO CUSTOMER FOUND ...' if the email is not registered - in that case
    ask the customer to double-check the spelling; do NOT guess or retry
    with variations you invented.
    """
    try:
        with get_ro_pool().connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            row = cur.execute(
                "SELECT id, name, email, phone FROM customers"
                " WHERE lower(email) = lower(%s)",
                (email.strip(),),
            ).fetchone()
    except psycopg.Error:
        return _DB_ERROR

    if row is None:
        return (
            f"NO CUSTOMER FOUND for email '{email}'. Ask the customer to "
            "double-check the spelling. Do not invent alternative spellings."
        )
    phone = row["phone"] or "not on file"
    return (
        f"customer_id: {row['id']}\n"
        f"  name: {row['name']}\n"
        f"  email: {row['email']}\n"
        f"  phone: {phone}"
    )


@tool("get_order_status", args_schema=GetOrderStatusArgs)
def get_order_status(order_id: str) -> str:
    """Get the full current status of ONE specific order.

    WHAT: Returns product, amount paid, status, tracking, and dates for a
    single order id.

    USE WHEN: The customer mentions a specific order id ('where is
    ord_1003?'), or you already found the id via list_recent_orders and
    need its details.

    DO NOT USE: When you only have an email (find_customer first) or when
    the customer says 'my orders' without an id (list_recent_orders).

    PREREQUISITES: A concrete order id in the 'ord_NNNN' format.

    RETURNS: Labeled lines (order_id / product / total / status / tracking /
    ordered / delivered). status is one of: processing, shipped, delivered,
    cancelled. Fields that do not apply yet read 'not yet ...'.
    'NO ORDER FOUND ...' if the id does not exist - re-check the id with
    the customer or use list_recent_orders to find the right one.
    """
    try:
        with get_ro_pool().connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            row = cur.execute(
                "SELECT o.id, o.quantity, o.total_usd, o.status, o.tracking_no,"
                " o.ordered_at, o.delivered_at, p.name AS product_name"
                " FROM orders o JOIN products p ON p.id = o.product_id"
                " WHERE o.id = %s",
                (order_id,),
            ).fetchone()
    except psycopg.Error:
        return _DB_ERROR

    if row is None:
        return (
            f"NO ORDER FOUND with id '{order_id}'. Either re-check the id "
            "with the customer, or call list_recent_orders with their "
            "customer_id to see their actual orders."
        )
    return _fmt_order(row)


@tool("list_recent_orders", args_schema=ListRecentOrdersArgs)
def list_recent_orders(customer_id: str, limit: int = 5) -> str:
    """List a customer's most recent orders, newest first.

    WHAT: A compact list of the customer's orders with id, product, status,
    and date - enough to identify WHICH order the customer means.

    USE WHEN: The customer talks about their orders without giving an id
    ('where's my stuff?', 'my last order'), or an id they gave was not
    found and you need to show them their real ones.

    DO NOT USE: For full details of one known order (get_order_status gives
    tracking and delivery dates) or to find the customer_id itself
    (find_customer).

    PREREQUISITES: The customer_id ('cust_NNN'), from find_customer.

    RETURNS: 'Showing N of M orders ...' followed by one line per order:
    id, product, status, date. If M > N the remainder is summarised, not
    hidden - mention to the customer that older orders exist.
    'NO ORDERS FOUND ...' if the customer exists but has never ordered.
    """
    try:
        with get_ro_pool().connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            total = cur.execute(
                "SELECT count(*) AS n FROM orders WHERE customer_id = %s",
                (customer_id,),
            ).fetchone()["n"]
            rows = cur.execute(
                "SELECT o.id, o.status, o.ordered_at, p.name AS product_name"
                " FROM orders o JOIN products p ON p.id = o.product_id"
                " WHERE o.customer_id = %s"
                " ORDER BY o.ordered_at DESC"
                " LIMIT %s",
                (customer_id, limit),
            ).fetchall()
    except psycopg.Error:
        return _DB_ERROR

    if total == 0:
        return (
            f"NO ORDERS FOUND for customer '{customer_id}'. If the customer "
            "insists they ordered, they may have used a different account "
            "email - ask which email the order confirmation went to."
        )

    # HONEST TRUNCATION: always say how much of the whole this is. A model
    # shown 5 rows with no count will confidently tell the customer they
    # have exactly 5 orders.
    lines = [f"Showing {len(rows)} of {total} orders (newest first):"]
    for r in rows:
        lines.append(
            f"  {r['id']}: {r['product_name']} - {r['status']}"
            f" - ordered {r['ordered_at'].strftime('%b %d, %Y')}"
        )
    if total > len(rows):
        lines.append(
            f"  ...and {total - len(rows)} older order(s) not shown. Mention "
            "this; call again with a higher limit if the customer asks."
        )
    return "\n".join(lines)


# All three, in the order the agent will receive them (Phase 4 imports this).
ORDER_TOOLS = [find_customer, get_order_status, list_recent_orders]


# ---------------------------------------------------------------------------
# Smoke test: every tool, happy path AND edge cases. No LLM involved -
# .invoke() calls the tool exactly the way the agent framework will.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("find_customer / known email", find_customer, {"email": "DANA@example.com"}),
        ("find_customer / unknown email", find_customer, {"email": "nobody@example.com"}),
        ("get_order_status / real order", get_order_status, {"order_id": "ord_1003"}),
        ("get_order_status / missing order", get_order_status, {"order_id": "ord_9999"}),
        ("list_recent_orders / cust_001", list_recent_orders, {"customer_id": "cust_001"}),
        ("list_recent_orders / truncation", list_recent_orders,
         {"customer_id": "cust_002", "limit": 1}),
        ("list_recent_orders / no orders... unknown cust", list_recent_orders,
         {"customer_id": "cust_999"}),
    ]
    for label, t, args in cases:
        print(f"\n=== {label} ===")
        print(t.invoke(args))

    # And one validation bounce: what the MODEL would see if it sent a
    # malformed id. Note this never reached our function or the database.
    print("\n=== get_order_status / malformed id (schema rejects) ===")
    try:
        get_order_status.invoke({"order_id": "ORD 1003"})
    except Exception as e:
        print(f"validation error returned to the model:\n{str(e)[:200]}")
