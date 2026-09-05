"""The HTTP API: ShopSense as a service other programs can call.

WHY WRAP THE GRAPH IN AN API AT ALL
    Without this file the only way to talk to ShopSense is a Python CLI
    that has to BE the application: it imports the graph, opens the
    database pools, and holds everything in one process. That is fine
    for a demo and wrong for a product:

      * a mobile app, a website widget, or a Slack bot cannot `import
        graph.build` - they can only make HTTP requests
      * the UI and the brain cannot be scaled, deployed, or restarted
        separately when they are the same process
      * nothing else can reuse the system - an API is the difference
        between a demo and a component

    So this file exposes the same graph over HTTP. Nothing about the
    agents changes; we are adding a door, not a room.

WHAT AN API IS (from zero)
    A web API is a set of URLs a program can call, sending and receiving
    JSON instead of clicking buttons. Each URL is an ENDPOINT, and each
    has a METHOD describing intent:

        GET   /health              "give me something"  (no side effects)
        POST  /chat                "do something"       (changes state)

    FastAPI turns Python functions into endpoints, validates the incoming
    JSON against Pydantic models, and generates live documentation. Visit
    /docs on a running server and you get a page where you can try every
    endpoint by hand - which is how you should test this file.

THE STATELESSNESS RULE
    This server keeps NO conversation in memory. Every request carries a
    thread_id, and the conversation lives in Postgres behind the
    checkpointer. That is what lets you run ten copies of this API behind
    a load balancer: request 1 can land on one server and request 2 on
    another, and the conversation still works. An API that remembers
    things in a Python dict cannot be scaled without losing them.

Run it:
    uvicorn api.main:app --reload
    then open http://localhost:8000/docs
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph.build import build_graph, get_checkpointer
from graph.memory import save_profile
from graph.state import INTERNAL, elapsed_seconds, usage_totals
from obs.costlog import load_runs, record_run, trace_config, tracing_status
from rag.retriever import load_chunks

# Module-level handles, filled once at startup by the lifespan below.
GRAPH: Any = None
POLICY_INDEX: list[dict] = []

# The browser UI lives next to this package, not inside it. Resolved from
# __file__ rather than the working directory: uvicorn is started from the
# repo root locally and from /app in the container, and a relative path
# would silently serve nothing in one of them.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown, run ONCE for the whole server process.

    Building the graph opens database connection pools and constructs the
    Bedrock client. Doing that per REQUEST would add seconds to every call
    (we measured exactly this: client construction alone was 5.2s) and
    would open a new pool per request until the database refused
    connections.

    Everything before `yield` runs at startup; everything after runs at
    shutdown. This is FastAPI's replacement for the older @app.on_event
    decorators.
    """
    global GRAPH, POLICY_INDEX
    GRAPH = build_graph(checkpointer=get_checkpointer())
    # The citation index: section_id -> title + file. Built once because the
    # docs are read-only at runtime, and the UI needs it on every page load
    # to turn a bare "[RET-1]" in an answer into a named rule.
    POLICY_INDEX = [
        {"section_id": c.section_id, "title": c.title, "source": c.source}
        for c in load_chunks()
    ]
    yield
    # Pools are closed by db.pool's atexit handler; nothing to do here yet.


app = FastAPI(
    title="ShopSense Support API",
    description=(
        "A multi-agent customer-support service. Ask about orders and "
        "policies; refunds pause for human approval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
#
# These are not paperwork - they are the CONTRACT. FastAPI uses them to
# validate incoming JSON (a bad request gets a precise 422 error before our
# code runs), to serialise responses, and to generate the /docs page. One
# definition, three jobs - the same argument as the tool schemas in Phase 2.
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="What the customer said.",
        examples=["Where is my order ord_1003?"],
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation id. Omit it to start a new conversation - the "
            "response tells you the id to send next time."
        ),
    )


class PendingApproval(BaseModel):
    """Returned when the graph paused for a human decision.

    Everything here comes straight off the parked RefundRequest. The
    approver is being asked to authorise money leaving the business, so
    the card they see names WHAT, for WHOM, and WHY - an approval screen
    that shows only an amount is a rubber stamp with extra steps.
    """

    refund_id: str
    order_id: str
    customer_id: str
    customer_name: str
    product_name: str
    amount_usd: str
    reason: str
    prompt: str


class ChatResponse(BaseModel):
    thread_id: str
    answer: str | None = Field(
        description="The reply to show the customer. Null while awaiting approval."
    )
    pending_approval: PendingApproval | None = Field(
        default=None,
        description=(
            "Present when the run is FROZEN awaiting a human decision. "
            "Call POST /approve with the same thread_id to resume it."
        ),
    )
    llm_calls: int
    tokens: int
    cost_usd: float
    seconds: float


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    approved_by: str = Field(
        min_length=1,
        description=(
            "Who decided. In production this comes from the authenticated "
            "session, never from the request body - the refunds table "
            "requires a name because someone must be accountable."
        ),
    )
    note: str = Field(default="", max_length=500)


class Message(BaseModel):
    role: str
    text: str


class ConversationState(BaseModel):
    """Where a conversation stands right now, without running anything.

    The UI needs this for two moments POST /chat cannot cover: restoring a
    page that was reloaded mid-approval, and noticing that someone else
    has since made the decision. Both are reads of the checkpointer, so
    neither costs a model call.
    """

    thread_id: str
    exists: bool
    held: bool = Field(
        description="True while the run is frozen awaiting a human decision."
    )
    pending_approval: PendingApproval | None = None
    customer_id: str | None = None
    gate_blocked: bool = False


class PolicySection(BaseModel):
    """One citable rule. The UI resolves '[RET-1]' against these."""

    section_id: str
    title: str
    source: str


# ---------------------------------------------------------------------------
# Endpoints
#
# NOTE THE MISSING `async`. FastAPI supports both `def` and `async def`:
#   async def -> runs ON the event loop. Correct only if every slow call
#                inside is awaited, or the whole server stalls.
#   def       -> FastAPI runs it in a THREAD POOL, so slow blocking work
#                cannot freeze other requests.
# Our graph is synchronous (LangGraph .invoke, psycopg, boto3), so plain
# `def` is the correct choice. Writing `async def` here would be the
# classic FastAPI performance bug: one customer's 10-second conversation
# blocking every other customer's request.
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check for the load balancer.

    DELIBERATELY DOES NOT TOUCH the database or Bedrock. A health check
    that calls its dependencies turns one slow dependency into a restart
    loop: the check times out, the platform kills a perfectly healthy
    container, the replacement checks the same slow dependency, and you
    have turned a degradation into an outage.

    It answers one question only: is this process alive and serving?
    """
    return {"status": "ok", "tracing": tracing_status()}


@app.post("/chat", response_model=ChatResponse, tags=["conversation"])
def chat(req: ChatRequest) -> ChatResponse:
    """Send a customer message; get an answer, or a pause for approval."""
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    config = trace_config(thread_id)

    # Read the usage counters BEFORE running, so we can log this turn's
    # delta rather than the conversation's running total. Reading them
    # from the checkpointer (not an in-process counter) is what keeps this
    # correct when the server has several workers or gets restarted
    # mid-conversation.
    before = GRAPH.get_state(config).values
    since_usage = len(before.get("usage") or [])
    since_timings = len(before.get("timings") or [])

    result = GRAPH.invoke({"messages": [HumanMessage(content=req.message)]}, config)
    return _to_response(result, thread_id, since_usage, since_timings)


@app.post("/approve", response_model=ChatResponse, tags=["conversation"])
def approve(req: ApprovalRequest) -> ChatResponse:
    """Resume a frozen run with a human's decision.

    THIS IS WHERE interrupt() PAYS OFF OVER HTTP. The graph stopped, its
    state was written to Postgres, and the earlier request returned. This
    call can arrive minutes later, from a different device, hitting a
    different server instance - and the run continues from exactly where
    it paused. That is only possible because "where we are" is
    checkpointed data rather than a Python call stack.
    """
    config = trace_config(req.thread_id)
    state = GRAPH.get_state(config)
    if not state.values:
        raise HTTPException(404, f"No conversation with thread_id {req.thread_id!r}")
    if _pending_for(state) is None:
        raise HTTPException(
            409,
            "This conversation is not waiting for an approval. Nothing to resume.",
        )

    since_usage = len(state.values.get("usage") or [])
    since_timings = len(state.values.get("timings") or [])

    result = GRAPH.invoke(
        Command(resume={
            "approved": req.approved,
            "approved_by": req.approved_by,
            "note": req.note,
        }),
        config,
    )
    return _to_response(result, req.thread_id, since_usage, since_timings)


@app.get("/conversations/{thread_id}", response_model=list[Message], tags=["conversation"])
def transcript(thread_id: str) -> list[Message]:
    """The conversation so far, straight from the checkpointer.

    Tool calls, tool results, and the refund gate's internal notes are all
    filtered out: they are machinery, not conversation, and belong in the
    trace rather than in front of a customer.

    The notes are the subtle one. "APPROVED, confirm this to the customer"
    is an AIMessage like any other, so nothing about its TYPE says to hide
    it - and for a while the customer read it, reference id and all, sitting
    just above the reply written from it. Hence the name stamped on it.
    """
    values = GRAPH.get_state(trace_config(thread_id)).values
    if not values:
        raise HTTPException(404, f"No conversation with thread_id {thread_id!r}")

    out: list[Message] = []
    for m in values.get("messages", []):
        if isinstance(m, HumanMessage):
            out.append(Message(role="customer", text=m.text))
        elif isinstance(m, AIMessage) and not m.tool_calls and m.name != INTERNAL:
            out.append(Message(role="assistant", text=m.text))
    return out


@app.get(
    "/conversations/{thread_id}/state",
    response_model=ConversationState,
    tags=["conversation"],
)
def conversation_state(thread_id: str) -> ConversationState:
    """Is this conversation frozen, and if so, on what?

    WHY THE UI CANNOT LIVE WITHOUT THIS. POST /chat tells you a run froze,
    but only the caller that froze it, and only once. Two things happen
    afterwards that no chat response can report:

      * the customer reloads the page - the pending approval is in
        Postgres, not in their tab, and must be read back
      * someone else approves it from another device - the customer's page
        has to find out, and polling a read is how

    Both are reads of the checkpointer, so neither costs a model call.
    Unknown threads answer exists=false rather than 404: a stale id in a
    browser is an ordinary Monday, not an error worth a red log line.
    """
    state = GRAPH.get_state(trace_config(thread_id))
    if not state.values:
        return ConversationState(thread_id=thread_id, exists=False, held=False)

    pending = _pending_for(state)
    return ConversationState(
        thread_id=thread_id,
        exists=True,
        held=pending is not None,
        pending_approval=pending,
        customer_id=state.values.get("customer_id"),
        gate_blocked=bool(state.values.get("gate_blocked")),
    )


@app.get("/policies", response_model=list[PolicySection], tags=["ops"])
def policies() -> list[PolicySection]:
    """Every citable section id, with the rule it names and the file it lives in.

    Answers turn up carrying bare markers like "[RET-1]". On their own
    those are noise to a customer; resolved against this list they become
    "Return window (returns.md)" - the rule the answer is standing on,
    named. Cheap to serve because it is computed once at startup.
    """
    return [PolicySection(**p) for p in POLICY_INDEX]


@app.post("/conversations/{thread_id}/close", tags=["conversation"])
def close(thread_id: str) -> dict:
    """End a session and distil the customer's long-term profile.

    An API has no natural "the customer closed the tab" moment, so the
    caller declares it. Profile distillation costs a model call, which is
    exactly why it happens HERE and not on every turn: no customer should
    ever wait for the system to write notes about them.
    """
    values = GRAPH.get_state(trace_config(thread_id)).values
    if not values:
        raise HTTPException(404, f"No conversation with thread_id {thread_id!r}")
    if not values.get("customer_id"):
        return {"saved": False, "reason": "customer was never identified"}
    profile = save_profile(values)
    return {"saved": bool(profile), "profile": profile}


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    """Aggregate cost and latency across every logged turn."""
    runs = load_runs()
    if not runs:
        return {"turns": 0}
    secs = sorted(r.get("seconds", 0.0) for r in runs)
    return {
        "turns": len(runs),
        "conversations": len({r["thread_id"] for r in runs}),
        "total_cost_usd": round(sum(r["cost_usd"] for r in runs), 4),
        "mean_cost_usd": round(sum(r["cost_usd"] for r in runs) / len(runs), 4),
        "latency_p50_s": secs[len(secs) // 2],
        "latency_p95_s": secs[min(int(len(secs) * 0.95), len(secs) - 1)],
    }


# ---------------------------------------------------------------------------
# Shared response building
# ---------------------------------------------------------------------------


def _approval_from_payload(payload: dict) -> PendingApproval:
    """Turn one interrupt() payload into the API's approval card."""
    r = payload["refund"]
    return PendingApproval(
        refund_id=r["refund_id"],
        order_id=r["order_id"],
        customer_id=r["customer_id"],
        customer_name=r["customer_name"],
        product_name=r["product_name"],
        # Decimal is not JSON. Serialising it as a string rather than a
        # float is deliberate: money that has been through a binary float
        # is money you cannot reconcile.
        amount_usd=str(r["amount_usd"]),
        reason=r["reason"],
        prompt=payload["prompt"],
    )


def _pending_for(state: Any) -> PendingApproval | None:
    """The approval a SAVED state is frozen on, or None if it is running.

    A checkpointed state records its pause as a task carrying interrupts.
    Reading it here - rather than in each caller - keeps /approve's 409
    and the UI's poll agreeing on one definition of "frozen".
    """
    for task in state.tasks or ():
        for itr in task.interrupts or ():
            return _approval_from_payload(itr.value)
    return None


def _to_response(
    result: dict, thread_id: str, since_usage: int, since_timings: int
) -> ChatResponse:
    """Turn raw graph output into the API's contract."""
    pending = None
    answer = None

    if "__interrupt__" in result:
        # The run is frozen. There is no answer yet - and saying so with
        # an explicit null is better than inventing a placeholder, because
        # the caller must render an approval UI, not a chat bubble.
        pending = _approval_from_payload(result["__interrupt__"][0].value)
    else:
        msg = next(
            (m for m in reversed(result.get("messages", []))
             if isinstance(m, AIMessage) and not m.tool_calls
             and m.name != INTERNAL),
            None,
        )
        answer = msg.text if msg else None
        record_run(
            result,
            thread_id=thread_id,
            since=since_usage,
            since_timings=since_timings,
        )

    calls, tin, tout, usd = usage_totals(result)
    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        pending_approval=pending,
        llm_calls=calls,
        tokens=tin + tout,
        cost_usd=round(usd, 4),
        seconds=round(elapsed_seconds(result), 2),
    )


# ---------------------------------------------------------------------------
# The browser UI - mounted LAST, and that is not a style preference.
#
# A mount at "/" matches every path under it. Starlette tries routes in the
# order they were registered, so every endpoint above still wins; move this
# line up and it swallows /chat, /health and /docs, and the API answers
# "404 Not Found" in HTML for the rest of its life.
#
# html=True serves index.html for "/" and falls back to it for unknown
# paths, so a bookmarked deep link still lands on the app.
#
# NO CORS MIDDLEWARE, deliberately: the page and the API are the same
# origin because they are the same process. CORS is the tax you pay for
# hosting them apart, and we are not.
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
