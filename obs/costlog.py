"""Observability: what did that conversation do, and what did it cost?

TWO QUESTIONS, TWO TOOLS - DO NOT CONFUSE THEM

    "WHY did it answer that?"     -> LangSmith traces
        Every node, every prompt, every tool result, nested and timed.
        You open it when something is WRONG and you need to replay the
        decision chain. Rich, remote, and only useful one run at a time.

    "WHAT does this cost us?"     -> this file
        One flat line per turn, appended locally. You open it when you
        need a NUMBER: cost per conversation, which node dominates,
        whether yesterday's prompt edit doubled the bill. Cheap, local,
        and only useful in aggregate.

    A trace cannot answer "what is our average cost per conversation"
    without a lot of clicking, and a cost log cannot tell you why the
    supervisor routed to the wrong specialist. Build both; reach for the
    right one.

WHY JSONL (one JSON object per line, appended)
    * append-only - concurrent turns cannot corrupt each other's lines
    * schemaless - adding a field next month breaks nothing that reads
      the old lines
    * greppable - `grep refund runs/*.jsonl` is a complete query engine
      at this scale, and needs no service to be running
    A database would be the right call at a million turns. At demo scale
    it would be infrastructure you maintain for no answer you cannot
    already get.

WHAT IS NOT LOGGED, ON PURPOSE
    Message text. The cost log answers money questions, and money
    questions do not need the customer's words - so it stores counts, not
    content. (The redacted trigger log in guards/ is the only local file
    that keeps message text, and it keeps it masked.)

Run directly to see the analysis over everything logged so far:

    python -m obs.costlog
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from graph.state import ShopSenseState

RUN_LOG = Path(__file__).resolve().parent.parent / "runs" / "cost.jsonl"


def _pct(values: list[float], p: int) -> float:
    """The p-th percentile, nearest-rank. Small enough not to need numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round(p / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(idx, 0)]


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


def tracing_status() -> str:
    """Human-readable summary of whether traces are being recorded.

    Deliberately reads os.environ rather than mirroring these into config:
    the langsmith client reads them itself, so a second copy could
    disagree with the one that actually matters.
    """
    on = os.environ.get(
        "LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "")
    ).lower() in {"1", "true", "yes"}
    if not on:
        return "off (set LANGSMITH_TRACING=true in .env to record traces)"
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    key = os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        return "ON but LANGSMITH_API_KEY is empty - traces will fail silently"
    return f"on -> project '{project}' at smith.langchain.com"


def trace_config(thread_id: str, *, customer_id: str | None = None) -> dict:
    """The config dict for graph.invoke(), with trace labels attached.

    thread_id is REQUIRED by the checkpointer. The tags and metadata are
    free extras that make traces findable later: without them you get a
    wall of identically-named runs and no way to ask "show me the refund
    conversations" or "show me everything cust_001 did".

    Label at write time. You cannot retro-tag a trace you already lost.
    """
    tags = ["shopsense", f"guardrails:{os.environ.get('SHOPSENSE_GUARDRAILS_MODE', 'monitor')}"]
    metadata = {"thread_id": thread_id}
    if customer_id:
        tags.append(f"customer:{customer_id}")
        metadata["customer_id"] = customer_id
    return {
        "configurable": {"thread_id": thread_id},
        "tags": tags,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# The cost log
# ---------------------------------------------------------------------------


def record_run(
    state: ShopSenseState,
    *,
    thread_id: str,
    question: str = "",
    since: int = 0,
    since_timings: int = 0,
) -> dict:
    """Append one line describing the turn that just finished.

    `since` IS LOAD-BEARING, and the bug it fixes is easy to repeat:
    state["usage"] has an APPEND reducer, so it holds every call made in
    the whole CONVERSATION, not this turn. Logging it wholesale records
    turn 1 once, again inside turn 2, again inside turn 3 - and the
    "total spend" is then wildly inflated by re-counting.

    So the caller passes how many usage records it has already logged, and
    we slice past them. The lesson generalises: with an accumulating
    field, every reader must decide whether it wants SO FAR or JUST NOW.
    The cost footer wants so far. A per-turn log wants just now.

    Returns the record so callers can display it without re-reading.
    Never raises: metrics must not be able to break the product.
    """
    fresh = (state.get("usage") or [])[since:]
    calls = len(fresh)
    tin = sum(r["input_tokens"] for r in fresh)
    tout = sum(r["output_tokens"] for r in fresh)
    usd = sum(r["cost_usd"] for r in fresh)

    # Timings accumulate exactly like usage, so they need the same
    # so-far/just-now slicing. Same trap, same fix.
    fresh_timings = (state.get("timings") or [])[since_timings:]
    seconds = sum(t["seconds"] for t in fresh_timings)

    by_node: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0, "seconds": 0.0}
    )
    for r in fresh:
        node = by_node[r["node"]]
        node["calls"] += 1
        node["cost_usd"] = round(node["cost_usd"] + r["cost_usd"], 6)
    for t in fresh_timings:
        # A node can appear here with zero LLM calls (the output rail, a
        # blocked input gate) - those are pure latency, and pretending
        # they are free is how a "cheap" turn still feels slow.
        by_node[t["node"]]["seconds"] = round(
            by_node[t["node"]]["seconds"] + t["seconds"], 3
        )

    record = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thread_id": thread_id,
        "customer_id": state.get("customer_id"),
        # Length, not content: enough to correlate cost with question size,
        # without keeping the customer's words in a metrics file.
        "question_chars": len(question),
        "llm_calls": calls,
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": round(usd, 6),
        "seconds": round(seconds, 3),
        "by_node": dict(by_node),
        # Guardrail activity is cost-adjacent: a blocked turn is a turn you
        # did NOT pay a specialist for, and that shows up here.
        "gate_flags": state.get("gate_flags") or [],
        "refund_pending": bool(state.get("pending_refund")),
    }

    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
    return record


def load_runs() -> list[dict]:
    """Every logged turn. Skips malformed lines rather than dying on one."""
    if not RUN_LOG.exists():
        return []
    runs = []
    for line in RUN_LOG.read_text(encoding="utf-8").splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line from a killed process
    return runs


def analyze(runs: list[dict] | None = None) -> str:
    """The report you actually want: where the money goes."""
    runs = load_runs() if runs is None else runs
    if not runs:
        return "No runs logged yet. Have a conversation first (python -m graph.build)."

    total = sum(r["cost_usd"] for r in runs)
    calls = sum(r["llm_calls"] for r in runs)
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in runs)
    threads = {r["thread_id"] for r in runs}

    seconds = [r.get("seconds", 0.0) for r in runs]
    total_s = sum(seconds)

    by_node: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0, "seconds": 0.0}
    )
    for r in runs:
        for node, stats in (r.get("by_node") or {}).items():
            by_node[node]["calls"] += stats["calls"]
            by_node[node]["cost_usd"] += stats["cost_usd"]
            by_node[node]["seconds"] += stats.get("seconds", 0.0)

    lines = [
        "ShopSense cost report",
        "=" * 62,
        f"  turns logged      : {len(runs)}  across {len(threads)} conversation(s)",
        f"  total spend       : ${total:.4f}",
        f"  per turn (mean)   : ${total / len(runs):.4f}",
        f"  per conversation  : ${total / len(threads):.4f}",
        f"  llm calls         : {calls}   ({calls / len(runs):.1f} per turn)",
        f"  tokens            : {tokens:,}",
        "",
        # PERCENTILES, NOT AVERAGES, for latency. A mean hides the tail,
        # and the tail is what people complain about: p95 is roughly "the
        # worst experience 1 customer in 20 has". An app can have a fine
        # average and still feel unreliable.
        f"  latency  p50      : {_pct(seconds, 50):.1f}s",
        f"           p95      : {_pct(seconds, 95):.1f}s"
        + ("   <- the tail customers actually notice" if len(seconds) > 3 else ""),
        f"           worst    : {max(seconds):.1f}s" if seconds else "",
        "",
        "  where the money and the TIME go",
        "  " + "-" * 58,
        f"  {'node':16} {'calls':>6} {'cost':>9} {'$ share':>8} {'secs':>7} {'s share':>8}",
    ]
    for node, s in sorted(by_node.items(), key=lambda kv: -kv[1]["cost_usd"]):
        share = 100 * s["cost_usd"] / total if total else 0
        s_share = 100 * s["seconds"] / total_s if total_s else 0
        lines.append(
            f"  {node:16} {s['calls']:>6} {s['cost_usd']:>9.4f} {share:>7.1f}%"
            f" {s['seconds']:>7.1f} {s_share:>7.1f}%"
        )

    # The insight this table exists to surface: call COUNT and call COST
    # are different rankings. The router runs constantly and costs little;
    # a specialist runs rarely and costs a lot. Optimising the wrong one is
    # the classic first mistake.
    if by_node:
        most_calls = max(by_node.items(), key=lambda kv: kv[1]["calls"])[0]
        most_cost = max(by_node.items(), key=lambda kv: kv[1]["cost_usd"])[0]
        lines += [
            "",
            f"  most CALLS : {most_calls}",
            f"  most COST  : {most_cost}",
        ]
        if most_calls != most_cost:
            lines.append(
                f"  -> {most_calls} is the busiest node but {most_cost} is the "
                "expensive one.\n     Optimise where the dollars are, not "
                "where the traffic is."
            )

    blocked = sum(1 for r in runs if any("blocked" in f for f in r.get("gate_flags", [])))
    if blocked:
        lines += [
            "",
            f"  guardrails blocked {blocked} turn(s) - each one is a turn you did",
            "  not pay a specialist to answer.",
        ]

    expensive = sorted(runs, key=lambda r: -r["cost_usd"])[:3]
    lines += ["", "  most expensive turns"]
    for r in expensive:
        lines.append(
            f"    ${r['cost_usd']:.4f}  {r['llm_calls']} calls  "
            f"{r['at'][:16]}  thread {r['thread_id']}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import enable_utf8_console

    enable_utf8_console()
    print(f"langsmith tracing : {tracing_status()}")
    print(f"cost log          : {RUN_LOG}")
    print()
    print(analyze())
