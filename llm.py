"""The only place a Claude client is constructed.

Why a factory instead of `ChatAnthropic(...)` scattered across nodes:

1.  The model rules live in ONE place. `claude-opus-5` rejects `temperature`
    with HTTP 400; `claude-haiku-4-5` accepts it. Every node constructing its
    own client is a node that can get this wrong. Here, the capability flag in
    config.py decides, and a node literally cannot send a bad parameter.

2.  Roles, not models. Call sites say get_llm("router") — *what the call is
    for* — and config decides which model that means today. When we move
    routing from Opus to Haiku in Phase 4, zero call sites change.

3.  Usage extraction is subtle enough to centralise (see usage_from below —
    LangChain and the raw Anthropic API disagree about what "input_tokens"
    means, and the difference is exactly the cached tokens).

Run directly for a live smoke test (needs ANTHROPIC_API_KEY in .env):

    python llm.py
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from config import cost_usd, get_settings

Role = Literal["agent", "router"]


def get_llm(
    role: Role = "agent",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatAnthropic:
    """Build a Claude client for a given ROLE (not a given model).

    role="agent"   the specialists doing real reasoning (order/policy agents)
    role="router"  high-volume single-label classification (supervisor, gates)

    `temperature` is honoured only if the resolved model supports it; on a
    model that would 400, we raise HERE, at construction, with an explanation —
    not five tool calls deep inside a graph run.
    """
    settings = get_settings()
    settings.require_api_key()  # fail with instructions, not a cryptic 401

    model = settings.model_agent if role == "agent" else settings.model_router

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or settings.max_tokens,
        # `effort` controls how hard the model thinks (low..max). This is the
        # cost/quality dial on the Claude 5 generation — the role temperature
        # used to play, now that sampling params are gone.
        "reasoning_effort": settings.effort,
    }

    if temperature is not None:
        if not settings.supports_temperature(model):
            raise ValueError(
                f"temperature={temperature} was requested, but {model!r} rejects "
                f"the temperature parameter (HTTP 400). Either drop it, or point "
                f"this role at a model that supports it (e.g. claude-haiku-4-5 "
                f"for deterministic routing)."
            )
        kwargs["temperature"] = temperature

    # NOTE deliberately absent:
    #   - `thinking`: on claude-opus-5 thinking is ON by default (adaptive).
    #     We pass nothing and get the recommended behaviour for free.
    #   - `anthropic_api_key`: the SDK reads ANTHROPIC_API_KEY from the
    #     environment itself; passing it around in code is one more place a
    #     secret could leak into a log.
    return ChatAnthropic(**kwargs)


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class CallUsage(NamedTuple):
    """Token accounting for one LLM call, split the way BILLING splits it."""

    uncached_input: int  # billed at the full input rate
    cache_creation: int  # billed at 1.25x input rate
    cache_read: int  # billed at 0.10x input rate
    output: int  # billed at the output rate (includes thinking tokens!)
    model: str

    @property
    def total_input(self) -> int:
        return self.uncached_input + self.cache_creation + self.cache_read

    @property
    def cost(self) -> float:
        return cost_usd(
            self.model,
            input_tokens=self.uncached_input,
            output_tokens=self.output,
            cache_creation_tokens=self.cache_creation,
            cache_read_tokens=self.cache_read,
        )

    def __str__(self) -> str:  # the cost-footer line
        cached = f" ({self.cache_read} cached)" if self.cache_read else ""
        return (
            f"{self.total_input} in{cached} / {self.output} out"
            f" / ${self.cost:.4f}"
        )


def usage_from(message: AIMessage, model: str) -> CallUsage:
    """Extract billing-shaped usage from a LangChain response.

    THE TRAP this function exists for: LangChain and the raw Anthropic API
    define `input_tokens` differently.

        raw API   : input_tokens = the UNCACHED REMAINDER only
        LangChain : usage_metadata["input_tokens"] = base + cache_read
                    + cache_creation  (it re-totals; verified in
                    langchain_anthropic.chat_models._create_usage_metadata)

    Our pricing function wants the split three ways (each part bills at a
    different rate), so we subtract the cache parts back out of LangChain's
    total. Feed LangChain's number in directly and every cached token would be
    double-counted — once at full rate, once at its cache rate.
    """
    um = message.usage_metadata or {}
    details = um.get("input_token_details") or {}

    cache_read = details.get("cache_read") or 0
    # 5m/1h TTL-specific keys replace the generic one when present.
    cache_creation = (
        (details.get("ephemeral_5m_input_tokens") or 0)
        + (details.get("ephemeral_1h_input_tokens") or 0)
    ) or (details.get("cache_creation") or 0)

    total_input = um.get("input_tokens", 0)
    uncached = max(total_input - cache_read - cache_creation, 0)

    return CallUsage(
        uncached_input=uncached,
        cache_creation=cache_creation,
        cache_read=cache_read,
        output=um.get("output_tokens", 0),
        model=model,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    print(f"model_agent = {settings.model_agent}, effort = {settings.effort}")

    try:
        llm = get_llm("agent")
    except RuntimeError as e:
        # No key yet: explain and exit 0 - an unconfigured machine is not broken.
        print(f"\n{e}")
        raise SystemExit(0)

    print("calling Claude...")
    reply = llm.invoke("Reply with exactly five words about e-commerce refunds.")

    print(f"\nreply : {reply.text()}")
    u = usage_from(reply, settings.model_agent)
    print(f"usage : {u}")
    print(
        f"detail: uncached={u.uncached_input} cache_write={u.cache_creation} "
        f"cache_read={u.cache_read} output={u.output}"
    )

    # Sanity: prove the temperature guard works on the live config too.
    try:
        get_llm("agent", temperature=0.0)
        print("temperature guard: FAILED TO FIRE (should not happen on opus-5)")
    except ValueError as e:
        print(f"temperature guard: ok - {str(e)[:60]}...")
