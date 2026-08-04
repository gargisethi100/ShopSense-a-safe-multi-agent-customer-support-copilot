"""The only place an LLM client is constructed. (Rewired for Bedrock, 3.5.)

THE PAYOFF, DELIVERED
    Phase 0 promised: "when the provider changes, zero call sites change."
    Phase 3.5 collected: this file swapped ChatAnthropic for Bedrock's
    ChatBedrockConverse, config swapped model ids and credentials - and
    NOT ONE other file was touched. Every future node keeps calling
    get_llm("agent") in blissful ignorance of what's behind the door.

WHAT CHANGED UNDER THE HOOD
    * Provider: langchain-aws ChatBedrockConverse -> Amazon Bedrock's
      Converse API -> Claude. Credentials come from boto3's environment
      contract (AWS_BEARER_TOKEN_BEDROCK or an IAM key pair + AWS_REGION);
      we never handle them in code - one less place a secret can leak.
    * Model ids: Bedrock's `anthropic.`-prefixed ids (or us./eu./apac.
      inference-profile variants). NEVER GUESSED: run  python llm.py list
      to see what your account+region actually serves.
    * Effort: Bedrock supports the Claude effort dial, but langchain-aws
      has no named parameter for it. Provider-specific extras ride in
      `additional_model_request_fields`, passed through verbatim to the
      API. The live smoke test is the verifier: if your region rejects
      the field, the error names it and the fix is one commented line.

Run:
    python llm.py         live smoke test (needs credentials in .env)
    python llm.py list    discover the Claude model ids your account offers
"""

from __future__ import annotations

import os
import sys
from typing import Literal, NamedTuple

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage

from config import cost_usd, get_settings

Role = Literal["agent", "router"]


def get_llm(
    role: Role = "agent",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatBedrockConverse:
    """Build a Claude-on-Bedrock client for a given ROLE (not a given model).

    role="agent"   the specialists doing real reasoning (order/policy agents)
    role="router"  high-volume single-label classification (supervisor, gates)

    `temperature` is honoured only if the resolved model supports it; on a
    model that would reject it we raise HERE, at construction, with an
    explanation - not five tool calls deep inside a graph run.
    """
    settings = get_settings()
    settings.require_aws_credentials()  # a sentence, not a NoCredentialsError

    model = settings.model_agent if role == "agent" else settings.model_router

    kwargs: dict = {
        "model": model,
        # Passed EXPLICITLY, not left to boto3's env-var chain: in testing,
        # botocore raised NoRegionError despite AWS_REGION being set in the
        # environment. Explicit beats implicit the moment implicit fails once.
        "region_name": os.environ["AWS_REGION"],
        "max_tokens": max_tokens or settings.max_tokens,
    }

    # The effort dial (thinking depth). No first-class langchain-aws
    # parameter, so it rides in the provider-passthrough field - and ONLY
    # for models that accept it. claude-haiku-4-5 answers a request carrying
    # output_config with a ValidationException, which is why the capability
    # lives in config's pricing table (verified by a real call) instead of
    # in an assumption here.
    if settings.supports_effort(model):
        kwargs["additional_model_request_fields"] = {
            "output_config": {"effort": settings.effort},
        }

    if temperature is not None:
        if not settings.supports_temperature(model):
            raise ValueError(
                f"temperature={temperature} was requested, but {model!r} rejects "
                f"the temperature parameter. Either drop it, or point this role "
                f"at a model that supports it (e.g. anthropic.claude-haiku-4-5 "
                f"for deterministic routing)."
            )
        kwargs["temperature"] = temperature

    # NOTE deliberately absent: credentials. Those stay on boto3's
    # environment contract (AWS_BEARER_TOKEN_BEDROCK / key pair) - passing
    # them around in code is one more place a secret could leak into a log.
    return ChatBedrockConverse(**kwargs)


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

    THE TRAP this function exists for (survives the Bedrock switch intact,
    because it lives at the LangChain layer): LangChain's standardized
    usage_metadata defines `input_tokens` as the TOTAL prompt - base +
    cache_read + cache_creation - while billing rates the three parts
    differently. So we subtract the cache parts back out. Feed the raw
    total into pricing and every cached token gets double-counted.

    ChatBedrockConverse fills the same standardized shape (that is the
    point of LangChain's abstraction); cache detail keys may simply be
    absent until prompt caching is in play - .get(...) treats absent as 0.
    """
    um = message.usage_metadata or {}
    details = um.get("input_token_details") or {}

    cache_read = details.get("cache_read") or 0
    # TTL-specific keys replace the generic one when a provider reports them.
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
# Model discovery - because guessing Bedrock ids is how you get 4-hour
# debugging sessions over a missing prefix.
# ---------------------------------------------------------------------------


def list_claude_models() -> None:
    """Print the Claude ids THIS account+region actually serves."""
    settings = get_settings()
    settings.require_aws_credentials()

    client = boto3.client("bedrock", region_name=os.environ["AWS_REGION"])
    print("foundation models (provider: anthropic)")
    print("-" * 46)
    try:
        resp = client.list_foundation_models(byProvider="anthropic")
        for m in resp.get("modelSummaries", []):
            print(f"  {m['modelId']}")
    except Exception as e:
        print(f"  could not list: {type(e).__name__}: {e}")

    print("\ninference profiles (cross-region ids, if your account uses them)")
    print("-" * 46)
    try:
        resp = client.list_inference_profiles()
        for p in resp.get("inferenceProfileSummaries", []):
            pid = p.get("inferenceProfileId", "")
            if "anthropic" in pid:
                print(f"  {pid}")
    except Exception as e:
        print(f"  could not list: {type(e).__name__}: {e}")

    print(
        "\nPut your chosen id into .env as SHOPSENSE_MODEL_AGENT / _ROUTER.\n"
        "If it's not in config.py's MODEL_PRICING (geo prefix aside), add it "
        "there with its price first - startup validation insists."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        try:
            list_claude_models()
        except RuntimeError as e:
            print(f"\n{e}")
        raise SystemExit(0)

    settings = get_settings()
    print(f"model_agent  = {settings.model_agent}")
    print(f"model_router = {settings.model_router}")
    print(f"effort       = {settings.effort} "
          f"(agent sends it: {settings.supports_effort(settings.model_agent)}, "
          f"router: {settings.supports_effort(settings.model_router)})")

    try:
        llm = get_llm("agent")
    except RuntimeError as e:
        # No credentials yet: explain and exit 0 - unconfigured is not broken.
        print(f"\n{e}")
        raise SystemExit(0)

    print("calling Claude via Bedrock...")
    try:
        reply = llm.invoke("Reply with exactly five words about e-commerce refunds.")
    except Exception as e:
        msg = str(e)
        print(f"\nCALL FAILED: {type(e).__name__}: {msg[:300]}")
        if "model" in msg.lower() and ("identifier" in msg.lower() or "found" in msg.lower()):
            print(
                "\nLikely a model-id problem. Run  python llm.py list  and put "
                "an id your account actually serves into .env "
                "(SHOPSENSE_MODEL_AGENT / _ROUTER)."
            )
        if "output_config" in msg or "additionalModelRequestFields" in msg:
            print(
                "\nLikely the effort passthrough. Comment out the "
                "additional_model_request_fields block in get_llm() and re-run."
            )
        raise SystemExit(1)

    print(f"\nreply : {reply.text}")
    u = usage_from(reply, settings.model_agent)
    print(f"usage : {u}")
    print(
        f"detail: uncached={u.uncached_input} cache_write={u.cache_creation} "
        f"cache_read={u.cache_read} output={u.output}"
    )

    # Also exercise the router role - it's a different model with different
    # capabilities, so "the agent works" does not imply "the router works".
    print("\ncalling the router model...")
    router_reply = get_llm("router", temperature=0.0).invoke("Reply with one word: ok")
    print(f"reply : {router_reply.text}")
    print(f"usage : {usage_from(router_reply, settings.model_router)}")

    # Sanity: the temperature guard fires only on models that reject it, so
    # assert against the CONFIGURED capability rather than a hardcoded model.
    guard_should_fire = not settings.supports_temperature(settings.model_agent)
    try:
        get_llm("agent", temperature=0.0)
        fired = False
    except ValueError:
        fired = True
    print(
        f"\ntemperature guard: {'fired' if fired else 'silent'} "
        f"(expected {'fired' if guard_should_fire else 'silent'}) - "
        f"{'ok' if fired == guard_should_fire else 'MISMATCH'}"
    )
