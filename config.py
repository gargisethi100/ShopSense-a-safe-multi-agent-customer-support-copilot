"""Single source of truth for configuration.

Two jobs:

1. Turn environment variables into a validated, typed object, and fail LOUDLY
   at startup if something is missing or malformed — rather than at 2am inside
   a graph node, five tool calls deep, as an AttributeError on None.

2. Hold the facts about each Claude model that the rest of the app must not
   guess at: what it costs, and whether it accepts `temperature`.

RULE: this is the only module allowed to read os.environ for SHOPSENSE_* vars.
Everything else imports `get_settings()`. That way there is exactly one place
to look when you ask "where does this value come from?", and exactly one place
a typo can hide.

Run it directly to see your resolved config with secrets redacted:

    python config.py
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from typing import Literal, NamedTuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Model facts
# ---------------------------------------------------------------------------

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelPricing(NamedTuple):
    """What one model costs, and one capability flag that changes our code.

    Prices are US dollars per MILLION tokens, matching how Anthropic publishes
    them. Every price here is a fact about the vendor, not a preference — if
    Anthropic changes it, this table is the one place to edit.
    """

    input_per_mtok: float
    output_per_mtok: float

    # `temperature` was REMOVED from the Claude Opus 5 / Sonnet 5 generation.
    # Sending it does not get ignored — it returns HTTP 400 and the call fails.
    # Older/smaller models still accept it. llm.py reads this flag so that no
    # individual node can get it wrong.
    supports_temperature: bool

    # Promotional pricing has an end date. A cost tracker that ignores this
    # over-reports today and under-reports after it lapses.
    intro_input_per_mtok: float | None = None
    intro_output_per_mtok: float | None = None
    intro_until: date | None = None

    def rates(self, on: date | None = None) -> tuple[float, float]:
        """(input, output) $/Mtok in effect on a given day. Defaults to today."""
        on = on or date.today()
        if self.intro_until and on <= self.intro_until:
            return (
                self.intro_input_per_mtok or self.input_per_mtok,
                self.intro_output_per_mtok or self.output_per_mtok,
            )
        return self.input_per_mtok, self.output_per_mtok


# BEDROCK model ids (Phase 3.5): Claude on Bedrock carries an `anthropic.`
# prefix. Ids are COMPLETE as written - never append a date suffix.
#
# PRICING CAVEAT: Bedrock is partner-operated and AWS SETS ITS OWN PRICES.
# The numbers below are Anthropic's first-party list prices, used as a
# starting approximation - verify against https://aws.amazon.com/bedrock/pricing/
# for your region and edit here. (The cost pipeline is the lesson; the
# constants are yours to true-up.)
MODEL_PRICING: dict[str, ModelPricing] = {
    "anthropic.claude-opus-5": ModelPricing(
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        supports_temperature=False,
    ),
    "anthropic.claude-sonnet-5": ModelPricing(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        supports_temperature=False,
        intro_input_per_mtok=2.00,
        intro_output_per_mtok=10.00,
        intro_until=date(2026, 8, 31),
    ),
    "anthropic.claude-haiku-4-5": ModelPricing(
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        supports_temperature=True,
    ),
}


def _pricing_key(model: str) -> str:
    """Normalise a Bedrock model id to its MODEL_PRICING key.

    Some accounts must call Claude through a cross-region "inference
    profile" whose id prepends a geography: us.anthropic.claude-opus-5,
    eu.anthropic..., apac.anthropic... . Same model, same price - so
    pricing lookups strip that one prefix. Everything else must match
    exactly; unknown ids should FAIL, not silently price as $0.
    """
    first, _, rest = model.partition(".")
    if first in {"us", "eu", "apac"} and rest.startswith("anthropic."):
        return rest
    return model

# Prompt caching is billed at a multiple of the normal INPUT rate.
# Writing to cache costs more than a normal token; reading is ~10x cheaper.
# Ignoring these is how cost dashboards end up wrong in both directions.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def cost_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    on: date | None = None,
) -> float:
    """Dollar cost of one Claude call.

    IMPORTANT: `input_tokens` from the API is the UNCACHED REMAINDER only.
    The full prompt is input + cache_creation + cache_read. A cost function
    that reads only `input_tokens` silently under-reports every cached run —
    which is exactly the runs you were hoping to measure.
    """
    key = _pricing_key(model)
    if key not in MODEL_PRICING:
        raise KeyError(
            f"No pricing for model {model!r}. Add it to MODEL_PRICING in config.py "
            f"before using it, so cost reporting can never silently read as $0."
        )
    in_rate, out_rate = MODEL_PRICING[key].rates(on)
    per_token_in = in_rate / 1_000_000
    per_token_out = out_rate / 1_000_000
    return (
        input_tokens * per_token_in
        + cache_creation_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * per_token_in * CACHE_READ_MULTIPLIER
        + output_tokens * per_token_out
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Typed view of the environment.

    Note the two naming regimes, which mirror the two sections of .env.example:

      * AWS credentials (AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID /
        AWS_REGION) are NOT fields here at all: those names belong to boto3,
        which reads them from the environment itself. Mirroring them into
        Settings would create two sources of truth that can disagree. We only
        CHECK their presence, at point of use, via require_aws_credentials().
      * everything of ours is auto-prefixed SHOPSENSE_ by env_prefix below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHOPSENSE_",
        extra="ignore",  # don't explode on unrelated vars in the environment
        frozen=True,  # config is read-only after load; no spooky action at a distance
    )

    # --- credentials ------------------------------------------------------
    # (none stored here - see the class docstring: AWS credentials belong to
    #  boto3's environment contract; we validate presence in
    #  require_aws_credentials() at the moment they're actually needed.)

    # --- database (Phase 1) -----------------------------------------------
    # One database, three roles. The privilege boundary is in Postgres, not in
    # a prompt, which is what makes it impossible rather than discouraged.
    db_url_admin: str = ""  # owner: migrations + seeding. Never used at runtime.
    db_url_ro: str = ""  # agent_ro: SELECT on three tables, nothing else.
    db_url_writer: str = ""  # refund_writer: the single write path.

    # --- models -----------------------------------------------------------
    model_agent: str = "anthropic.claude-opus-5"
    model_router: str = "anthropic.claude-opus-5"

    # Caps thinking AND the visible answer TOGETHER on this model generation.
    # Too small does not mean "cheaper", it means "truncated mid-sentence".
    max_tokens: int = Field(default=4096, ge=256, le=128_000)

    effort: Effort = "medium"

    # --- guardrails (Phase 7) ---------------------------------------------
    guardrails_mode: Literal["monitor", "enforce"] = "monitor"

    # --- app --------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- validation -------------------------------------------------------

    @field_validator("model_agent", "model_router")
    @classmethod
    def _model_must_be_priced(cls, v: str) -> str:
        """Reject a model we have no pricing for.

        This looks pedantic until you realise the alternative: the app runs
        fine, the cost footer quietly reports $0.00, and you discover the real
        number on an invoice. Failing at startup is the cheaper bug.
        (Geo prefixes like us./eu. are normalised away first - see
        _pricing_key - so an inference-profile id of a priced model passes.)
        """
        if _pricing_key(v) not in MODEL_PRICING:
            known = ", ".join(sorted(MODEL_PRICING))
            raise ValueError(
                f"Unknown model {v!r}. Known: {known} (a us./eu./apac. prefix "
                f"on any of these is also fine). If this is a new model, add "
                f"it to MODEL_PRICING in config.py (with its real prices) "
                f"rather than removing this check."
            )
        return v

    # --- capability helpers -----------------------------------------------

    def pricing(self, model: str) -> ModelPricing:
        return MODEL_PRICING[_pricing_key(model)]

    def supports_temperature(self, model: str) -> bool:
        """True if this model accepts `temperature`.

        The Claude 5 generation (opus-5 / sonnet-5) rejects the parameter
        outright; claude-haiku-4-5 accepts it — which is why moving the
        router to Haiku also buys deterministic (temperature=0) routing.
        """
        return MODEL_PRICING[_pricing_key(model)].supports_temperature

    # --- "you need a credential now" helpers ------------------------------
    # These exist so a missing value produces a sentence telling you what to
    # do, at the moment it matters, instead of a cryptic SDK error deep in
    # a network call.

    def require_aws_credentials(self) -> None:
        """Check the boto3 credential environment before any Bedrock call.

        Deliberately reads os.environ (not Settings fields): these names are
        boto3's contract, and boto3 will read them itself — we only verify
        they exist so the failure is a sentence, not a NoCredentialsError
        five tool-calls deep. Either auth style passes:
          * AWS_BEARER_TOKEN_BEDROCK          (Bedrock API key), or
          * AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY   (IAM key pair)
        AWS_REGION is required either way (model ids are per-region).
        """
        bearer = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
        pair = os.environ.get("AWS_ACCESS_KEY_ID", "") and os.environ.get(
            "AWS_SECRET_ACCESS_KEY", ""
        )
        if not (bearer or pair):
            raise RuntimeError(
                "No AWS credentials found for Bedrock.\n"
                "  Set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key), or\n"
                "  AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, in .env\n"
                "  (NOT .env.example). Then set a budget alarm in AWS Billing."
            )
        if not os.environ.get("AWS_REGION"):
            raise RuntimeError(
                "AWS_REGION is empty. Set it in .env to the region your "
                "Bedrock access lives in (e.g. us-east-1). Model ids are "
                "per-region; `python llm.py list` shows what yours offers."
            )

    def require_db_url(self, which: Literal["admin", "ro", "writer"]) -> str:
        value = {
            "admin": self.db_url_admin,
            "ro": self.db_url_ro,
            "writer": self.db_url_writer,
        }[which]
        if not value:
            raise RuntimeError(
                f"SHOPSENSE_DB_URL_{which.upper()} is empty.\n"
                "  - 'admin' comes from Neon when you create the project.\n"
                "  - 'ro' and 'writer' exist only after db/roles.sql is applied."
            )
        return value

    # --- introspection ----------------------------------------------------

    @property
    def tracing_enabled(self) -> bool:
        """Whether LangSmith will trace.

        Read straight from os.environ rather than mirrored into a field: the
        langsmith client reads LANGSMITH_TRACING itself, so duplicating it here
        would create two sources of truth that can disagree.
        """
        return os.environ.get(
            "LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "")
        ).lower() in {"1", "true", "yes"}

    def summary(self) -> str:
        """Human-readable, secret-safe dump. Safe to log or paste into an issue."""
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            key_state = "bedrock api key set"
        elif os.environ.get("AWS_ACCESS_KEY_ID"):
            key_state = "iam key pair set"
        else:
            key_state = "MISSING"
        region = os.environ.get("AWS_REGION", "MISSING")

        def db(v: str) -> str:
            if not v:
                return "not set"
            # Show enough to identify the host, never the password.
            tail = v.rsplit("@", 1)[-1]
            return f"set (…@{tail[:40]})"

        in_rate, out_rate = MODEL_PRICING[_pricing_key(self.model_agent)].rates()
        lines = [
            "ShopSense configuration",
            "-" * 46,
            f"  aws credentials   : {key_state}",
            f"  aws region        : {region}",
            f"  model_agent       : {self.model_agent}  (${in_rate}/${out_rate} per Mtok today)",
            f"  model_router      : {self.model_router}",
            f"  max_tokens        : {self.max_tokens}   (thinking + answer combined)",
            f"  effort            : {self.effort}",
            f"  temperature usable: {self.supports_temperature(self.model_agent)}",
            f"  guardrails_mode   : {self.guardrails_mode}",
            f"  log_level         : {self.log_level}",
            f"  langsmith tracing : {'on' if self.tracing_enabled else 'off'}",
            f"  db admin          : {db(self.db_url_admin)}",
            f"  db read-only      : {db(self.db_url_ro)}",
            f"  db writer         : {db(self.db_url_writer)}",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load once, reuse everywhere.

    Cached because .env is read from disk on construction, and because a single
    shared frozen instance means every module provably sees identical values.
    """
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(settings.summary())

    # NOTE: keep printed output ASCII-only. The default Windows console
    # codepage is cp1252, so an em-dash here renders as a replacement char.
    # Source-code comments can be Unicode; terminal output should not be.
    print("\nWorked example - cost of a small call on each model")
    print("-" * 46)
    for model in MODEL_PRICING:
        usd = cost_usd(model, input_tokens=1_200, output_tokens=400)
        print(f"  {model:18} 1,200 in / 400 out -> ${usd:.6f}")

    print("\nWhy cached tokens must be counted separately")
    print("-" * 46)
    naive = cost_usd("anthropic.claude-opus-5", input_tokens=500, output_tokens=400)
    honest = cost_usd(
        "anthropic.claude-opus-5",
        input_tokens=500,
        output_tokens=400,
        cache_read_tokens=20_000,
    )
    print(f"  reading only input_tokens : ${naive:.6f}")
    print(f"  counting 20k cached tokens: ${honest:.6f}")
    print(f"  under-reported by         : ${honest - naive:.6f}")
