"""Shared test setup: two tiers of test, and when each is allowed to run.

EVALS ARE NOT UNIT TESTS
    A unit test asserts an exact value: add(2,2) == 4, forever. An eval
    asserts a BEHAVIOUR of a non-deterministic system: "this question
    reaches the order specialist". The same input can produce different
    words every run, so an eval that string-matches an answer is a test
    that fails for no reason and gets deleted within a week.

    Rules that follow, applied throughout tests/:
      * assert on the DECISION (which route, which tool, is it cited),
        never on phrasing
      * where a judgement is fuzzy, allow a THRESHOLD rather than
        demanding perfection - "at least 6 of 7" is a real quality bar;
        "all 7 forever" is a flaky build
      * pin what you can: temperature=0 on the router makes routing
        reproducible, which is exactly why the router lives on a model
        that accepts it

TWO TIERS
    FREE  no network, no credentials, no cost. Retrieval ranking, guardrail
          patterns, the citation auditor, graph wiring. These run on every
          push, in every environment, in under a second.
    LIVE  calls Bedrock (marked @pytest.mark.live) and/or the database
          (@pytest.mark.db). Real money, real latency.

    Split so CI can gate cheaply and always: a contributor without
    credentials still gets the free tier, and a pull request cannot break
    retrieval or guardrails without turning something red.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: needs Bedrock credentials (costs money)")
    config.addinivalue_line("markers", "db: needs a seeded Postgres database")


def _has_bedrock() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    ) and bool(os.environ.get("AWS_REGION"))


def _has_db() -> bool:
    from config import get_settings

    return bool(get_settings().db_url_ro)


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip - don't fail - when a tier's prerequisites are absent.

    A missing API key is a fact about the ENVIRONMENT, not a defect in the
    code. Failing here would train everyone to ignore red builds, which is
    the only truly unrecoverable state for a test suite.
    """
    skip_live = pytest.mark.skip(reason="no Bedrock credentials in environment")
    skip_db = pytest.mark.skip(reason="no database configured")
    live_ok, db_ok = _has_bedrock(), _has_db()
    for item in items:
        if "live" in item.keywords and not live_ok:
            item.add_marker(skip_live)
        if "db" in item.keywords and not db_ok:
            item.add_marker(skip_db)


@pytest.fixture(scope="session")
def retriever():
    from rag.retriever import get_retriever

    return get_retriever()
