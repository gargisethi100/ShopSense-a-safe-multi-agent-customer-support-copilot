"""Connection pools — how the app holds its two database keycards.

THE PROBLEM THIS FILE SOLVES
    Opening a database connection is expensive: a network round trip to Neon,
    a TLS handshake, a password check — easily 100-300ms, worse when Neon is
    waking from idle. A single customer question triggers several queries; if
    every query opened a fresh connection, the agent would spend more time
    on handshakes than on answers.

THE FIX: A POOL
    A pool opens a few connections ONCE, keeps them alive, and lends them
    out. Borrow one, run your query, hand it back (it stays open for the
    next borrower). Think: shared office cars with the keycard in the glove
    box, instead of buying a new car per errand and burning it afterwards.

WHY *TWO* POOLS
    A connection is opened AS a specific role and stays that role for life.
    Since we have two runtime identities (agent_ro / refund_writer), we need
    two separate pools. Which pool a piece of code borrows from IS its
    privilege level:

        pool.get_ro_pool()      -> read-only keycard  (the specialists' tools)
        pool.get_writer_pool()  -> append-only keycard (the refund tool ONLY)

WHY THE ADMIN URL IS *NEVER* POOLED
    A pool means "kept open for the lifetime of the app". The master keycard
    must not sit in a running, internet-facing app all day. Admin work
    (schema, seeding) is a one-off task on your laptop: open, do the job,
    close. `admin_connection()` below gives exactly that and nothing more.

Run directly for a smoke test (needs the RO/WRITER URLs in .env, and
schema.sql + roles.sql already applied on Neon):

    python -m db.pool
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

from config import get_settings


def _make_pool(url: str, name: str, *, autocommit: bool) -> ConnectionPool:
    """One place holds the pool sizing so both pools provably match."""
    return ConnectionPool(
        conninfo=url,
        # Lower bound: keep 1 connection warm so the first question after a
        # quiet spell doesn't pay the full handshake. Upper bound: Neon's
        # free tier allows ~100 connections total; a small demo app taking
        # 4 + 4 of them is a polite neighbour and still plenty of parallelism.
        min_size=1,
        max_size=4,
        # Shows up in Neon's monitoring and in error messages, so you can
        # tell WHICH pool misbehaved.
        name=name,
        # `autocommit=True` for reads: without it, psycopg silently starts a
        # transaction on the first SELECT and holds it until commit. A held
        # transaction on a borrowed-and-returned connection is a classic
        # "idle in transaction" leak that blocks Neon from suspending.
        # The writer keeps transactions (autocommit=False): an INSERT should
        # be atomic - it commits when the `with` block exits cleanly and
        # rolls back if an exception escapes. That behaviour IS the safety.
        kwargs={"autocommit": autocommit},
        # Open on construction. We construct lazily (see lru_cache below),
        # so "on construction" means "on first actual use", not "on import".
        open=True,
    )


@lru_cache(maxsize=1)
def get_ro_pool() -> ConnectionPool:
    """The read-only pool (role: agent_ro). What every specialist tool uses.

    @lru_cache makes this LAZY and a SINGLETON in one line:
      * lazy      - the pool (and its network connections) is created the
                    first time somebody asks, not when the module is
                    imported. `import db.pool` in a test must not require
                    a live database.
      * singleton - every caller gets the SAME pool object, so the app
                    holds one set of connections, not one per module.
    """
    url = get_settings().require_db_url("ro")  # explains itself if missing
    return _make_pool(url, name="shopsense-ro", autocommit=True)


@lru_cache(maxsize=1)
def get_writer_pool() -> ConnectionPool:
    """The append-only pool (role: refund_writer).

    Exactly ONE call site may use this: the refund tool, after a human
    approval. If you find a second import of this function anywhere else in
    the codebase, that is a bug by definition - treat it like a failed audit.
    """
    url = get_settings().require_db_url("writer")
    return _make_pool(url, name="shopsense-writer", autocommit=False)


@atexit.register
def close_pools() -> None:
    """Shut the pools down cleanly when the process ends.

    A pool runs BACKGROUND THREADS (workers that open connections, plus a
    scheduler that recycles idle ones). Python won't exit while those are
    alive, so without this you get a 5-second hang and a stern
    "couldn't stop thread 'shopsense-ro-worker-0'" warning on every script.

    Registered with @atexit so nothing has to remember to call it. Note it
    asks the CACHE whether a pool was ever built (`cache_info().currsize`)
    rather than calling get_ro_pool() - which would helpfully CREATE a pool
    at shutdown just to close it. Cleanup must never allocate.
    """
    for getter in (get_ro_pool, get_writer_pool):
        if getter.cache_info().currsize:
            getter().close()


@contextmanager
def admin_connection() -> Iterator[psycopg.Connection]:
    """A one-shot ADMIN connection for setup scripts (seed.py). Never pooled.

    Usage:
        with admin_connection() as conn:
            conn.execute(schema_sql)

    The `with` closes the connection at block exit - the master keycard is
    back in the drawer the moment the job is done. autocommit=True because
    schema files manage their own statement flow, and a half-open
    transaction around a DROP TABLE helps nobody.
    """
    url = get_settings().require_db_url("admin")
    conn = psycopg.connect(url, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Smoke test: prove both keycards work AND that their limits are real.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ro = get_ro_pool()
    except RuntimeError as e:
        # URL not configured yet -> print the instructions and leave quietly.
        print(f"\n{e}")
        raise SystemExit(0)

    print("read-only pool")
    print("-" * 46)
    with ro.connection() as conn:
        # current_user asks Postgres "who am I on this connection?" -
        # the ground truth of which keycard we're holding, straight from
        # the database, not from our own config.
        who = conn.execute("SELECT current_user").fetchone()[0]
        print(f"  connected as        : {who}")

        n = conn.execute("SELECT count(*) FROM customers").fetchone()[0]
        print(f"  customers visible   : {n}  (0 until seed.py runs - fine)")

        # Now the important part: watch the database say no.
        try:
            conn.execute("DELETE FROM orders")
            print("  DELETE FROM orders  : SUCCEEDED  <- WRONG! roles.sql not applied?")
        except psycopg.errors.InsufficientPrivilege:
            print("  DELETE FROM orders  : permission denied  <- correct, by design")

    try:
        writer = get_writer_pool()
    except RuntimeError as e:
        print(f"\n{e}")
        raise SystemExit(0)

    print("\nwriter pool")
    print("-" * 46)
    with writer.connection() as conn:
        who = conn.execute("SELECT current_user").fetchone()[0]
        print(f"  connected as        : {who}")
        try:
            conn.execute("SELECT count(*) FROM customers")
            print("  SELECT customers    : SUCCEEDED  <- WRONG! roles.sql not applied?")
        except psycopg.errors.InsufficientPrivilege:
            print("  SELECT customers    : permission denied  <- correct: the writer")
            print("                        cannot even READ customer data")

    print("\nBoth keycards verified. Every 'permission denied' above is the")
    print("least-privilege design working - not an error to fix.")
