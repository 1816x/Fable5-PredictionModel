"""Integration tests: parsers + store + jobs against the real schema.

Skipped unless ``EDGE_TEST_DATABASE_URL`` points at a Postgres with
``infra/schema.sql`` applied (see tests/conftest.py). These tests verify the
F0 invariants end to end: idempotent schedule sync, doubleheader-safe event
matching, append-only odds snapshots with dedupe, and the closing-line flag.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from app.ingestion import store
from app.ingestion.parsers import parse_transactions
from app.jobs import snapshot_odds, sync_schedule

from conftest import load_fixture

pytestmark = pytest.mark.integration

CAPTURE_TS = datetime(2026, 7, 8, 18, 0, tzinfo=timezone.utc)


class FakeMlbClient:
    def get_schedule(self, date_iso: str):
        return load_fixture("mlb_schedule.json")


class FakeOddsClient:
    """Slate fixture (h2h only) + per-event F5 for event A, mirroring how
    The Odds API splits featured vs additional markets across endpoints."""

    def get_mlb_odds(self, **kwargs):
        return load_fixture("odds_api_mlb.json")

    def get_event_odds(self, event_id, **kwargs):
        f5 = load_fixture("odds_api_event_f5.json")
        if event_id == f5["id"]:
            return f5
        # Books without the market simply don't appear in the response.
        for event in load_fixture("odds_api_mlb.json"):
            if event["id"] == event_id:
                return {**event, "bookmakers": []}
        raise AssertionError(f"unexpected event id {event_id}")


def _scalar(engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def test_schedule_sync_is_idempotent(db):
    first = sync_schedule.run("2026-07-08", client=FakeMlbClient(), engine=db)
    assert first["games_in_feed"] == 3
    assert first["events_created"] == 3

    second = sync_schedule.run("2026-07-08", client=FakeMlbClient(), engine=db)
    assert second["events_created"] == 0
    assert second["events_refreshed"] == 3

    assert _scalar(db, "SELECT count(*) FROM events") == 3
    # Padres/Dodgers appear twice (doubleheader) but teams dedupe by name.
    assert _scalar(db, "SELECT count(*) FROM teams") == 4
    assert (
        _scalar(
            db,
            "SELECT count(*) FROM events WHERE external_ids ? 'mlb_game_pk'",
        )
        == 3
    )


def test_snapshot_odds_matches_schedule_and_creates_unknown(db):
    sync_schedule.run("2026-07-08", client=FakeMlbClient(), engine=db)
    summary = snapshot_odds.run(
        client=FakeOddsClient(), engine=db, captured_at=CAPTURE_TS
    )

    # Yankees@RedSox and the doubleheader opener match the schedule; the
    # Cubs@Cardinals event is not in the slate fixture and gets created.
    assert summary["events_matched"] == 2
    assert summary["events_created"] == 1
    assert summary["events_started_skipped"] == 0
    # Event A: 4 moneyline (pinnacle + fanduel) + 2 F5 via per-event call;
    # Dodgers and Cardinals games: 2 moneyline each, no F5 quoted.
    assert summary["snapshots_inserted"] == 10
    assert summary["f5_events_fetched"] == 3
    assert summary["f5_errors"] == []
    f5_rows = _scalar(
        db, "SELECT count(*) FROM odds_snapshots WHERE market = 'f5_moneyline'"
    )
    assert f5_rows == 2
    assert _scalar(db, "SELECT count(*) FROM events") == 4

    # The matched event now carries BOTH external identities.
    merged = _scalar(
        db,
        """
        SELECT count(*) FROM events
        WHERE external_ids ->> 'mlb_game_pk' = '745001'
          AND external_ids ->> 'the_odds_api_id' = 'e912aa27b1c4f03d8e5a6b7c8d9e0f1a'
        """,
    )
    assert merged == 1

    # Doubleheader safety: the 20:05Z odds event must attach to the 20:10Z
    # opener (5 min away), never to the 02:10Z nightcap.
    opener_snapshots = _scalar(
        db,
        """
        SELECT count(*) FROM odds_snapshots s
        JOIN events e ON e.id = s.event_id
        WHERE e.external_ids ->> 'mlb_game_pk' = :pk
        """,
        pk="745002",
    )
    nightcap_snapshots = _scalar(
        db,
        """
        SELECT count(*) FROM odds_snapshots s
        JOIN events e ON e.id = s.event_id
        WHERE e.external_ids ->> 'mlb_game_pk' = :pk
        """,
        pk="745003",
    )
    assert opener_snapshots == 2
    assert nightcap_snapshots == 0

    # implied_prob is a generated column: 1 / 2.05 rounded to 6 decimals.
    implied = _scalar(
        db,
        "SELECT implied_prob FROM odds_snapshots WHERE price_decimal = 2.050 LIMIT 1",
    )
    assert implied == Decimal("0.487805")

    # Re-running with the same capture instant inserts nothing (dedupe key).
    rerun = snapshot_odds.run(client=FakeOddsClient(), engine=db, captured_at=CAPTURE_TS)
    assert rerun["snapshots_inserted"] == 0
    assert _scalar(db, "SELECT count(*) FROM odds_snapshots") == 10


def test_no_f5_skips_per_event_calls(db):
    class NoF5Client(FakeOddsClient):
        def get_event_odds(self, event_id, **kwargs):
            raise AssertionError("per-event endpoint must not be called with include_f5=False")

    summary = snapshot_odds.run(
        include_f5=False, client=NoF5Client(), engine=db, captured_at=CAPTURE_TS
    )
    assert summary["snapshots_inserted"] == 8  # moneyline only
    assert summary["f5_events_fetched"] == 0


def test_f5_failure_never_kills_the_moneyline_rows(db):
    from app.ingestion.odds_client import OddsApiError

    class FlakyF5Client(FakeOddsClient):
        def get_event_odds(self, event_id, **kwargs):
            raise OddsApiError("simulated 500")

    summary = snapshot_odds.run(
        client=FlakyF5Client(), engine=db, captured_at=CAPTURE_TS
    )
    assert summary["snapshots_inserted"] == 8  # all moneyline rows landed
    assert len(summary["f5_errors"]) == 3


def test_f5_timeout_never_kills_the_moneyline_rows(db):
    # A raw httpx timeout on the per-event F5 call (NOT wrapped as
    # OddsApiError) must be caught too — otherwise it sinks the whole run and
    # the already-fetched slate moneyline rows are lost.
    import httpx

    class TimeoutF5Client(FakeOddsClient):
        def get_event_odds(self, event_id, **kwargs):
            raise httpx.TimeoutException("simulated read timeout")

    summary = snapshot_odds.run(
        client=TimeoutF5Client(), engine=db, captured_at=CAPTURE_TS
    )
    assert summary["snapshots_inserted"] == 8  # all moneyline rows landed
    assert len(summary["f5_errors"]) == 3


def test_closing_flag_only_within_window_and_never_duplicated(db):
    late_capture = datetime(2026, 7, 8, 19, 50, tzinfo=timezone.utc)
    summary = snapshot_odds.run(
        closing_window_min=20,
        client=FakeOddsClient(),
        engine=db,
        captured_at=late_capture,
    )
    assert summary["snapshots_inserted"] == 10

    # Only the 20:05Z Dodgers game falls inside the 20-minute closing window.
    assert _scalar(db, "SELECT count(*) FROM odds_snapshots WHERE is_closing") == 2
    closing_ok = _scalar(
        db,
        """
        SELECT count(*) FROM odds_snapshots s
        JOIN events e ON e.id = s.event_id
        WHERE s.is_closing AND e.external_ids ->> 'the_odds_api_id' = :oid
        """,
        oid="f7c3d2e1a0b9c8d7e6f5a4b3c2d1e0f9",
    )
    assert closing_ok == 2

    # A second closing run minutes later: the partial unique index rejects a
    # second closing row per outcome; non-closing events snapshot normally.
    second = snapshot_odds.run(
        closing_window_min=20,
        client=FakeOddsClient(),
        engine=db,
        captured_at=datetime(2026, 7, 8, 19, 55, tzinfo=timezone.utc),
    )
    assert second["snapshots_inserted"] == 8  # 10 minus the 2 closing dupes
    assert _scalar(db, "SELECT count(*) FROM odds_snapshots WHERE is_closing") == 2


def test_pregame_only_skips_started_events(db):
    after_first_pitch = datetime(2026, 7, 8, 23, 30, tzinfo=timezone.utc)
    summary = snapshot_odds.run(
        client=FakeOddsClient(), engine=db, captured_at=after_first_pitch
    )
    # 23:10Z and 20:05Z already started; only the 00:15Z game snapshots.
    assert summary["events_started_skipped"] == 2
    assert summary["snapshots_inserted"] == 2


def test_odds_snapshots_are_append_only(db):
    snapshot_odds.run(client=FakeOddsClient(), engine=db, captured_at=CAPTURE_TS)
    with pytest.raises(Exception, match="append-only"):
        with db.begin() as conn:
            conn.execute(text("UPDATE odds_snapshots SET price_decimal = 3.0"))
    with pytest.raises(Exception, match="append-only"):
        with db.begin() as conn:
            conn.execute(text("DELETE FROM odds_snapshots"))


def _upsert_transactions(conn, tables, rows):
    """Helper: upsert players from a transactions batch, then the txns —
    the exact order the sync_transactions job runs (players first)."""
    sport_id = store.get_sport_id(conn, tables)
    player_cache = store.load_player_cache(conn, tables)
    store.bulk_upsert_players(
        conn, tables, sport_id,
        [
            {"mlb_person_id": r.mlb_person_id, "full_name": r.full_name, "pitch_hand": None}
            for r in rows
        ],
        player_cache,
    )
    team_by_mlb_id = store.load_team_cache_by_mlb_id(conn, tables, sport_id)
    return store.bulk_upsert_transactions(conn, tables, rows, player_cache, team_by_mlb_id)


def test_bulk_upsert_transactions_idempotent_and_resolves_teams(db):
    tables = store.reflect_tables(
        db, ("sports", "teams", "players", "player_transactions")
    )
    rows = list(parse_transactions(load_fixture("mlb_transactions.json")).rows)
    with db.begin() as conn:
        sport_id = store.get_sport_id(conn, tables)
        # Yankees (147) exists with its MLB id; Angels/Cubs do NOT — a from/to
        # team that is unknown resolves to NULL (audit-only, never fabricated).
        store.get_or_create_team(conn, tables, sport_id, "New York Yankees", 147)
        store.get_or_create_team(conn, tables, sport_id, "Boston Red Sox", 111)

    with db.begin() as conn:
        first = _upsert_transactions(conn, tables, rows)
    assert first == 4  # the 4 well-formed rows
    assert _scalar(db, "SELECT count(*) FROM player_transactions") == 4
    # Auto-upserted a player who may never appear in a boxscore (season on IL).
    assert _scalar(
        db, "SELECT count(*) FROM players WHERE mlb_person_id = 592450"
    ) == 1
    # Judge's from_team resolves to the Yankees row; a known team id present.
    assert _scalar(
        db,
        "SELECT count(*) FROM player_transactions t JOIN teams tm "
        "ON t.from_team_id = tm.id WHERE t.mlb_transaction_id = 900001 "
        "AND tm.name = 'New York Yankees'",
    ) == 1
    # Trout's Angels (108) is unknown -> from_team_id NULL, not fabricated.
    assert _scalar(
        db,
        "SELECT from_team_id FROM player_transactions WHERE mlb_transaction_id = 900002",
    ) is None

    # Re-run: idempotent by mlb_transaction_id (no duplicate rows).
    with db.begin() as conn:
        second = _upsert_transactions(conn, tables, rows)
    assert second == 4
    assert _scalar(db, "SELECT count(*) FROM player_transactions") == 4


def test_bulk_upsert_transactions_do_update_on_correction(db):
    tables = store.reflect_tables(
        db, ("sports", "teams", "players", "player_transactions")
    )
    rows = list(parse_transactions(load_fixture("mlb_transactions.json")).rows)
    with db.begin() as conn:
        _upsert_transactions(conn, tables, rows)
    # The feed re-emits transaction 900001 with a corrected description
    # (frozen dataclass -> rebuild the one row via dataclasses.replace).
    import dataclasses

    corrected = [
        dataclasses.replace(r, description="CORRECTED move")
        if r.mlb_transaction_id == 900001 else r
        for r in rows
    ]
    with db.begin() as conn:
        _upsert_transactions(conn, tables, corrected)
    assert _scalar(db, "SELECT count(*) FROM player_transactions") == 4
    assert _scalar(
        db,
        "SELECT description FROM player_transactions WHERE mlb_transaction_id = 900001",
    ) == "CORRECTED move"
