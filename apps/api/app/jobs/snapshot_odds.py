"""Odds snapshot cron (F0): archive the current MLB lines.

Two-phase fetch, mirroring how The Odds API splits its endpoints:
1. One slate-wide call for the moneyline (featured market, cheap).
2. One per-event call for the F5 moneyline (additional market — ONLY served
   by the per-event endpoint; requesting it slate-wide 422s). Controlled by
   ``Settings.odds_include_f5`` / ``--no-f5`` because it costs ~2 credits per
   pregame event per run (docs/02 credit plan).

Appends one immutable row per (event, book, market, side) to
``odds_snapshots``. Events resolve to the schedule by The Odds API id, then
by team pair + closest start time (doubleheader-safe); unmatched events are
created and merged later by the schedule sync.

Pregame archive only: events whose start time is already past are skipped
(in-play lines are out of the MVP's scope, docs/01).

With ``--closing-window-min N``, rows for events starting within the next N
minutes are flagged ``is_closing`` — the partial unique index keeps at most
one closing row per outcome, so re-runs are safe.

A failed F5 fetch for one event never kills the run: the moneyline rows
still land and the error is reported in the summary.

Archiving own snapshots from day 1 is the F0 exit criterion: it is what
makes an honest backtest possible later (docs/06).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.db.engine import make_engine
from app.ingestion import store
from app.ingestion.odds_client import OddsApiError, OddsClient
from app.ingestion.parsers import OddsEvent, parse_odds_event


def _with_f5(ev: OddsEvent, client: OddsClient, summary: dict[str, Any]) -> OddsEvent:
    """Merge the event's F5 outcomes (per-event endpoint) into ``ev``."""
    try:
        f5_event = parse_odds_event(client.get_event_odds(ev.source_id))
    except (OddsApiError, httpx.HTTPError) as exc:
        # A timeout/connection error on ONE per-event F5 call must never sink
        # the whole run: the slate moneyline rows are already captured and the
        # F5 line can't be re-fetched later, so we log and move on. httpx.HTTPError
        # is the base of TimeoutException/ConnectError/etc. (non-2xx is already
        # wrapped as OddsApiError by the client).
        summary["f5_errors"].append(f"{ev.source_id}: {exc}")
        return ev
    summary["f5_events_fetched"] += 1
    summary["outcomes_skipped"].extend(f5_event.skipped)
    return OddsEvent(
        source_id=ev.source_id,
        home_team=ev.home_team,
        away_team=ev.away_team,
        commence_time=ev.commence_time,
        outcomes=ev.outcomes + f5_event.outcomes,
        skipped=ev.skipped,
    )


def run(
    *,
    closing_window_min: int | None = None,
    include_f5: bool | None = None,
    client: OddsClient | None = None,
    engine: Engine | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Snapshot the current slate's odds; returns a summary dict."""
    settings = get_settings()
    client = client or OddsClient()
    engine = engine or make_engine(settings.database_url)
    captured_at = captured_at or datetime.now(timezone.utc)
    if include_f5 is None:
        include_f5 = settings.odds_include_f5

    payload = client.get_mlb_odds()
    tables = store.reflect_tables(engine)
    summary: dict[str, Any] = {
        "job": "snapshot_odds",
        "captured_at": captured_at.isoformat(),
        "include_f5": include_f5,
        "events_in_feed": len(payload),
        "events_matched": 0,
        "events_created": 0,
        "events_restamped": 0,
        "restamp_ambiguous": [],
        "events_started_skipped": 0,
        "snapshots_inserted": 0,
        "f5_events_fetched": 0,
        "f5_errors": [],
        "outcomes_skipped": [],
    }

    # Every id in this cycle's payload (started events included): tier 2.5
    # treats an id absent from this set as superseded by The Odds API's
    # reissue, and one still present as a live, distinct game.
    live_ids = frozenset(str(raw["id"]) for raw in payload)

    with engine.begin() as conn:
        sport_id = store.get_sport_id(conn, tables)
        for raw_event in payload:
            ev = parse_odds_event(raw_event)
            summary["outcomes_skipped"].extend(ev.skipped)
            if ev.commence_time <= captured_at:
                summary["events_started_skipped"] += 1
                continue
            if include_f5:
                ev = _with_f5(ev, client, summary)
            if not ev.outcomes:
                continue
            event_id, action = store.find_or_create_event_for_odds(
                conn, tables, sport_id, ev, live_odds_ids=live_ids
            )
            if action in ("created", "created_ambiguous"):
                summary["events_created"] += 1
                if action == "created_ambiguous":
                    summary["restamp_ambiguous"].append(
                        {
                            "source_id": ev.source_id,
                            "home": ev.home_team,
                            "away": ev.away_team,
                            "commence_time": ev.commence_time.isoformat(),
                        }
                    )
            else:
                summary["events_matched"] += 1
                if action == "restamped":
                    summary["events_restamped"] += 1
            is_closing = (
                closing_window_min is not None
                and ev.commence_time <= captured_at + timedelta(minutes=closing_window_min)
            )
            summary["snapshots_inserted"] += store.insert_odds_snapshots(
                conn, tables, event_id, ev.outcomes, captured_at, is_closing
            )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closing-window-min",
        type=int,
        default=None,
        help="Flag is_closing on events starting within the next N minutes",
    )
    parser.add_argument(
        "--no-f5",
        action="store_true",
        help="Skip the per-event F5 calls (free-tier credit saver)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                closing_window_min=args.closing_window_min,
                include_f5=False if args.no_f5 else None,
            )
        )
    )
