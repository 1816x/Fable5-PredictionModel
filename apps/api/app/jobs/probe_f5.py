"""One-off probe: does The Odds API serve the F5 moneyline, and does the
sharp reference book (Pinnacle) quote it?

Read-only, no database. Answers the load-bearing question docs/02 §51-58 left
open before committing to the paid 20K plan and the F5 market pivot: the F5
gate uses Pinnacle as the no-vig reference (docs/00 decision #6), so if
Pinnacle doesn't publish ``h2h_1st_5_innings`` here, the F5 prior can't be
built as designed.

For the next N pregame events it calls the per-event endpoint (the only one
serving additional markets) and reports, per event, which bookmakers returned
a parseable F5 home/away price and whether ``pinnacle`` is among them.

Cost: ~2 credits per pregame event with data (~6 on the free tier for the
default N=3). Usage::

    python -m app.jobs.probe_f5 [--events N]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from app.ingestion.odds_client import F5_MARKET, OddsApiError, OddsClient
from app.ingestion.parsers import parse_odds_event

SHARP_BOOK = "pinnacle"  # docs/00 decision #6: the no-vig reference book


def run(*, events: int = 3, client: OddsClient | None = None) -> dict[str, Any]:
    client = client or OddsClient()
    now = datetime.now(timezone.utc)
    slate = client.get_mlb_odds()
    pregame = [
        raw for raw in slate
        if parse_odds_event(raw).commence_time > now
    ][:events]

    summary: dict[str, Any] = {
        "job": "probe_f5",
        "f5_market": F5_MARKET,
        "sharp_book": SHARP_BOOK,
        "events_in_slate": len(slate),
        "events_probed": 0,
        "probed": [],
        "errors": [],
    }
    for raw in pregame:
        ev0 = parse_odds_event(raw)
        try:
            payload = client.get_event_odds(ev0.source_id)
        except OddsApiError as exc:
            summary["errors"].append(f"{ev0.source_id}: {exc}")
            continue
        summary["events_probed"] += 1
        # Raw book keys present in the per-event response (any market), and the
        # subset that produced a parseable F5 home/away price.
        raw_books = sorted({b["key"] for b in payload.get("bookmakers", [])})
        parsed = parse_odds_event(payload)
        f5_books = sorted({
            o.book_key for o in parsed.outcomes if o.market == "f5_moneyline"
        })
        summary["probed"].append({
            "event": f"{ev0.away_team} @ {ev0.home_team}",
            "commence_time": ev0.commence_time.isoformat(),
            "books_in_response": raw_books,
            "books_with_f5": f5_books,
            "pinnacle_has_f5": SHARP_BOOK in f5_books,
            "skipped": list(parsed.skipped)[:6],
        })

    probed = summary["probed"]
    n_pin = sum(1 for p in probed if p["pinnacle_has_f5"])
    summary["verdict"] = {
        # Capability question: does Pinnacle quote F5 here AT ALL? One event is
        # enough to answer yes (some games post F5 later, so requiring every
        # event would give a false no). The coverage count below judges how
        # reliably, so the pivot decision isn't made on a single fluke.
        "pinnacle_publishes_f5": n_pin > 0,
        "pinnacle_f5_coverage": f"{n_pin}/{summary['events_probed']}",
        "any_book_publishes_f5": any(p["books_with_f5"] for p in probed),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=3,
                    help="how many pregame events to probe (cost ~2 credits each)")
    args = ap.parse_args()
    print(json.dumps(run(events=args.events), indent=2))


if __name__ == "__main__":
    main()
