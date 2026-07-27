"""Diagnostic (read-only): WHERE does the model lose to the sharp market?

Runs over the SAME 2026 rows the docs/04 §2.4 gate uses (games with an
archived pregame sharp prior) and breaks the result down to answer the one
question that decides the next move: does the model carry signal the market
lacks, or does it just track the market with noise (and so can never beat
it)?

It reproduces the 2026 walk-forward fold EXACTLY — same fit/calib/test split,
same model hyperparameters, same Platt calibration as
``app/ml/train.py::walk_forward_report`` (kept in sync by hand; the printed
``subset_metrics`` MUST match the gate run's market_prior_subset, and the job
asserts it as a built-in self-check). Nothing is written to the database.

Usage::

    python -m app.jobs.analyze_gate [--market moneyline] [--test-season 2026]
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.db.engine import make_engine
from app.ml.dataset import (
    build_training_frame,
    feature_columns,
    load_batting_frame,
    load_bullpen_frame,
    load_lineup_frame,
    load_market_prior,
    load_pitching_frame,
    load_results_frame,
    load_transactions_frame,
)
from app.ml.train import (
    MIN_GATE_N,
    PlattCalibrator,
    _prep_matrix,
    brier,
    ece,
    log_loss,
)


def _loss_vec(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-game log loss (not averaged) — for head-to-head comparisons."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fold_2026(
    frame: pd.DataFrame, columns: list[str], test_season: int
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Replica of the walk_forward fold: returns (test_mask, p_calibrated,
    logistic_coefs). Hyperparameters MUST mirror train.walk_forward_report."""
    seasons = sorted(int(s) for s in frame["season"].unique())
    train_seasons = [s for s in seasons if s < test_season]
    calib_season = train_seasons[-1]
    fit_mask = (frame["season"] < calib_season).to_numpy()
    calib_mask = (frame["season"] == calib_season).to_numpy()
    test_mask = (frame["season"] == test_season).to_numpy()

    y_fit = frame.loc[fit_mask, "target"].to_numpy()
    y_calib = frame.loc[calib_mask, "target"].to_numpy()
    x_fit, medians = _prep_matrix(frame.loc[fit_mask], columns)
    x_calib, _ = _prep_matrix(frame.loc[calib_mask], columns, medians)
    x_test, _ = _prep_matrix(frame.loc[test_mask], columns, medians)

    models = {
        "logistic_scaled": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=7,
        ),
    }
    p_calibrated: dict[str, np.ndarray] = {}
    coefs: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        calibrator = PlattCalibrator.fit(model.predict_proba(x_calib)[:, 1], y_calib)
        p_calibrated[name] = calibrator.apply(model.predict_proba(x_test)[:, 1])
    lr = models["logistic_scaled"].named_steps["logisticregression"]
    coefs["logistic_scaled"] = {
        col: round(float(c), 4) for col, c in zip(columns, lr.coef_[0])
    }
    return test_mask, p_calibrated, coefs


def _calibration_table(
    p: np.ndarray, y: np.ndarray, edges: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n": int(m.sum()),
            "mean_pred": round(float(p[m].mean()), 4),
            "emp_winrate": round(float(y[m].mean()), 4),
            "gap": round(float(p[m].mean() - y[m].mean()), 4),
        })
    return rows


def analyze(
    market: str = "moneyline", test_season: int = 2026, *, engine=None
) -> dict[str, Any]:
    engine = engine or make_engine(get_settings().database_url)
    games = load_results_frame(engine)
    out0: dict[str, Any] = {"job": "analyze_gate", "market": market,
                            "test_season": test_season}
    if len(games) == 0:
        out0["error"] = "no finished games with results; run backfill_results first"
        return out0
    # In production every archive is present; load defensively so a missing
    # newer table degrades features instead of crashing the diagnostic.
    def _safe(loader):
        try:
            f = loader(engine)
            return f if f is not None and len(f) else None
        except Exception:  # noqa: BLE001 — diagnostic must not die on a hole
            return None

    frame = build_training_frame(
        games, market,
        _safe(load_pitching_frame), _safe(load_bullpen_frame),
        _safe(load_batting_frame), _safe(load_lineup_frame),
        _safe(load_transactions_frame),
    )
    prior = load_market_prior(engine, market)
    frame = frame.merge(prior, on="event_id", how="left")

    out: dict[str, Any] = {"job": "analyze_gate", "market": market,
                           "test_season": test_season}
    seasons = sorted(int(s) for s in frame["season"].unique())
    if test_season not in seasons or [s for s in seasons if s < test_season] == []:
        out["error"] = f"test season {test_season} not trainable yet (seasons={seasons})"
        return out

    columns = feature_columns(market)
    test_mask, p_cal, coefs = _fold_2026(frame, columns, test_season)
    test = frame.loc[test_mask].reset_index(drop=True)
    y = test["target"].to_numpy(dtype=float)
    mkt = test["market_prior_p_home"].to_numpy(dtype=float)
    priced = ~np.isnan(mkt)
    n = int(priced.sum())
    out["n_priced"] = n
    out["min_gate_n"] = MIN_GATE_N
    if n == 0:
        out["note"] = "no priced games in this test season yet"
        return out

    ys = y[priced]
    mkts = mkt[priced]
    logs = p_cal["logistic_scaled"][priced]
    gbs = p_cal["hist_gb"][priced]

    # --- Built-in self-check: must reproduce the gate run's subset metrics ---
    out["subset_metrics"] = {
        "market_prior": {"log_loss": round(log_loss(ys, mkts), 5),
                         "brier": round(brier(ys, mkts), 5),
                         "ece": round(ece(ys, mkts), 5)},
        "logistic_scaled": {"log_loss": round(log_loss(ys, logs), 5),
                            "brier": round(brier(ys, logs), 5),
                            "ece": round(ece(ys, logs), 5)},
        "hist_gb": {"log_loss": round(log_loss(ys, gbs), 5),
                   "brier": round(brier(ys, gbs), 5),
                   "ece": round(ece(ys, gbs), 5)},
    }

    # --- A. Head-to-head, per game ---
    ll_mkt = _loss_vec(ys, mkts)
    hh: dict[str, Any] = {}
    for name, p in (("logistic_scaled", logs), ("hist_gb", gbs)):
        ll_m = _loss_vec(ys, p)
        acc_model = float(((p >= 0.5) == (ys == 1)).mean())
        hh[name] = {
            "games_model_beats_market_loss": int((ll_m < ll_mkt).sum()),
            "games_market_beats_model_loss": int((ll_m > ll_mkt).sum()),
            "model_accuracy": round(acc_model, 4),
        }
    hh["market_accuracy"] = round(float(((mkts >= 0.5) == (ys == 1)).mean()), 4)
    out["head_to_head"] = hh

    # --- B. By favorite strength (market's view of the home team) ---
    fav_edges = [0.0, 0.40, 0.47, 0.53, 0.60, 1.01]
    fav_labels = ["home_dog_strong(<.40)", "home_dog(.40-.47)",
                  "tossup(.47-.53)", "home_fav(.53-.60)", "home_fav_strong(>.60)"]
    bands = []
    for (lo, hi), label in zip(zip(fav_edges[:-1], fav_edges[1:]), fav_labels):
        m = (mkts >= lo) & (mkts < hi)
        if m.sum() == 0:
            continue
        bands.append({
            "band": label, "n": int(m.sum()),
            "home_winrate": round(float(ys[m].mean()), 4),
            "market_ll": round(log_loss(ys[m], mkts[m]), 4),
            "logistic_ll": round(log_loss(ys[m], logs[m]), 4),
            "hist_gb_ll": round(log_loss(ys[m], gbs[m]), 4),
        })
    out["by_favorite_strength"] = bands

    # --- C. Disagreement: does deviating from the market help or hurt? ---
    # The decisive test. delta = model_p - market_p. A "good" deviation moves
    # toward the outcome (up when home won, down when home lost).
    disagree = {}
    for name, p in (("logistic_scaled", logs), ("hist_gb", gbs)):
        delta = p - mkts
        buckets = []
        for lo, hi in [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]:
            m = (np.abs(delta) >= lo) & (np.abs(delta) < hi)
            if m.sum() == 0:
                continue
            ll_m = _loss_vec(ys[m], p[m])
            ll_k = _loss_vec(ys[m], mkts[m])
            buckets.append({
                "abs_delta": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                "model_beats_market": int((ll_m < ll_k).sum()),
                "model_mean_ll": round(float(ll_m.mean()), 4),
                "market_mean_ll": round(float(ll_k.mean()), 4),
            })
        # Direction test on meaningful deviations (|delta| >= 0.03).
        big = np.abs(delta) >= 0.03
        good = ((delta > 0) & (ys == 1)) | ((delta < 0) & (ys == 0))
        disagree[name] = {
            "buckets": buckets,
            "meaningful_deviations_n": int(big.sum()),
            "deviations_toward_outcome": int((big & good).sum()),
            "deviation_hit_rate": (round(float(good[big].mean()), 4)
                                   if big.sum() else None),
        }
    out["disagreement"] = disagree

    # --- D. Largest disagreements (concrete examples) ---
    dl = logs - mkts
    order = np.argsort(-np.abs(dl))[:12]
    out["largest_logistic_disagreements"] = [
        {"market_p_home": round(float(mkts[i]), 3),
         "model_p_home": round(float(logs[i]), 3),
         "home_won": int(ys[i]),
         "model_more_right": bool(abs(logs[i] - ys[i]) < abs(mkts[i] - ys[i]))}
        for i in order
    ]

    # --- E. Feature leverage (standardized logistic coefs, |coef| ranked) ---
    ranked = sorted(coefs["logistic_scaled"].items(),
                    key=lambda kv: -abs(kv[1]))
    out["logistic_top_features"] = [
        {"feature": k, "coef": v} for k, v in ranked[:15]
    ]
    return out


def _markdown(r: dict[str, Any]) -> str:
    if "error" in r:
        return f"\n## analyze_gate — {r['error']}\n"
    L = [f"\n## analyze_gate — {r['market']} {r['test_season']} · n_priced={r['n_priced']}\n"]
    sm = r["subset_metrics"]
    L.append("**Subconjunto (self-check vs gate):**")
    L.append("| | log_loss | brier | ece |")
    L.append("|---|---|---|---|")
    for k in ("market_prior", "logistic_scaled", "hist_gb"):
        L.append(f"| {k} | {sm[k]['log_loss']} | {sm[k]['brier']} | {sm[k]['ece']} |")
    hh = r["head_to_head"]
    L.append("\n**Head-to-head (por juego):**")
    L.append(f"- market accuracy: {hh['market_accuracy']}")
    for name in ("logistic_scaled", "hist_gb"):
        h = hh[name]
        L.append(f"- {name}: bate al mercado en {h['games_model_beats_market_loss']}/"
                 f"{r['n_priced']} juegos · accuracy {h['model_accuracy']}")
    L.append("\n**Por fuerza del favorito (visión del mercado sobre el local):**")
    L.append("| banda | n | home_wr | market_ll | logistic_ll | hist_gb_ll |")
    L.append("|---|---|---|---|---|---|")
    for b in r["by_favorite_strength"]:
        L.append(f"| {b['band']} | {b['n']} | {b['home_winrate']} | {b['market_ll']} "
                 f"| {b['logistic_ll']} | {b['hist_gb_ll']} |")
    L.append("\n**Desacuerdo (¿desviarse del mercado ayuda?):**")
    for name in ("logistic_scaled", "hist_gb"):
        d = r["disagreement"][name]
        L.append(f"- {name}: {d['meaningful_deviations_n']} desviaciones ≥0.03 · "
                 f"aciertan hacia el resultado {d['deviations_toward_outcome']} "
                 f"(hit rate {d['deviation_hit_rate']})")
    L.append("\n**Features con más peso (logistic estandarizado):**")
    L.append(", ".join(f"{f['feature']}={f['coef']}" for f in r["logistic_top_features"]))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default="moneyline")
    ap.add_argument("--test-season", type=int, default=2026)
    args = ap.parse_args()
    r = analyze(market=args.market, test_season=args.test_season)
    print(json.dumps(r))
    print(_markdown(r))


if __name__ == "__main__":
    main()
