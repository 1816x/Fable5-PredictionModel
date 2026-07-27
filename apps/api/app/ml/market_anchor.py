"""Market-anchored blend: the sharp line nudged by the model.

Instead of predicting p_home from scratch (which analyze_gate showed is
anti-signal vs the sharp market — the model's departures hit the outcome only
44% of the time), this anchors on the market prior and lets the model apply a
BOUNDED correction in log-odds space:

    p_blend = sigmoid( logit(p_market) + beta * (logit(p_model) - logit(p_market)) )

- beta = 0  -> the pure sharp line (baseline; the docs/04 §2.4 gate target).
- beta = 1  -> the pure model.
- 0 < beta < 1 -> the line, nudged toward the model by a fraction beta.

HONEST LIMITATION: we have no historical odds archive (free tier; own
snapshots started 2026-07-08), so beta CANNOT be fit out-of-time — it can only
be swept on the same priced subset it is scored on. A sweep here is therefore
an IN-SAMPLE upper bound: if even the in-sample-best beta cannot beat the pure
line (beta=0) materially, the model adds nothing the market lacks. Any beta
that does beat it is a candidate to validate out-of-time as the archive grows,
never a validated production weight on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ml.train import _logit, brier, ece, log_loss

_EPS = 1e-6


def blend(p_model: np.ndarray, p_market: np.ndarray, beta: float) -> np.ndarray:
    """Anchor on the market line, nudge toward the model by fraction ``beta``.

    Pure function in log-odds space (see module docstring). ``beta`` is not
    clamped: values outside [0, 1] extrapolate, which the caller may want for
    diagnostics, but the sweep grid stays within [0, 1].
    """
    lm = _logit(np.asarray(p_market, dtype=float))
    lp = _logit(np.asarray(p_model, dtype=float))
    z = lm + beta * (lp - lm)
    return 1.0 / (1.0 + np.exp(-z))


def sweep(
    p_model: np.ndarray,
    p_market: np.ndarray,
    y: np.ndarray,
    betas: list[float] | None = None,
) -> dict[str, Any]:
    """Score the blend across a beta grid; return the curve and the argmin.

    ``beats_market`` on each grid point compares against beta=0 (the pure
    line), which is exactly ``market``'s own log loss. ``best`` is the argmin
    of log loss over the grid — IN-SAMPLE (see module docstring), so it is an
    upper bound, not a validated operating point.
    """
    if betas is None:
        betas = [round(float(b), 2) for b in np.linspace(0.0, 1.0, 21)]
    # Cast every value to a plain Python type: numpy scalars survive into the
    # summary and break json.dumps at the end of the Actions run.
    y = np.asarray(y, dtype=float)
    market_ll = float(log_loss(y, np.asarray(p_market, dtype=float)))
    curve = []
    for b in betas:
        b = float(b)
        ll = float(log_loss(y, blend(p_model, p_market, b)))
        p = blend(p_model, p_market, b)
        curve.append({
            "beta": b,
            "log_loss": round(ll, 5),
            "brier": round(float(brier(y, p)), 5),
            "ece": round(float(ece(y, p)), 5),
            "beats_market": bool(ll < market_ll - _EPS),
        })
    best = min(curve, key=lambda r: r["log_loss"])
    return {
        "n": int(len(y)),
        "market_log_loss": round(market_ll, 5),
        "curve": curve,
        "best": {
            "beta": float(best["beta"]),
            "log_loss": float(best["log_loss"]),
            "delta_vs_market": round(float(best["log_loss"]) - market_ll, 5),
            "beats_market": bool(best["beta"] > 0 and best["log_loss"] < market_ll - _EPS),
        },
    }
