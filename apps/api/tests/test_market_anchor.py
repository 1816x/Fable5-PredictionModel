"""market_anchor: the blend math, verified with hand-calculated numbers.

Unit-only (no DB) — always runs. Guards the one property that matters: the
model does exactly what the formula says, so a sweep result can be trusted.
"""

import math

import numpy as np
import pytest

from app.ml.market_anchor import blend, sweep


def test_beta_zero_is_the_pure_market_line():
    p_market = np.array([0.6, 0.42, 0.71])
    out = blend(np.array([0.9, 0.1, 0.5]), p_market, 0.0)
    assert np.allclose(out, p_market, atol=1e-9)


def test_beta_one_is_the_pure_model():
    p_model = np.array([0.9, 0.1, 0.5])
    out = blend(p_model, np.array([0.6, 0.42, 0.71]), 1.0)
    assert np.allclose(out, p_model, atol=1e-9)


def test_half_beta_symmetric_logits_cancel_to_half():
    # logit(0.6) = +0.405465, logit(0.4) = -0.405465 -> z = 0 -> sigmoid = 0.5.
    out = blend(np.array([0.6]), np.array([0.4]), 0.5)
    assert out[0] == \
        pytest.approx(0.5, abs=1e-9)


def test_blend_is_monotone_in_beta_toward_the_model():
    # p_model > p_market: bigger beta pulls the blend up, bounded by the two.
    vals = [blend(np.array([0.8]), np.array([0.5]), b)[0]
            for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals[0] == pytest.approx(0.5, abs=1e-9)
    assert vals[-1] == pytest.approx(0.8, abs=1e-9)
    assert all(a < b for a, b in zip(vals, vals[1:]))  # strictly increasing
    assert all(0.5 <= v <= 0.8 for v in vals)


def test_sweep_hand_calculated_endpoints():
    # y=[1,0], market=[0.6,0.6], model=[0.9,0.1] (model deviates toward truth).
    y = np.array([1.0, 0.0])
    market = np.array([0.6, 0.6])
    model = np.array([0.9, 0.1])
    r = sweep(model, market, y, betas=[0.0, 1.0])

    # beta=0 == pure line: -(ln0.6 + ln0.4)/2 = 0.71356.
    market_ll = -(math.log(0.6) + math.log(0.4)) / 2
    assert r["market_log_loss"] == pytest.approx(round(market_ll, 5))
    assert r["curve"][0]["log_loss"] == pytest.approx(round(market_ll, 5))
    assert r["curve"][0]["beats_market"] is False  # beta=0 never beats itself

    # beta=1 == pure model: -(ln0.9 + ln0.9)/2 = 0.10536.
    model_ll = -(math.log(0.9) + math.log(0.9)) / 2
    assert r["curve"][1]["log_loss"] == pytest.approx(round(model_ll, 5))
    assert r["curve"][1]["beats_market"] is True

    assert r["best"]["beta"] == 1.0
    assert r["best"]["beats_market"] is True
    assert r["best"]["delta_vs_market"] < 0


def test_sweep_when_model_is_noise_best_is_beta_zero():
    # model points the WRONG way on both games -> no beta > 0 helps.
    y = np.array([1.0, 0.0])
    market = np.array([0.6, 0.4])
    model = np.array([0.2, 0.8])
    r = sweep(model, market, y)
    assert r["best"]["beta"] == 0.0
    assert r["best"]["beats_market"] is False
