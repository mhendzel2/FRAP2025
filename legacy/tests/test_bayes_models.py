import numpy as np

from frap_bayes_models import (
    hb_1exp_forward,
    hb_2exp_ordered_forward,
    hb_reaction_dominant_forward,
    hb_stretched_forward,
    ordered_rate_constants,
)


def test_ordered_rate_constants_enforce_fast_component():
    k_slow = np.array([0.1, 0.2])
    delta_k = np.array([0.05, 0.01])
    slow, fast = ordered_rate_constants(k_slow, delta_k)
    assert np.all(fast > slow)


def test_hb_1exp_forward_monotonic_recovery():
    t = np.linspace(0, 5, 20)
    y = hb_1exp_forward(t, I0=0.0, I_inf=1.0, k=0.5)
    assert y[0] == 0.0
    assert y[-1] < 1.0
    assert np.all(np.diff(y) >= 0)


def test_hb_2exp_ordered_forward_bounds():
    t = np.linspace(0, 10, 40)
    y = hb_2exp_ordered_forward(
        t,
        I0=0.1,
        I_inf=0.95,
        k_slow=0.08,
        delta_k=0.25,
        f_fast=0.6,
    )
    assert np.isclose(y[0], 0.1)
    assert y.max() <= 0.95 + 1e-9


def test_hb_stretched_forward_starts_at_i0():
    t = np.linspace(0, 6, 50)
    y = hb_stretched_forward(t, I0=0.05, I_inf=0.90, tau=2.0, alpha=0.8)
    assert np.isclose(y[0], 0.05)
    assert y[-1] > y[0]


def test_reaction_dominant_matches_single_exp_form():
    t = np.linspace(0, 4, 25)
    y1 = hb_reaction_dominant_forward(t, I0=0.0, I_inf=1.0, k_off=0.4)
    y2 = hb_1exp_forward(t, I0=0.0, I_inf=1.0, k=0.4)
    assert np.allclose(y1, y2)
