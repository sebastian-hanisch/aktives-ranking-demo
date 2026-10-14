"""Orakel: Schätzer gegen scipy, Laplace-Kovarianz gegen finite Differenzen, Paarwahl gegen Aufzählen mit math.erf, Eigenschaften der Strategien über feste Läufe."""

import math

import numpy as np
import pytest

import akt_model as M
import akt_strategies as S


def _data(seed, n=5, games=60):
    rng = np.random.default_rng(seed)
    s = np.linspace(-1, 1, n)
    W = np.zeros((n, n))
    for _ in range(games):
        a, b = rng.choice(n, 2, replace=False)
        if rng.random() < 1 / (1 + math.exp(-(s[a] - s[b]))):
            W[a, b] += 1
        else:
            W[b, a] += 1
    return W, W + W.T


def test_bt_equals_the_scipy_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(5):
        W, G = _data(seed)
        n = len(W)
        i, j = np.triu_indices(n, 1)
        lam = 0.2

        def neg(s, W=W, i=i, j=j, lam=lam):
            d = s[i] - s[j]
            return float(np.sum(W[i, j] * np.logaddexp(0, -d) + W[j, i] * np.logaddexp(0, d)) + 0.5 * lam * s @ s)

        res = optimize.minimize(neg, np.zeros(n), method="BFGS", options={"gtol": 1e-10})
        ours, _ = M.fit_bt(W, G, lam, iters=40)
        assert ours == pytest.approx(res.x, abs=1e-5), seed


def test_laplace_covariance_is_the_inverse_hessian_of_the_negative_log_posterior():
    W, G = _data(1)
    n, lam = len(W), 0.2
    s, Sigma = M.fit_bt(W, G, lam, iters=40)
    i, j = np.triu_indices(n, 1)

    def neg(x):
        d = x[i] - x[j]
        return float(np.sum(W[i, j] * np.logaddexp(0, -d) + W[j, i] * np.logaddexp(0, d)) + 0.5 * lam * x @ x)

    H = np.zeros((n, n))
    eps = 1e-4
    for a in range(n):
        for b in range(n):
            ea, eb = np.eye(n)[a] * eps, np.eye(n)[b] * eps
            H[a, b] = (neg(s + ea + eb) - neg(s + ea - eb) - neg(s - ea + eb) + neg(s - ea - eb)) / (4 * eps * eps)
    assert Sigma == pytest.approx(np.linalg.inv(H), abs=2e-3)


def test_pair_choice_equals_a_scalar_enumeration_with_math_erf():
    rng = np.random.default_rng(3)
    n = 6
    s = rng.normal(0, 1, n)
    A = rng.normal(0, 0.3, (n, n))
    Sigma = A @ A.T + 0.2 * np.eye(n)
    best_m, best_q = (-1.0, None), (-1.0, None)
    for a in range(n):
        for b in range(a + 1, n):
            v = Sigma[a, a] + Sigma[b, b] - 2 * Sigma[a, b]
            m = 0.5 * math.erfc(abs(s[a] - s[b]) / math.sqrt(v) / math.sqrt(2.0))
            p = 1 / (1 + math.exp(-(s[a] - s[b])))
            q = p * (1 - p) * v
            if m > best_m[0]:
                best_m = (m, (a, b))
            if q > best_q[0]:
                best_q = (q, (a, b))
    assert S.select_pairs(M.misorder_probability(s, Sigma), np.zeros(15), 1) == [best_m[1]]
    assert S.select_pairs(M.information_score(s, Sigma), np.zeros(15), 1) == [best_q[1]]


def test_with_many_games_every_strategy_recovers_the_true_order():
    for name in ("Rundenturnier", "Aktiv: Information", "Schweizer System (vereinfacht)"):
        r = S.play_strategy(name, 6, 0.8, 600, 0.05, 1, 4)
        assert r.accuracy[-1] > 0.95, name


def test_the_naive_rule_concentrates_more_than_the_information_rule_over_ten_runs():
    mx_f = np.mean([S.concentration(S.play_strategy("Aktiv: Fehlordnung", 10, 0.3, 200, 0.1, 1, sd))[0] for sd in range(10)])
    mx_i = np.mean([S.concentration(S.play_strategy("Aktiv: Information", 10, 0.3, 200, 0.1, 1, sd))[0] for sd in range(10)])
    assert mx_f > 2 * mx_i


def test_mini_instance_picks_a_neighbour_pair_by_hand():
    s = np.array([1.0, 0.0, -1.0])
    q = M.information_score(s, np.eye(3))
    p1 = 1 / (1 + math.exp(-1.0))
    assert q[0, 1] == pytest.approx(p1 * (1 - p1) * 2.0) and q[0, 1] > q[0, 2]
    assert S.select_pairs(q, np.zeros(3), 1)[0] in {(0, 1), (1, 2)}
