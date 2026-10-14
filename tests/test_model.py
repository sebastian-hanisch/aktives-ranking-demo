"""Schätzer, Laplace-Kovarianz, Paarmaße und die Auswahlwerte der aktiven Paarung an Handwerten."""

import math

import numpy as np
import pytest

import akt_model as M
from conftest import counts


def test_sigmoid_is_stable():
    assert M.sigmoid(0.0) == 0.5 and M.sigmoid(-800.0) == 0.0 and M.sigmoid(800.0) == 1.0


def test_bt_ranks_by_wins_and_has_zero_mean_through_the_prior():
    W, G = counts(3, {(0, 1): 8, (1, 0): 2, (1, 2): 7, (2, 1): 3, (0, 2): 9, (2, 0): 1})
    s, Sigma = M.fit_bt(W, G, lam=0.1, iters=20)
    assert s[0] > s[1] > s[2] and abs(s.mean()) < 1e-9 and np.allclose(Sigma, Sigma.T)


def test_gradient_is_zero_at_the_solution():
    W, G = counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1})
    s, _ = M.fit_bt(W, G, lam=0.2, iters=30)
    P = M.sigmoid(s[:, None] - s[None, :])
    grad = (W - G * P).sum(axis=1) - 0.2 * s
    assert np.abs(grad).max() < 1e-8


def test_without_games_the_estimate_is_zero_and_the_covariance_is_the_prior():
    W, G = counts(4, {})
    s, Sigma = M.fit_bt(W, G, lam=0.5)
    assert np.allclose(s, 0.0) and np.allclose(Sigma, np.eye(4) / 0.5)


def test_covariance_is_the_inverse_of_the_hessian_by_hand_for_two_players():
    W, G = counts(2, {(0, 1): 3, (1, 0): 1})
    s, Sigma = M.fit_bt(W, G, lam=0.1, iters=30)
    p = float(M.sigmoid(s[0] - s[1]))
    h = 4 * p * (1 - p)
    H = np.array([[h + 0.1, -h], [-h, h + 0.1]])
    assert Sigma == pytest.approx(np.linalg.inv(H), abs=1e-9)


def test_more_games_shrink_the_uncertainty():
    W1, G1 = counts(3, {(0, 1): 2, (1, 0): 1, (1, 2): 2, (2, 1): 1})
    W2, G2 = counts(3, {(0, 1): 20, (1, 0): 10, (1, 2): 20, (2, 1): 10})
    _, S1 = M.fit_bt(W1, G1, 0.1, iters=20)
    _, S2 = M.fit_bt(W2, G2, 0.1, iters=20)
    assert np.diag(S2).sum() < np.diag(S1).sum()


def test_pair_accuracy_winner_and_top_k_by_hand():
    truth = [0.0, 1.0, 2.0, 3.0]
    assert M.pair_accuracy([0, 1, 2, 3], truth) == 1.0 and M.pair_accuracy([3, 2, 1, 0], truth) == 0.0 and M.pair_accuracy([5, 5, 5, 5], truth) == 0.5
    assert M.pair_accuracy([0, 0, 1, 1], truth) == pytest.approx((4 + 2 * 0.5) / 6)
    assert M.winner_correct([0, 1, 2, 3], truth) == 1.0 and M.winner_correct([3, 1, 2, 0], truth) == 0.0
    assert M.winner_correct([0, 1, 3, 3], truth) == 0.5 and M.winner_correct([1, 1, 1, 1], truth) == 0.25
    assert M.top_k_overlap([0, 1, 2, 3], truth, 2) == 1.0 and M.top_k_overlap([3, 2, 0, 1], truth, 2) == 0.0 and M.top_k_overlap([0, 3, 2, 1], truth, 2) == 0.5
    assert M.top_k_overlap([0, 1, 2], [0, 1, 2], 10) == 1.0


def test_pair_variance_by_hand():
    Sigma = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.5], [0.0, 0.5, 3.0]])
    V = M.pair_variance(Sigma)
    assert V[0, 1] == pytest.approx(1.0 + 2.0 - 0.4) and V[1, 2] == pytest.approx(2.0 + 3.0 - 1.0) and np.allclose(V, V.T)


def test_misorder_probability_by_hand():
    s = np.array([1.0, 0.0, -1.0])
    Sigma = np.eye(3)
    m = M.misorder_probability(s, Sigma)
    # Var(s_i − s_j) = 2: Nachbarn z = 1/√2, außen z = 2/√2
    assert m[0, 1] == pytest.approx(0.5 * math.erfc(1 / math.sqrt(2) / math.sqrt(2))) and m[0, 2] == pytest.approx(0.5 * math.erfc(2 / math.sqrt(2) / math.sqrt(2)))
    assert m[0, 1] > m[0, 2] and m[0, 1] == m[1, 0]


def test_information_score_prefers_close_and_uncertain_pairs():
    s = np.array([1.0, 0.0, -1.0])
    q = M.information_score(s, np.eye(3))
    assert q[0, 1] == pytest.approx(float(M.sigmoid(1.0)) * (1 - float(M.sigmoid(1.0))) * 2.0)
    assert q[0, 1] > q[0, 2]
    wide = np.diag([1.0, 1.0, 9.0])
    assert M.information_score(s, wide)[1, 2] > M.information_score(s, np.eye(3))[1, 2]          # mehr Unsicherheit, mehr Information


def test_newton_does_not_diverge_for_separable_data_and_a_small_prior():
    """Regression: Spieler 0 gewinnt alle Partien, Spieler 3 verliert alle; ohne Rückwärtsschritt schwang das volle Newton-Verfahren auf Werte um 10^3."""
    W, G = counts(4, {(0, 1): 6, (0, 2): 6, (0, 3): 6, (1, 2): 4, (1, 3): 6, (2, 3): 6, (2, 1): 2})
    s = np.zeros(4)
    for _ in range(25):
        s, Sigma = M.fit_bt(W, G, lam=0.02, s0=s, iters=1)
    assert np.abs(s).max() < 12.0 and s[0] > s[1] > s[2] > s[3]
    assert M.log_posterior(s, W, 0.02) >= M.log_posterior(np.zeros(4), W, 0.02)
    assert np.isfinite(Sigma).all()


def test_log_posterior_by_hand():
    W, G = counts(2, {(0, 1): 1})
    assert M.log_posterior(np.zeros(2), W, 0.5) == pytest.approx(-math.log(2.0))
    assert M.log_posterior(np.array([1.0, -1.0]), W, 0.5) == pytest.approx(-math.log1p(math.exp(-2.0)) - 0.25 * 2.0)
