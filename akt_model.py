"""Schätzer und Maße: Bradley-Terry mit Ridge (Newton, warm) und seine Laplace-Kovarianz, Paartreffer, Sieger richtig, Top-k-Treffer. Nur numpy; Stärken in θ-Einheiten.

Daten: Siegmatrix W[i, j] (Siege von i gegen j) und Partienmatrix G = W + Wᵀ. MAP: maximiert Σ_{i<j} [W_ij ln σ(s_i − s_j) + W_ji ln σ(s_j − s_i)] − λ/2 |s|²;
die Hesse-Matrix des negativen Zielwerts ist H = Σ G_ij p_ij (1 − p_ij) (e_i − e_j)(e_i − e_j)ᵀ + λ I, die Kovarianz (Laplace) Σ = H⁻¹."""

import math

import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def log_posterior(s: np.ndarray, W: np.ndarray, lam: float) -> float:
    """Σ_{i≠j} W_ij ln σ(s_i − s_j) − λ/2 |s|² (bis auf eine Konstante)."""
    d = s[:, None] - s[None, :]
    return float(-(W * np.logaddexp(0.0, -d)).sum() - 0.5 * lam * s @ s)


def fit_bt(W: np.ndarray, G: np.ndarray, lam: float, s0=None, iters: int = 4) -> tuple:
    """Rückgabe: (s, Σ). `s0` ist der Startwert (warm). Newton-Schritte mit Rückwärtsschritt (Halbieren der Schrittweite, bis der Zielwert nicht sinkt): das volle Newton-Verfahren
    schwingt bei wenigen Partien und kleinem λ (Spieler ohne Sieg oder ohne Niederlage). Vollständige n × n-Rechnung (n ≤ 16)."""
    n = len(W)
    s = np.zeros(n) if s0 is None else np.asarray(s0, dtype=float).copy()
    eye = np.eye(n)
    f = log_posterior(s, W, lam)
    for _ in range(max(iters, 1)):
        P = sigmoid(s[:, None] - s[None, :])
        np.fill_diagonal(P, 0.5)
        grad = (W - G * P).sum(axis=1) - lam * s                       # Σ_j (W_ij − G_ij p_ij) − λ s_i
        h = G * P * (1.0 - P)
        np.fill_diagonal(h, 0.0)
        H = lam * eye + np.diag(h.sum(axis=1)) - h
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-6:
            fn = log_posterior(s + t * step, W, lam)
            if fn >= f - 1e-12:
                break
            t *= 0.5
        s = s + t * step
        f = log_posterior(s, W, lam)
    P = sigmoid(s[:, None] - s[None, :])
    np.fill_diagonal(P, 0.5)
    h = G * P * (1.0 - P)
    np.fill_diagonal(h, 0.0)
    H = lam * eye + np.diag(h.sum(axis=1)) - h
    return s, np.linalg.inv(H)


def pair_accuracy(est, truth) -> float:
    """Anteil richtig geordneter Spielerpaare; ein Gleichstand des Schätzers zählt halb (0,5 = Zufall)."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    iu = np.triu_indices(len(est), 1)
    se, st = np.sign(est[iu[0]] - est[iu[1]]), np.sign(truth[iu[0]] - truth[iu[1]])
    return float(np.where(se == 0, 0.5, (se == st).astype(float)).mean())


def winner_correct(est, truth) -> float:
    """1, wenn der Schätzer den wahren Besten allein an die Spitze setzt, 0,5 bei Gleichstand an der Spitze unter zwei, sonst 0 (bei mehr Gleichständen 1/Zahl)."""
    est = np.asarray(est, dtype=float)
    best = int(np.argmax(truth))
    top = est.max()
    tied = int((est == top).sum())
    return (1.0 / tied) if est[best] == top else 0.0


def top_k_overlap(est, truth, k: int) -> float:
    """Anteil der wahren Top-k, die der Schätzer unter seinen Top-k hat."""
    k = min(k, len(est))
    return len(set(np.argsort(-np.asarray(est), kind="stable")[:k].tolist()) & set(np.argsort(-np.asarray(truth), kind="stable")[:k].tolist())) / k


def pair_variance(Sigma: np.ndarray) -> np.ndarray:
    """Var(s_i − s_j) für alle Paare als Matrix."""
    d = np.diag(Sigma)
    return d[:, None] + d[None, :] - 2.0 * Sigma


def misorder_probability(s: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """P(Reihenfolge von i und j ist falsch) = Φ(−|s_i − s_j| / sd(s_i − s_j)) (Normalnäherung) als Matrix."""
    var = np.maximum(pair_variance(Sigma), 1e-12)
    z = np.abs(s[:, None] - s[None, :]) / np.sqrt(var)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def information_score(s: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """p (1 − p) · Var(s_i − s_j): Unsicherheit der Differenz mal Ungewissheit des Ausgangs, als Matrix."""
    p = sigmoid(s[:, None] - s[None, :])
    return p * (1.0 - p) * pair_variance(Sigma)
