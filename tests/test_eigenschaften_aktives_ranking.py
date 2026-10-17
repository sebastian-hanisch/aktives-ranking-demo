"""Eigenschaftstests (metamorph) zum aktiven Ranking: Aussagen, die für das Verfahren gelten müssen, ohne dass das richtige Ergebnis bekannt ist (ORACLE-PLAYBOOK §5.2).

Jede Eigenschaft läuft über 20 bis 40 Zufallsinstanzen mit festem Seed plus Randfälle. Gilt eine Aussage nur unter Voraussetzungen, stehen sie beim Test:
- Umnummerieren der Spieler ändert die Wahl nur bis auf die Namen, solange keine Gleichstände vorkommen (Gleichstände entscheidet der Gleichstandsbrecher `noise`, der an der
  Paarposition hängt, und bei gleichem `noise` der kleinste Index in der triu-Reihenfolge; beides ist eine Konvention, keine Eigenschaft des Verfahrens).
- Der Strom der Würfe hängt an der Zeile der Partie, nicht an Budget oder Messraster: Eine kürzere Partiefolge ist ein Anfangsstück der längeren."""

import math

import numpy as np
import pytest

import akt_constants as C
import akt_model as M
import akt_strategies as S


def _counts(rng, n, games, spread=1.0):
    s = np.linspace(-spread, spread, n)
    W = np.zeros((n, n))
    for _ in range(games):
        a, b = rng.choice(n, 2, replace=False)
        if rng.random() < 1.0 / (1.0 + math.exp(-(s[a] - s[b]))):
            W[a, b] += 1
        else:
            W[b, a] += 1
    return W, W + W.T


def _sym_scores(rng, n):
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def _pair_set(pairs):
    return [frozenset(p) for p in pairs]


# --- 1. gleicher Seed, gleiche Paarungsfolge ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", C.STRATEGIES)
def test_same_seed_gives_the_same_pairing_sequence(strategy):
    for seed in (0, 1, 2, 17, 999999):
        for round_size in (1, 3):
            a = S.play_strategy(strategy, 7, 0.3, 36, 0.1, round_size, seed)
            b = S.play_strategy(strategy, 7, 0.3, 36, 0.1, round_size, seed)
            assert a.sequence == b.sequence
            assert np.array_equal(a.W, b.W) and np.array_equal(a.s_hat, b.s_hat)


# --- 2. Budget wird nie überschritten, jede Paarung hat zwei verschiedene Spieler -----------------------------------------------------------

def test_budget_is_never_exceeded_and_every_pairing_has_two_different_players():
    rng = np.random.default_rng(20261017)
    for _ in range(40):
        n = int(rng.integers(3, 10))
        budget = int(rng.integers(1, 61))
        strategy = C.STRATEGIES[int(rng.integers(0, len(C.STRATEGIES)))]
        round_size = int(rng.choice(C.ROUND_CHOICES))
        seed = int(rng.integers(0, 10 ** 6))
        r = S.play_strategy(strategy, n, 0.3, budget, 0.1, round_size, seed)
        assert len(r.sequence) == budget == r.budget
        assert r.G.sum() == 2 * budget and r.W.sum() == budget
        assert np.array_equal(r.G, r.G.T) and (np.diag(r.G) == 0).all()
        for a, b, a_wins in r.sequence:
            assert a != b and 0 <= a < n and 0 <= b < n
            assert isinstance(a_wins, bool)
        assert r.games[-1] == budget and (r.games <= budget).all()


def test_no_player_plays_twice_in_one_round():
    """Parallele Runde: die k Paare der aktiven Regel und die Paare des Schweizer Systems haben keinen Spieler gemeinsam (auch wenn die letzte Runde vom Budget gekappt wird)."""
    rng = np.random.default_rng(5)
    for _ in range(25):
        n = int(rng.integers(4, 11))
        budget = int(rng.integers(1, 50))
        seed = int(rng.integers(0, 10 ** 6))
        for strategy, size in (("Aktiv: Information", int(rng.choice((3, 6)))), ("Aktiv: Fehlordnung", 3), ("Schweizer System (vereinfacht)", None)):
            k = n // 2 if size is None else max(1, min(size, n // 2))
            r = S.play_strategy(strategy, n, 0.3, budget, 0.1, size or 1, seed)
            for start in range(0, budget, k):
                players = [p for a, b, _ in r.sequence[start:start + k] for p in (a, b)]
                assert len(set(players)) == len(players), (strategy, n, budget, seed, start)


# --- 3. Anfangsstück: Budget und Messraster ändern die Wahl nicht -----------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", C.STRATEGIES)
def test_a_shorter_budget_and_another_measurement_grid_give_a_prefix_of_the_same_sequence(strategy):
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = int(rng.integers(4, 9))
        seed = int(rng.integers(0, 10 ** 6))
        round_size = int(rng.choice(C.ROUND_CHOICES))
        long_run = S.play_strategy(strategy, n, 0.3, 45, 0.1, round_size, seed)
        short = S.play_strategy(strategy, n, 0.3, 21, 0.1, round_size, seed)
        other_grid = S.play_strategy(strategy, n, 0.3, 45, 0.1, round_size, seed, record_every=7)
        assert short.sequence == long_run.sequence[:21]
        assert other_grid.sequence == long_run.sequence


# --- 4. Umnummerieren der Spieler --------------------------------------------------------------------------------------------------------------

def test_relabelling_the_players_relabels_the_chosen_pairs_when_there_are_no_ties():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(4, 10))
        score = _sym_scores(rng, n)
        p = rng.permutation(n)                                             # neuer Spieler k ist der alte Spieler p[k]
        score2 = score[np.ix_(p, p)]
        npairs = n * (n - 1) // 2
        for size in (1, 2, n // 2):
            old = S.select_pairs(score, np.zeros(npairs), size)
            new = S.select_pairs(score2, np.zeros(npairs), size)
            assert _pair_set(old) == _pair_set([(int(p[a]), int(p[b])) for a, b in new])


def test_relabelling_commutes_with_the_estimate_the_covariance_and_both_selection_scores():
    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.integers(3, 8))
        lam = float(rng.choice((0.05, 0.1, 0.5)))
        W, G = _counts(rng, n, int(rng.integers(5, 80)))
        p = rng.permutation(n)
        W2, G2 = W[np.ix_(p, p)], G[np.ix_(p, p)]
        s, Sigma = M.fit_bt(W, G, lam, iters=40)
        s2, Sigma2 = M.fit_bt(W2, G2, lam, iters=40)
        assert s2 == pytest.approx(s[p], abs=1e-6)
        assert Sigma2 == pytest.approx(Sigma[np.ix_(p, p)], abs=1e-6)
        assert M.misorder_probability(s2, Sigma2) == pytest.approx(M.misorder_probability(s, Sigma)[np.ix_(p, p)], abs=1e-6)
        assert M.information_score(s2, Sigma2) == pytest.approx(M.information_score(s, Sigma)[np.ix_(p, p)], abs=1e-6)


def test_the_swiss_round_is_relabelled_with_the_players_when_the_tiebreaker_is_relabelled_too():
    rng = np.random.default_rng(6)
    for _ in range(30):
        n = int(rng.integers(4, 11))
        W, G = _counts(rng, n, int(rng.integers(0, 40)))
        noise = rng.random(n) + 0.01                                        # paarweise verschieden: keine Gleichstände
        p = rng.permutation(n)
        old = S.swiss_round(W, G, noise)
        new = S.swiss_round(W[np.ix_(p, p)], G[np.ix_(p, p)], noise[p])
        assert _pair_set(old) == _pair_set([(int(p[a]), int(p[b])) for a, b in new])


# --- 5. Spiegeln aller Ergebnisse kehrt die Schätzung um ---------------------------------------------------------------------------------------

def test_mirroring_all_results_negates_the_strengths_and_keeps_the_covariance():
    rng = np.random.default_rng(8)
    for _ in range(25):
        n = int(rng.integers(3, 8))
        lam = float(rng.choice((0.02, 0.1, 1.0)))
        W, G = _counts(rng, n, int(rng.integers(0, 70)))
        s, Sigma = M.fit_bt(W, G, lam, iters=40)
        sm, Sigmam = M.fit_bt(W.T, G, lam, iters=40)
        assert sm == pytest.approx(-s, abs=1e-6)
        assert Sigmam == pytest.approx(Sigma, abs=1e-6)
    # Randfall: keine Partie -> Schätzung null, Kovarianz λ⁻¹ I (Spiegeln ändert nichts)
    s0, S0 = M.fit_bt(np.zeros((4, 4)), np.zeros((4, 4)), 0.1)
    assert (s0 == 0).all() and S0 == pytest.approx(np.eye(4) / 0.1)


# --- 6. Gleichstandsregel -------------------------------------------------------------------------------------------------------------------

def test_ties_in_the_score_are_broken_by_the_noise_then_by_the_triu_index():
    n = 6
    npairs = n * (n - 1) // 2
    flat = np.full((n, n), 0.3)
    assert S.select_pairs(flat, np.zeros(npairs), 1) == [(0, 1)]                                  # alles gleich: kleinster triu-Index
    assert S.select_pairs(flat, np.zeros(npairs), n // 2) == [(0, 1), (2, 3), (4, 5)]
    i, j = S.pair_list(n)
    rng = np.random.default_rng(9)
    for _ in range(20):
        noise = rng.permutation(npairs).astype(float)                                              # verschieden: das größte `noise` gewinnt
        k = int(np.argmax(noise))
        assert S.select_pairs(flat, noise, 1) == [(int(i[k]), int(j[k]))]


@pytest.mark.parametrize("strategy", C.ACTIVE)
def test_before_any_game_the_active_rule_picks_the_same_pair_as_the_random_rule(strategy):
    """Vor der ersten Partie sind alle Paare gleich unsicher (Schätzung 0, Kovarianz λ⁻¹ I): die Wahl ist reiner Gleichstand und folgt dem Gleichstandsbrecher der Zeile 0,
    derselbe Wurf, der bei „Zufällig“ das Paar bestimmt."""
    for seed in range(20):
        for n in (3, 6, 9):
            active = S.play_strategy(strategy, n, 0.3, 1, 0.1, 1, seed)
            rand = S.play_strategy("Zufällig", n, 0.3, 1, 0.1, 1, seed)
            assert active.sequence[0][:2] == rand.sequence[0][:2]


# --- 7. Information-Regel und Fehlordnung an Handwerten mit bekannter Unsicherheit -------------------------------------------------------------

def test_information_rule_picks_the_pair_with_the_largest_variance_of_the_difference_by_hand():
    s = np.zeros(4)                                                        # alle Siegwahrscheinlichkeiten 1/2, p(1−p) = 1/4
    Sigma = np.diag([1.0, 2.0, 3.0, 4.0])
    q = M.information_score(s, Sigma)
    # Var(s_i − s_j) = Σ_ii + Σ_jj: 3, 4, 5, 5, 6, 7 für (01), (02), (03), (12), (13), (23)
    for (a, b), v in {(0, 1): 3, (0, 2): 4, (0, 3): 5, (1, 2): 5, (1, 3): 6, (2, 3): 7}.items():
        assert q[a, b] == pytest.approx(0.25 * v)
    assert S.select_pairs(q, np.zeros(6), 1) == [(2, 3)]
    assert S.select_pairs(q, np.zeros(6), 2) == [(2, 3), (0, 1)]
    # Kovarianz zwischen 2 und 3 (Σ_23 = 2,5) senkt Var(s_2 − s_3) auf 3 + 4 − 5 = 2: nun ist (1, 3) mit 6 vorn
    Sigma2 = Sigma.copy()
    Sigma2[2, 3] = Sigma2[3, 2] = 2.5
    q2 = M.information_score(s, Sigma2)
    assert q2[2, 3] == pytest.approx(0.25 * 2.0)
    assert S.select_pairs(q2, np.zeros(6), 1) == [(1, 3)]


def test_misorder_rule_by_hand_close_pair_and_large_uncertainty():
    s = np.array([0.0, 1.0, 4.0])
    m = M.misorder_probability(s, np.eye(3))
    assert m[0, 1] == pytest.approx(0.5 * math.erfc((1.0 / math.sqrt(2.0)) / math.sqrt(2.0)))        # Φ(−1/√2)
    assert S.select_pairs(m, np.zeros(3), 1) == [(0, 1)]                                              # kleinster Abstand bei gleicher Unsicherheit
    # Spieler 2 ist unsicher (Σ_22 = 9): Abstände 0,5 / 0,5 / 1,0, aber sd = 0,141 / 3,0 / 3,0 -> (1, 2) hat die größte Fehlordnungswahrscheinlichkeit
    s2 = np.array([0.0, 0.5, 1.0])
    Sigma = np.diag([0.01, 0.01, 9.0])
    m2 = M.misorder_probability(s2, Sigma)
    assert m2[1, 2] == pytest.approx(0.5 * math.erfc((0.5 / math.sqrt(9.01)) / math.sqrt(2.0)))
    assert S.select_pairs(m2, np.zeros(3), 1) == [(1, 2)]
    assert S.select_pairs(M.information_score(s2, Sigma), np.zeros(3), 1) == [(1, 2)]


# --- 8. Maße: Invarianz unter positiver affiner Umrechnung, Spiegelung ------------------------------------------------------------------------

def test_order_measures_ignore_positive_affine_changes_and_mirror_under_negation():
    rng = np.random.default_rng(12)
    for _ in range(40):
        n = int(rng.integers(3, 13))
        truth = np.sort(rng.normal(size=n))
        est = rng.normal(size=n)
        a, b = float(rng.uniform(0.1, 20.0)), float(rng.normal(scale=5.0))
        assert M.pair_accuracy(a * est + b, truth) == pytest.approx(M.pair_accuracy(est, truth))
        assert M.pair_accuracy(est, a * truth + b) == pytest.approx(M.pair_accuracy(est, truth))
        assert M.pair_accuracy(-est, truth) == pytest.approx(1.0 - M.pair_accuracy(est, truth))          # keine Gleichstände
        assert M.pair_accuracy(truth, truth) == 1.0 and M.pair_accuracy(-truth, truth) == 0.0
        assert M.winner_correct(a * est + b, truth) == M.winner_correct(est, truth)
        assert M.top_k_overlap(a * est + b, truth, 3) == pytest.approx(M.top_k_overlap(est, truth, 3))
        assert 0.0 <= M.top_k_overlap(est, truth, 3) <= 1.0 and M.winner_correct(est, truth) in (0.0, 1.0)
    # Randfall: vollständiger Gleichstand des Schätzers = Zufall
    assert M.pair_accuracy(np.zeros(5), np.arange(5.0)) == 0.5
    assert M.winner_correct(np.zeros(4), np.arange(4.0)) == 0.25
