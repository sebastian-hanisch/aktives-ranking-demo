"""Paarwahl und Lauf: Rundenturnier, Zufall, Schweizer Runde, aktive Auswahl, Budget, Messpunkte, Konzentration."""

import numpy as np
import pytest

import akt_constants as C
import akt_strategies as S
from conftest import counts


def _run(strategy, n=8, budget=80, seed=1, delta=0.3, round_size=1):
    return S.play_strategy(strategy, n, delta, budget, 0.1, round_size, seed)


@pytest.mark.parametrize("strategy", C.STRATEGIES)
def test_every_strategy_plays_exactly_the_budget_and_records_the_games(strategy):
    r = _run(strategy, budget=50)
    assert r.budget == 50 and r.G.sum() == 100 and r.W.sum() == 50
    assert np.array_equal(r.G, r.W + r.W.T) and (np.diag(r.G) == 0).all()
    assert r.games[-1] == 50 and (np.diff(r.games) > 0).all() and len(r.accuracy) == len(r.games)


def test_same_inputs_same_run_and_other_seed_other_run():
    a, b, c = _run("Aktiv: Information", seed=3), _run("Aktiv: Information", seed=3), _run("Aktiv: Information", seed=4)
    assert a.sequence == b.sequence and not np.array_equal(a.W, c.W)


def test_all_strategies_share_the_truth_and_the_first_outcome_stream():
    runs = [_run(name, seed=7) for name in C.STRATEGIES]
    assert all(np.array_equal(runs[0].strengths, r.strengths) for r in runs)


def test_round_robin_plays_every_pair_before_repeating():
    n = 6
    r = S.play_strategy("Rundenturnier", n, 0.3, 15, 0.1, 1, 2)
    assert r.G[np.triu_indices(n, 1)].tolist() == [1.0] * 15
    r2 = S.play_strategy("Rundenturnier", n, 0.3, 30, 0.1, 1, 2)
    assert r2.G[np.triu_indices(n, 1)].tolist() == [2.0] * 15


def test_swiss_round_pairs_every_player_once_and_avoids_repeats_when_possible():
    W, G = counts(6, {(0, 1): 1, (2, 3): 1, (4, 5): 1})
    noise = np.linspace(0, 1, 6)
    pairs = S.swiss_round(W, G, noise)
    players = [p for pr in pairs for p in pr]
    assert sorted(players) == list(range(6)) and len(pairs) == 3
    assert all(G[a, b] == 0 for a, b in pairs)


def test_swiss_round_pairs_players_with_equal_points_and_leaves_one_out_for_an_odd_field():
    W, G = counts(5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (4, 1): 1})
    pairs = S.swiss_round(W, G, np.zeros(5))
    assert len(pairs) == 2 and pairs[0][0] == 0                            # der Punktbeste (3 Siege) spielt zuerst
    assert len({p for pr in pairs for p in pr}) == 4


def test_select_pairs_takes_the_best_disjoint_pairs():
    score = np.array([[0, 5, 4, 1], [5, 0, 3, 2], [4, 3, 0, 9], [1, 2, 9, 0]], dtype=float)
    assert S.select_pairs(score, np.zeros(6), 1) == [(2, 3)]
    assert S.select_pairs(score, np.zeros(6), 2) == [(2, 3), (0, 1)]
    assert S.select_pairs(score, np.zeros(6), 5) == [(2, 3), (0, 1)]       # mehr als zwei disjunkte Paare gibt es nicht


def test_select_pairs_breaks_ties_with_the_noise():
    score = np.zeros((3, 3))
    noise = np.array([0.1, 0.9, 0.5])                                      # Paare in triu-Reihenfolge: (0,1), (0,2), (1,2)
    assert S.select_pairs(score, noise, 1) == [(0, 2)]


def test_active_strategy_plays_each_pair_before_any_pair_twice_at_the_start():
    n = 5
    r = S.play_strategy("Aktiv: Information", n, 0.3, 10, 0.1, 1, 4)
    assert r.G[np.triu_indices(n, 1)].max() <= 2


def test_parallel_rounds_never_use_a_player_twice_in_a_round():
    r = S.play_strategy("Aktiv: Information", 8, 0.3, 40, 0.1, 4, 5)
    for k in range(0, 40, 4):
        players = [p for a, b, _ in r.sequence[k:k + 4] for p in (a, b)]
        assert len(set(players)) == len(players)


def test_games_to_target_finds_the_first_measurement_point_or_the_deckel():
    r = _run("Rundenturnier", n=6, budget=60)
    first = S.games_to_target(r, target=0.0)
    assert first == int(r.games[0])
    assert S.games_to_target(r, target=1.1) == C.NOT_REACHED and S.games_to_target(r, target=1.1, not_reached=7) == 7


def test_concentration_by_hand():
    r = _run("Rundenturnier", n=6, budget=15)
    mx, near = S.concentration(r, gap=1)
    assert mx == 1.0 and near == pytest.approx(5 / 15)                      # 5 Paare mit Rangabstand 1 unter 15
    assert S.concentration(r, gap=10)[1] == pytest.approx(1.0)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        S.play_strategy("Unbekannt", 8, 0.3, 10, 0.1, 1, 1)
    with pytest.raises(ValueError):
        S.play_strategy("Zufällig", 2, 0.3, 10, 0.1, 1, 1)
    with pytest.raises(ValueError):
        S.play_strategy("Zufällig", 8, 0.3, 0, 0.1, 1, 1)


def test_smallest_valid_instances_run():
    r = S.play_strategy("Aktiv: Information", 3, 0.3, 5, 0.1, 1, 1)                  # 3 Spieler sind erlaubt
    assert r.budget == 5 and r.G.sum() == 10
    r1 = S.play_strategy("Rundenturnier", 8, 0.3, 1, 0.1, 1, 1)                      # eine Partie ist erlaubt
    assert r1.G.sum() == 2 and list(r1.games) == [1]


def test_measurement_points_are_multiples_of_the_step_and_the_last_is_the_budget():
    r = S.play_strategy("Aktiv: Information", 8, 0.3, 50, 0.1, 1, 2, record_every=12)
    assert list(r.games) == [12, 24, 36, 48, 50]
    r6 = S.play_strategy("Rundenturnier", 8, 0.3, 36, 0.1, 6, 2, record_every=12)
    assert list(r6.games) == [12, 24, 36]


def test_games_to_target_counts_equality_as_reached():
    run = type("R", (), {"accuracy": np.array([0.5, 0.9, 1.0]), "games": np.array([12, 24, 36])})()
    assert S.games_to_target(run, target=0.9) == 24
