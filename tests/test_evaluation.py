"""Analyse des gezeigten Laufs und die Verdikt-Zweige; die Experimente selbst prüft test_claims.py."""

import numpy as np
import pytest

import akt_constants as C
import akt_evaluation as ev
from akt_scenario import Settings


def test_german_number_formatting():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.pct(0.256) == "26" and ev.pct(0.256, 1) == "25,6"


def test_mean_skips_nan():
    assert ev._mean([1.0, float("nan"), 3.0]) == 2.0 and np.isnan(ev._mean([float("nan")]))


def test_analysis_covers_every_strategy_with_consistent_numbers():
    a = ev.analyse(Settings(n=8, budget=60, seed=2))
    assert set(a.runs) == set(C.STRATEGIES) == set(a.final) == set(a.to_target) == set(a.conc)
    for name, run in a.runs.items():
        assert run.budget == 60 and a.final[name][0] == run.accuracy[-1] and 0.0 <= a.final[name][0] <= 1.0
        assert a.conc[name][0] == run.G.max()


def test_all_strategies_see_the_same_truth():
    a = ev.analyse(Settings(n=8, budget=40, seed=5))
    first = a.runs[C.STRATEGIES[0]].strengths
    assert all(np.array_equal(first, r.strengths) for r in a.runs.values())


def _fake(final_ri, final_ai, conc=None):
    s = Settings(n=8, budget=100)
    final = {k: (0.8, 1.0, 1.0) for k in C.STRATEGIES}
    final["Rundenturnier"] = (final_ri, 0.0, 0.0)
    final["Aktiv: Information"] = (final_ai, 1.0, 1.0)
    cc = {k: (5.0, 0.3) for k in C.STRATEGIES}
    cc.update(conc or {})
    return ev.Analysis(s, {}, final, {k: 100 for k in C.STRATEGIES}, cc)


def test_verdict_flags_no_measurable_gain_only_for_a_small_difference():
    assert "nichts Messbares" in ev.verdict_analysis(_fake(0.90, 0.905))[1]
    assert "nichts Messbares" in ev.verdict_analysis(_fake(0.90, 0.895))[1]
    assert "nichts Messbares" not in ev.verdict_analysis(_fake(0.90, 0.92))[1]
    assert "nichts Messbares" in ev.verdict_analysis(_fake(0.90, 0.909))[1]
    assert "nichts Messbares" not in ev.verdict_analysis(_fake(0.90, 0.911))[1]


def test_verdict_names_the_best_strategy_and_the_most_concentrated_pair():
    a = _fake(0.8, 0.95, {"Aktiv: Fehlordnung": (99.0, 0.9)})
    text = ev.verdict_analysis(a)[1]
    assert "am besten **Aktiv: Information**" in text and "99 bei „Aktiv: Fehlordnung“ (Rundenturnier: 5)" in text


def test_short_name_drops_the_parenthesis():
    assert ev._short("Schweizer System (vereinfacht)") == "Schweizer System" and ev._short("Aktiv: Information") == "Aktiv: Information"


def test_interpolation_helper_reads_values_at_budgets():
    a = ev.analyse(Settings(n=6, budget=48, seed=1))
    run = a.runs["Rundenturnier"]
    assert ev._at(run, int(run.games[0]), "accuracy") == pytest.approx(run.accuracy[0])
    mid = (run.games[0] + run.games[1]) / 2
    assert ev._at(run, mid, "accuracy") == pytest.approx((run.accuracy[0] + run.accuracy[1]) / 2)


def _main_with(medians, not_reached=None, acc240=None):
    out = {}
    for k in C.STRATEGIES:
        out[k] = {"to_target_median": medians.get(k, 100), "not_reached": (not_reached or {}).get(k, 0), "at": {240: {"accuracy": (acc240 or {}).get(k, 0.5), "winner": 0.5, "top3": 0.5}}}
    return out


def test_target_verdict_compares_the_random_pairing_with_the_round_robin():
    base = {"Rundenturnier": 120, "Schweizer System (vereinfacht)": 96, "Aktiv: Information": 90}
    same = ev.verdict_target(_main_with({**base, "Zufällig": 132}))[1]
    assert "braucht ebenso viele Partien wie das feste Rundenturnier" in same and "spart 25 %" in same and "gleichauf mit dem Schweizer System" in same
    assert "ebenso viele" in ev.verdict_target(_main_with({**base, "Zufällig": 108}))[1]
    assert "mehr Partien als das feste Rundenturnier" in ev.verdict_target(_main_with({**base, "Zufällig": 133}))[1]
    assert "weniger Partien als das feste Rundenturnier" in ev.verdict_target(_main_with({**base, "Zufällig": 107}))[1]
    off = ev.verdict_target(_main_with({**base, "Zufällig": 120, "Aktiv: Information": 80, "Schweizer System (vereinfacht)": 98}))[1]
    assert "gleichauf" not in off and "neben dem Schweizer System" in off


def test_narrow_verdict_branches_and_counts():
    nr = {"Rundenturnier": 28, "Aktiv: Information": 25, "Aktiv: Fehlordnung": 52}
    acc = {"Rundenturnier": 0.5, "Zufällig": 0.5, "Schweizer System (vereinfacht)": 0.5, "Aktiv: Fehlordnung": 0.4, "Aktiv: Information": 0.5078125}
    t = ev.verdict_narrow(_main_with({}, nr, acc))[1]
    assert "Rundenturnier 28, bei der Information-Regel 25, bei der naiven Regel 52" in t and "gleichauf mit dem Rundenturnier" in t and "hinter allen anderen" in t
    assert "Fehlordnung 0,400" in t and "Information 0,508" in t
    vor = ev.verdict_narrow(_main_with({}, nr, {**acc, "Aktiv: Information": 0.515625}))[1]
    assert "liegt vor dem Rundenturnier" in vor and "gleichauf" not in vor
    hinter = ev.verdict_narrow(_main_with({}, nr, {**acc, "Aktiv: Information": 0.484375}))[1]
    assert "Die Information-Regel liegt hinter dem Rundenturnier" in hinter
    tie = ev.verdict_narrow(_main_with({}, nr, {**acc, "Aktiv: Fehlordnung": 0.5}))[1]
    assert "hinter allen anderen" not in tie
    not_worst = ev.verdict_narrow(_main_with({}, nr, {**acc, "Zufällig": 0.3}))[1]
    assert "hinter allen anderen" not in not_worst
