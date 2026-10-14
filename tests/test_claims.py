"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import akt_constants as C
import akt_evaluation as ev

RR, RND, SW, AF, AI = C.STRATEGIES


@pytest.fixture(scope="module")
def main():
    return ev.sweep_main()


@pytest.fixture(scope="module")
def narrow():
    return ev.sweep_main(C.DELTA_NARROW)


@pytest.fixture(scope="module")
def rounds():
    return {r["round"]: r for r in ev.sweep_rounds()}


def test_the_measurement_grid_runs_from_the_step_to_the_full_budget(main):
    grid = main[RR]["grid"]
    assert grid[0] == C.RECORD_EVERY and grid[-1] == C.EXP_BUDGET and set(grid[1:] - grid[:-1]) == {C.RECORD_EVERY}
    assert all(main[k]["accuracy"].shape == grid.shape for k in C.STRATEGIES)


# --- Experiment 1: Partien bis 0,90 -----------------------------------------------------------------------------------------------------------

def test_games_to_the_target(main):
    med = {k: main[k]["to_target_median"] for k in C.STRATEGIES}
    assert med[RR] == pytest.approx(120, abs=12) and med[RND] == pytest.approx(120, abs=12) and med[SW] == pytest.approx(96, abs=12)
    assert med[AF] == pytest.approx(126, abs=18) and med[AI] == pytest.approx(90, abs=12)
    assert med[AI] < med[RR] and med[AI] < med[RND]
    assert abs(med[RND] - med[RR]) <= 12                                        # zufällige Paarung ≈ festes Rundenturnier (Befund, nicht Annahme)
    assert abs(med[AI] - med[SW]) <= 18
    assert 1 - med[AI] / med[RR] == pytest.approx(0.25, abs=0.12)
    assert main[AI]["not_reached"] == 0 and main[SW]["not_reached"] == 0 and main[RR]["not_reached"] == 0 and 0 < main[AF]["not_reached"] <= 6


# --- Experiment 2: festes Budget --------------------------------------------------------------------------------------------------------------

def test_accuracy_at_fixed_budgets(main):
    expected = {RR: (0.846, 0.892, 0.929, 0.958), RND: (0.841, 0.891, 0.926, 0.959), SW: (0.855, 0.900, 0.937, 0.960), AF: (0.860, 0.904, 0.937, 0.967),
                AI: (0.859, 0.912, 0.944, 0.971)}
    for name, vals in expected.items():
        got = [main[name]["at"][b]["accuracy"] for b in C.CHECKS]
        assert got == pytest.approx(vals, abs=0.006), name
    assert main[AI]["at"][240]["accuracy"] > main[RR]["at"][240]["accuracy"] + 0.01
    assert main[AI]["at"][120]["accuracy"] > main[RR]["at"][120]["accuracy"] + 0.01


def test_winner_and_top_three(main):
    w = {k: main[k]["at"][240]["winner"] for k in C.STRATEGIES}
    assert w[AI] == pytest.approx(0.73, abs=0.08) and w[RR] == pytest.approx(0.60, abs=0.08) and w[SW] == pytest.approx(0.68, abs=0.08) and w[AF] == pytest.approx(0.77, abs=0.08)
    assert w[AI] > w[RR] + 0.08
    assert main[AI]["at"][60]["winner"] > main[RR]["at"][60]["winner"] + 0.04            # Preset „Wenige Partien“
    assert main[AI]["at"][60]["winner"] == pytest.approx(0.53, abs=0.08) and main[RR]["at"][60]["winner"] == pytest.approx(0.42, abs=0.08)
    t = {k: main[k]["at"][120]["top3"] for k in C.STRATEGIES}
    assert t[AI] > t[RR] and t[AI] > t[RND] and t[AI] == pytest.approx(0.83, abs=0.06) and t[RR] == pytest.approx(0.81, abs=0.06)


# --- Experiment 3: Konzentration --------------------------------------------------------------------------------------------------------------

def test_the_naive_rule_concentrates_the_games(main):
    assert main[AF]["max_pair"] == pytest.approx(144.8, abs=12) and main[AI]["max_pair"] == pytest.approx(26.0, abs=5) and main[RR]["max_pair"] == pytest.approx(8.0, abs=0.5)
    assert main[SW]["max_pair"] == pytest.approx(10.6, abs=1.5) and main[RND]["max_pair"] == pytest.approx(14.4, abs=2.5)
    assert main[AF]["near_share"] == pytest.approx(0.86, abs=0.04) and main[AI]["near_share"] == pytest.approx(0.64, abs=0.05)
    assert main[RR]["near_share"] == pytest.approx(0.32, abs=0.02) and main[SW]["near_share"] == pytest.approx(0.34, abs=0.03) and main[RND]["near_share"] == pytest.approx(0.33, abs=0.03)


# --- Experiment 4: enges Feld -----------------------------------------------------------------------------------------------------------------

def test_a_narrow_field_gives_the_active_rule_no_advantage(narrow):
    med = {k: narrow[k]["to_target_median"] for k in C.STRATEGIES}
    assert med[AF] == C.NOT_REACHED and all(med[k] < C.NOT_REACHED for k in (RR, RND, SW, AI))
    assert narrow[AF]["not_reached"] == pytest.approx(52, abs=6)
    for k in (RR, RND, SW, AI):
        assert narrow[k]["not_reached"] == pytest.approx(26, abs=6) and 0.35 <= narrow[k]["not_reached"] / len(C.SWEEP_SEEDS) <= 0.55, k   # „rund jeder zweite“
    assert narrow[AF]["not_reached"] > narrow[AI]["not_reached"] + 15
    at = {k: narrow[k]["at"][240]["accuracy"] for k in C.STRATEGIES}
    assert (at[RR], at[RND], at[SW], at[AF], at[AI]) == pytest.approx((0.828, 0.825, 0.823, 0.775, 0.828), abs=0.008)
    assert abs(at[AI] - at[RR]) <= 0.01                                        # kein Vorsprung
    assert at[AF] < min(at[k] for k in (RR, RND, SW, AI)) - 0.04


# --- Experiment 5: Runden ---------------------------------------------------------------------------------------------------------------------

def test_parallel_rounds_cost_a_little(rounds):
    assert rounds[1]["at240"] == pytest.approx(0.944, abs=0.006) and rounds[6]["at240"] == pytest.approx(0.935, abs=0.006)
    assert rounds[1]["at120"] == pytest.approx(0.912, abs=0.006) and rounds[6]["at120"] == pytest.approx(0.902, abs=0.006)
    assert rounds[1]["at240"] > rounds[6]["at240"] and rounds[1]["to_target_median"] <= rounds[6]["to_target_median"]


# --- Verdikte nennen die Zahlen aus den Daten -------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(main, narrow, rounds):
    t = ev.verdict_target(main)[1]
    assert "Rundenturnier **120**" in t and "Aktiv: Information **90**" in t
    c = ev.verdict_concentration(main)[1]
    assert "bis zu **145 Mal**" in c and "(Information-Regel 26, Rundenturnier 8)" in c
    assert "**Sieger richtig** 0,73 gegen 0,60 beim Rundenturnier und 0,68 beim Schweizer System" in ev.verdict_budget(main)[1]
    assert "braucht ebenso viele Partien wie das feste Rundenturnier" in t and "spart 25 %" in t
    n = ev.verdict_narrow(narrow)[1]
    assert "Rundenturnier 28, bei der Information-Regel 25, bei der naiven Regel 52" in n and "gleichauf mit dem Rundenturnier" in n and "Die naive Regel liegt hinter allen anderen" in n
    r = ev.verdict_rounds(list(rounds.values()))[1]
    assert "0,944" in r and "0,935" in r and "90 gegen 96" in r
