"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ
BASE = 1500.0

N_MIN, N_MAX, DEFAULT_N = 6, 16, 12
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.01, 0.3
BUDGET_MIN, BUDGET_MAX, BUDGET_STEP, DEFAULT_BUDGET = 20, 600, 10, 120
STRATEGIES = ("Rundenturnier", "Zufällig", "Schweizer System (vereinfacht)", "Aktiv: Fehlordnung", "Aktiv: Information")
DEFAULT_STRATEGY = STRATEGIES[4]
ACTIVE = (STRATEGIES[3], STRATEGIES[4])
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.01, 1.0, 0.01, 0.1
ROUND_CHOICES, DEFAULT_ROUND = (1, 3, 6), 1
SEED_MAX, DEFAULT_SEED = 999999, 3

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_DELTA, EXP_BUDGET = 12, 0.3, 480
DELTA_NARROW = 0.12
RECORD_EVERY = 12                                          # Partien zwischen zwei Messpunkten der Verläufe
CHECKS = (60, 120, 240, 480)
TARGET = 0.90                                              # Paartreffer, bis zu dem gezählt wird
NOT_REACHED = EXP_BUDGET + 120                             # „Ziel nicht erreicht“ (Deckel der Partienzahl, 600)
NEIGHBOUR_GAP = 2                                          # „Nachbarn“: Rangabstand höchstens so groß
EXP_LAM = 0.1

COLORS = {"Rundenturnier": "#9aa0a6", "Zufällig": "#c9b6e0", "Schweizer System (vereinfacht)": "#8e6bb0", "Aktiv: Fehlordnung": "#c0392b", "Aktiv: Information": "#2F6B65"}
COLOR_TRUTH = "#e67e22"

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "budget": DEFAULT_BUDGET, "strategy": DEFAULT_STRATEGY, "lam": DEFAULT_LAM, "round": DEFAULT_ROUND, "seed": DEFAULT_SEED}
PRESETS = {
    "Rundenturnier gegen aktiv": dict(_BASE),
    "Wenige Partien": {**_BASE, "budget": 60},
    "Enges Feld": {**_BASE, "delta": DELTA_NARROW, "budget": 240},
    "Die naive Regel": {**_BASE, "strategy": STRATEGIES[3], "budget": 240},
}
PRESET_HELP = {
    "Rundenturnier gegen aktiv": "12 Spieler, 120 Partien: Die Information-Regel erreicht denselben Paartreffer mit etwa einem Viertel weniger Partien als das feste Rundenturnier.",
    "Wenige Partien": "Nur 60 Partien (5 je Spieler): Der Unterschied zwischen den Strategien ist klein, aber „Sieger richtig“ ist bei der aktiven Paarung häufiger.",
    "Enges Feld": "Stärkeabstand 0,12: Es gibt kaum etwas zu trennen: rund jeder zweite Lauf erreicht den Paartreffer 0,90 in 480 Partien nicht, und die aktive Paarung hat keinen Vorsprung.",
    "Die naive Regel": "Immer die Paarung mit der größten Fehlordnungswahrscheinlichkeit: Sie spielt dieselben Nachbarn immer wieder, das Budget wird verschwendet.",
}
