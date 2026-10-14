"""Szenario: Reglerwerte (Settings); alle fünf Strategien laufen mit denselben Reglerwerten auf derselben Wahrheit und denselben Würfen."""

from dataclasses import dataclass

import akt_constants as C


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    budget: int = C.DEFAULT_BUDGET
    strategy: str = C.DEFAULT_STRATEGY        # die Strategie, deren Paare und Rangfolge im Detail gezeigt werden
    lam: float = C.DEFAULT_LAM
    round_size: int = C.DEFAULT_ROUND         # Paare je Runde der aktiven Strategien
    seed: int = C.DEFAULT_SEED
