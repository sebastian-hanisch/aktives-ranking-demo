"""Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Alle Strategien sehen dieselbe Wahrheit (Stärken) und dieselben Würfe der Ausgänge; Hauptmaß ist der Paartreffer (Anteil richtig geordneter Spielerpaare, Gleichstand halb). Dazu die
Partien bis zum Paartreffer 0,90 und die Konzentration der Partien auf wenige Paare."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import akt_constants as C
import akt_model as M
import akt_strategies as S
from akt_scenario import Settings

NAN = float("nan")


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    runs: dict                  # Strategie -> Run
    final: dict                 # Strategie -> (Paartreffer, Sieger richtig, Top-3) am Ende
    to_target: dict             # Strategie -> Partien bis Paartreffer 0,90 (NOT_REACHED, wenn nicht erreicht)
    conc: dict                  # Strategie -> (größte Zahl Partien eines Paares, Anteil Nachbarpartien)


def analyse(s: Settings) -> Analysis:
    runs = {name: S.play_strategy(name, s.n, s.delta, s.budget, s.lam, s.round_size, s.seed) for name in C.STRATEGIES}
    final = {k: (float(r.accuracy[-1]), float(r.winner[-1]), float(r.top3[-1])) for k, r in runs.items()}
    return Analysis(s, runs, final, {k: S.games_to_target(r) for k, r in runs.items()}, {k: S.concentration(r) for k, r in runs.items()})


def verdict_analysis(a: Analysis):
    s = a.settings
    ref = a.final["Rundenturnier"][0]
    best = max(C.STRATEGIES, key=lambda k: a.final[k][0])
    text = (f"Paartreffer nach {s.budget} Partien: " + ", ".join(f"{k.split(' (')[0]} {de(a.final[k][0], 3)}" for k in C.STRATEGIES) + f"; am besten **{best}**. ")
    top_conc = max(C.STRATEGIES, key=lambda k: a.conc[k][0])
    text += f"Die größte Zahl Partien eines Paares: {int(a.conc[top_conc][0])} bei „{top_conc}“ (Rundenturnier: {int(a.conc['Rundenturnier'][0])})."
    gain = a.final["Aktiv: Information"][0] - ref
    if abs(gain) < 0.01:
        text += " **In diesem einen Lauf unterscheiden sich die Paartreffer von Information-Regel und Rundenturnier um weniger als 0,01** (ein Lauf sagt wenig; die Experimente unten mitteln über 60)."
    return "info", text


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=2048)
def _run(strategy: str, delta: float, seed: int, round_size: int = 1) -> S.Run:
    return S.play_strategy(strategy, C.EXP_N, delta, C.EXP_BUDGET, C.EXP_LAM, round_size, seed)


def _at(run: S.Run, games: int, field: str) -> float:
    return float(np.interp(games, run.games, getattr(run, field)))


def sweep_main(delta: float = C.EXP_DELTA, strategies=C.STRATEGIES, round_size: int = 1) -> dict:
    """Je Strategie über die festen Läufe: mittlere Verläufe (Paartreffer, Sieger, Top-3) an den Messpunkten, Werte bei den Budgets CHECKS, Partien bis 0,90 (Median, Mittel, Zahl der
    Läufe ohne Erreichen) und Konzentration (größte Zahl Partien eines Paares, Anteil Nachbarpartien)."""
    out = {}
    for name in strategies:
        rs = [_run(name, delta, sd, round_size if name in C.ACTIVE else 1) for sd in C.SWEEP_SEEDS]
        grid = np.arange(C.RECORD_EVERY, C.EXP_BUDGET + 1, C.RECORD_EVERY)
        acc = np.mean([[_at(r, g, "accuracy") for g in grid] for r in rs], axis=0)
        to = [S.games_to_target(r) for r in rs]
        conc = [S.concentration(r) for r in rs]
        out[name] = {"grid": grid, "accuracy": acc,
                     "at": {b: {"accuracy": _mean([_at(r, b, "accuracy") for r in rs]), "winner": _mean([_at(r, b, "winner") for r in rs]),
                                "top3": _mean([_at(r, b, "top3") for r in rs])} for b in C.CHECKS},
                     "to_target_median": float(np.median(to)), "to_target_mean": float(np.mean(to)), "not_reached": int(sum(t >= C.NOT_REACHED for t in to)),
                     "max_pair": _mean([c[0] for c in conc]), "near_share": _mean([c[1] for c in conc])}
    return out


def sweep_rounds(sizes=C.ROUND_CHOICES, delta: float = C.EXP_DELTA) -> list:
    """Experiment 5: Die Information-Regel mit 1, 3 und 6 Paaren je Runde (Paare einer Runde ohne gemeinsame Spieler)."""
    rows = []
    for k in sizes:
        res = sweep_main(delta, ("Aktiv: Information",), round_size=k)["Aktiv: Information"]
        rows.append({"round": k, "at120": res["at"][120]["accuracy"], "at240": res["at"][240]["accuracy"], "winner240": res["at"][240]["winner"],
                     "to_target_median": res["to_target_median"], "max_pair": res["max_pair"]})
    return rows


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def _short(name: str) -> str:
    return name.split(" (")[0]


def verdict_target(main):
    t = {k: v["to_target_median"] for k, v in main.items()}
    saved = 1.0 - t["Aktiv: Information"] / t["Rundenturnier"]
    rnd = t["Zufällig"] - t["Rundenturnier"]
    rnd_text = ("braucht ebenso viele Partien wie das feste Rundenturnier" if abs(rnd) <= 12 else
                ("braucht mehr Partien als das feste Rundenturnier (Wiederholungen)" if rnd > 0 else "braucht weniger Partien als das feste Rundenturnier"))
    return "info", (f"Median der Partien bis zum Paartreffer {de(C.TARGET)} (12 Spieler, Abstand {de(C.EXP_DELTA, 1)}): " + ", ".join(f"{_short(k)} **{int(t[k])}**" for k in C.STRATEGIES) +
                    f". Die Information-Regel spart {pct(saved)} % der Partien gegen das Rundenturnier und liegt "
                    f"{'gleichauf mit' if abs(t['Aktiv: Information'] - t['Schweizer System (vereinfacht)']) <= 12 else 'neben'} dem Schweizer System; die zufällige Paarung {rnd_text}.")


def verdict_budget(main):
    a240 = {k: v["at"][240] for k, v in main.items()}
    return "info", (f"Nach 240 Partien: Paartreffer " + ", ".join(f"{_short(k)} {de(a240[k]['accuracy'], 3)}" for k in C.STRATEGIES) + f"; **Sieger richtig** {de(a240['Aktiv: Information']['winner'])} "
                    f"gegen {de(a240['Rundenturnier']['winner'])} beim Rundenturnier und {de(a240['Schweizer System (vereinfacht)']['winner'])} beim Schweizer System. Der Vorsprung der aktiven Paarung "
                    "scheint bei der Spitze größer zu sein als bei der ganzen Rangfolge (mit 60 Läufen nicht gesichert).")


def verdict_concentration(main):
    f, i = main["Aktiv: Fehlordnung"], main["Aktiv: Information"]
    r = main["Rundenturnier"]
    return "warning", (f"Die naive Regel spielt im Mittel über die Läufe das meistgespielte Paar **{de(f['max_pair'], 0)} Mal** (Information-Regel {de(i['max_pair'], 0)}, Rundenturnier {de(r['max_pair'], 0)}), "
                       f"{pct(f['near_share'])} % ihrer Partien gehen zwischen Spielern mit Rangabstand höchstens {C.NEIGHBOUR_GAP} (Rundenturnier {pct(r['near_share'])} %). "
                       "Die naive Regel konzentriert die Partien auf wenige Nachbarpaare, die sich kaum trennen lassen; bei engem Feld kostet das Genauigkeit (Experiment 4).")


def verdict_narrow(main_narrow):
    nr = {k: v["not_reached"] for k, v in main_narrow.items()}
    a = {k: v["at"][240]["accuracy"] for k, v in main_narrow.items()}
    runs = len(C.SWEEP_SEEDS)
    gain = a["Aktiv: Information"] - a["Rundenturnier"]
    tail = ("Die Information-Regel liegt gleichauf mit dem Rundenturnier: Wo es kaum etwas zu trennen gibt, ist in diesen Läufen kein Vorteil der aktiven Paarung messbar. "
            if abs(gain) <= 0.01 else
            ("Die Information-Regel liegt vor dem Rundenturnier. " if gain > 0 else "Die Information-Regel liegt hinter dem Rundenturnier. "))
    worst = "Aktiv: Fehlordnung"
    tail += ("Die naive Regel liegt hinter allen anderen." if all(a[worst] < a[k] for k in C.STRATEGIES if k != worst) else "")
    return "warning", (f"Bei engem Feld (Abstand {de(C.DELTA_NARROW)}) erreicht in {C.EXP_BUDGET} Partien ein großer Teil der Läufe den Paartreffer {de(C.TARGET)} nicht: Läufe ohne Erreichen von {runs} "
                       f"beim Rundenturnier {nr['Rundenturnier']}, bei der Information-Regel {nr['Aktiv: Information']}, bei der naiven Regel {nr['Aktiv: Fehlordnung']}. Paartreffer nach 240 "
                       f"Partien: " + ", ".join(f"{_short(k)} {de(a[k], 3)}" for k in C.STRATEGIES) + ". " + tail)


def verdict_rounds(rows):
    r = {x["round"]: x for x in rows}
    lo, hi = r[min(r)], r[max(r)]
    return "info", (f"Mit {min(r)} Paar je Runde (nacheinander) erreicht die Information-Regel nach 240 Partien {de(lo['at240'], 3)}, mit {max(r)} Paaren je Runde (parallel) {de(hi['at240'], 3)}; "
                    f"Median der Partien bis {de(C.TARGET)}: {int(lo['to_target_median'])} gegen {int(hi['to_target_median'])}. Parallele Runden kosten Partien, weil spätere Paare die Ergebnisse "
                    "der Runde nicht kennen und spielerfremd gewählt werden müssen.")


def verdict_top(main):
    t = {k: v["at"][120] for k, v in main.items()}
    return "info", (f"Nach 120 Partien: Top-3-Treffer " + ", ".join(f"{_short(k)} {de(t[k]['top3'])}" for k in C.STRATEGIES) + f"; Sieger richtig {de(t['Aktiv: Information']['winner'])} bei der "
                    f"Information-Regel, {de(t['Rundenturnier']['winner'])} beim Rundenturnier. Beim Sieger ist der Abstand der Information-Regel zum Rundenturnier am größten, aber mit 60 Läufen nicht gesichert.")
