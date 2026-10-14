"""Paarwahl und Lauf: Rundenturnier, zufällige Paarung, vereinfachtes Schweizer System und die aktive Paarung (nach Fehlordnung oder nach Information). Zufall nur über SplitMix64
(Generator der Linie); die Zufallszahlen eines Laufs haben eine feste Anordnung, die nicht von der Strategie abhängt, damit alle Strategien auf derselben Wahrheit und denselben
Würfen verglichen werden:

  Zeile g (die g-te Partie): Spalte 0 = Ausgang, Spalten 1 … = Gleichstandsbrecher je Paar (bzw. je Spieler beim Schweizer System).

Wahrheit: n Spieler mit gleichabständigen Stärken (Generator, Mittel 0, aufsteigend: Spieler n − 1 ist der Beste); P(i schlägt j) = σ(s_i − s_j)."""

import functools
from dataclasses import dataclass

import numpy as np

import akt_constants as C
import akt_model as M
import turnier_generator as T

_SALT = 0xA17C


@dataclass(frozen=True)
class Run:
    strategy: str
    n: int
    strengths: np.ndarray       # wahre Stärken (aufsteigend)
    W: np.ndarray
    G: np.ndarray
    s_hat: np.ndarray           # Schätzung am Ende
    Sigma: np.ndarray           # Laplace-Kovarianz am Ende
    sequence: tuple             # ((a, b, a_gewinnt), ...) in Spielreihenfolge
    games: np.ndarray           # Zahl der Partien an den Messpunkten
    accuracy: np.ndarray        # Paartreffer an den Messpunkten
    winner: np.ndarray          # „Sieger richtig“ an den Messpunkten
    top3: np.ndarray            # Top-3-Treffer an den Messpunkten

    @property
    def budget(self) -> int:
        return len(self.sequence)


@functools.lru_cache(maxsize=None)
def pair_list(n: int) -> tuple:
    """Die Paare i < j in der Reihenfolge von triu (nicht zu verändern)."""
    i, j = np.triu_indices(n, 1)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def swiss_round(W: np.ndarray, G: np.ndarray, noise: np.ndarray) -> list:
    """Eine Runde des vereinfachten Schweizer Systems: nach Punkten sortiert (Gleichstand: Gleichstandsbrecher), jeder freie Spieler gegen den nächsten freien, bevorzugt gegen einen
    Gegner, gegen den er am seltensten gespielt hat. Bei ungerader Zahl setzt einer aus."""
    n = len(W)
    score = W.sum(axis=1) + 1e-3 * noise[:n]
    order = [int(x) for x in np.argsort(-score, kind="stable")]
    pairs, used = [], set()
    for a in order:
        if a in used:
            continue
        cand = [b for b in order if b != a and b not in used]
        if not cand:
            continue
        least = min(G[a, b] for b in cand)
        b = next(b for b in cand if G[a, b] == least)
        pairs.append((a, b))
        used.update((a, b))
    return pairs


def select_pairs(score: np.ndarray, noise: np.ndarray, size: int) -> list:
    """Die `size` Paare mit den größten Werten der Matrix `score` (i < j), ohne dass ein Spieler zweimal vorkommt; Gleichstände durch `noise` (je Paar in der Reihenfolge von triu)."""
    n = len(score)
    i, j = pair_list(n)
    val = score[i, j] + 1e-9 * noise[:len(i)]
    chosen, used = [], set()
    for idx in np.argsort(-val, kind="stable"):
        a, b = int(i[idx]), int(j[idx])
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used.update((a, b))
        if len(chosen) >= size:
            break
    return chosen


def play_strategy(strategy: str, n: int, delta: float, budget: int, lam: float = C.DEFAULT_LAM, round_size: int = 1, seed: int = 0,
                  record_every: int = C.RECORD_EVERY) -> Run:
    if strategy not in C.STRATEGIES:
        raise ValueError(f"unbekannte Strategie: {strategy}")
    if n < 3 or budget < 1:
        raise ValueError("Mindestens 3 Spieler und eine Partie")
    s = T.make_strengths(n, delta)
    i_all, j_all = pair_list(n)
    npairs = len(i_all)
    width = 1 + max(npairs, n)
    rng = T.SplitMix64(int(seed) ^ _SALT)
    U = rng.uniform((budget + n) * width).reshape(budget + n, width)
    order = np.argsort(U[0, 1:1 + npairs], kind="stable")                       # feste Reihenfolge der Paare des Rundenturniers
    W = np.zeros((n, n))
    G = np.zeros((n, n))
    s_hat = np.zeros(n)
    Sigma = np.eye(n) / lam
    seq = []
    games, acc, win, top3 = [], [], [], []
    next_record = record_every

    def play(a: int, b: int):
        row = len(seq)
        a_wins = bool(U[row, 0] < M.sigmoid(s[a] - s[b]))
        if a_wins:
            W[a, b] += 1
        else:
            W[b, a] += 1
        G[a, b] += 1
        G[b, a] += 1
        seq.append((a, b, a_wins))

    while len(seq) < budget:
        row = len(seq)
        if strategy == "Rundenturnier":
            k = int(order[row % npairs])
            pairs = [(int(i_all[k]), int(j_all[k]))]
        elif strategy == "Zufällig":
            k = int(np.argmax(U[row, 1:1 + npairs]))
            pairs = [(int(i_all[k]), int(j_all[k]))]
        elif strategy == "Schweizer System (vereinfacht)":
            pairs = swiss_round(W, G, U[row, 1:1 + n])
        else:
            score = M.misorder_probability(s_hat, Sigma) if strategy == "Aktiv: Fehlordnung" else M.information_score(s_hat, Sigma)
            pairs = select_pairs(score, U[row, 1:1 + npairs], max(1, min(round_size, n // 2)))
        for a, b in pairs:
            if len(seq) >= budget:
                break
            play(a, b)
        if strategy in C.ACTIVE:
            s_hat, Sigma = M.fit_bt(W, G, lam, s_hat, iters=3)
        while len(seq) >= next_record:
            if strategy not in C.ACTIVE:
                s_hat, Sigma = M.fit_bt(W, G, lam, s_hat, iters=8)
            games.append(len(seq))
            acc.append(M.pair_accuracy(s_hat, s))
            win.append(M.winner_correct(s_hat, s))
            top3.append(M.top_k_overlap(s_hat, s, 3))
            next_record += record_every
    s_hat, Sigma = M.fit_bt(W, G, lam, s_hat, iters=8)
    if not games or games[-1] != len(seq):
        games.append(len(seq))
        acc.append(M.pair_accuracy(s_hat, s))
        win.append(M.winner_correct(s_hat, s))
        top3.append(M.top_k_overlap(s_hat, s, 3))
    return Run(strategy, n, s, W, G, s_hat, Sigma, tuple(seq), np.array(games), np.array(acc), np.array(win), np.array(top3))


def games_to_target(run: Run, target: float = C.TARGET, not_reached: int = C.NOT_REACHED) -> int:
    """Zahl der Partien am ersten Messpunkt mit Paartreffer ≥ target; sonst `not_reached`."""
    hit = np.nonzero(run.accuracy >= target)[0]
    return int(run.games[hit[0]]) if len(hit) else not_reached


def concentration(run: Run, gap: int = C.NEIGHBOUR_GAP) -> tuple:
    """(größte Zahl Partien eines Paares, Anteil der Partien zwischen Spielern, deren wahrer Rangabstand höchstens `gap` ist)."""
    n = run.n
    total = float(run.G.sum() / 2.0)
    near = sum(run.G[i, j] for i in range(n) for j in range(i + 1, n) if j - i <= gap)
    return float(run.G.max()), float(near / total) if total else 0.0
