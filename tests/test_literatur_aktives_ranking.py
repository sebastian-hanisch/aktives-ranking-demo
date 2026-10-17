"""Literatur-Referenzwerte zum aktiven Ranking (ORACLE-PLAYBOOK §5.1). Nur Werte, die in der genannten Quelle eingesehen wurden; keiner stammt aus dem Code der Demo.

Wert 1: Bradley-Terry-Maximum-Likelihood-Schätzung der Zeitschriften-Zitationsdaten (Stigler 1994, übernommen aus Agresti 2002, S. 448).
  Quelle: Turner, H., Firth, D. (2012): Bradley-Terry models in R: The BradleyTerry2 package. Journal of Statistical Software 48(9); Vignette des Pakets BradleyTerry2,
  Abschnitt 2.1 „Example: Analysis of journal citations“, https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html (Referenzsoftware BTm).
  Fundstelle: Zählmatrix „cited/citing“ (4 × 4: Biometrika, Comm Statist, JASA, JRSS-B) und die gedruckten Koeffizienten von `citeModel` (Biometrika = 0 als Identifikation):
  Comm Statist −2.9491, JASA −0.4796, JRSS-B 0.2690; mit Bezugsjournal JASA: Biometrika 0.4796, Comm Statist −2.4695, JRSS-B 0.7485. Gewinner = zitiertes Journal, Verlierer =
  zitierendes Journal; die Diagonale (Selbstzitate) geht nicht ein. Eingesehen am 2026-10-17.
  Grenze: Standardfehler dieses Modells stehen in der eingesehenen Fassung nicht; die Laplace-Kovarianz der Demo ist daher nicht gegen die Quelle geprüft. Die Daten des
  Baseball-Beispiels (42 Zeilen) waren nicht vollständig einsehbar; kein Test.
Die ML-Schätzung ist hier der Grenzwert des Ridge-Schätzers `fit_bt` für λ → 0 (die Verschiebung ist unbestimmt; verglichen werden Differenzen)."""

import numpy as np
import pytest

import akt_model as M

# Zeilen: zitiert (Gewinner), Spalten: zitierend (Verlierer); Reihenfolge Biometrika, Comm Statist, JASA, JRSS-B (Quelle, Abschnitt 2.1)
CITATIONS = np.array([[714, 730, 498, 221],
                      [33, 425, 68, 17],
                      [320, 813, 1072, 142],
                      [284, 276, 325, 188]], dtype=float)


def _wins():
    W = CITATIONS.copy()
    np.fill_diagonal(W, 0.0)
    return W, W + W.T


def test_bradley_terry_maximum_likelihood_matches_the_published_journal_citation_fit():
    W, G = _wins()
    s, _ = M.fit_bt(W, G, 1e-8, iters=100)
    assert (s - s[0])[1:] == pytest.approx([-2.9491, -0.4796, 0.2690], abs=6e-5)           # Bezug Biometrika
    assert (s - s[2]) == pytest.approx([0.4796, -2.4695, 0.0, 0.7485], abs=1.2e-4)         # Bezug JASA (gedruckt auf vier Stellen)
