# Aktives Ranking: welche Paarung als Nächstes? – Streamlit-Demo

Zwölftes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `bradley-terry-demo`, `glicko-trueskill-demo` und `nicht-transitiv-demo`: Ein fester Spielplan spielt jedes Paar gleich
oft, obwohl manche Paare nach wenigen Partien klar entschieden sind und andere nicht. Das **aktive Ranking** wählt die nächste Partie nach der Unsicherheit der Rangfolge. Gemessen wird,
**wie viele Partien das spart** gegen das Rundenturnier, die zufällige Paarung und ein vereinfachtes Schweizer System, **warum die naheliegende Regel das Budget verschwendet**, **wann es
nichts zu gewinnen gibt** und was **parallele Runden** kosten. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Stärken aus dem Generator Version 1, wortgleich); alles ist
simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wie viele Partien spart die aktive Paarung? **Etwa ein Viertel:** 12 Spieler, Stärkeabstand 0,3, Median der Partien bis zum Paartreffer 0,90 bei der Information-Regel **90** gegen **120**
beim Rundenturnier (**25 %**), gleichauf mit dem vereinfachten Schweizer System (96). Die zufällige Paarung braucht ebenso viele Partien wie das Rundenturnier (120). Bei festem Budget ist der
Vorsprung klein, aber stetig: Paartreffer **0,944** gegen 0,929 nach 240 Partien; der **wahre Beste** steht in 0,73 der Läufe oben gegen 0,60. **Die naheliegende Regel verschwendet das Budget:**
Wer immer das Paar mit der größten Fehlordnungswahrscheinlichkeit spielt, spielt dasselbe Paar bis zu **145 Mal** (Information-Regel 26, Rundenturnier 8), 86 % ihrer Partien gehen zwischen
nahen Nachbarn, und bei engem Feld (Abstand 0,12) liegt sie hinter allen anderen (0,775 gegen 0,828 nach 240 Partien). **Bei engem Feld gewinnt auch die Information-Regel nichts:** gleichauf
mit dem Rundenturnier (0,828); rund jeder zweite Lauf erreicht 0,90 in 480 Partien nicht. Parallele Runden (6 Paare je Runde) kosten rund 0,01 Paartreffer.

## Modell

- **Daten:** Siegmatrix W (Siege von i gegen j) und Partienmatrix G = W + Wᵀ; wahre Siegwahrscheinlichkeit σ(s_i − s_j) mit den Stärken des Generators (12 Spieler, gleichmäßiger
  Abstand δ), Ausgang und Gleichstandsbrecher als feste Würfe je Partie, damit alle Strategien dieselbe Wahrheit und dieselben Würfe sehen.
- **Schätzer:** Bradley-Terry mit Ridge λ (MAP), **gedämpftes Newton-Verfahren** (Schrittweite halbiert, bis der Zielwert nicht sinkt), nach jeder Partie (bzw. Runde) warm mit drei Schritten
  neu angepasst; **Unsicherheit** nach Laplace Σ = H⁻¹ mit H = Σ G_ij p_ij (1 − p_ij) (e_i − e_j)(e_i − e_j)ᵀ + λ I.
- **Paarwahl:** *Fehlordnung* m_ij = Φ(−|ŝ_i − ŝ_j| / √v_ij) mit v_ij = Var(ŝ_i − ŝ_j) („naive Regel“); *Information* q_ij = p̂_ij (1 − p̂_ij) v_ij; bei k Paaren je Runde die k größten
  Werte ohne gemeinsame Spieler. Vergleich: **Rundenturnier** (jedes Paar einmal in zufälliger Reihenfolge, dann von vorn), **Zufällig** (gleichverteiltes Paar), **Schweizer System
  (vereinfacht)** (nach Punkten sortiert, nächster freier Spieler, bevorzugt der am seltensten gespielte Gegner).
- **Maße:** *Paartreffer* (Anteil richtig geordneter Paare, Gleichstand zählt ½), *Sieger richtig*, *Top-3-Treffer*, *Partien bis 0,90* (erster Messpunkt, alle 12 Partien; „nicht
  erreicht“ zählt als 600), *Konzentration* (größte Zahl Partien eines Paares, Anteil der Partien zwischen Spielern mit wahrem Rangabstand ≤ 2).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Spielern, Stärkeabstand 0,3, 480 Partien, λ = 0,1,
  unabhängig vom Regler. Die Zahlen sind plattformunabhängig (der Generator hängt nur von Seed und Position ab).
- **Gleiche Würfe:** Alle Strategien einer Instanz sehen dieselben Stärken, dieselben Ausgänge je Partienummer und denselben Gleichstandsbrecher.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie).
- **Erster Aufruf:** Die drei Läufe über die 60 Instanzen (Hauptexperiment, enges Feld, Runden) rechnen etwa eine Minute und werden danach zwischengespeichert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 12 Spieler, δ = 0,3, 480 Partien, λ = 0,1, wo nichts anderes steht.

| Frage | Befund | Test |
|---|---|---|
| Median der Partien bis Paartreffer 0,90 | Rundenturnier **120**, Zufällig **120**, Schweizer **96**, Fehlordnung **126**, Information **90**; Ersparnis der Information-Regel 25 %; Läufe ohne Erreichen: nur Fehlordnung (2 von 60). | `test_games_to_the_target` |
| Paartreffer nach 60 / 120 / 240 / 480 Partien | Rundenturnier 0,846 / 0,892 / 0,929 / 0,958; Information **0,859 / 0,912 / 0,944 / 0,971**; Schweizer 0,855 / 0,900 / 0,937 / 0,960; Fehlordnung 0,860 / 0,904 / 0,937 / 0,967; Zufällig 0,841 / 0,891 / 0,926 / 0,959. Vorsprung der Information-Regel nach 120 und 240 Partien über 0,01. | `test_accuracy_at_fixed_budgets` |
| Sieger richtig nach 240 Partien; Top-3 nach 120 | **0,73** (Information) gegen 0,60 (Rundenturnier), 0,68 (Schweizer), 0,77 (Fehlordnung); nach 60 Partien 0,53 gegen 0,42; Top-3 nach 120: 0,83 gegen 0,81. | `test_winner_and_top_three` |
| Konzentration nach 480 Partien | größte Zahl Partien eines Paares: Fehlordnung **145**, Information 26, Zufällig 14, Schweizer 11, Rundenturnier 8; Anteil der Nachbarpartien 0,86 / 0,64 / 0,33 / 0,34 / 0,32. | `test_the_naive_rule_concentrates_the_games` |
| Enges Feld (Abstand 0,12) | Paartreffer nach 240 Partien: Rundenturnier 0,828, Zufällig 0,825, Schweizer 0,823, Information **0,828**, Fehlordnung **0,775**; Läufe ohne Erreichen von 0,90 in 480 Partien: 28 / 26 / 25 / 25 / **52** von 60 (beim Rundenturnier 29 auf dem Linux-Läufer der CI: ein einzelner Lauf fällt je nach Plattform auf die andere Seite von 0,90). | `test_a_narrow_field_gives_the_active_rule_no_advantage` |
| Runden (Information-Regel) 1 / 3 / 6 Paare je Runde | Paartreffer nach 240 Partien **0,944 / 0,939 / 0,935**, nach 120 Partien 0,912 / 0,909 / 0,902; Median bis 0,90: 90 / 96 / 96. | `test_parallel_rounds_cost_a_little` |
| Verdikte nennen dieselben Zahlen | Die live berechneten Texte der App geben die obigen Werte wieder. | `test_verdicts_quote_the_measured_numbers` |

Die Reihenfolge der drei besten Regeln untereinander ist beim Paartreffer eng (0,937 / 0,937 / 0,944 nach 240 Partien) und bei „Sieger richtig“ mit 60 Läufen **nicht belastbar**
(Standardfehler rund 0,06): Belastbar ist „aktiv vor Rundenturnier“ bei der Spitze, nicht die Rangfolge der aktiven Regeln.

## Befunde und Korrekturen gegenüber dem Plan

- **Der Schätzer divergierte in der Vorab-Messung.** Das volle Newton-Verfahren schwingt bei wenigen Partien und kleinem λ, sobald ein Spieler keinen Sieg oder keine Niederlage hat; die
  Schätzungen liefen auf Werte um ±2000. Entdeckt hat es der Orakel-Test `test_with_many_games_every_strategy_recovers_the_true_order` (Schweizer System nach 600 Partien nur 0,867
  statt über 0,95). **Behoben** mit Rückwärtsschritt (gedämpftes Newton-Verfahren, geprüft gegen den BFGS von scipy und mit `test_newton_does_not_diverge_for_separable_data_and_a_small_prior`).
- **Zwei Aussagen der Vorab-Messung fielen weg**, weil sie am Schätzfehler hingen: (1) „die zufällige Paarung ist schlechter als das Rundenturnier“ (156 gegen 120 Partien; gemessen sind
  **120 gegen 120**), (2) „bei engem Feld erreicht niemand 0,90“ (Rundenturnier 0,804, Zufällig 0,760 nach 240 Partien; gemessen **0,828 und 0,825**, und rund jeder zweite Lauf erreicht 0,90).
  Geblieben ist: die naive Regel liegt bei engem Feld hinter allen anderen.
- **Die Ersparnis ist ein Viertel, nicht ein Drittel:** Der Entwurf des Preset-Textes sagte „etwa ein Drittel“; gemessen sind 25 % (90 gegen 120 Partien).
- Die Zahlen verschieben sich um wenige Hundertstel gegen die Vorab-Messung (60 statt 30 Läufe, SplitMix64-Strom).

## Abgrenzung

- **Bradley-Terry** (Stück 5): das Modell und seine Schätzung bei vollständigem Plan; hier das gleiche Modell, aber die Partien werden nach seiner Unsicherheit gewählt.
- **Glicko und TrueSkill** (Stück 8): Unsicherheit je Spieler; hier die Unsicherheit der Differenz zweier Spieler (mit Kovarianz) als Maß für die Paarwahl.
- **Nicht-Transitivität** (Stück 11): Bei Zyklen ist „die richtige Rangfolge“ nicht eindeutig; diese Demo setzt eine transitive Wahrheit voraus.
- **Nachfolger (nur beim Namen):** Rangintervalle und Shrinkage (Abbruchregel statt festem Budget).

## Ehrliche Grenzen

- **Das Modell stimmt:** Wahrheit und Schätzer sind beide Bradley-Terry; bei falschem Modell steuert die Paarwahl falsch. Nicht gemessen.
- **12 Spieler, ein Abstand (0,3) und der enge Fall (0,12):** Andere Feldgrößen und Stärkeverteilungen sind nur über die Regler zu sehen, nicht über die 60 festen Läufe.
- **Das Schweizer System ist vereinfacht** (ohne Farbe, Schwimmer, Freilos der FIDE).
- **Feste Budgets:** Eine Abbruchregel („aufhören, wenn die Rangfolge sicher genug ist“) ist nicht umgesetzt.
- **Sequenzielle und parallele Partien sind Extreme:** Echte Turniere spielen Runden, in denen manche Paare gleichzeitig laufen; Experiment 5 misst 1, 3 und 6 Paare.
- **Rechenzeit:** Die Paarwahl rechnet nach jeder Partie eine 12 × 12-Hesse-Matrix; für große Felder wäre eine andere Umsetzung nötig.
- **Alles simuliert:** keine Aussage über einen Sport oder eine Plattform.

## Quellen

- Maystre, L., Grossglauser, M. (2017): Just sort it! A simple and effective approach to active preference learning. *ICML 2017* (nicht eingesehen).
- Jamieson, K. G., Nowak, R. (2011): Active ranking using pairwise comparisons. *Advances in Neural Information Processing Systems 24* (nicht eingesehen).
- Glickman, M. E., Jensen, S. T. (2005): Adaptive paired comparison design. *Journal of Statistical Planning and Inference* 127, 279–293 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** die Regeln des Schweizer Systems der FIDE und die Literatur zu Bandit-Verfahren für Rangfolgen; die Zahlen dieser Demo sind Simulation.

## Tests

88 Tests, gut eine Minute (nach dem ersten Aufruf der Läufe; die AppTest-Rauchtests rechnen die Experimente einmal) (die Experimente rechnen über 60 feste Läufe): Einheiten gegen Handwerte (`test_model.py`: Schätzer, Laplace-Kovarianz, Paartreffer, Sieger, Auswahlwerte;
`test_strategies.py`: Läufe, Paarungen, Schweizer System), Generator (`test_generator.py`, wortgleich), **Orakel** (`test_oracle_aktives_ranking.py`: Schätzer gegen den BFGS von scipy,
Laplace-Kovarianz gegen finite Differenzen der Hesse-Matrix, Paarwahl gegen eine skalare Aufzählung mit `math.erfc`, Eigenschaften der Strategien über feste Läufe), Maße und Verdikte
(`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`), Presets, Szenario und PDF (`test_presets.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `akt_`-Module; schnelle Testdateien zuerst, Überlebende gegen die Claims-Tests): **115 Stellen, 104 getötet, 11 äquivalent.** Im ersten, schnellen Lauf überlebten 34 Stellen; 21 davon sind gegen die Claims-Tests getötet (das Vorzeichen des Gleichstandsbrechers im Schweizer System, das Vorzeichen im Rückwärtsschritt des Newton-Verfahrens, die Zählung der nicht erreichten Läufe), eine Stelle (`while len(seq) <= budget`) endet in einer Endlosschleife und gilt als getötet. **Zwei echte Lücken** waren das Messgitter der Experimente (letzter Messpunkt 480) und die Randfälle der Eingabeprüfung (3 Spieler, eine Partie); sie sind mit Tests geschlossen und auf einer Kopie einzeln getötet, ebenso die Verdikt-Zweige (Abstand der zufälligen Paarung zum Rundenturnier, Vorsprung der Information-Regel bei engem Feld) und der Messpunkt bei „Partien bis 0,90“ mit Gleichheit. **11 sind äquivalent:** `<` gegen `<=` bei stetigen Würfen und am Gleichheitsrand von 0,01, der leere innere Bereich der Nachbarschleife und die Diagonale mit G = 0, die Grenze `n // 2` (zwei Paare ohne gemeinsamen Spieler gibt es ohnehin höchstens n // 2), die Vorzeichen-Zweige hinter einem schon abgefangenen Gleichstand und der Stoppwert der `arange` (481 gegen 482).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `akt_model.py` | Bradley-Terry (gedämpftes Newton), Laplace-Kovarianz, Paartreffer, Auswahlwerte |
| `akt_strategies.py` | die fünf Strategien, ein Lauf, Schweizer System, Konzentration |
| `akt_scenario.py` | Reglerwerte (`Settings`) |
| `akt_evaluation.py` | Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `akt_presets.py`, `akt_constants.py` | Permalink, Presets, Grenzen |
| `akt_visualization.py`, `akt_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Abbruchregel** (aufhören, wenn die Rangfolge sicher genug ist) und **Rangintervalle**.
- **Remis, Punktedifferenzen, Heimvorteil** (nur Sieg und Niederlage).
- **Das vollständige Schweizer System der FIDE.**

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
