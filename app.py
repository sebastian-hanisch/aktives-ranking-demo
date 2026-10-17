"""Aktives Ranking – welche Paarung als Nächstes? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes Stück der Ranking-Linie: Statt einen festen Spielplan abzuarbeiten, wählt die aktive Paarung die nächste Partie nach der Unsicherheit der Rangfolge. Sechs Experimente messen,
wie viele Partien das spart (gegen Rundenturnier, zufällige Paarung und ein vereinfachtes Schweizer System), warum die naive Regel die Partien konzentriert, wann es kaum etwas zu
gewinnen gibt und was parallele Runden kosten. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import akt_constants as C
import akt_evaluation as ev
from akt_pdf_export import build_pdf
from akt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from akt_scenario import Settings
from akt_visualization import (
    build_budget,
    build_concentration,
    build_curves,
    build_pair_heatmap,
    build_ranking,
    build_rounds,
    build_to_target,
    build_trajectories,
)

st.set_page_config(page_title="Aktives Ranking – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _main():
    return ev.sweep_main()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _narrow():
    return ev.sweep_main(C.DELTA_NARROW)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _rounds():
    return ev.sweep_rounds()


st.title("🎯 Aktives Ranking: welche Paarung als Nächstes?")
st.markdown(
    """
Ein fester Spielplan spielt jedes Paar gleich oft, obwohl manche Paare nach wenigen Partien klar entschieden sind und andere nicht. Das **aktive Ranking** wählt die nächste Partie
nach der **Unsicherheit der Rangfolge**. Diese Demo ist das **zwölfte Stück der Ranking-Linie** und misst, **wie viele Partien das spart** gegen das Rundenturnier, die zufällige
Paarung und ein **vereinfachtes Schweizer System**, warum die naheliegende Regel die Partien **konzentriert** und wann es **kaum etwas zu gewinnen** gibt.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wer spielt als Nächstes?", expanded=True):
    st.markdown(
        """
1. **Schätzen:** Nach jeder Partie wird eine Stärke je Spieler geschätzt (Bradley-Terry mit Regularisierung) und eine **Unsicherheit** dazu (Laplace-Näherung).
2. **Fehlordnung:** Für jedes Paar die Wahrscheinlichkeit, dass seine Reihenfolge falsch ist: klein bei großem Abstand und kleiner Unsicherheit. Die naive Regel spielt immer das
   Paar mit der größten Fehlordnungswahrscheinlichkeit, und das sind zwei Spieler, die sich nur schwer trennen lassen.
3. **Information:** Die bessere Regel gewichtet die Unsicherheit der Differenz mit der Ungewissheit des Ausgangs, p (1 − p): Ein Paar bringt wenig, wenn der Ausgang fast sicher ist
   oder wenn die Differenz schon genau bekannt ist.
4. **Vergleichsmaßstab:** das feste **Rundenturnier**, die **zufällige** Paarung und ein **vereinfachtes Schweizer System** (nach Punkten sortiert, Nachbarn, möglichst keine
   Wiederholung); alle auf derselben Wahrheit und denselben Würfen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler gerankt werden.")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider", format="%.2f",
                      help="Stärkedifferenz zweier benachbarter Spieler (logistische Einheiten). Klein heißt: kaum etwas zu trennen.")
    budget = st.slider("Budget in Partien", *bounds("budget_slider"), step=C.BUDGET_STEP, key="budget_slider", help="Wie viele Partien insgesamt gespielt werden.")
    strategy = st.selectbox("Strategie im Detail", C.STRATEGIES, key="strategy_select", help="Für die Heatmap der gespielten Paare und die Rangfolge mit Intervallen. Die Verläufe zeigen alle.")
    round_size = st.select_slider("Paare je Runde (aktiv)", C.ROUND_CHOICES, key="round_select",
                                  help="Wie viele Paare die aktive Paarung je Runde gemeinsam wählt (ohne gemeinsame Spieler); 1 = jede Partie nacheinander.")
    lam = st.slider("Ridge-Regularisierung λ", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider", format="%.2f",
                    help="Prior N(0, 1/λ) auf jeder Stärke; hält die Schätzung bei wenigen Partien endlich.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "budget_slider": int(budget), "strategy_select": strategy, "lam_slider": round(float(lam), 2),
                   "round_select": int(round_size), "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(budget), strategy, round(float(lam), 2), int(round_size), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Rangfolge nach dem Budget")
cols = st.columns(len(C.STRATEGIES))
for col, name in zip(cols, C.STRATEGIES):
    acc, win, top = a.final[name]
    col.metric(name.split(" (")[0], ev.de(acc, 3), delta=f"Sieger {ev.de(win, 0)} · Top-3 {ev.de(top)}", delta_color="off",
               help="Anteil richtig geordneter Spielerpaare nach dem Budget (0,5 = Zufall); darunter: Sieger richtig (1 = der wahre Beste steht oben) und Top-3-Treffer.")

st.plotly_chart(build_trajectories(a), width="stretch", key="trajectory_chart")
st.caption("Paartreffer über die Partien, eine Linie je Strategie, alle auf derselben Wahrheit und denselben Würfen (ein einzelner Lauf: die Experimente unten mitteln über 60).")
cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_pair_heatmap(a), width="stretch", key="pair_chart")
    st.caption(f"Wie oft jedes Paar bei „{settings.strategy}“ gespielt wurde (Rang nach wahrer Stärke); eine Linie nahe der Diagonale heißt: nur Nachbarn.")
with cr:
    st.plotly_chart(build_ranking(a), width="stretch", key="ranking_chart")
    st.caption(f"Geschätzte Stärke mit 95-%-Intervall bei „{settings.strategy}“ gegen die wahre Stärke (Raute).")

rows = []
for name in C.STRATEGIES:
    acc, win, top = a.final[name]
    mx, near = a.conc[name]
    reached = a.to_target[name]
    rows.append({"Strategie": name, "Paartreffer": round(acc, 3), "Sieger richtig": round(win, 2), "Top-3": round(top, 2),
                 "Partien bis 0,90": "nicht erreicht" if reached >= C.NOT_REACHED else reached, "größte Zahl Partien eines Paares": int(mx), "Anteil Nachbarpartien": f"{ev.pct(near)} %"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.download_button("📄 PDF des aktuellen Laufs", data=build_pdf(a), file_name="aktives-ranking.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viele Partien spart die aktive Paarung?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

main = _main()
st.subheader("🔬 Experiment 1: Wie viele Partien bis zum Paartreffer 0,90?")
st.caption("12 Spieler (feste Instanz, unabhängig vom Regler), Stärkeabstand 0,3, bis zu 480 Partien, 60 feste Läufe; Median der Partien am ersten Messpunkt (alle 12 Partien) mit "
           "Paartreffer ≥ 0,90; „nicht erreicht“ zählt als 600.")
st.plotly_chart(build_to_target(main), width="stretch", key="exp1_chart")
verdict(*ev.verdict_target(main))
st.dataframe(pd.DataFrame([{"Strategie": k, "Median": int(v["to_target_median"]), "Mittel": round(v["to_target_mean"]), "Läufe ohne Erreichen": v["not_reached"]} for k, v in main.items()]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Bei festem Budget")
st.caption("Mittlerer Paartreffer über die Partien (dieselben Läufe) und die Werte bei 60, 120, 240 und 480 Partien.")
st.plotly_chart(build_curves(main), width="stretch", key="exp2_chart")
st.plotly_chart(build_budget(main, "accuracy", "Paartreffer"), width="stretch", key="exp2b_chart")
verdict(*ev.verdict_budget(main))

st.subheader("🔬 Experiment 3: Wohin gehen die Partien?")
st.caption("Nach 480 Partien: die größte Zahl Partien, die ein einziges Paar bekommen hat (Mittel über 60 Läufe), und der Anteil der Partien zwischen Spielern, deren wahrer Rangabstand "
           f"höchstens {C.NEIGHBOUR_GAP} ist.")
st.plotly_chart(build_concentration(main), width="stretch", key="exp3_chart")
verdict(*ev.verdict_concentration(main))
st.dataframe(pd.DataFrame([{"Strategie": k, "größte Zahl Partien eines Paares": round(v["max_pair"], 1), "Anteil Nachbarpartien": f"{ev.pct(v['near_share'])} %"} for k, v in main.items()]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Enges Feld")
st.caption(f"Dieselbe Instanz mit Stärkeabstand {ev.de(C.DELTA_NARROW)} statt 0,3: Es gibt kaum etwas zu trennen.")
narrow = _narrow()
st.plotly_chart(build_curves(narrow), width="stretch", key="exp4_chart")
verdict(*ev.verdict_narrow(narrow))

st.subheader("🔬 Experiment 5: Runden statt einzelner Partien")
st.caption("Die Information-Regel wählt je Runde 1, 3 oder 6 Paare ohne gemeinsame Spieler; Schätzung und Wahl erst nach der Runde.")
rows5 = _rounds()
st.plotly_chart(build_rounds(rows5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_rounds(rows5))

st.subheader("🔬 Experiment 6: Sieger und Top-3 statt der ganzen Rangfolge")
st.caption("Dieselben Läufe: Anteil der Läufe, in denen der wahre Beste oben steht, und Top-3-Treffer bei 60, 120, 240 und 480 Partien.")
st.plotly_chart(build_budget(main, "winner", "Sieger richtig (Anteil der Läufe)"), width="stretch", key="exp6_chart")
st.plotly_chart(build_budget(main, "top3", "Top-3-Treffer"), width="stretch", key="exp6b_chart")
verdict(*ev.verdict_top(main))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Stärken sind transitiv** | Mit Zyklen ist „richtige Rangfolge“ nicht eindeutig; die Paarwahl nach einer Skala kann dieselben Paare immer wieder spielen. | Nicht-Transitivität |
| **Es gibt etwas zu trennen** | Bei engem Feld ist in 60 Läufen kein Vorsprung der aktiven Paarung messbar (Experiment 4); die Fehlordnungsregel liegt dort sogar hinter der zufälligen Paarung (Experiment 4, Konzentration in Experiment 3). | – |
| **Ergebnisse sind sofort bekannt** | Parallele Runden kosten Partien (Experiment 5); echte Turniere spielen Runden. | – |
| **Nur Sieg und Niederlage** | Eine Punktedifferenz oder ein Remis trägt mehr Information je Partie; hier nicht gemessen. | Tore, Remis, Heimvorteil |
| **Das Schweizer System ist vereinfacht** | Der Vergleichsmaßstab sortiert nach Punkten und meidet Wiederholungen, ohne die Regeln der FIDE (Farbe, Schwimmer, Freilos). | Schweizer System |
| **Das Modell stimmt** | Die Paarwahl nutzt Bradley-Terry und seine Unsicherheit; ein falsches Modell steuert die Partien falsch. | – |
| **Das Budget ist bekannt** | Wer aufhört, wenn die Unsicherheit klein genug ist, braucht eine Abbruchregel; hier läuft das feste Budget. | Rangintervalle und Shrinkage |
| **Alles ist simuliert** | Stärken und Ausgänge sind Konstruktionen; die Zahlen gelten für 12 Spieler und diese Abstände. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Spieler $i = 1, \dots, n$ mit Stärken $s_i$; $P(i \text{ schlägt } j) = \sigma(s_i - s_j)$. Daten: Siege $W_{ij}$, Partien $G_{ij}$.

**Schätzer (MAP).** $\hat s = \arg\max_s \sum_{i<j}\bigl[W_{ij}\ln\sigma(s_i - s_j) + W_{ji}\ln\sigma(s_j - s_i)\bigr] - \tfrac{\lambda}{2}\lVert s\rVert^2$; gedämpftes Newton-Verfahren (Schrittweite halbiert, bis der Zielwert nicht sinkt), nach jeder Partie
(bzw. Runde) mit drei Schritten warm neu angepasst. **Unsicherheit** (Laplace): $\Sigma = H^{-1}$ mit $H = \sum_{i<j} G_{ij} p_{ij}(1 - p_{ij})(e_i - e_j)(e_i - e_j)^\top + \lambda I$.

**Differenz.** $v_{ij} = \operatorname{Var}(\hat s_i - \hat s_j) = \Sigma_{ii} + \Sigma_{jj} - 2\Sigma_{ij}$.

**Fehlordnung.** $m_{ij} = \Phi\bigl(-|\hat s_i - \hat s_j|/\sqrt{v_{ij}}\bigr)$; die naive Regel spielt $\arg\max_{i<j} m_{ij}$.

**Information.** $q_{ij} = \hat p_{ij}(1 - \hat p_{ij})\,v_{ij}$ mit $\hat p_{ij} = \sigma(\hat s_i - \hat s_j)$; die Regel spielt $\arg\max_{i<j} q_{ij}$ (bei $k$ Paaren je Runde die $k$ größten
Werte ohne gemeinsame Spieler).

**Rundenturnier.** Jedes Paar einmal in zufälliger Reihenfolge, dann von vorn. **Zufällig.** Je Partie ein gleichverteiltes Paar. **Schweizer System (vereinfacht).** Nach Punkten
sortiert; jeder freie Spieler gegen den nächsten freien, bevorzugt gegen einen, gegen den er am seltensten gespielt hat.

**Maße.** *Paartreffer:* Anteil der Paare $(i, j)$, die $\hat s$ wie die Wahrheit ordnet (Gleichstand zählt $\tfrac12$). *Partien bis 0,90:* kleinste Zahl Partien an einem Messpunkt
(alle 12 Partien) mit Paartreffer $\ge 0{,}9$. *Konzentration:* $\max_{i<j} G_{ij}$ und der Anteil der Partien zwischen Spielern mit Rangabstand $\le 2$.

**Zufall.** SplitMix64-Strom des Generators; Ausgänge und Gleichstandsbrecher in fester Anordnung je Partie, damit alle Strategien dieselben Würfe sehen.

**Literatur.**
- Maystre, L., Grossglauser, M. (2017): Just sort it! A simple and effective approach to active preference learning. *ICML 2017* (eingesehen 2026-10-17, arXiv-Fassung 1502.05556v2, Abschnitte 4.1 und 4.3: Die Unsicherheitsstichprobe wählt das Paar mit dem kleinsten Abstand der Schätzwerte und konzentriert sich mit wachsender Partienzahl auf wenige schwer trennbare Paare; die Paarwahlregeln dieser Demo sind nicht die Regeln der Arbeit).
- Jamieson, K. G., Nowak, R. (2011): Active ranking using pairwise comparisons. *Advances in Neural Information Processing Systems 24* (eingesehen 2026-10-17, arXiv-Fassung 1109.3701v2, Zusammenfassung, Abschnitte 1 und 5: Rahmen ist die Einbettung der Objekte in $\mathbb{R}^d$, bei Fehlern mit dauerhafter Wirkung („persistent errors“), nicht das Bradley-Terry-Modell; die Demo übernimmt daraus keine Formel).
- Glickman, M. E., Jensen, S. T. (2005): Adaptive paired comparison design. *Journal of Statistical Planning and Inference* 127, 279–293 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
