"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import akt_constants as C
import akt_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trajectories(a: ev.Analysis):
    """Paartreffer über die Partien, eine Linie je Strategie, auf derselben Wahrheit."""
    fig = go.Figure()
    for name in C.STRATEGIES:
        r = a.runs[name]
        fig.add_trace(go.Scatter(x=r.games, y=r.accuracy, mode="lines", name=name, line=dict(color=C.COLORS[name], width=3 if name in C.ACTIVE else 2),
                                 hovertemplate=name + ", %{x} Partien: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=C.TARGET, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="Ziel 0,90", annotation_position="bottom right")
    fig.update_xaxes(title_text="Partien")
    fig.update_yaxes(title_text="Anteil richtig geordneter Paare", range=[0.4, 1.0])
    return _base(fig, height=400)


def build_pair_heatmap(a: ev.Analysis):
    """Wie oft jedes Paar gespielt wurde (Strategie aus dem Regler); Spieler nach der wahren Stärke geordnet (Beste oben)."""
    r = a.runs[a.settings.strategy]
    n = r.n
    order = np.arange(n)[::-1]
    G = r.G[np.ix_(order, order)].astype(float)
    np.fill_diagonal(G, np.nan)
    labels = [f"{i + 1}" for i in range(n)]
    fig = go.Figure(go.Heatmap(z=G, x=labels, y=labels, colorscale="Teal", colorbar=dict(title="Partien"), hovertemplate="Rang %{y} gegen Rang %{x}: %{z:.0f} Partien<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Rang nach wahrer Stärke (1 = Bester)")
    fig.update_xaxes(title_text="Rang nach wahrer Stärke")
    return _base(fig, height=420)


def build_ranking(a: ev.Analysis):
    """Geschätzte Stärke mit 95-%-Intervall (Laplace) je Spieler am Ende, nach wahrer Stärke geordnet, die wahre Stärke als Punkt."""
    r = a.runs[a.settings.strategy]
    n = r.n
    sd = np.sqrt(np.maximum(np.diag(r.Sigma), 0.0))
    s_hat = r.s_hat - r.s_hat.mean()
    truth = r.strengths - r.strengths.mean()
    x = [f"{n - i}" for i in range(n)]
    order = np.arange(n)[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=C.BASE + C.SCALE * s_hat[order], mode="markers", name="geschätzt", marker=dict(color=C.COLORS[a.settings.strategy], size=9),
                             error_y=dict(type="data", array=C.SCALE * 1.96 * sd[order], visible=True),
                             hovertemplate="Rang %{x}: geschätzt %{y:.0f} Punkte<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=C.BASE + C.SCALE * truth[order], mode="markers", name="wahre Stärke", marker=dict(color=C.COLOR_TRUTH, size=8, symbol="diamond"),
                             hovertemplate="Rang %{x}: wahr %{y:.0f} Punkte<extra></extra>"))
    fig.update_xaxes(title_text="Rang nach wahrer Stärke (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Elo-Punkte (um 1500)")
    return _base(fig, height=380)


def build_curves(main, title="mittlerer Paartreffer"):
    fig = go.Figure()
    for name in C.STRATEGIES:
        if name not in main:
            continue
        v = main[name]
        fig.add_trace(go.Scatter(x=v["grid"], y=v["accuracy"], mode="lines", name=name, line=dict(color=C.COLORS[name], width=3 if name in C.ACTIVE else 2),
                                 hovertemplate=name + ", %{x} Partien: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=C.TARGET, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="Ziel 0,90", annotation_position="bottom right")
    fig.update_xaxes(title_text="Partien")
    fig.update_yaxes(title_text=title, range=[0.5, 1.0])
    return _base(fig, height=400)


def build_to_target(main):
    names = list(C.STRATEGIES)
    vals = [main[k]["to_target_median"] for k in names]
    fig = go.Figure(go.Bar(x=[k.split(" (")[0] for k in names], y=vals, marker_color=[C.COLORS[k] for k in names], text=[("nicht erreicht" if v >= C.NOT_REACHED else f"{v:.0f}") for v in vals],
                           textposition="auto", hovertemplate="%{x}: %{y:.0f} Partien (Median)<extra></extra>"))
    fig.update_yaxes(title_text="Partien bis Paartreffer 0,90 (Median)")
    return _base(fig, height=340)


def build_concentration(main):
    names = list(C.STRATEGIES)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[k.split(" (")[0] for k in names], y=[main[k]["max_pair"] for k in names], marker_color=[C.COLORS[k] for k in names], name="größte Zahl Partien eines Paares",
                         text=[f"{main[k]['max_pair']:.0f}" for k in names], textposition="auto", hovertemplate="%{x}: %{y:.0f}<extra></extra>"))
    fig.update_yaxes(title_text="größte Zahl Partien eines Paares (480 Partien)")
    return _base(fig, height=320)


def build_budget(main, metric: str, label: str):
    fig = go.Figure()
    for name in C.STRATEGIES:
        fig.add_trace(go.Bar(x=[f"{b} Partien" for b in C.CHECKS], y=[main[name]["at"][b][metric] for b in C.CHECKS], name=name, marker_color=C.COLORS[name],
                             hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=label, range=[0.0, 1.0])
    return _base(fig)


def build_rounds(rows):
    x = [f"{r['round']} Paar{'e' if r['round'] > 1 else ''} je Runde" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[r["at120"] for r in rows], name="nach 120 Partien", marker_color="#9cc5bf"))
    fig.add_trace(go.Bar(x=x, y=[r["at240"] for r in rows], name="nach 240 Partien", marker_color="#2F6B65"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Paartreffer", range=[0.8, 1.0])
    return _base(fig, height=320)
