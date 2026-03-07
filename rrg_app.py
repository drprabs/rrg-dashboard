# ============================================================
# Sector Rotation / RRG Dashboard
# pip install dash plotly yfinance pandas numpy
# Run: python rrg_app.py  →  open http://127.0.0.1:8050
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ─── DEFAULT SECTORS ─────────────────────────────────────────
DEFAULT_SECTORS = [
    {"ticker": "XLE",  "color": "#FF5722", "name": "Energy"},
    {"ticker": "XLB",  "color": "#795548", "name": "Materials"},
    {"ticker": "XLI",  "color": "#607D8B", "name": "Industrials"},
    {"ticker": "XLY",  "color": "#FFA500", "name": "Cons. Discr."},
    {"ticker": "XLP",  "color": "#28A745", "name": "Cons. Staples"},
    {"ticker": "XLV",  "color": "#DC3545", "name": "Healthcare"},
    {"ticker": "XLF",  "color": "#1E90FF", "name": "Financials"},
    {"ticker": "XLK",  "color": "#00BCD4", "name": "Technology"},
    {"ticker": "XLC",  "color": "#9C27B0", "name": "Comm. Svcs."},
    {"ticker": "XLU",  "color": "#6366F1", "name": "Utilities"},
    {"ticker": "XLRE", "color": "#009688", "name": "Real Estate"},
]

# ─── QUADRANT COLORS ─────────────────────────────────────────
Q_LEADING   = "rgba(40,167,69,0.12)"
Q_IMPROVING = "rgba(30,144,255,0.12)"
Q_LAGGING   = "rgba(220,53,69,0.12)"
Q_WEAKENING = "rgba(255,193,7,0.12)"

# ─── DATA FETCH & RRG COMPUTATION ────────────────────────────
def fetch_and_compute(sectors, benchmark, timeframe, lookback, history_len):
    interval = "1wk" if timeframe == "Weekly" else "1mo"
    lb = lookback

    tickers = [benchmark] + [s["ticker"] for s in sectors]
    raw = yf.download(tickers, period="5y", interval=interval,
                      auto_adjust=True, progress=False)["Close"]

    # Ensure consistent columns even for single ticker
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    raw = raw.dropna(how="all")

    bench = raw[benchmark]

    results = []
    for s in sectors:
        tk = s["ticker"]
        if tk not in raw.columns:
            continue
        sec = raw[tk].dropna()
        # align to benchmark index
        combined = pd.concat([bench, sec], axis=1, join="inner").dropna()
        b = combined.iloc[:, 0]
        p = combined.iloc[:, 1]

        # % change over lookback period
        b_chg = b.pct_change(lb)
        p_chg = p.pct_change(lb)

        # Relative Strength ratio
        rs = (1 + p_chg) / (1 + b_chg)

        # RS Momentum = (current RS / previous RS) * 100
        rs_mom = (rs / rs.shift(1)) * 100

        df_s = pd.DataFrame({
            "rs":  rs  * 100,   # scaled to % like TrendSpider
            "mom": rs_mom
        }).dropna()

        # Keep only last N periods
        df_s = df_s.iloc[-history_len:]
        results.append({"meta": s, "data": df_s})

    return results


# ─── BUILD FIGURE ─────────────────────────────────────────────
def build_figure(results, history_len, last_dot_size, prev_dot_size):
    fig = go.Figure()

    all_x, all_y = [], []
    for r in results:
        df = r["data"]
        all_x += df["rs"].tolist()
        all_y += df["mom"].tolist()

    if not all_x:
        return fig

    pad = 1
    min_x = np.floor(min(all_x)) - pad
    max_x = np.ceil(max(all_x))  + pad
    min_y = np.floor(min(all_y)) - pad
    max_y = np.ceil(max(all_y))  + pad

    # ── Quadrant shading ──────────────────────────────────────
    for (x0, x1, y0, y1, col) in [
        (min_x, 100,   100,   max_y, Q_IMPROVING),
        (100,   max_x, 100,   max_y, Q_LEADING),
        (min_x, 100,   min_y, 100,   Q_LAGGING),
        (100,   max_x, min_y, 100,   Q_WEAKENING),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=col, line_width=0, layer="below")

    # ── Crosshair lines ───────────────────────────────────────
    for axis, val in [("x", 100), ("y", 100)]:
        line_kw = dict(type="line", line=dict(color="rgba(180,180,180,0.6)",
                                              width=1.5, dash="dot"))
        if axis == "x":
            fig.add_shape(**line_kw, x0=val, x1=val, y0=min_y, y1=max_y)
        else:
            fig.add_shape(**line_kw, x0=min_x, x1=max_x, y0=val, y1=val)

    # ── Quadrant labels ───────────────────────────────────────
    labels = [
        ("Leading",   max_x, max_y, "right", "top",    "#28A745"),
        ("Improving", min_x, max_y, "left",  "top",    "#1E90FF"),
        ("Weakening", max_x, min_y, "right", "bottom", "#FFC107"),
        ("Lagging",   min_x, min_y, "left",  "bottom", "#DC3545"),
    ]
    for txt, x, y, xanchor, yanchor, col in labels:
        fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>",
                           showarrow=False, font=dict(size=13, color=col),
                           xanchor=xanchor, yanchor=yanchor, opacity=0.85)

    # ── Sector scatter traces ─────────────────────────────────
    for r in results:
        df    = r["data"]
        meta  = r["meta"]
        color = meta["color"]
        label = meta.get("name", meta["ticker"])

        n = len(df)
        sizes = [prev_dot_size] * n
        if n > 0:
            sizes[-1] = last_dot_size

        # line trace (path)
        fig.add_trace(go.Scatter(
            x=df["rs"], y=df["mom"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=sizes,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "RS: %{x:.2f}<br>"
                "Momentum: %{y:.2f}<extra></extra>"
            )
        ))

        # Label at last point
        if n > 0:
            fig.add_annotation(
                x=df["rs"].iloc[-1], y=df["mom"].iloc[-1],
                text=f" {meta['ticker']}",
                showarrow=False, font=dict(size=10, color=color),
                xanchor="left"
            )

    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        xaxis=dict(title="Relative Strength (%)", range=[min_x, max_x],
                   gridcolor="rgba(128,128,128,0.2)",
                   zeroline=False, showline=True,
                   linecolor="rgba(128,128,128,0.4)"),
        yaxis=dict(title="RS Momentum", range=[min_y, max_y],
                   gridcolor="rgba(128,128,128,0.2)",
                   zeroline=False, showline=True,
                   linecolor="rgba(128,128,128,0.4)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(128,128,128,0.3)",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=50, b=60),
        height=620,
        title=dict(text="Sector Rotation — Relative Rotation Graph (RRG)",
                   font=dict(size=16), x=0.5)
    )
    return fig


# ─── DASH APP ─────────────────────────────────────────────────
app = dash.Dash(__name__, title="Sector RRG")
server = app.server   # for deployment (gunicorn/Render/Railway)

sector_options = [{"label": f"{s['ticker']} – {s['name']}", "value": s["ticker"]}
                  for s in DEFAULT_SECTORS]
default_enabled = ["XLE", "XLI", "XLY", "XLV", "XLF", "XLK"]

app.layout = html.Div(style={"backgroundColor": "#0d1117", "minHeight": "100vh",
                              "padding": "20px", "fontFamily": "Inter, sans-serif"}, children=[

    html.H2("📊 Sector Rotation — RRG Dashboard",
            style={"color": "#e6edf3", "marginBottom": "4px"}),
    html.P("Relative Rotation Graph — Leading / Weakening / Lagging / Improving",
           style={"color": "#8b949e", "marginTop": 0, "marginBottom": "20px"}),

    # ── Controls ──────────────────────────────────────────────
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "20px",
                    "marginBottom": "20px"}, children=[

        html.Div([
            html.Label("Sectors", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Dropdown(id="sectors", options=sector_options,
                         value=default_enabled, multi=True,
                         style={"width": "420px", "backgroundColor": "#161b22"},
                         className="dark-dropdown")
        ]),

        html.Div([
            html.Label("Benchmark", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Input(id="benchmark", value="VTI", type="text",
                      style={"width": "80px", "backgroundColor": "#161b22",
                             "color": "#e6edf3", "border": "1px solid #30363d",
                             "borderRadius": "6px", "padding": "6px"})
        ]),

        html.Div([
            html.Label("Timeframe", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Dropdown(id="timeframe", options=["Weekly", "Monthly"],
                         value="Weekly", clearable=False,
                         style={"width": "130px", "backgroundColor": "#161b22"})
        ]),

        html.Div([
            html.Label("Lookback (periods)", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Input(id="lookback", value=13, type="number", min=1, max=52,
                      style={"width": "80px", "backgroundColor": "#161b22",
                             "color": "#e6edf3", "border": "1px solid #30363d",
                             "borderRadius": "6px", "padding": "6px"})
        ]),

        html.Div([
            html.Label("History Tail (bars)", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Slider(id="history", min=2, max=52, step=1, value=8,
                       marks={2: "2", 13: "13", 26: "26", 52: "52"},
                       tooltip={"placement": "bottom"},
                       className="dark-slider")
        ], style={"width": "200px"}),

        html.Div([
            html.Label("Last Dot Size", style={"color": "#8b949e", "fontSize": "12px"}),
            dcc.Slider(id="last-dot", min=4, max=20, step=1, value=10,
                       marks={4: "4", 12: "12", 20: "20"},
                       tooltip={"placement": "bottom"})
        ], style={"width": "160px"}),

        html.Div([
            html.Button("🔄 Refresh", id="refresh", n_clicks=0,
                        style={"backgroundColor": "#238636", "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "padding": "10px 20px", "cursor": "pointer",
                               "fontSize": "14px", "marginTop": "16px"})
        ]),
    ]),

    # ── Chart ──────────────────────────────────────────────────
    dcc.Loading(type="circle", color="#58a6ff", children=[
        dcc.Graph(id="rrg-chart", config={"displayModeBar": True,
                                          "scrollZoom": True})
    ]),

    # ── Quadrant Legend ───────────────────────────────────────
    html.Div(style={"display": "flex", "gap": "24px", "marginTop": "12px",
                    "flexWrap": "wrap"}, children=[
        html.Div([html.Span("●", style={"color": "#28A745", "fontSize": "18px"}),
                  html.Span(" Leading — RS↑ Momentum↑", style={"color": "#8b949e", "fontSize": "13px"})]),
        html.Div([html.Span("●", style={"color": "#1E90FF", "fontSize": "18px"}),
                  html.Span(" Improving — RS↓ Momentum↑", style={"color": "#8b949e", "fontSize": "13px"})]),
        html.Div([html.Span("●", style={"color": "#FFC107", "fontSize": "18px"}),
                  html.Span(" Weakening — RS↑ Momentum↓", style={"color": "#8b949e", "fontSize": "13px"})]),
        html.Div([html.Span("●", style={"color": "#DC3545", "fontSize": "18px"}),
                  html.Span(" Lagging — RS↓ Momentum↓", style={"color": "#8b949e", "fontSize": "13px"})]),
    ]),

    html.P("Rotation is typically clockwise: Leading → Weakening → Lagging → Improving → Leading",
           style={"color": "#484f58", "fontSize": "12px", "marginTop": "8px"}),
])


# ─── CALLBACK ─────────────────────────────────────────────────
@app.callback(
    Output("rrg-chart", "figure"),
    Input("refresh", "n_clicks"),
    State("sectors", "value"),
    State("benchmark", "value"),
    State("timeframe", "value"),
    State("lookback", "value"),
    State("history", "value"),
    State("last-dot", "value"),
    prevent_initial_call=False
)
def update_chart(_, selected_sectors, benchmark, timeframe,
                 lookback, history_len, last_dot_size):
    if not selected_sectors:
        return go.Figure()

    sectors = [s for s in DEFAULT_SECTORS if s["ticker"] in selected_sectors]
    lb = int(lookback) if lookback else (13 if timeframe == "Weekly" else 3)
    hl = int(history_len) if history_len else 8
    lds = int(last_dot_size) if last_dot_size else 10

    try:
        results = fetch_and_compute(sectors, benchmark, timeframe, lb, hl)
        return build_figure(results, hl, lds, 3)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="red", size=14))
        fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        return fig


if __name__ == "__main__":
    app.run(debug=False)

