# ============================================================
# Sector Rotation / RRG Dashboard  +  Sector/SPY Ratio Chart
# pip install dash plotly yfinance pandas numpy scipy
# Run: python rrg_app.py  →  open http://127.0.0.1:8050
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

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

Q_LEADING   = "rgba(40,167,69,0.12)"
Q_IMPROVING = "rgba(30,144,255,0.12)"
Q_LAGGING   = "rgba(220,53,69,0.12)"
Q_WEAKENING = "rgba(255,193,7,0.12)"

# ─── SHARED DATA FETCH ───────────────────────────────────────
def fetch_prices(sectors, benchmark, timeframe, period="5y"):
    interval = "1d" if timeframe == "Daily" else "1wk" if timeframe == "Weekly" else "1mo"
    tickers = [benchmark] + [s["ticker"] for s in sectors]
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    return raw.dropna(how="all")

# ─── RRG COMPUTATION ─────────────────────────────────────────
def fetch_and_compute(sectors, benchmark, timeframe, lookback, history_len):
    raw   = fetch_prices(sectors, benchmark, timeframe)
    bench = raw[benchmark]
    results = []
    for s in sectors:
        tk = s["ticker"]
        if tk not in raw.columns:
            continue
        combined = pd.concat([bench, raw[tk]], axis=1, join="inner").dropna()
        b, p = combined.iloc[:, 0], combined.iloc[:, 1]
        b_chg = b.pct_change(lookback)
        p_chg = p.pct_change(lookback)
        rs      = (1 + p_chg) / (1 + b_chg)
        rs_mom  = (rs / rs.shift(1)) * 100
        df_s = pd.DataFrame({"rs": rs * 100, "mom": rs_mom}).dropna()
        results.append({"meta": s, "data": df_s.iloc[-history_len:]})
    return results

# ─── SECTOR / SPY RATIO LINE CHART ───────────────────────────
def build_ratio_chart(sectors, benchmark, timeframe, period, normalize):
    raw = fetch_prices(sectors, benchmark, timeframe, period=period)
    if benchmark not in raw.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Benchmark '{benchmark}' not found.",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color="red", size=14))
        fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        return fig

    spy = raw[benchmark].dropna()
    fig = go.Figure()

    for s in sectors:
        tk = s["ticker"]
        if tk not in raw.columns:
            continue
        sec   = raw[tk].dropna()
        combo = pd.concat([spy, sec], axis=1, join="inner").dropna()
        ratio = combo.iloc[:, 1] / combo.iloc[:, 0]

        if normalize:
            ratio = (ratio / ratio.iloc[0]) * 100   # base 100

        # Add SPY baseline only once (normalized = flat 100 line)
                # ── Smooth the ratio line with cubic spline ──────────
        y_raw = ratio.values
        n     = len(y_raw)

        if n >= 4:
            t        = np.linspace(0, 1, n)
            t_smooth = np.linspace(0, 1, n * 20)          # 20x more points = silky smooth
            spl_y    = make_interp_spline(t, y_raw, k=3)  # k=3 = cubic
            y_smooth = spl_y(t_smooth)
            # Interpolate dates numerically then map back
            x_num    = np.linspace(0, 1, n)
            x_smooth_num = np.linspace(0, 1, n * 20)
            x_dates  = pd.date_range(ratio.index[0], ratio.index[-1], periods=n * 20)
        else:
            y_smooth = y_raw
            x_dates  = ratio.index

        fig.add_trace(go.Scatter(
            x=x_dates, y=y_smooth,
            mode="lines",
            name=f"{tk} ({s['name']})",
            line=dict(color=s["color"], width=2),
            hovertemplate=(
                f"<b>{tk} / {benchmark}</b><br>"
                "%{x|%Y-%m-%d}<br>"
                "Ratio: %{y:.3f}<extra></extra>"
            )
        ))


    # Baseline at 100 (normalized) or at mean
    if normalize:
        fig.add_hline(y=100, line_dash="dot",
                      line_color="rgba(180,180,180,0.5)", line_width=1.5,
                      annotation_text=f"{benchmark} baseline",
                      annotation_font_color="rgba(180,180,180,0.6)")

    ylabel = f"Sector / {benchmark} (Base 100)" if normalize else f"Sector / {benchmark} Ratio"
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        title=dict(text=f"Sector Relative Strength vs {benchmark}",
                   font=dict(size=16), x=0.5),
        xaxis=dict(title="Date", gridcolor="rgba(128,128,128,0.2)",
                   showline=True, linecolor="rgba(128,128,128,0.4)"),
        yaxis=dict(title=ylabel, gridcolor="rgba(128,128,128,0.2)",
                   showline=True, linecolor="rgba(128,128,128,0.4)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(128,128,128,0.3)",
                    borderwidth=1, font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        height=620,
    )
    return fig

# ─── RRG FIGURE (unchanged) ──────────────────────────────────
def build_figure(results, history_len, last_dot_size, prev_dot_size):
    fig = go.Figure()
    all_x, all_y = [], []
    for r in results:
        df = r["data"]
        all_x += df["rs"].tolist()
        all_y += df["mom"].tolist()
    if not all_x:
        return fig

    pad   = 1
    min_x = np.floor(min(all_x)) - pad
    max_x = np.ceil(max(all_x))  + pad
    min_y = np.floor(min(all_y)) - pad
    max_y = np.ceil(max(all_y))  + pad

    for (x0, x1, y0, y1, col) in [
        (min_x, 100,   100,   max_y, Q_IMPROVING),
        (100,   max_x, 100,   max_y, Q_LEADING),
        (min_x, 100,   min_y, 100,   Q_LAGGING),
        (100,   max_x, min_y, 100,   Q_WEAKENING),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=col, line_width=0, layer="below")

    for axis, val in [("x", 100), ("y", 100)]:
        line_kw = dict(type="line", line=dict(color="rgba(180,180,180,0.6)",
                                              width=1.5, dash="dot"))
        if axis == "x":
            fig.add_shape(**line_kw, x0=val, x1=val, y0=min_y, y1=max_y)
        else:
            fig.add_shape(**line_kw, x0=min_x, x1=max_x, y0=val, y1=val)

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

    for r in results:
        df    = r["data"]
        meta  = r["meta"]
        color = meta["color"]
        label = meta.get("name", meta["ticker"])
        n     = len(df)
        x_raw = df["rs"].values
        y_raw = df["mom"].values

        if n >= 4:
            t        = np.linspace(0, 1, n)
            t_smooth = np.linspace(0, 1, n * 20)
            spl_x    = make_interp_spline(t, x_raw, k=3)
            spl_y    = make_interp_spline(t, y_raw, k=3)
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
        else:
            x_smooth = x_raw
            y_smooth = y_raw

        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth, mode="lines", name=label,
            showlegend=False, line=dict(color=color, width=1.8), hoverinfo="skip"
        ))

        sizes = [prev_dot_size] * n
        if n > 0:
            sizes[-1] = last_dot_size

        fig.add_trace(go.Scatter(
            x=x_raw, y=y_raw, mode="markers", name=label,
            marker=dict(color=color, size=sizes,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{label}</b><br>RS: %{{x:.2f}}<br>"
                "Momentum: %{y:.2f}<extra></extra>"
            )
        ))

        if n > 0:
            fig.add_annotation(
                x=x_raw[-1], y=y_raw[-1], text=f" {meta['ticker']}",
                showarrow=False, font=dict(size=10, color=color), xanchor="left"
            )

    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        xaxis=dict(title="Relative Strength (%)", range=[min_x, max_x],
                   gridcolor="rgba(128,128,128,0.2)", zeroline=False,
                   showline=True, linecolor="rgba(128,128,128,0.4)"),
        yaxis=dict(title="RS Momentum", range=[min_y, max_y],
                   gridcolor="rgba(128,128,128,0.2)", zeroline=False,
                   showline=True, linecolor="rgba(128,128,128,0.4)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(128,128,128,0.3)",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=50, b=60),
        height=620,
        title=dict(text="Sector Rotation — Relative Rotation Graph (RRG)",
                   font=dict(size=16), x=0.5)
    )
    return fig

# ─── DASH APP ─────────────────────────────────────────────────
app  = dash.Dash(__name__, title="Sector RRG")
server = app.server

sector_options   = [{"label": f"{s['ticker']} – {s['name']}", "value": s["ticker"]}
                    for s in DEFAULT_SECTORS]
default_enabled  = [s["ticker"] for s in DEFAULT_SECTORS]

CTRL_STYLE  = {"backgroundColor": "#161b22", "color": "#e6edf3",
               "border": "1px solid #30363d", "borderRadius": "6px", "padding": "6px"}
LABEL_STYLE = {"color": "#8b949e", "fontSize": "12px"}

# ── Shared controls (reused in both tabs) ──────────────────────
shared_controls = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "20px"},
    children=[
        html.Div([
            html.Label("Sectors", style=LABEL_STYLE),
            dcc.Dropdown(id="sectors", options=sector_options,
                         value=default_enabled, multi=True,
                         style={"width": "420px", "backgroundColor": "#161b22"})
        ]),
        html.Div([
            html.Label("Benchmark", style=LABEL_STYLE),
            dcc.Input(id="benchmark", value="SPY", type="text",
                      style={"width": "80px", **CTRL_STYLE})
        ]),
        html.Div([
            html.Label("Timeframe", style=LABEL_STYLE),
            dcc.Dropdown(id="timeframe", options=["Daily", "Weekly", "Monthly"],
                         value="Weekly", clearable=False,
                         style={"width": "130px", "backgroundColor": "#161b22"})
        ]),
        html.Div([
            html.Label("Lookback (periods)", style=LABEL_STYLE),
            dcc.Input(id="lookback", value=13, type="number", min=1, max=60,
                      style={"width": "80px", **CTRL_STYLE})
        ]),
        html.Div([
            html.Label("History Tail (bars)", style=LABEL_STYLE),
            dcc.Slider(id="history", min=2, max=60, step=1, value=8,
                       marks={2: "2", 13: "13", 26: "26", 52: "52", 60: "60"},
                       tooltip={"placement": "bottom"})
        ], style={"width": "220px"}),
        html.Div([
            html.Label("Last Dot Size", style=LABEL_STYLE),
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
    ]
)

# ── Ratio-chart-specific controls ──────────────────────────────
ratio_controls = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "16px"},
    children=[
        html.Div([
            html.Label("Period", style=LABEL_STYLE),
            dcc.Dropdown(id="ratio-period",
                         options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                         value="1y", clearable=False,
                         style={"width": "100px", "backgroundColor": "#161b22"})
        ]),
        html.Div([
            html.Label("Normalize to 100", style=LABEL_STYLE),
            dcc.RadioItems(id="ratio-normalize",
                           options=[{"label": " Yes", "value": True},
                                    {"label": " No",  "value": False}],
                           value=True, inline=True,
                           labelStyle={"color": "#e6edf3", "marginRight": "12px"})
        ]),
    ]
)

app.layout = html.Div(
    style={"backgroundColor": "#0d1117", "minHeight": "100vh",
           "padding": "20px", "fontFamily": "Inter, sans-serif"},
    children=[
        html.H2("📊 Sector Rotation Dashboard",
                style={"color": "#e6edf3", "marginBottom": "4px"}),
        html.P("RRG + Sector / Benchmark Ratio Line Chart",
               style={"color": "#8b949e", "marginTop": 0, "marginBottom": "20px"}),

        shared_controls,

        dcc.Tabs(
            id="tabs", value="tab-rrg",
            colors={"border": "#30363d", "primary": "#58a6ff",
                    "background": "#161b22"},
            children=[
                # ── Tab 1: RRG ───────────────────────────────
                dcc.Tab(label="📡 RRG Chart", value="tab-rrg",
                        style={"color": "#8b949e", "backgroundColor": "#161b22"},
                        selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117"},
                        children=[
                            dcc.Loading(type="circle", color="#58a6ff", children=[
                                dcc.Graph(id="rrg-chart",
                                          config={"displayModeBar": True, "scrollZoom": True})
                            ]),
                            html.Div(
                                style={"display": "flex", "gap": "24px",
                                       "marginTop": "12px", "flexWrap": "wrap"},
                                children=[
                                    html.Div([html.Span("●", style={"color": "#28A745", "fontSize": "18px"}),
                                              html.Span(" Leading — RS↑ Momentum↑", style={"color": "#8b949e", "fontSize": "13px"})]),
                                    html.Div([html.Span("●", style={"color": "#1E90FF", "fontSize": "18px"}),
                                              html.Span(" Improving — RS↓ Momentum↑", style={"color": "#8b949e", "fontSize": "13px"})]),
                                    html.Div([html.Span("●", style={"color": "#FFC107", "fontSize": "18px"}),
                                              html.Span(" Weakening — RS↑ Momentum↓", style={"color": "#8b949e", "fontSize": "13px"})]),
                                    html.Div([html.Span("●", style={"color": "#DC3545", "fontSize": "18px"}),
                                              html.Span(" Lagging — RS↓ Momentum↓", style={"color": "#8b949e", "fontSize": "13px"})]),
                                ]
                            ),
                            html.P(
                                "Rotation is typically clockwise: Leading → Weakening → Lagging → Improving → Leading",
                                style={"color": "#484f58", "fontSize": "12px", "marginTop": "8px"}
                            ),
                        ]),

                # ── Tab 2: Ratio Line Chart ───────────────────
                dcc.Tab(label="📈 Sector / SPY Ratio", value="tab-ratio",
                        style={"color": "#8b949e", "backgroundColor": "#161b22"},
                        selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117"},
                        children=[
                            html.Div(style={"marginTop": "16px"}, children=[
                                ratio_controls,
                                dcc.Loading(type="circle", color="#58a6ff", children=[
                                    dcc.Graph(id="ratio-chart",
                                              config={"displayModeBar": True, "scrollZoom": True})
                                ]),
                                html.P(
                                    "Lines above 100 = outperforming benchmark | "
                                    "Lines below 100 = underperforming | Normalized at start date.",
                                    style={"color": "#484f58", "fontSize": "12px", "marginTop": "8px"}
                                ),
                            ])
                        ]),
            ]
        ),
    ]
)

# ─── CALLBACK: RRG ───────────────────────────────────────────
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
def update_rrg(_, selected_sectors, benchmark, timeframe,
               lookback, history_len, last_dot_size):
    if not selected_sectors:
        return go.Figure()
    sectors = [s for s in DEFAULT_SECTORS if s["ticker"] in selected_sectors]
    lb  = int(lookback)      if lookback      else (20 if timeframe == "Daily" else 13)
    hl  = int(history_len)   if history_len   else 8
    lds = int(last_dot_size) if last_dot_size else 10
    try:
        results = fetch_and_compute(sectors, benchmark or "SPY", timeframe, lb, hl)
        return build_figure(results, hl, lds, 3)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="red", size=14))
        fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        return fig

# ─── CALLBACK: RATIO LINE CHART ──────────────────────────────
@app.callback(
    Output("ratio-chart", "figure"),
    Input("refresh", "n_clicks"),
    Input("ratio-period", "value"),
    Input("ratio-normalize", "value"),
    State("sectors", "value"),
    State("benchmark", "value"),
    State("timeframe", "value"),
    prevent_initial_call=False
)
def update_ratio(_, period, normalize, selected_sectors, benchmark, timeframe):
    if not selected_sectors:
        return go.Figure()
    sectors = [s for s in DEFAULT_SECTORS if s["ticker"] in selected_sectors]
    try:
        return build_ratio_chart(sectors, benchmark or "SPY", timeframe,
                                 period or "1y", normalize)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="red", size=14))
        fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        return fig

if __name__ == "__main__":
    app.run(debug=False)
