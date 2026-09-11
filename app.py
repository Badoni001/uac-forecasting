"""
app.py  —  UAC Predictive Forecasting · Project 2
Redesigned UI: dark/light mode, 6-colour palette,
graphs as the hero element.

Run: streamlit run app.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_prep import load_and_prepare, train_test_split
from models import (
    evaluate,
    naive_forecast,
    moving_average_forecast,
    exp_smoothing_forecast,
    random_forest_forecast,
    gradient_boosting_forecast,
    build_future_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME STATE  (light / dark toggle)
# ─────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True          # start in dark mode

# ─────────────────────────────────────────────────────────────────────────────
# 6-COLOUR PALETTE  (same in both modes — graphs always pop)
# ─────────────────────────────────────────────────────────────────────────────
#  C1  Electric Blue   — historical line / primary series
#  C2  Vivid Coral     — forecast line / warnings
#  C3  Mint Green      — confidence band / positive metrics
#  C4  Gold Amber      — secondary series / moving avg
#  C5  Violet Purple   — third series / GB model
#  C6  Sky Cyan        — accent / naive / MA model

C1 = "#4FC3F7"   # Electric Blue
C2 = "#FF6B6B"   # Vivid Coral
C3 = "#69F0AE"   # Mint Green
C4 = "#FFD54F"   # Gold Amber
C5 = "#CE93D8"   # Violet Purple
C6 = "#80DEEA"   # Sky Cyan
C_ERR = "#FF5252"  # bright red for errors / surges

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLOURS  (UI chrome, not graphs)
# ─────────────────────────────────────────────────────────────────────────────
def theme():
    if st.session_state.dark_mode:
        return dict(
            bg        = "#0D1117",
            sidebar   = "#161B22",
            card      = "#1C2128",
            card_border= "#30363D",
            text      = "#E6EDF3",
            subtext   = "#8B949E",
            input_bg  = "#21262D",
            input_text= "#E6EDF3",
            tab_active= C1,
            hero_grad = "linear-gradient(135deg,#0D1117 0%,#161B40 50%,#0D1117 100%)",
            plot_bg   = "#161B22",
            plot_paper= "#161B22",
            grid_color= "#21262D",
            font_color= "#E6EDF3",
        )
    else:
        return dict(
            bg        = "#F6F8FA",
            sidebar   = "#FFFFFF",
            card      = "#FFFFFF",
            card_border= "#D0D7DE",
            text      = "#1F2328",
            subtext   = "#57606A",
            input_bg  = "#F6F8FA",
            input_text= "#1F2328",
            tab_active= "#0969DA",
            hero_grad = "linear-gradient(135deg,#0969DA 0%,#1A4CA6 50%,#0550AE 100%)",
            plot_bg   = "#FFFFFF",
            plot_paper= "#F6F8FA",
            grid_color= "#E6EDF3",
            font_color= "#1F2328",
        )

T = theme()

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Global background ── */
.stApp {{
    background-color: {T['bg']};
    color: {T['text']};
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {T['sidebar']};
    border-right: 1px solid {T['card_border']};
}}
section[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
}}

/* ── ALL input / select / slider labels visible ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] input {{
    background-color: {T['input_bg']} !important;
    color: {T['input_text']} !important;
    border: 1px solid {T['card_border']} !important;
    border-radius: 8px !important;
}}

/* ── Selectbox dropdown container — the problematic one ── */
section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {{
    background-color: {T['input_bg']} !important;
    border: 1.5px solid {C1} !important;
    border-radius: 10px !important;
    color: {T['input_text']} !important;
}}
/* Text INSIDE the select box */
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div {{
    color: {T['input_text']} !important;
}}
/* Dropdown menu options */
[data-baseweb="popover"] [role="option"] {{
    background-color: {T['card']} !important;
    color: {T['text']} !important;
}}
[data-baseweb="popover"] [role="option"]:hover {{
    background-color: {T['input_bg']} !important;
    color: {C1} !important;
}}
/* Selected option highlight */
[data-baseweb="popover"] [aria-selected="true"] {{
    background-color: rgba(79,195,247,0.13) !important;
    color: {C1} !important;
    font-weight: 600 !important;
}}

/* ── Radio buttons (model selector) ── */
.stRadio > div {{
    gap: 8px;
}}
.stRadio label {{
    background-color: {T['input_bg']} !important;
    border: 1.5px solid {T['card_border']} !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    color: {T['text']} !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all .2s;
    display: block !important;
    width: 100% !important;
}}
.stRadio label:hover {{
    border-color: {C1} !important;
    color: {C1} !important;
}}
input[type="radio"]:checked + div {{
    background-color: rgba(79,195,247,0.13) !important;
    border-color: {C1} !important;
    color: {C1} !important;
}}

/* ── Metric cards ── */
div[data-testid="stMetric"] {{
    background-color: {T['card']};
    border: 1px solid {T['card_border']};
    border-radius: 14px;
    padding: 16px 20px 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}}
div[data-testid="stMetricLabel"] p {{
    color: {T['subtext']} !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}}
div[data-testid="stMetricValue"] {{
    color: {T['text']} !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}}

/* ── Tab bar ── */
button[data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {T['subtext']} !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all .2s;
}}
button[data-baseweb="tab"]:hover {{
    color: {T['text']} !important;
    background-color: {T['input_bg']} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {T['tab_active']} !important;
    border-bottom: 3px solid {T['tab_active']} !important;
    background-color: {T['input_bg']} !important;
}}

/* ── Section cards (wrap charts) ── */
.chart-card {{
    background-color: {T['card']};
    border: 1px solid {T['card_border']};
    border-radius: 16px;
    padding: 20px 24px 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}}

/* ── Section heading ── */
.section-title {{
    color: {T['text']};
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 4px;
}}
.section-sub {{
    color: {T['subtext']};
    font-size: .82rem;
    margin-bottom: 14px;
}}

/* ── Divider ── */
hr {{
    border-color: {T['card_border']} !important;
}}

/* ── Slider track accent ── */
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background-color: {C1} !important;
    border-color: {C1} !important;
}}
.stSlider [data-baseweb="slider"] div[style*="background"] {{
    background-color: {C1} !important;
}}

/* ── Success / error banners ── */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border-left-width: 4px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT TEMPLATE  (reused on every chart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_layout(title="", height=420, legend_bottom=True):
    T2 = theme()
    layout = dict(
        title=dict(text=title, font=dict(color=T2["font_color"], size=14), x=0.01),
        height=height,
        paper_bgcolor=T2["plot_paper"],
        plot_bgcolor=T2["plot_bg"],
        font=dict(color=T2["font_color"], family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=T2["grid_color"], gridwidth=1,
            linecolor=T2["card_border"],
            tickfont=dict(color=T2["subtext"]),
            title_font=dict(color=T2["subtext"]),
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor=T2["grid_color"], gridwidth=1,
            linecolor=T2["card_border"],
            tickfont=dict(color=T2["subtext"]),
            title_font=dict(color=T2["subtext"]),
            showgrid=True, zeroline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=T2["card"], bordercolor=T2["card_border"],
            font_color=T2["text"],
        ),
        margin=dict(l=50, r=20, t=50, b=60 if legend_bottom else 30),
        legend=dict(
            orientation="h", y=-0.18 if legend_bottom else 1.02,
            x=0, font=dict(color=T2["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_prepare()

df = get_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Dark / Light toggle at the very top
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown(f"### 📈 UAC Forecast")
        st.markdown(
            f"<span style='color:{T['subtext']};font-size:.78rem'>"
            "HHS · Office of Refugee Resettlement</span>",
            unsafe_allow_html=True,
        )
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        mode_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(mode_label, help="Toggle dark / light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()

    # ── What to forecast ──
    st.markdown(
        f"<p style='color:{C1};font-weight:700;font-size:.85rem;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
        "🎯 What to forecast</p>",
        unsafe_allow_html=True,
    )
    target = st.radio(
        label="target",
        options=["hhs_load", "hhs_discharges"],
        format_func=lambda x: (
            "👥  HHS Care Load" if x == "hhs_load" else "🏠  Daily Discharges"
        ),
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model selector ──
    st.markdown(
        f"<p style='color:{C1};font-weight:700;font-size:.85rem;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
        "🤖 Algorithm</p>",
        unsafe_allow_html=True,
    )
    model_choice = st.radio(
        label="model",
        options=[
            "Random Forest",
            "Gradient Boosting",
            "Exponential Smoothing",
            "Moving Average",
            "Naive",
        ],
        label_visibility="collapsed",
    )

    # Colour badge per model
    MODEL_COLORS = {
        "Random Forest":        C3,
        "Gradient Boosting":    C5,
        "Exponential Smoothing":C4,
        "Moving Average":       C6,
        "Naive":                C2,
    }
    mc = MODEL_COLORS[model_choice]
    # Convert mc hex to rgba for badge background
    mc_rgba = "rgba(105,240,174,0.13)"  # default mint green
    MC_RGBA_MAP = {
        C3: "rgba(105,240,174,0.13)",   # Random Forest - mint
        C5: "rgba(206,147,216,0.13)",   # Gradient Boosting - purple
        C4: "rgba(255,213,79,0.13)",    # Exp Smoothing - amber
        C6: "rgba(128,222,234,0.13)",   # Moving Average - cyan
        C2: "rgba(255,107,107,0.13)",   # Naive - coral
    }
    mc_rgba = MC_RGBA_MAP.get(mc, "rgba(105,240,174,0.13)")

    st.markdown(
        f"<div style='background:{mc_rgba};border:1.5px solid {mc};"
        f"border-radius:8px;padding:6px 12px;margin-top:4px;"
        f"color:{mc};font-weight:600;font-size:.82rem'>"
        f"Selected → {model_choice}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Horizon ──
    st.markdown(
        f"<p style='color:{C1};font-weight:700;font-size:.85rem;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
        "📅 Forecast Horizon</p>",
        unsafe_allow_html=True,
    )
    horizon = st.slider(
        "Days ahead", min_value=7, max_value=90, value=30,
        label_visibility="collapsed",
    )
    st.markdown(
        f"<span style='color:{T['subtext']};font-size:.8rem'>"
        f"Forecasting next <b style='color:{C4}'>{horizon} days</b></span>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Surge threshold ──
    st.markdown(
        f"<p style='color:{C2};font-weight:700;font-size:.85rem;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
        "🚨 Surge Alert</p>",
        unsafe_allow_html=True,
    )
    surge_threshold = st.slider(
        "Threshold", min_value=1000, max_value=12000,
        value=8000, step=500, label_visibility="collapsed",
    )
    st.markdown(
        f"<span style='color:{T['subtext']};font-size:.8rem'>"
        f"Alert if forecast exceeds <b style='color:{C2}'>{surge_threshold:,}</b></span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        f"<span style='color:{T['subtext']};font-size:.75rem'>"
        "Data: HHS UAC Program · Jan 2023 – Dec 2025<br>"
        "720 reporting days · 1,075 after interpolation</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE TRAIN / TEST
# ─────────────────────────────────────────────────────────────────────────────
train_df, test_df, X_train, X_test, y_train, y_test = train_test_split(df, target)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
label_name = "HHS Care Load" if target == "hhs_load" else "Daily Discharges"
T2 = theme()

st.markdown(f"""
<div style="background:{T2['hero_grad']};padding:28px 36px;border-radius:18px;
            margin-bottom:24px;border:1px solid {T2['card_border']};
            box-shadow:0 8px 32px rgba(0,0,0,0.3)">
    <div style="display:flex;align-items:center;gap:16px">
        <span style="font-size:2.4rem">📈</span>
        <div>
            <h1 style="color:#FFFFFF;margin:0;font-size:1.6rem;font-weight:800">
                Predictive Forecasting of Care Load & Placement Demand
            </h1>
            <p style="color:{C1};margin:4px 0 0;font-size:.9rem">
                Forecasting &nbsp;<b style="color:{mc}">{label_name}</b>
                &nbsp;·&nbsp; Algorithm: <b style="color:{mc}">{model_choice}</b>
                &nbsp;·&nbsp; Horizon: <b style="color:{C4}">{horizon} days</b>
                &nbsp;·&nbsp; U.S. Department of Health and Human Services
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN SELECTED MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_model(model_name, target, horizon):
    train_df2, test_df2, X_tr, X_te, y_tr, y_te = train_test_split(df, target)
    ts = train_df2[target]

    if model_name == "Naive":
        preds = naive_forecast(ts, len(y_te))
        lower = preds * 0.90; upper = preds * 1.10
        fp = naive_forecast(ts, horizon)
        fl = fp * 0.90; fu = fp * 1.10

    elif model_name == "Moving Average":
        preds = moving_average_forecast(ts, len(y_te))
        lower = preds * 0.90; upper = preds * 1.10
        fp = moving_average_forecast(ts, horizon)
        fl = fp * 0.90; fu = fp * 1.10

    elif model_name == "Exponential Smoothing":
        preds, lower, upper = exp_smoothing_forecast(ts, len(y_te))
        fp, fl, fu = exp_smoothing_forecast(df[target].dropna(), horizon)

    elif model_name == "Random Forest":
        preds, lower, upper, mdl = random_forest_forecast(X_tr, y_tr, X_te)
        ff = build_future_features(df, target, horizon)
        fp = mdl.predict(ff)
        tree_p = np.array([t.predict(ff) for t in mdl.estimators_])
        fl = np.percentile(tree_p, 5, axis=0)
        fu = np.percentile(tree_p, 95, axis=0)

    else:  # Gradient Boosting
        preds, lower, upper, mdl = gradient_boosting_forecast(X_tr, y_tr, X_te)
        ff = build_future_features(df, target, horizon)
        fp = mdl.predict(ff)
        fl = fp * 0.92; fu = fp * 1.08

    metrics = evaluate(y_te, preds, model_name)
    fd = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return preds, lower, upper, y_te, test_df2.index, fp, fl, fu, fd, metrics

with st.spinner(f"⚙️  Running {model_choice}…"):
    (preds, lower, upper, y_te, test_idx,
     fp, fl, fu, fd, metrics) = run_model(model_choice, target, horizon)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Forecast Accuracy",  f"{metrics['Accuracy (%)']:.1f}%")
c2.metric("📉 MAE",                f"{metrics['MAE']:,.0f}")
c3.metric("📊 RMSE",               f"{metrics['RMSE']:,.0f}")
c4.metric("📐 MAPE",               f"{metrics['MAPE']:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Future Forecast",
    "🔍  Model vs Actual",
    "🏆  Model Comparison",
    "🚨  Surge Warning",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — FUTURE FORECAST
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"<p class='section-title'>Next {horizon}-day Forecast — {model_choice}</p>"
        f"<p class='section-sub'>Shaded band = 90 % confidence interval &nbsp;|&nbsp; "
        f"Dashed line = forecast &nbsp;|&nbsp; Solid line = last 180 days of real data</p>",
        unsafe_allow_html=True,
    )

    hist = df[target].iloc[-180:]
    fig1 = go.Figure()

    # Confidence band (filled area)
    fig1.add_trace(go.Scatter(
        x=list(fd) + list(fd[::-1]),
        y=list(fu) + list(fl[::-1]),
        fill="toself",
        fillcolor="rgba(105,240,174,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90 % confidence band",
        hoverinfo="skip",
    ))

    # Historical
    fig1.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical (last 180 days)",
        mode="lines",
        line=dict(color=C1, width=2),
    ))

    # Forecast line
    fig1.add_trace(go.Scatter(
        x=fd, y=fp,
        name=f"{model_choice} forecast",
        mode="lines+markers",
        line=dict(color=mc, width=2.5, dash="dot"),
        marker=dict(size=5, color=mc, line=dict(color=T2["card"], width=1)),
    ))

    # Surge line
    fig1.add_hline(
        y=surge_threshold,
        line=dict(color=C2, dash="dash", width=1.5),
        annotation=dict(
            text=f"⚠ Surge threshold: {surge_threshold:,}",
            font=dict(color=C2, size=11),
            bgcolor=T2["card"],
        ),
    )

    fig1.update_layout(**plot_layout(height=460))
    fig1.update_yaxes(title_text=label_name)
    fig1.update_xaxes(title_text="Date")
    st.plotly_chart(fig1, use_container_width=True)

    # Forecast table with colour-coded column
    st.markdown(
        "<p class='section-title' style='margin-top:20px'>Forecast Table</p>",
        unsafe_allow_html=True,
    )
    ftbl = pd.DataFrame({
        "Date":              fd.strftime("%Y-%m-%d"),
        f"{label_name} (forecast)": fp.round(0).astype(int),
        "Lower bound":       fl.round(0).astype(int),
        "Upper bound":       fu.round(0).astype(int),
        "Surge risk":        ["⚠ YES" if v >= surge_threshold else "✅ No"
                              for v in fp],
    })
    st.dataframe(ftbl, use_container_width=True, height=280)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL vs ACTUAL
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p class='section-title'>Predicted vs Actual — Test Period</p>"
        "<p class='section-sub'>Electric blue = real values the model never trained on  "
        "|  Coloured dashed = model's predictions for those same days</p>",
        unsafe_allow_html=True,
    )

    fig2 = go.Figure()

    # CI band on test
    fig2.add_trace(go.Scatter(
        x=list(test_idx) + list(test_idx[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor=mc_rgba,
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence band", hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=test_idx, y=y_te.values,
        name="Actual", mode="lines",
        line=dict(color=C1, width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=test_idx, y=preds,
        name=f"{model_choice} predicted", mode="lines",
        line=dict(color=mc, width=2.2, dash="dot"),
    ))
    fig2.update_layout(**plot_layout(height=430))
    fig2.update_yaxes(title_text="Children")
    st.plotly_chart(fig2, use_container_width=True)

    # Residuals bar chart
    st.markdown(
        "<p class='section-title' style='margin-top:8px'>Forecast Error  (Actual − Predicted)</p>"
        "<p class='section-sub'>Bars above zero = model under-predicted  "
        "|  below zero = model over-predicted</p>",
        unsafe_allow_html=True,
    )
    residuals = np.array(y_te) - np.array(preds)
    colors_res = [C3 if r >= 0 else C2 for r in residuals]
    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(
        x=test_idx, y=residuals,
        marker_color=colors_res,
        name="Residual",
        hovertemplate="Date: %{x}<br>Error: %{y:,.0f}<extra></extra>",
    ))
    fig_res.add_hline(y=0, line=dict(color=C4, width=1))
    fig_res.update_layout(**plot_layout(height=280, legend_bottom=False))
    fig_res.update_yaxes(title_text="Error")
    st.plotly_chart(fig_res, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<p class='section-title'>All 5 Models — Head-to-Head Comparison</p>"
        "<p class='section-sub'>Lower MAE / RMSE / MAPE = better  "
        "|  Higher Accuracy % = better  "
        "|  Green highlight = best value in column</p>",
        unsafe_allow_html=True,
    )

    @st.cache_data
    def compare_all(target):
        tr, te, Xtr, Xte, ytr, yte = train_test_split(df, target)
        ts = tr[target]; res = []
        res.append(evaluate(yte, naive_forecast(ts, len(yte)), "Naive"))
        res.append(evaluate(yte, moving_average_forecast(ts, len(yte)), "Moving Average"))
        p, _, _ = exp_smoothing_forecast(ts, len(yte))
        res.append(evaluate(yte, p, "Exp. Smoothing"))
        p, _, _, _ = random_forest_forecast(Xtr, ytr, Xte)
        res.append(evaluate(yte, p, "Random Forest"))
        p, _, _, _ = gradient_boosting_forecast(Xtr, ytr, Xte)
        res.append(evaluate(yte, p, "Gradient Boosting"))
        return pd.DataFrame(res)

    with st.spinner("Comparing all models…"):
        comp = compare_all(target)

    best = comp.loc[comp["Accuracy (%)"].idxmax(), "Model"]
    st.success(f"🏆  Best model on this target: **{best}**")

    # Styled table
    st.dataframe(
        comp.style
            .highlight_max(subset=["Accuracy (%)"], color="#1A3A2A", props="color:#69F0AE;font-weight:700")
            .highlight_min(subset=["MAE","RMSE","MAPE"], color="#1A3A2A", props="color:#69F0AE;font-weight:700"),
        use_container_width=True, height=220,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side bar charts
    BAR_COLORS = [C2, C4, C5, C3, C6]  # one per model

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=comp["Model"], y=comp["MAE"],
            marker_color=BAR_COLORS,
            text=comp["MAE"].round(0).astype(int),
            textposition="outside",
            textfont=dict(color=T2["font_color"]),
        ))
        fig_mae.update_layout(
            **plot_layout("MAE by Model  (lower = better)", height=340, legend_bottom=False)
        )
        fig_mae.update_yaxes(title_text="MAE")
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_b:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=comp["Model"], y=comp["Accuracy (%)"],
            marker_color=BAR_COLORS,
            text=comp["Accuracy (%)"].round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(color=T2["font_color"]),
        ))
        fig_acc.update_layout(
            **plot_layout("Accuracy % by Model  (higher = better)", height=340, legend_bottom=False)
        )
        fig_acc.update_yaxes(title_text="Accuracy %")
        st.plotly_chart(fig_acc, use_container_width=True)

    # RMSE comparison
    st.markdown("<p class='section-title'>RMSE Comparison</p>", unsafe_allow_html=True)
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        x=comp["Model"], y=comp["RMSE"],
        marker_color=BAR_COLORS,
        text=comp["RMSE"].round(0).astype(int),
        textposition="outside",
        textfont=dict(color=T2["font_color"]),
    ))
    fig_rmse.update_layout(
        **plot_layout("RMSE — penalises large errors more", height=320, legend_bottom=False)
    )
    fig_rmse.update_yaxes(title_text="RMSE")
    st.plotly_chart(fig_rmse, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — SURGE WARNING
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f"<p class='section-title'>Surge Early Warning System</p>"
        f"<p class='section-sub'>Flags any forecast day where predicted value exceeds "
        f"the threshold of <b style='color:{C2}'>{surge_threshold:,}</b>. "
        f"Adjust the threshold in the sidebar.</p>",
        unsafe_allow_html=True,
    )

    surge_mask = fp >= surge_threshold
    surge_days = fd[surge_mask]
    surge_vals = fp[surge_mask]
    surge_prob = len(surge_days) / horizon * 100

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("⚠ Surge days detected",       len(surge_days))
    sc2.metric("🎲 Breach probability",        f"{surge_prob:.1f}%")
    sc3.metric(
        "⏱ Surge lead time",
        f"{(surge_days[0] - pd.Timestamp.today()).days} days"
        if len(surge_days) > 0 else "No surge",
    )

    if len(surge_days) > 0:
        st.error(
            f"⚠️  **SURGE ALERT** — {len(surge_days)} of the next {horizon} days "
            f"are forecast to exceed **{surge_threshold:,}** people in care.  "
            f"First surge day: **{surge_days[0].strftime('%B %d, %Y')}**"
        )
    else:
        st.success(
            f"✅  No surge detected in the next {horizon} days.  "
            f"All forecast values stay below **{surge_threshold:,}**."
        )

    # Surge chart
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=fd, y=fp,
        name="Forecast", mode="lines+markers",
        line=dict(color=C3, width=2.5),
        marker=dict(size=5, color=C3),
    ))
    if len(surge_days) > 0:
        fig_s.add_trace(go.Scatter(
            x=surge_days, y=surge_vals,
            name="⚠ Surge days", mode="markers",
            marker=dict(color=C2, size=14, symbol="x-thin-open", line=dict(width=3)),
        ))
    fig_s.add_hline(
        y=surge_threshold,
        line=dict(color=C2, dash="dash", width=2),
        annotation=dict(
            text=f"⚠ Alert: {surge_threshold:,}",
            font=dict(color=C2, size=12),
            bgcolor=T2["card"],
        ),
    )
    fig_s.update_layout(**plot_layout(height=400))
    fig_s.update_yaxes(title_text="Forecast value")
    fig_s.update_xaxes(title_text="Date")
    st.plotly_chart(fig_s, use_container_width=True)

    if len(surge_days) > 0:
        st.markdown("<p class='section-title'>Surge Days Detail</p>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Date":                   surge_days.strftime("%Y-%m-%d"),
            "Forecast value":         surge_vals.round(0).astype(int),
            "Exceeds threshold by":   (surge_vals - surge_threshold).round(0).astype(int),
        }), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='color:{T['subtext']};font-size:.78rem;text-align:center'>"
    "UAC Predictive Forecasting · Project 2 · Unified Mentor · "
    "U.S. Department of Health and Human Services</p>",
    unsafe_allow_html=True,
)