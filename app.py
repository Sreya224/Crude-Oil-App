import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Crude Oil Volatility Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; }
    .metric-card {
        background: #1A2E40;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00B4D8;
        margin: 5px 0;
    }
    .event-card {
        background: #1A2E40;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #F4A261;
        margin: 8px 0;
    }
    .model-card {
        background: #1A2E40;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    h1, h2, h3 { color: #00B4D8 !important; }
    .stMetric label { color: #A0AEC0 !important; }
    .stMetric value { color: #FFFFFF !important; }
    div[data-testid="metric-container"] {
        background: #1A2E40;
        border: 1px solid #2D4A5E;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ Oil Volatility Dashboard")
    st.markdown("**Iran-US Geopolitical Tensions**")
    st.markdown("---")

    st.markdown("### 📅 Date Range")
    start_date = st.date_input("Start Date", value=datetime(2023, 1, 30))
    end_date   = st.date_input("End Date",   value=datetime(2026, 4, 30))

    st.markdown("### ⚙️ GARCH Settings")
    garch_p = st.selectbox("GARCH p", [1, 2], index=0)
    garch_q = st.selectbox("GARCH q", [1, 2], index=0)

    st.markdown("### 📊 Display Options")
    show_brent    = st.checkbox("Show Brent Price",    value=True)
    show_wti      = st.checkbox("Show WTI Price",      value=True)
    show_vol      = st.checkbox("Show Volatility",     value=True)
    show_garch    = st.checkbox("Show GARCH Forecast", value=True)

    st.markdown("---")
    st.markdown("**Project:** Crude Oil Volatility Study")
    st.markdown("**Period:** 2023–2026")
    st.markdown("**Models:** 9 trained models")

# ── Load and cache data ────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_price_data(start, end):
    tickers = ['CL=F', 'BZ=F', '^OVX']
    data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    data.columns = ['OVX', 'Brent', 'WTI']
    data = data.dropna(how='all').reset_index()
    data.rename(columns={'Date': 'date'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data['daily_return']  = data['WTI'].pct_change()
    data['volatility_5d'] = data['daily_return'].rolling(5).std()
    data['volatility_20d']= data['daily_return'].rolling(20).std()
    return data

@st.cache_data(ttl=3600)
def run_garch(returns_series, p=1, q=1):
    clean = returns_series.dropna() * 100
    model = arch_model(clean, vol='Garch', p=p, q=q, dist='normal')
    fit   = model.fit(disp='off')
    cond_vol = fit.conditional_volatility / 100
    return fit, cond_vol, fit.params, fit.aic, fit.bic

# ── Geopolitical events ────────────────────────────────────────
EVENTS = [
    {"date": "2023-04-03", "event": "OPEC Surprise Production Cut",        "color": "#F4A261"},
    {"date": "2023-10-07", "event": "Hamas-Israel Conflict Outbreak",       "color": "#E63946"},
    {"date": "2024-01-12", "event": "US Strikes on Yemen Houthis",          "color": "#F4A261"},
    {"date": "2024-04-01", "event": "Iran-Israel Direct Attack",            "color": "#E63946"},
    {"date": "2024-10-01", "event": "Iran Missiles on Israel",              "color": "#E63946"},
    {"date": "2026-02-05", "event": "US-Iran Nuclear Talks Announced",      "color": "#06D6A0"},
    {"date": "2026-02-24", "event": "US Seizes Iranian Oil Tanker",         "color": "#F4A261"},
    {"date": "2026-03-12", "event": "OPEC Emergency Meeting",               "color": "#F4A261"},
    {"date": "2026-04-21", "event": "US Seizes Second Iranian Tanker",      "color": "#F4A261"},
]

MODEL_RESULTS = {
    "GARCH(1,1)":    {"rmse": 0.0146, "r2": 0.6507, "mae": 0.0105, "type": "Econometric",   "color": "#0096C7"},
    "GARCH-X":       {"rmse": 0.0168, "r2": 0.5374, "mae": 0.0116, "type": "Econometric",   "color": "#48CAE4"},
    "BiLSTM":        {"rmse": 0.0234, "r2": 0.5082, "mae": 0.0127, "type": "Deep Learning", "color": "#E63946"},
    "Transformer":   {"rmse": 0.0283, "r2": 0.5354, "mae": 0.0128, "type": "Deep Learning", "color": "#F4A261"},
    "GRU":           {"rmse": 0.0252, "r2": 0.4282, "mae": 0.0131, "type": "Deep Learning", "color": "#FF6B6B"},
    "LightGBM":      {"rmse": 0.0207, "r2": 0.2998, "mae": 0.0135, "type": "Tree ML",       "color": "#06D6A0"},
    "XGBoost":       {"rmse": 0.0210, "r2": 0.2762, "mae": 0.0139, "type": "Tree ML",       "color": "#40E0D0"},
    "LSTM":          {"rmse": 0.0305, "r2": 0.2134, "mae": 0.0148, "type": "Deep Learning", "color": "#FFB703"},
    "Random Forest": {"rmse": 0.0224, "r2": 0.1781, "mae": 0.0148, "type": "Tree ML",       "color": "#A8DADC"},
}

SHAP_FEATURES = {
    "vol_lag1":                0.008816,
    "OVX":                     0.001605,
    "return_lag5":             0.001281,
    "trend_iran_nuclear_deal": 0.001179,
    "OVX_lag1":                0.001072,
    "return_lag2":             0.000382,
    "return_lag1":             0.000275,
    "trend_iran_military":     0.000261,
    "GPR_global":              0.000198,
    "avg_sentiment":           0.000065,
}

ABLATION = {
    "Price Features":     {"change": +0.0106, "color": "#E63946"},
    "Article Count":      {"change": +0.0023, "color": "#F4A261"},
    "Google Trends":      {"change": +0.0009, "color": "#FFB703"},
    "Sentiment Features": {"change": +0.0009, "color": "#06D6A0"},
    "Event Features":     {"change": +0.0004, "color": "#0096C7"},
    "GPR Features":       {"change": +0.0002, "color": "#48CAE4"},
}

# ── Load data ──────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    df = load_price_data(str(start_date), str(end_date))

if df.empty:
    st.error("No data loaded. Please check date range.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────
st.markdown("# 🛢️ Crude Oil Volatility — Iran-US Geopolitical Tensions")
st.markdown(f"**Data Period:** {start_date} → {end_date} &nbsp;|&nbsp; **Trading Days:** {len(df)}")
st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────
latest = df.dropna().iloc[-1]
prev   = df.dropna().iloc[-2] if len(df) > 1 else latest

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("WTI Price (USD)",    f"${latest['WTI']:.2f}",  f"{((latest['WTI']-prev['WTI'])/prev['WTI']*100):.2f}%")
with col2:
    st.metric("Brent Price (USD)",  f"${latest['Brent']:.2f}", f"{((latest['Brent']-prev['Brent'])/prev['Brent']*100):.2f}%")
with col3:
    st.metric("OVX Index",          f"{latest['OVX']:.2f}",   f"{(latest['OVX']-prev['OVX']):.2f}")
with col4:
    st.metric("5d Volatility",      f"{latest['volatility_5d']:.4f}", None)
with col5:
    wti_min = df['WTI'].min(); wti_max = df['WTI'].max()
    st.metric("WTI Range",          f"${wti_min:.0f} – ${wti_max:.0f}", f"Δ${wti_max-wti_min:.0f}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Volatility",
    "⚡ GARCH Forecast",
    "🏆 Model Comparison",
    "🌍 Event Study",
    "🔍 Explainability"
])

# ── TAB 1: Price & Volatility ─────────────────────────────────
with tab1:
    st.subheader("Crude Oil Price & Volatility")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("WTI & Brent Crude Oil Price (USD)", "5-Day & 20-Day Rolling Volatility", "OVX — Crude Oil Volatility Index"),
        vertical_spacing=0.08, row_heights=[0.45, 0.3, 0.25])

    # Price chart
    if show_wti:
        fig.add_trace(go.Scatter(x=df['date'], y=df['WTI'], name='WTI', line=dict(color='#00B4D8', width=1.8)), row=1, col=1)
    if show_brent:
        fig.add_trace(go.Scatter(x=df['date'], y=df['Brent'], name='Brent', line=dict(color='#F4A261', width=1.8)), row=1, col=1)

    # Shock event lines
    df_range = df[df['date'].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    for ev in EVENTS:
        ev_date = pd.Timestamp(ev['date'])
        if pd.Timestamp(start_date) <= ev_date <= pd.Timestamp(end_date):
            for row in [1, 2, 3]:
                fig.add_vline(x=ev_date, line_dash="dash", line_color=ev['color'], opacity=0.5, row=row, col=1)

    # Volatility
    if show_vol:
        fig.add_trace(go.Scatter(x=df['date'], y=df['volatility_5d'],  name='5d Vol',  line=dict(color='#06D6A0', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['volatility_20d'], name='20d Vol', line=dict(color='#FFB703', width=1.5, dash='dot')), row=2, col=1)

    # OVX
    fig.add_trace(go.Scatter(x=df['date'], y=df['OVX'], name='OVX', line=dict(color='#E63946', width=1.5), fill='tozeroy', fillcolor='rgba(230,57,70,0.1)'), row=3, col=1)

    fig.update_layout(
        height=650, template='plotly_dark',
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A',
        legend=dict(bgcolor='#1A2E40', bordercolor='#2D4A5E'),
        font=dict(color='#E0E0E0')
    )
    fig.update_xaxes(gridcolor='#1A2E40', zeroline=False)
    fig.update_yaxes(gridcolor='#1A2E40', zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # Event legend
    st.markdown("**Geopolitical Shock Events (dashed lines):**")
    cols = st.columns(3)
    for i, ev in enumerate(EVENTS):
        ev_date = pd.Timestamp(ev['date'])
        if pd.Timestamp(start_date) <= ev_date <= pd.Timestamp(end_date):
            with cols[i % 3]:
                st.markdown(f'<div class="event-card"><b style="color:{ev["color"]}">{ev["date"]}</b><br>{ev["event"]}</div>', unsafe_allow_html=True)

# ── TAB 2: GARCH Forecast ─────────────────────────────────────
with tab2:
    st.subheader(f"GARCH({garch_p},{garch_q}) Volatility Forecast")

    returns = df['daily_return'].dropna()

    if len(returns) < 30:
        st.warning("Not enough data for GARCH. Select a wider date range.")
    else:
        with st.spinner(f"Fitting GARCH({garch_p},{garch_q}) model..."):
            fit, cond_vol, params, aic, bic = run_garch(df.set_index('date')['daily_return'], garch_p, garch_q)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("AIC",    f"{aic:.2f}")
        with c2: st.metric("BIC",    f"{bic:.2f}")
        with c3: st.metric("α (shock sensitivity)",  f"{params.get('alpha[1]', 0):.4f}")
        with c4: st.metric("β (persistence)",        f"{params.get('beta[1]', 0):.4f}")

        alpha = params.get('alpha[1]', 0)
        beta  = params.get('beta[1]', 0)
        st.info(f"**α + β = {alpha+beta:.4f}** — {'High volatility persistence (close to 1)' if alpha+beta > 0.95 else 'Moderate persistence'}")

        # GARCH chart
        garch_dates = df.dropna(subset=['daily_return'])['date'].values
        min_len = min(len(garch_dates), len(cond_vol))

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Actual vs GARCH Conditional Volatility", "Daily Returns"),
            vertical_spacing=0.1, row_heights=[0.6, 0.4])

        actual_vol = df['volatility_5d'].dropna()
        actual_dates = df.dropna(subset=['volatility_5d'])['date'].values

        fig2.add_trace(go.Scatter(x=actual_dates, y=actual_vol, name='Actual Volatility (5d)', line=dict(color='#00B4D8', width=1.5)), row=1, col=1)
        if show_garch:
            fig2.add_trace(go.Scatter(x=garch_dates[:min_len], y=cond_vol[:min_len], name=f'GARCH({garch_p},{garch_q}) Conditional Vol', line=dict(color='#E63946', width=1.5, dash='dot')), row=1, col=1)

        fig2.add_trace(go.Bar(x=df['date'], y=df['daily_return'], name='Daily Return',
            marker_color=['#E63946' if r < 0 else '#06D6A0' for r in df['daily_return'].fillna(0)],
            opacity=0.7), row=2, col=1)

        fig2.update_layout(height=550, template='plotly_dark', paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A', font=dict(color='#E0E0E0'), legend=dict(bgcolor='#1A2E40'))
        fig2.update_xaxes(gridcolor='#1A2E40')
        fig2.update_yaxes(gridcolor='#1A2E40')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**GARCH Model Parameters:**")
        param_df = pd.DataFrame({'Parameter': list(params.index), 'Value': [f"{v:.6f}" for v in params.values]})
        st.dataframe(param_df, use_container_width=True, hide_index=True)

# ── TAB 3: Model Comparison ───────────────────────────────────
with tab3:
    st.subheader("9-Model Volatility Forecasting Comparison")

    sorted_models = sorted(MODEL_RESULTS.items(), key=lambda x: x[1]['rmse'])

    # RMSE bar chart
    names  = [m[0] for m in sorted_models]
    rmses  = [m[1]['rmse'] for m in sorted_models]
    colors = [m[1]['color'] for m in sorted_models]
    types  = [m[1]['type'] for m in sorted_models]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=names, y=rmses,
        marker_color=colors,
        text=[f"{r:.4f}" for r in rmses],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
    ))
    fig3.add_hline(y=min(rmses), line_dash="dash", line_color="#FFD700", annotation_text=f"Best: {min(rmses):.4f}", annotation_position="top right")
    fig3.update_layout(
        title="Test RMSE — Lower is Better",
        height=420, template='plotly_dark',
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A',
        font=dict(color='#E0E0E0'),
        yaxis_title="RMSE", xaxis_title="Model"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Full metrics table
    st.markdown("**Complete Evaluation Metrics:**")
    metrics_data = []
    for name, m in sorted_models:
        metrics_data.append({
            "Model": name, "Type": m['type'],
            "RMSE": f"{m['rmse']:.4f}",
            "MAE":  f"{m['mae']:.4f}",
            "R²":   f"{m['r2']:.4f}",
            "Rank": sorted_models.index((name, m)) + 1
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # R² chart
    fig4 = go.Figure()
    r2_sorted = sorted(MODEL_RESULTS.items(), key=lambda x: x[1]['r2'], reverse=True)
    fig4.add_trace(go.Bar(
        x=[m[0] for m in r2_sorted],
        y=[m[1]['r2'] for m in r2_sorted],
        marker_color=[m[1]['color'] for m in r2_sorted],
        text=[f"{m[1]['r2']:.4f}" for m in r2_sorted],
        textposition='outside'
    ))
    fig4.update_layout(
        title="R² Score — Higher is Better (% Variance Explained)",
        height=380, template='plotly_dark',
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A',
        font=dict(color='#E0E0E0'),
        yaxis_title="R²", xaxis_title="Model"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Key finding box
    st.markdown("""
    <div class="metric-card">
    <b style="color:#00B4D8">Key Finding:</b> GARCH(1,1) achieves the best RMSE (0.0146) and R² (0.6507),
    outperforming all deep learning and tree-based models during the extreme Iran-US escalation test period.
    This challenges the assumption that complex models always outperform classical econometric approaches
    during unprecedented geopolitical regimes.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4: Event Study ────────────────────────────────────────
with tab4:
    st.subheader("Geopolitical Event Study — Abnormal Volatility Analysis")

    event_results = [
        {"date": "2023-04-03", "event": "OPEC Surprise Cut",           "abnormal_vol": -0.0011, "p_value": 0.4376, "significant": False},
        {"date": "2023-10-07", "event": "Hamas-Israel Conflict",       "abnormal_vol":  0.0056, "p_value": 0.0041, "significant": True},
        {"date": "2024-01-12", "event": "US Strikes Yemen",            "abnormal_vol": -0.0015, "p_value": 0.5006, "significant": False},
        {"date": "2024-04-01", "event": "Iran-Israel Direct Attack",   "abnormal_vol": -0.0050, "p_value": 0.0001, "significant": True},
        {"date": "2024-10-01", "event": "Iran Missiles on Israel",     "abnormal_vol": -0.0001, "p_value": 0.9432, "significant": False},
        {"date": "2026-02-05", "event": "US-Iran Talks Announced",     "abnormal_vol":  0.0013, "p_value": 0.5009, "significant": False},
        {"date": "2026-02-24", "event": "US Seizes Iranian Tanker",    "abnormal_vol": -0.0013, "p_value": 0.5869, "significant": False},
        {"date": "2026-03-12", "event": "OPEC Emergency Meeting",      "abnormal_vol":  0.0124, "p_value": 0.0638, "significant": False},
        {"date": "2026-04-21", "event": "US Seizes Tanker 2",          "abnormal_vol": -0.0007, "p_value": 0.9077, "significant": False},
    ]

    # Summary stats
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Events Tested", "9")
    with c2: st.metric("Significant Events",  "2", "p < 0.05")
    with c3: st.metric("Event Window",        "±5 Trading Days")

    # Abnormal vol chart
    fig5 = go.Figure()
    bar_colors = ['#E63946' if e['significant'] else '#2D4A5E' for e in event_results]
    border_colors = ['#FFD700' if e['significant'] else '#2D4A5E' for e in event_results]

    fig5.add_trace(go.Bar(
        x=[e['event'] for e in event_results],
        y=[e['abnormal_vol'] for e in event_results],
        marker_color=bar_colors,
        marker_line_color=border_colors,
        marker_line_width=2,
        text=[f"p={e['p_value']:.4f}{'*' if e['significant'] else ''}" for e in event_results],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Abnormal Vol: %{y:.4f}<extra></extra>'
    ))
    fig5.add_hline(y=0, line_color='white', line_width=1)
    fig5.update_layout(
        title="Mean Abnormal Volatility ±5 Days (Red = Statistically Significant, p<0.05)",
        height=420, template='plotly_dark',
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A',
        font=dict(color='#E0E0E0'),
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Results table
    result_df = pd.DataFrame([{
        "Date": e['date'], "Event": e['event'],
        "Abnormal Vol": f"{e['abnormal_vol']:+.4f}",
        "p-value": f"{e['p_value']:.4f}",
        "Significant": "✅ YES" if e['significant'] else "❌ NO"
    } for e in event_results])
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="event-card">
    <b style="color:#F4A261">Interpretation:</b><br>
    Only 2 of 9 events produced statistically significant abnormal volatility.
    The <b>Hamas-Israel conflict</b> caused a positive spike (+0.0056) — a genuinely unexpected shock.
    The <b>Iran-Israel direct attack</b> showed negative abnormal volatility (-0.0050) suggesting
    markets had partially anticipated this escalation. Most 2026 Iran-US events were not significant,
    suggesting markets adapted to prolonged tensions.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5: Explainability ─────────────────────────────────────
with tab5:
    st.subheader("Model Explainability — SHAP Feature Importance")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # SHAP bar chart
        features = list(SHAP_FEATURES.keys())
        values   = list(SHAP_FEATURES.values())
        colors_shap = ['#E63946' if f == 'vol_lag1' else '#00B4D8' if f == 'OVX' else '#F4A261' if 'trend' in f else '#06D6A0' for f in features]

        fig6 = go.Figure(go.Bar(
            x=values, y=features,
            orientation='h',
            marker_color=colors_shap,
            text=[f"{v:.6f}" for v in values],
            textposition='outside'
        ))
        fig6.update_layout(
            title="SHAP Feature Importance (LightGBM)",
            height=420, template='plotly_dark',
            paper_bgcolor='#0D1B2A', plot_bgcolor='#0D1B2A',
            font=dict(color='#E0E0E0'),
            xaxis_title="Mean |SHAP Value|"
        )
        fig6.update_yaxes(autorange='reversed')
        st.plotly_chart(fig6, use_container_width=True)

    with col_b:
        st.markdown("**Feature Group Importance**")
        st.markdown("*Ablation Study Results (RMSE increase when group removed):*")

        for group, data in ABLATION.items():
            pct = data['change'] / 0.0106 * 100
            st.markdown(f"""
            <div class="model-card">
                <b style="color:{data['color']}">{group}</b><br>
                RMSE +{data['change']:.4f} when removed<br>
                <div style="background:#2D4A5E;border-radius:4px;margin-top:5px">
                    <div style="background:{data['color']};width:{pct:.0f}%;height:8px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Granger causality result
    st.markdown("---")
    st.markdown("### Granger Causality Test Results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-card">
        <b style="color:#00B4D8">Sentiment → Volatility</b><br><br>
        Lag 1: p = 0.9724 — not significant<br>
        Lag 2: p = 0.9438 — not significant<br>
        Lag 3: p = 0.2744 — not significant<br><br>
        <b style="color:#E63946">Conclusion: Sentiment does NOT predict volatility</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
        <b style="color:#00B4D8">Volatility → Sentiment</b><br><br>
        Lag 1: p = 0.3304 — not significant<br>
        Lag 2: p = 0.5961 — not significant<br>
        Lag 3: p = 0.3159 — not significant<br><br>
        <b style="color:#E63946">Conclusion: Volatility does NOT predict sentiment</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
    <b style="color:#06D6A0">Key Explainability Finding:</b>
    <code>vol_lag1</code> (yesterday's volatility) dominates all models with normalized importance of 0.90.
    This confirms <b>volatility clustering</b> — not geopolitical sentiment — is the primary driver of crude oil volatility.
    FinBERT sentiment SHAP = 0.000065, effectively zero across all tree models.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#A0AEC0;font-size:12px">
Crude Oil Volatility Study — Iran-US Geopolitical Tensions (2023–2026)<br>
Department of Computer Science & Engineering | Sahrdaya College of Engineering
</div>
""", unsafe_allow_html=True)
