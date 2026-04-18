import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&display=swap');

:root {
    --bg:         #F7F6F2;
    --surface:    #FFFFFF;
    --card:       #FFFFFF;
    --border:     #E8E5DE;
    --border2:    #D4CFBF;
    --accent:     #2D6A4F;
    --accent-lt:  #52B788;
    --accent-bg:  #EAF4EF;
    --text:       #1C1B18;
    --text2:      #4A4740;
    --muted:      #8C897E;
    --radius:     14px;
    --shadow:     0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.05);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"], .main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    max-width: 1180px !important;
    padding: 1.8rem 2.4rem 3rem !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(118deg, #1B4332 0%, #2D6A4F 55%, #40916C 100%);
    border-radius: 20px;
    padding: 2.8rem 3.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(45,106,79,0.28);
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -40px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 65%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 38%;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 2.9rem !important;
    font-weight: 700 !important;
    line-height: 1.08 !important;
    margin: 0 0 0.8rem !important;
    color: #FFFFFF !important;
    letter-spacing: -0.01em !important;
}
.hero h1 em { font-style: italic; color: rgba(255,255,255,0.65) !important; }
.hero p {
    color: rgba(255,255,255,0.65);
    font-size: 0.98rem;
    font-weight: 400;
    max-width: 520px;
    line-height: 1.7;
    margin: 0;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-lt));
    border-radius: 3px 3px 0 0;
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 5px;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.kpi-value .accent { color: var(--accent); }

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.8rem !important;
}
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    padding: 0.7rem 1.4rem !important;
    border: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    margin-bottom: -2px !important;
    transition: color .2s !important;
}
button[data-baseweb="tab"]:hover { color: var(--text) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabPanel"] { padding-top: 0 !important; }

/* ── Panel card ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.panel-icon {
    width: 34px; height: 34px;
    background: var(--accent-bg);
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.panel-sub {
    font-size: 0.81rem;
    color: var(--muted);
    margin-bottom: 0;
    margin-left: 44px;
}

/* ── Widget labels ── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stSlider"] > label p,
div[data-testid="stRadio"] > label p,
.stFileUploader label p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Radio */
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 8px !important; }
div[data-testid="stRadio"] div[role="radio"] {
    background: var(--bg) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 8px !important;
    padding: 5px 16px !important;
    color: var(--text2) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    transition: all .18s !important;
}
div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
    background: var(--accent-bg) !important;
    border-color: var(--accent-lt) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.8rem !important;
    width: 100% !important;
    transition: background .18s, transform .15s, box-shadow .15s !important;
    box-shadow: 0 2px 10px rgba(45,106,79,0.25) !important;
}
.stButton > button:hover {
    background: #245A42 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(45,106,79,0.35) !important;
}
[data-testid="stDownloadButton"] > button {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent-lt) !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
    width: 100% !important;
    transition: background .18s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--accent-bg) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Result badge */
.result-badge {
    background: var(--accent-bg);
    border: 1.5px solid var(--accent-lt);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-top: 1rem;
}
.result-badge .rb-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.result-badge .rb-name {
    font-family: 'Fraunces', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--accent);
}
.result-badge .rb-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
}

/* Segment legend grid */
.seg-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
    gap: 0.85rem;
    margin-top: 0.8rem;
}
.seg-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    transition: border-color .18s, box-shadow .18s;
}
.seg-card:hover { border-color: var(--border2); box-shadow: var(--shadow); }
.seg-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 7px;
    flex-shrink: 0;
}
.seg-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
}
.seg-desc { font-size: 0.76rem; color: var(--muted); line-height: 1.55; }

/* Chart wrap */
.chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem 0.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Fraunces', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.1rem;
}
.chart-sub { font-size: 0.73rem; color: var(--muted); margin-bottom: 0.4rem; }

/* History table */
.hist-row {
    display: grid;
    grid-template-columns: 80px 70px 72px 55px 90px 1fr;
    align-items: center;
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--text2);
    transition: background .15s;
}
.hist-row:hover { background: var(--bg); }
.hist-row:last-child { border-bottom: none; }
.hist-header {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--bg);
    border-radius: 10px 10px 0 0;
    border-bottom: 1px solid var(--border2) !important;
}
.seg-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--accent-bg);
    border: 1px solid rgba(82,183,136,0.35);
    color: var(--accent);
    font-size: 0.71rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
}

.divider { height: 1px; background: var(--border); margin: 1.8rem 0; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline       = joblib.load('models/customer_segmentation_pipeline.joblib')
    cluster_labels = joblib.load('models/cluster_labels_mapping.joblib')
    features       = joblib.load('models/feature_names.joblib')
    return pipeline, cluster_labels, features

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Constants ──────────────────────────────────────────────────────────────────
COLORS = ["#2D6A4F", "#52B788", "#C77B2E", "#1A6B9E", "#9B5DE5"]
SEG_COLORS = {
    "Affluent Enthusiasts":       COLORS[0],
    "Conservative High-Earners":  COLORS[1],
    "Budget Shoppers":            COLORS[2],
    "Value-Conscious":            COLORS[3],
    "Average Consumers":          COLORS[4],
}
SEGMENTS_INFO = {
    "Affluent Enthusiasts":      ("High income & high spending. Prime targets for luxury products and premium services.", COLORS[0]),
    "Conservative High-Earners": ("High income, low spending. Respond well to value-proposition and investment narratives.", COLORS[1]),
    "Budget Shoppers":           ("Lower income yet high spending. Benefit from payment plans and BNPL offers.",          COLORS[2]),
    "Value-Conscious":           ("Lower income & low spending. Price-sensitive; driven by deals and discounts.",         COLORS[3]),
    "Average Consumers":         ("Mid-range on both axes. Balanced offerings and loyalty programmes work best.",        COLORS[4]),
}

def light_fig(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#8C897E", size=11.5),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E8E5DE", borderwidth=1, font=dict(size=11)),
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
    )
    fig.update_xaxes(gridcolor="#F0EDE6", zeroline=False, linecolor="#E8E5DE")
    fig.update_yaxes(gridcolor="#F0EDE6", zeroline=False, linecolor="#E8E5DE")
    return fig

def fig_title(title, sub=""):
    txt = f"<b>{title}</b>"
    if sub:
        txt += f"<br><sup style='color:#8C897E;font-weight:400'>{sub}</sup>"
    return dict(title=dict(text=txt, font=dict(family="Fraunces", size=15, color="#1C1B18")))

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ &nbsp;ML-Powered Analytics</div>
    <h1>Customer <em>Segmentation</em></h1>
    <p>Identify behavioural clusters in your customer base to drive targeted strategy,
       personalised marketing, and smarter retention decisions.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ──────────────────────────────────────────────────────────────────
total_preds = len(st.session_state.history)
batch_runs  = sum(1 for h in st.session_state.history if h.get("source") == "batch")
st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-card">
        <div class="kpi-label">Algorithm</div>
        <div class="kpi-value">K<span class="accent">-</span>Means</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Segments</div>
        <div class="kpi-value"><span class="accent">5</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Predictions</div>
        <div class="kpi-value"><span class="accent">{total_preds}</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Batch Rows Run</div>
        <div class="kpi-value"><span class="accent">{batch_runs}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤  Single Prediction", "📂  Batch Prediction", "🕘  History"])

try:
    pipeline, cluster_labels, features = load_model()

    # ════════════════════════════════════════════
    # TAB 1 — Single Prediction
    # ════════════════════════════════════════════
    with tab1:
        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("""
            <div class="panel">
                <div class="panel-title"><span class="panel-icon">👤</span> Customer Details</div>
                <div class="panel-sub">Fill in the attributes below to classify a single customer.</div>
            </div>
            """, unsafe_allow_html=True)

            gender   = st.radio("Gender", options=["Female", "Male"], horizontal=True)
            age      = st.slider("Age", min_value=18, max_value=90, value=30)
            income   = st.slider("Annual Income (k$)", min_value=0, max_value=200, value=50)
            spending = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)
            run      = st.button("Predict Segment →")

        with col_result:
            if run:
                input_data = pd.DataFrame({
                    'Gender':                 [gender],
                    'Age':                    [age],
                    'Annual Income (k$)':     [income],
                    'Spending Score (1-100)': [spending]
                })
                cluster  = pipeline.predict(input_data)[0]
                segment  = cluster_labels[cluster]
                seg_color = SEG_COLORS.get(segment, COLORS[0])
                seg_desc  = SEGMENTS_INFO.get(segment, ("", COLORS[0]))[0]

                # Save to history
                st.session_state.history.append({
                    "time":     datetime.now().strftime("%H:%M:%S"),
                    "source":   "single",
                    "gender":   gender,
                    "age":      age,
                    "income":   income,
                    "spending": spending,
                    "segment":  segment,
                })

                st.markdown(f"""
                <div class="result-badge">
                    <div class="rb-label">Predicted Segment</div>
                    <div class="rb-name">{segment}</div>
                    <div class="rb-sub">{seg_desc}</div>
                </div>
                """, unsafe_allow_html=True)

                # Radar chart
                hex_c   = seg_color.lstrip('#')
                r, g, b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                norm_vals = [
                    round((age - 18) / 72 * 100),
                    round(income / 200 * 100),
                    spending
                ]
                cats = ['Age', 'Income', 'Spending Score']
                fig_radar = go.Figure(go.Scatterpolar(
                    r=norm_vals + [norm_vals[0]],
                    theta=cats + [cats[0]],
                    fill='toself',
                    fillcolor=f"rgba({r},{g},{b},0.15)",
                    line=dict(color=seg_color, width=2.5),
                    marker=dict(size=6, color=seg_color)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(
                            visible=True, range=[0,100],
                            gridcolor='#E8E5DE',
                            tickfont=dict(color='#8C897E', size=9),
                            tickvals=[25,50,75,100]
                        ),
                        angularaxis=dict(
                            gridcolor='#E8E5DE',
                            tickfont=dict(family='Plus Jakarta Sans', color='#4A4740', size=12)
                        )
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=250
                )
                st.markdown('<div class="chart-wrap"><div class="chart-title">Attribute Profile</div><div class="chart-sub">Normalised 0–100 scale across all three dimensions</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download
                single_df = pd.DataFrame([{
                    "Gender": gender, "Age": age,
                    "Annual Income (k$)": income,
                    "Spending Score (1-100)": spending,
                    "Segment": segment
                }])
                st.download_button(
                    "⬇  Download This Prediction as CSV",
                    data=single_df.to_csv(index=False),
                    file_name=f"prediction_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;
                            justify-content:center;height:340px;color:#C5C2B8;
                            font-family:'Fraunces',serif;font-size:1rem;font-style:italic;
                            text-align:center;gap:10px;">
                    <div style="font-size:2.5rem;">◈</div>
                    Fill in the form and click<br>
                    <b style="color:#8C897E;font-style:normal">Predict Segment</b> to see results.
                </div>
                """, unsafe_allow_html=True)

        # Segment legend
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Fraunces',serif;font-size:1rem;font-weight:600;
                    color:#1C1B18;margin-bottom:0.2rem;">Segment Definitions</div>
        <div style="font-size:0.78rem;color:#8C897E;margin-bottom:0.9rem;">
            Five behavioural clusters derived from income and spending patterns.
        </div>
        """, unsafe_allow_html=True)
        cards_html = '<div class="seg-grid">'
        for name, (desc, color) in SEGMENTS_INFO.items():
            cards_html += f"""
            <div class="seg-card">
                <div class="seg-name"><span class="seg-dot" style="background:{color};"></span>{name}</div>
                <div class="seg-desc">{desc}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 2 — Batch Prediction
    # ════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span class="panel-icon">📂</span> Upload Customer Data</div>
            <div class="panel-sub">CSV must contain: <b>Gender, Age, Annual Income (k$), Spending Score (1-100)</b></div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Drop CSV here or click to browse", type="csv", label_visibility="collapsed")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

            if all(c in df.columns for c in required):
                clusters      = pipeline.predict(df)
                df['Segment'] = [cluster_labels[c] for c in clusters]

                for _, row in df.iterrows():
                    st.session_state.history.append({
                        "time":     datetime.now().strftime("%H:%M:%S"),
                        "source":   "batch",
                        "gender":   row.get("Gender", "—"),
                        "age":      int(row.get("Age", 0)),
                        "income":   row.get("Annual Income (k$)", 0),
                        "spending": row.get("Spending Score (1-100)", 0),
                        "segment":  row["Segment"],
                    })

                seg_counts = df['Segment'].value_counts()
                top_seg    = seg_counts.index[0]

                st.markdown(f"""
                <div class="kpi-strip" style="grid-template-columns:repeat(3,1fr);margin-top:0.2rem;">
                    <div class="kpi-card">
                        <div class="kpi-label">Customers Uploaded</div>
                        <div class="kpi-value"><span class="accent">{len(df)}</span></div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Top Segment</div>
                        <div class="kpi-value" style="font-size:1rem;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;">{top_seg}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Segments Found</div>
                        <div class="kpi-value"><span class="accent">{df['Segment'].nunique()}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Row 1: Donut + Bar
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    fig_pie = px.pie(df, names='Segment', color_discrete_sequence=COLORS, hole=0.52)
                    fig_pie.update_traces(
                        textposition='outside',
                        textfont=dict(family="Plus Jakarta Sans", size=11),
                        marker_line=dict(color="#FFFFFF", width=2)
                    )
                    fig_pie = light_fig(fig_pie, 310)
                    fig_pie.update_layout(**fig_title("Segment Share", "Proportion of each cluster"))
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    bar_df = seg_counts.reset_index()
                    bar_df.columns = ['Segment', 'Count']
                    fig_bar = px.bar(bar_df, x='Count', y='Segment', orientation='h',
                                    color='Segment', color_discrete_sequence=COLORS)
                    fig_bar.update_traces(marker_line_width=0)
                    fig_bar = light_fig(fig_bar, 310)
                    fig_bar.update_layout(showlegend=False,
                                         **fig_title("Customer Count per Segment"),
                                         yaxis=dict(categoryorder='total ascending'))
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Row 2: Scatter + Box
                c3, c4 = st.columns(2, gap="medium")
                with c3:
                    fig_sc = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                                       color='Segment', symbol='Gender',
                                       color_discrete_sequence=COLORS, opacity=0.82)
                    fig_sc.update_traces(marker=dict(size=9, line=dict(width=0.8, color='rgba(255,255,255,0.7)')))
                    fig_sc = light_fig(fig_sc, 320)
                    fig_sc.update_layout(**fig_title("Income vs Spending", "Coloured by segment · shaped by gender"))
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with c4:
                    fig_age = px.box(df, x='Segment', y='Age', color='Segment',
                                    color_discrete_sequence=COLORS, points='outliers')
                    fig_age.update_traces(marker_size=4)
                    fig_age = light_fig(fig_age, 320)
                    fig_age.update_layout(showlegend=False,
                                          **fig_title("Age Distribution by Segment"),
                                          xaxis_tickangle=-18)
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_age, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Row 3: Violin full-width
                fig_vio = px.violin(df, x='Segment', y='Annual Income (k$)',
                                    color='Segment', box=True,
                                    color_discrete_sequence=COLORS)
                fig_vio = light_fig(fig_vio, 310)
                fig_vio.update_layout(showlegend=False,
                                      **fig_title("Income Distribution by Segment", "Violin + interquartile box"),
                                      xaxis_tickangle=-18)
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_vio, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Table + download
                with st.expander("📋  View full predictions table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇  Download Segmented Results as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Missing required columns. Please include: Gender, Age, Annual Income (k$), Spending Score (1-100)")

        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:260px;color:#C5C2B8;
                        font-family:'Fraunces',serif;font-size:0.98rem;font-style:italic;
                        text-align:center;gap:10px;background:var(--surface);
                        border:1px solid var(--border);border-radius:var(--radius);">
                <div style="font-size:2.2rem;">📂</div>
                Upload a CSV file to see<br>
                <b style="color:#8C897E;font-style:normal">batch segmentation results</b>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 3 — History
    # ════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span class="panel-icon">🕘</span> Prediction History</div>
            <div class="panel-sub">All predictions made during this session — from both single and batch modes.</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:260px;color:#C5C2B8;
                        font-family:'Fraunces',serif;font-size:0.98rem;font-style:italic;
                        text-align:center;gap:10px;background:var(--surface);
                        border:1px solid var(--border);border-radius:var(--radius);">
                <div style="font-size:2.2rem;">🕘</div>
                No predictions yet.<br>
                <b style="color:#8C897E;font-style:normal">Run single or batch predictions</b> to see them here.
            </div>
            """, unsafe_allow_html=True)
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            segs_seen = hist_df['segment'].value_counts()
            top_hist  = segs_seen.index[0] if len(segs_seen) else "—"
            n_single  = int((hist_df['source'] == 'single').sum())
            n_batch   = int((hist_df['source'] == 'batch').sum())

            st.markdown(f"""
            <div class="kpi-strip" style="margin-top:0;margin-bottom:1.4rem;">
                <div class="kpi-card">
                    <div class="kpi-label">Total Predictions</div>
                    <div class="kpi-value"><span class="accent">{len(hist_df)}</span></div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Single Predictions</div>
                    <div class="kpi-value"><span class="accent">{n_single}</span></div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Batch Rows</div>
                    <div class="kpi-value"><span class="accent">{n_batch}</span></div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Most Common</div>
                    <div class="kpi-value" style="font-size:0.92rem;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;">{top_hist}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Charts
            hc1, hc2 = st.columns(2, gap="medium")
            with hc1:
                fig_hpie = px.pie(hist_df, names='segment', color_discrete_sequence=COLORS, hole=0.48)
                fig_hpie.update_traces(
                    textposition='outside',
                    textfont=dict(family="Plus Jakarta Sans", size=11),
                    marker_line=dict(color="#FFFFFF", width=2)
                )
                fig_hpie = light_fig(fig_hpie, 290)
                fig_hpie.update_layout(**fig_title("Segment Frequency", "All predictions in session"))
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_hpie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with hc2:
                grp = hist_df.groupby(['segment','source']).size().reset_index(name='count')
                fig_hsrc = px.bar(grp, x='segment', y='count', color='source',
                                  color_discrete_map={'single':'#2D6A4F','batch':'#52B788'},
                                  barmode='group')
                fig_hsrc.update_traces(marker_line_width=0)
                fig_hsrc = light_fig(fig_hsrc, 290)
                fig_hsrc.update_layout(**fig_title("Single vs Batch by Segment"),
                                       xaxis_tickangle=-18, legend_title_text="Source")
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_hsrc, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if len(hist_df) > 1:
                fig_hsc = px.scatter(
                    hist_df, x='income', y='spending',
                    color='segment', symbol='source',
                    color_discrete_sequence=COLORS, opacity=0.8,
                    hover_data=['gender','age']
                )
                fig_hsc.update_traces(marker=dict(size=9, line=dict(width=0.8, color='rgba(255,255,255,0.7)')))
                fig_hsc = light_fig(fig_hsc, 300)
                fig_hsc.update_layout(**fig_title("All Predictions — Income vs Spending", "◆ = batch  ●  = single"))
                fig_hsc.update_xaxes(title="Annual Income (k$)")
                fig_hsc.update_yaxes(title="Spending Score (1-100)")
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_hsc, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Log table
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Fraunces',serif;font-size:0.95rem;font-weight:600;
                        color:#1C1B18;margin-bottom:0.8rem;">Prediction Log</div>
            """, unsafe_allow_html=True)

            log_container = """
            <div style="background:var(--surface);border:1px solid var(--border);
                        border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);">
            <div class="hist-row hist-header">
                <div>Time</div><div>Source</div><div>Gender</div>
                <div>Age</div><div>Income (k$)</div><div>Segment</div>
            </div>"""
            for rec in reversed(st.session_state.history[-120:]):
                seg_color = SEG_COLORS.get(rec['segment'], COLORS[0])
                src_icon  = "👤" if rec['source'] == 'single' else "📂"
                log_container += f"""
                <div class="hist-row">
                    <div style="color:#8C897E;font-size:0.77rem;">{rec['time']}</div>
                    <div>{src_icon} {rec['source']}</div>
                    <div>{rec['gender']}</div>
                    <div>{rec['age']}</div>
                    <div>{rec['income']}</div>
                    <div><span class="seg-pill">
                        <span style="width:7px;height:7px;border-radius:50%;
                            background:{seg_color};display:inline-block;flex-shrink:0;"></span>
                        {rec['segment']}
                    </span></div>
                </div>"""
            log_container += "</div>"
            st.markdown(log_container, unsafe_allow_html=True)

            # Download + clear
            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            hist_export = hist_df.rename(columns={
                'time':'Time','source':'Source','gender':'Gender',
                'age':'Age','income':'Annual Income (k$)',
                'spending':'Spending Score (1-100)','segment':'Segment'
            })
            st.download_button(
                "⬇  Download Full History as CSV",
                data=hist_export.to_csv(index=False),
                file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🗑  Clear All History"):
                st.session_state.history = []
                st.rerun()

except FileNotFoundError:
    st.markdown("""
    <div style="background:#FDECEA;border:1.5px solid #F5B7B1;
                border-radius:14px;padding:1.6rem 2rem;margin-top:1rem;">
        <div style="font-family:'Fraunces',serif;font-weight:700;font-size:1rem;
                    color:#C0392B;margin-bottom:.5rem;">⚠ Model files not found</div>
        <div style="font-size:.84rem;color:#6B4B49;line-height:1.6;">
            Ensure the trained pipeline, cluster labels, and feature names are saved inside
            <code style="background:rgba(0,0,0,0.06);padding:2px 8px;border-radius:5px;">models/</code>
            before launching the app.
        </div>
    </div>
    """, unsafe_allow_html=True)