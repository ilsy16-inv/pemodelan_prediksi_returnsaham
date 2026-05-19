import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Volatilitas Saham Energi LQ45",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS CUSTOM
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.6);
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 12px;
}

.kpi-card {
    background: linear-gradient(145deg, #1a1d2e, #1f2235);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-icon  { font-size: 1.8rem; margin-bottom: 6px; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #e2e8f0; margin: 4px 0; }
.kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

.kpi-green { border-top: 3px solid #10b981; }
.kpi-blue  { border-top: 3px solid #3b82f6; }
.kpi-purple{ border-top: 3px solid #8b5cf6; }
.kpi-orange{ border-top: 3px solid #f59e0b; }

.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 24px 0 14px 0;
    padding-left: 12px;
    border-left: 3px solid #6366f1;
}

.emiten-chip {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px;
    cursor: pointer;
}

.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.83rem;
    color: #93c5fd;
    margin: 10px 0;
}

.sidebar-section {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════════
WARNA = {
    "ADRO":"#6366f1","ADMR":"#10b981","ITMG":"#f59e0b",
    "PTBA":"#ef4444","MEDC":"#8b5cf6","AKRA":"#06b6d4",
}
EMITEN_NAMA = {
    "ADRO":"Adaro Energy","ADMR":"Adaro Minerals",
    "ITMG":"Indo Tambangraya","PTBA":"Bukit Asam",
    "MEDC":"Medco Energi","AKRA":"AKR Corporindo",
}
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        margin=dict(t=40, b=40, l=50, r=20),
    )
)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_teknikal():
    for p in ["saham_lq45_energi_TEKNIKAL.csv",
              "saham_lq45_energi_covid2020_TEKNIKAL.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_prediksi():
    for p in ["hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv",
              "hasil_prediksi_lstm_teknikal_t1.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_evaluasi():
    for p in ["hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv",
              "evaluasi_lstm_teknikal_t1.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()

df_tek  = load_teknikal()
df_pred = load_prediksi()
df_eval = load_evaluasi()
DATA_OK = len(df_tek) > 0

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px">
        <div style="font-size:2.5rem">📈</div>
        <div style="font-size:1rem; font-weight:700; color:#e2e8f0">Volatilitas Saham</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px">LQ45 Sektor Energi</div>
    </div>
    """, unsafe_allow_html=True)

    emiten_list = sorted(df_tek["Kode_Emiten"].unique().tolist()) if DATA_OK \
                  else list(EMITEN_NAMA.keys())

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**⚙️ Filter Emiten**")
    emiten_sel = st.multiselect(
        "Pilih Emiten", emiten_list, default=emiten_list,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if DATA_OK:
        tgl_min = df_tek["Tanggal"].min().date()
        tgl_max = df_tek["Tanggal"].max().date()
    else:
        import datetime
        tgl_min = datetime.date(2020, 3, 1)
        tgl_max = datetime.date(2026, 5, 19)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📅 Rentang Tanggal**")
    rentang = st.date_input("Rentang Tanggal", value=(tgl_min, tgl_max),
                            min_value=tgl_min, max_value=tgl_max,
                            label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🤖 Konfigurasi Model**")
    st.markdown("""
    <div style="font-size:0.8rem; color:#94a3b8; line-height:1.8">
    🔹 Model: LSTM 3 Layer<br>
    🔹 Lookback: 20 hari<br>
    🔹 Horizon: t+1 (besok)<br>
    🔹 Split: 80% / 20%<br>
    🔹 Fitur: 12 teknikal<br>
    🔹 Target: Volatilitas Return
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#94a3b8; line-height:1.9">
    👩‍🎓 <b style="color:#e2e8f0">Ilda Nurida</b><br>
    Manajemen Teknologi<br>
    SIMT — ITS Surabaya<br>
    <span style="color:#6366f1">2026</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FILTER
# ══════════════════════════════════════════════════════════════════
if DATA_OK and emiten_sel and len(rentang) == 2:
    mask = (df_tek["Kode_Emiten"].isin(emiten_sel) &
            (df_tek["Tanggal"].dt.date >= rentang[0]) &
            (df_tek["Tanggal"].dt.date <= rentang[1]))
    df_f = df_tek[mask].copy()
else:
    df_f = pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📈 Dashboard Prediksi Volatilitas Return Saham</div>
    <div class="hero-subtitle">Emiten LQ45 Sektor Energi Indonesia • Model LSTM Berbasis Data Teknikal</div>
    <div>
        <span class="hero-badge">🏭 6 Emiten</span>
        <span class="hero-badge">⏱ COVID 2020–2026</span>
        <span class="hero-badge">🤖 LSTM t+1</span>
        <span class="hero-badge">📊 Teknikal + Sentimen</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Overview",
    "📉  Analisis Harga",
    "🤖  Hasil Prediksi",
    "📋  Evaluasi Model",
])

# ────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────────
with tab1:
    if DATA_OK and len(df_f) > 0:
        # KPI Cards
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-icon">📰</div>
                <div class="kpi-value">{len(df_f):,}</div>
                <div class="kpi-label">Total Data</div>
                <div class="kpi-sub">hari trading</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card kpi-green">
                <div class="kpi-icon">🏭</div>
                <div class="kpi-value">{df_f["Kode_Emiten"].nunique()}</div>
                <div class="kpi-label">Emiten Aktif</div>
                <div class="kpi-sub">LQ45 Energi</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            vol_m = df_f["Volatilitas"].mean()
            st.markdown(f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{vol_m:.4f}</div>
                <div class="kpi-label">Volatilitas Rata2</div>
                <div class="kpi-sub">std dev log return</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            n_tahun = (df_f["Tanggal"].max() - df_f["Tanggal"].min()).days // 365
            st.markdown(f"""
            <div class="kpi-card kpi-orange">
                <div class="kpi-icon">📅</div>
                <div class="kpi-value">{n_tahun}+</div>
                <div class="kpi-label">Tahun Data</div>
                <div class="kpi-sub">COVID 2020 – kini</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Volatilitas per emiten — area chart
        st.markdown('<div class="section-header">📊 Tren Volatilitas per Emiten</div>', unsafe_allow_html=True)

        df_monthly = df_f.copy()
        df_monthly["Bulan"] = df_f["Tanggal"].dt.to_period("M").dt.to_timestamp()
        df_monthly = df_monthly.groupby(["Bulan","Kode_Emiten"])["Volatilitas"].mean().reset_index()

        fig_area = go.Figure()
        for kode in emiten_sel:
            df_k = df_monthly[df_monthly["Kode_Emiten"]==kode]
            fig_area.add_trace(go.Scatter(
                x=df_k["Bulan"], y=df_k["Volatilitas"],
                name=kode, mode="lines",
                line=dict(color=WARNA.get(kode,"#888"), width=2),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
                hovertemplate=f"<b>{kode}</b><br>%{{x|%b %Y}}<br>Vol: %{{y:.4f}}<extra></extra>",
            ))
        fig_area.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                               title=dict(text="Volatilitas Bulanan (Rata-rata)", font=dict(size=13)))
        st.plotly_chart(fig_area, use_container_width=True)

        # 2 kolom bawah
        col_l, col_r = st.columns([1.2, 0.8])
        with col_l:
            st.markdown('<div class="section-header">📌 Statistik per Emiten</div>', unsafe_allow_html=True)
            tbl = df_f.groupby("Kode_Emiten").agg(
                Hari=("Tanggal","count"),
                Close_Rata=("Close","mean"),
                Vol_Rata=("Volatilitas","mean"),
                Vol_Max=("Volatilitas","max"),
                RSI_Rata=("RSI","mean"),
            ).round(4).reset_index()
            tbl.columns = ["Emiten","Hari","Close Rata2","Vol Rata2","Vol Max","RSI Rata2"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown('<div class="section-header">🥧 Porsi Data</div>', unsafe_allow_html=True)
            vc = df_f["Kode_Emiten"].value_counts().reset_index()
            vc.columns = ["Emiten","Jumlah"]
            fig_donut = px.pie(vc, values="Jumlah", names="Emiten",
                               color="Emiten", color_discrete_map=WARNA,
                               hole=0.55)
            fig_donut.update_traces(textposition="outside", textinfo="label+percent")
            fig_donut.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                                    showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_donut, use_container_width=True)

        # Heatmap korelasi volatilitas
        st.markdown('<div class="section-header">🔥 Korelasi Volatilitas Antar Emiten</div>', unsafe_allow_html=True)
        pv = df_f.pivot_table(index="Tanggal", columns="Kode_Emiten",
                               values="Volatilitas").dropna()
        if len(pv.columns) > 1:
            corr = pv.corr().round(3)
            fig_heat = px.imshow(corr, text_auto=True,
                                  color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e293b"],[1,"#10b981"]],
                                  zmin=-1, zmax=1)
            fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"], height=340,
                                    coloraxis_colorbar=dict(title="Korelasi"))
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
        📂 <b>File data belum tersedia.</b><br><br>
        File yang dibutuhkan:<br>
        • <code>saham_lq45_energi_TEKNIKAL.csv</code> — dari notebook scraping<br>
        • <code>hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv</code> — dari notebook LSTM<br>
        • <code>hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv</code> — dari notebook LSTM
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB 2 — ANALISIS HARGA
# ────────────────────────────────────────────────────────────────
with tab2:
    if DATA_OK and len(df_f) > 0:
        kode2 = st.selectbox("Pilih Emiten",
                              [e for e in emiten_sel if e in df_f["Kode_Emiten"].unique()],
                              key="tab2")
        df_e2 = df_f[df_f["Kode_Emiten"]==kode2].sort_values("Tanggal")
        warna2 = WARNA.get(kode2,"#6366f1")

        # Grafik 3 panel
        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              subplot_titles=["Harga Close (Rp)",
                                              "Log Return Harian",
                                              f"Volatilitas Rolling 20 Hari"],
                              vertical_spacing=0.07,
                              row_heights=[0.5,0.25,0.25])

        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"], y=df_e2["Close"],
            name="Close", line=dict(color=warna2, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(warna2.lstrip('#')[0:2],16)},{int(warna2.lstrip('#')[2:4],16)},{int(warna2.lstrip('#')[4:6],16)},0.08)",
            hovertemplate="Rp %{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        colors_bar = [warna2 if v >= 0 else "#ef4444"
                      for v in df_e2["Log_Return"].fillna(0)]
        fig3.add_trace(go.Bar(
            x=df_e2["Tanggal"], y=df_e2["Log_Return"],
            name="Log Return", marker_color=colors_bar, opacity=0.8,
            hovertemplate="%{y:.4f}<extra></extra>",
        ), row=2, col=1)

        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"], y=df_e2["Volatilitas"],
            name="Volatilitas", line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.09)",
            hovertemplate="%{y:.5f}<extra></extra>",
        ), row=3, col=1)
        fig3.add_hline(y=df_e2["Volatilitas"].mean(), row=3, col=1,
                       line_dash="dot", line_color="#64748b",
                       annotation_text=f"Mean={df_e2['Volatilitas'].mean():.4f}",
                       annotation_font_color="#94a3b8")

        fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=600,
                            showlegend=False,
                            title=dict(text=f"{kode2} — {EMITEN_NAMA.get(kode2,'')}", font=dict(size=14)))
        st.plotly_chart(fig3, use_container_width=True)

        # Indikator teknikal
        st.markdown('<div class="section-header">📐 Indikator Teknikal</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["RSI"],
                line=dict(color=warna2, width=1.5), name="RSI",
                fill="tozeroy", fillcolor=f"rgba({int(warna2.lstrip('#')[0:2],16)},{int(warna2.lstrip('#')[2:4],16)},{int(warna2.lstrip('#')[4:6],16)},0.08)"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                               annotation_text="Overbought", annotation_font_color="#ef4444")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981",
                               annotation_text="Oversold", annotation_font_color="#10b981")
            fig_rsi.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                                   title="RSI (14 Hari)", yaxis_range=[0,100])
            st.plotly_chart(fig_rsi, use_container_width=True)

        with c2:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["Stoch_K"],
                line=dict(color="#6366f1", width=1.5), name="%K"))
            fig_stoch.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["Stoch_D"],
                line=dict(color="#f59e0b", width=1.5, dash="dot"), name="%D"))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="#ef4444",
                                  annotation_text="Overbought", annotation_font_color="#ef4444")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="#10b981",
                                  annotation_text="Oversold", annotation_font_color="#10b981")
            fig_stoch.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                                     title="Stochastic Oscillator (%K/%D)")
            st.plotly_chart(fig_stoch, use_container_width=True)

        # MA
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df_e2["Tanggal"], y=df_e2["Close"],
                                     line=dict(color="#475569",width=1), name="Close",opacity=0.5))
        for ma,warna_ma in [("MA_5","#10b981"),("MA_10","#f59e0b"),("MA_20","#8b5cf6")]:
            fig_ma.add_trace(go.Scatter(x=df_e2["Tanggal"], y=df_e2[ma],
                                         line=dict(color=warna_ma,width=1.5), name=ma))
        fig_ma.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                               title="Moving Average (5/10/20 Hari)")
        st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.info("📂 Data belum tersedia.")

# ────────────────────────────────────────────────────────────────
# TAB 3 — HASIL PREDIKSI
# ────────────────────────────────────────────────────────────────
with tab3:
    if len(df_pred) > 0:
        emiten_pred = [e for e in emiten_sel if e in df_pred["Kode_Emiten"].unique()]
        if emiten_pred:
            kode3 = st.selectbox("Pilih Emiten", emiten_pred, key="tab3")
            df_p3 = df_pred[df_pred["Kode_Emiten"]==kode3].sort_values("Tanggal")
            col_pred = next((c for c in df_p3.columns if "Pred" in c), None)
            warna3   = WARNA.get(kode3,"#6366f1")

            if col_pred and len(df_p3) > 0:
                # Grafik utama
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df_p3["Tanggal"], y=df_p3["Volatilitas_Aktual"],
                    name="Aktual", line=dict(color=warna3, width=2),
                    hovertemplate="Aktual: %{y:.5f}<extra></extra>"))
                fig_pred.add_trace(go.Scatter(
                    x=df_p3["Tanggal"], y=df_p3[col_pred],
                    name="Prediksi LSTM (t+1)",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="Prediksi: %{y:.5f}<extra></extra>"))
                fig_pred.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                                        title=dict(text=f"Prediksi vs Aktual — {kode3} (t+1)", font=dict(size=14)),
                                        legend=dict(orientation="h",y=1.05))
                st.plotly_chart(fig_pred, use_container_width=True)

                # KPI prediksi
                err = (df_p3["Volatilitas_Aktual"] - df_p3[col_pred]).abs()
                rmse = np.sqrt((err**2).mean())
                mae  = err.mean()
                mape = (err / df_p3["Volatilitas_Aktual"].replace(0,np.nan)).mean() * 100

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="kpi-card kpi-green">
                        <div class="kpi-icon">🎯</div>
                        <div class="kpi-value">{rmse:.5f}</div>
                        <div class="kpi-label">RMSE</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="kpi-card kpi-blue">
                        <div class="kpi-icon">📏</div>
                        <div class="kpi-value">{mae:.5f}</div>
                        <div class="kpi-label">MAE</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="kpi-card kpi-purple">
                        <div class="kpi-icon">📐</div>
                        <div class="kpi-value">{mape:.2f}%</div>
                        <div class="kpi-label">MAPE</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                cl, cr = st.columns(2)
                with cl:
                    fig_sc = px.scatter(df_p3, x="Volatilitas_Aktual", y=col_pred,
                                         opacity=0.5, color_discrete_sequence=[warna3],
                                         title="Scatter: Aktual vs Prediksi",
                                         labels={"Volatilitas_Aktual":"Aktual",col_pred:"Prediksi"})
                    mn = min(df_p3["Volatilitas_Aktual"].min(), df_p3[col_pred].min())
                    mx = max(df_p3["Volatilitas_Aktual"].max(), df_p3[col_pred].max())
                    fig_sc.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],
                                                 mode="lines",name="Sempurna",
                                                 line=dict(color="#475569",dash="dash",width=1)))
                    fig_sc.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
                    st.plotly_chart(fig_sc, use_container_width=True)

                with cr:
                    error_dist = df_p3["Volatilitas_Aktual"] - df_p3[col_pred]
                    fig_err = px.histogram(error_dist.rename("Error"), nbins=30,
                                            color_discrete_sequence=[warna3],
                                            title="Distribusi Error Prediksi")
                    fig_err.add_vline(x=0, line_color="#ef4444", line_dash="dash")
                    fig_err.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
                    st.plotly_chart(fig_err, use_container_width=True)

                # Tabel terbaru
                st.markdown('<div class="section-header">📋 20 Hari Prediksi Terbaru</div>', unsafe_allow_html=True)
                df_show = df_p3.tail(20)[["Tanggal","Volatilitas_Aktual",col_pred,"Error_Abs","Error_Pct"]].copy()
                df_show.columns = ["Tanggal","Aktual","Prediksi t+1","Error Abs","Error %"]
                df_show = df_show.round(6)
                st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="info-box">
        📂 File hasil prediksi belum tersedia.<br>
        Jalankan notebook <code>PREDIKSI_VOLATILITAS_LSTM_TEKNIKAL_v2.ipynb</code> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB 4 — EVALUASI MODEL
# ────────────────────────────────────────────────────────────────
with tab4:
    if len(df_eval) > 0:
        st.markdown('<div class="section-header">📋 Tabel Evaluasi LSTM</div>', unsafe_allow_html=True)
        st.dataframe(df_eval.round(6), use_container_width=True, hide_index=True)

        c1,c2,c3 = st.columns(3)
        for col, metrik, warna_m, ikon in [
            (c1,"RMSE","#6366f1","🎯"),
            (c2,"MAE","#10b981","📏"),
            (c3,"MAPE","#f59e0b","📐"),
        ]:
            if metrik in df_eval.columns:
                with col:
                    v = df_eval[metrik].mean()
                    suffix = "%" if metrik=="MAPE" else ""
                    fig_b = px.bar(
                        df_eval.sort_values(metrik),
                        x="Emiten", y=metrik,
                        color="Emiten", color_discrete_map=WARNA,
                        text_auto=".4f", title=f"{ikon} {metrik}")
                    fig_b.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                                         showlegend=False, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:-10px">
                    Rata-rata: <b style="color:#e2e8f0">{v:.5f}{suffix}</b>
                    </div>""", unsafe_allow_html=True)

        # Radar chart
        if all(m in df_eval.columns for m in ["RMSE","MAE","MAPE"]):
            st.markdown('<div class="section-header">🕸️ Radar Perbandingan Emiten</div>', unsafe_allow_html=True)
            df_n = df_eval.copy()
            for m in ["RMSE","MAE","MAPE"]:
                mx = df_n[m].max()
                df_n[m] = df_n[m]/mx if mx>0 else 0

            fig_r = go.Figure()
            cats = ["RMSE","MAE","MAPE"]
            for _,row in df_n.iterrows():
                vals = [row[m] for m in cats]+[row["RMSE"]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats+[cats[0]],
                    fill="toself", name=row["Emiten"],
                    line_color=WARNA.get(row["Emiten"],"#888"), opacity=0.7))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                    gridcolor="rgba(255,255,255,0.1)",
                                    linecolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(26,29,46,0.5)",
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                     linecolor="rgba(255,255,255,0.1)"),
                ),
                font=dict(color="#94a3b8", family="Inter"),
                height=420,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Normalisasi Metrik Error (lebih kecil = lebih baik)",
                           font=dict(color="#e2e8f0", size=13)),
            )
            st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
        📂 File evaluasi belum tersedia.<br>
        Jalankan notebook LSTM terlebih dahulu untuk menghasilkan file evaluasi.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    text-align:center;
    padding: 24px 0 12px;
    color: #334155;
    font-size: 0.78rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 32px;
">
    📈 Dashboard Prediksi Volatilitas Return Saham LQ45 Sektor Energi &nbsp;|&nbsp;
    LSTM Baseline (Teknikal) &nbsp;|&nbsp;
    Ilda Nurida &nbsp;•&nbsp; SIMT ITS Surabaya &nbsp;•&nbsp; 2026
</div>
""", unsafe_allow_html=True)
