import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ── Konfigurasi Halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Volatilitas Saham Energi LQ45",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    border-radius:16px; padding:28px 32px; margin-bottom:24px;
    border:1px solid rgba(255,255,255,0.08);
}
.hero h2 { color:#fff; font-size:1.7rem; margin:0 0 4px; }
.hero p  { color:rgba(255,255,255,0.6); margin:0; font-size:0.9rem; }
.badge {
    display:inline-block; background:rgba(99,102,241,0.2);
    color:#a5b4fc; border:1px solid rgba(99,102,241,0.35);
    border-radius:20px; padding:2px 11px; font-size:0.73rem;
    font-weight:600; margin:10px 4px 0 0;
}
.kcard {
    background:linear-gradient(145deg,#1a1d2e,#1f2235);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:20px; text-align:center;
}
.kcard .icon  { font-size:1.6rem; }
.kcard .val   { font-size:1.6rem; font-weight:700; color:#e2e8f0; margin:4px 0; }
.kcard .lbl   { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.kcard .sub   { font-size:0.78rem; color:#94a3b8; margin-top:3px; }
.kcard.blue   { border-top:3px solid #3b82f6; }
.kcard.green  { border-top:3px solid #10b981; }
.kcard.purple { border-top:3px solid #8b5cf6; }
.kcard.orange { border-top:3px solid #f59e0b; }
.sec { font-size:1rem; font-weight:600; color:#e2e8f0;
       margin:20px 0 12px; padding-left:10px;
       border-left:3px solid #6366f1; }
.ibox {
    background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.2);
    border-radius:10px; padding:14px 16px; font-size:0.82rem;
    color:#93c5fd; margin:10px 0;
}
.sbox {
    background:rgba(255,255,255,0.03); border-radius:10px;
    padding:13px; margin-bottom:10px;
    border:1px solid rgba(255,255,255,0.05);
}
.footer {
    text-align:center; padding:20px 0 8px;
    color:#334155; font-size:0.76rem;
    border-top:1px solid rgba(255,255,255,0.04);
    margin-top:28px;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ──────────────────────────────────────────────────────────────────
WARNA = {
    "ADRO":"#6366f1","ADMR":"#10b981","ITMG":"#f59e0b",
    "PTBA":"#ef4444","MEDC":"#8b5cf6","AKRA":"#06b6d4",
}
NAMA = {
    "ADRO":"Adaro Energy","ADMR":"Adaro Minerals",
    "ITMG":"Indo Tambangraya","PTBA":"Bukit Asam",
    "MEDC":"Medco Energi","AKRA":"AKR Corporindo",
}

def layout_dark(title="", height=350, margin_t=40, margin_b=30):
    """Helper layout dark — tanpa duplikasi margin."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.85)",
        font=dict(family="Inter", color="#94a3b8", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.08)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.08)", showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(255,255,255,0.08)"),
        margin=dict(t=margin_t, b=margin_b, l=50, r=20),
        height=height,
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=13, color="#e2e8f0"))
    return d

def hex_rgba(hex_color, alpha):
    """Konversi '#rrggbb' ke 'rgba(r,g,b,alpha)' — aman untuk Plotly 6.x"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Load Data ──────────────────────────────────────────────────────────────────
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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 18px">
        <div style="font-size:2.2rem">📈</div>
        <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0">Volatilitas Saham</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:3px">LQ45 Sektor Energi</div>
    </div>""", unsafe_allow_html=True)

    emiten_list = sorted(df_tek["Kode_Emiten"].unique().tolist()) if DATA_OK \
                  else list(NAMA.keys())

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("**⚙️ Filter Emiten**")
    emiten_sel = st.multiselect(
        "Emiten", emiten_list, default=emiten_list,
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

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("**📅 Rentang Tanggal**")
    rentang = st.date_input(
        "Rentang",
        value=(tgl_min, tgl_max),
        min_value=tgl_min,
        max_value=tgl_max,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.79rem;color:#94a3b8;line-height:1.9">
    🔹 <b style="color:#e2e8f0">Model:</b> LSTM 3 Layer<br>
    🔹 <b style="color:#e2e8f0">Lookback:</b> 20 hari<br>
    🔹 <b style="color:#e2e8f0">Horizon:</b> t+1 (besok)<br>
    🔹 <b style="color:#e2e8f0">Split:</b> 80% / 20%<br>
    🔹 <b style="color:#e2e8f0">Fitur X:</b> 12 teknikal<br>
    🔹 <b style="color:#e2e8f0">Target Y:</b> Volatilitas Return
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.79rem;color:#94a3b8;line-height:1.9">
    👩‍🎓 <b style="color:#e2e8f0">Ilda Nurida</b><br>
    Manajemen Teknologi<br>
    SIMT — ITS Surabaya<br>
    <span style="color:#6366f1;font-weight:600">2026</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Filter ─────────────────────────────────────────────────────────────────────
if DATA_OK and emiten_sel and len(rentang) == 2:
    mask = (
        df_tek["Kode_Emiten"].isin(emiten_sel) &
        (df_tek["Tanggal"].dt.date >= rentang[0]) &
        (df_tek["Tanggal"].dt.date <= rentang[1])
    )
    df_f = df_tek[mask].copy()
else:
    df_f = pd.DataFrame()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h2>📈 Dashboard Prediksi Volatilitas Return Saham</h2>
  <p>Emiten LQ45 Sektor Energi Indonesia &nbsp;•&nbsp; Model LSTM Baseline (Teknikal)</p>
  <div>
    <span class="badge">🏭 6 Emiten</span>
    <span class="badge">⏱ COVID 2020 – 2026</span>
    <span class="badge">🤖 LSTM t+1</span>
    <span class="badge">📊 Data Teknikal</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Overview",
    "📉  Analisis Harga",
    "🤖  Hasil Prediksi",
    "📋  Evaluasi Model",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    if DATA_OK and len(df_f) > 0:
        # KPI
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kcard blue">
              <div class="icon">📰</div>
              <div class="val">{len(df_f):,}</div>
              <div class="lbl">Total Data</div>
              <div class="sub">hari trading</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kcard green">
              <div class="icon">🏭</div>
              <div class="val">{df_f["Kode_Emiten"].nunique()}</div>
              <div class="lbl">Emiten Aktif</div>
              <div class="sub">LQ45 Energi</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            vm = df_f["Volatilitas"].mean()
            st.markdown(f"""
            <div class="kcard purple">
              <div class="icon">📊</div>
              <div class="val">{vm:.4f}</div>
              <div class="lbl">Volatilitas Rata2</div>
              <div class="sub">std dev log return</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            n_thn = (df_f["Tanggal"].max()-df_f["Tanggal"].min()).days // 365
            st.markdown(f"""
            <div class="kcard orange">
              <div class="icon">📅</div>
              <div class="val">{n_thn}+</div>
              <div class="lbl">Tahun Data</div>
              <div class="sub">COVID 2020 – kini</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Area chart tren volatilitas bulanan
        st.markdown('<div class="sec">📊 Tren Volatilitas Bulanan</div>', unsafe_allow_html=True)
        df_m = df_f.copy()
        df_m["Bulan"] = df_f["Tanggal"].dt.to_period("M").dt.to_timestamp()
        df_m = df_m.groupby(["Bulan","Kode_Emiten"])["Volatilitas"].mean().reset_index()

        fig_area = go.Figure()
        for kode in emiten_sel:
            dk   = df_m[df_m["Kode_Emiten"]==kode]
            wn   = WARNA.get(kode, "#888")
            fig_area.add_trace(go.Scatter(
                x=dk["Bulan"], y=dk["Volatilitas"],
                name=kode, mode="lines",
                line=dict(color=wn, width=2),
                fill="tozeroy",
                fillcolor=hex_rgba(wn, 0.07),
                hovertemplate=f"<b>{kode}</b><br>%{{x|%b %Y}}<br>Vol: %{{y:.4f}}<extra></extra>",
            ))
        fig_area.update_layout(**layout_dark("Rata-rata Volatilitas per Bulan", height=310))
        st.plotly_chart(fig_area, width="stretch")

        col_l, col_r = st.columns([1.3, 0.7])
        with col_l:
            st.markdown('<div class="sec">📌 Statistik per Emiten</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="sec">🥧 Porsi Data</div>', unsafe_allow_html=True)
            vc = df_f["Kode_Emiten"].value_counts().reset_index()
            vc.columns = ["Emiten","Jumlah"]
            fig_d = px.pie(vc, values="Jumlah", names="Emiten",
                           color="Emiten", color_discrete_map=WARNA, hole=0.52)
            fig_d.update_traces(textposition="outside", textinfo="label+percent")
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_d, width="stretch")

        # Heatmap korelasi
        st.markdown('<div class="sec">🔥 Korelasi Volatilitas Antar Emiten</div>', unsafe_allow_html=True)
        pv = df_f.pivot_table(index="Tanggal", columns="Kode_Emiten",
                               values="Volatilitas").dropna()
        if len(pv.columns) > 1:
            corr = pv.corr().round(3)
            fig_h = px.imshow(
                corr, text_auto=True,
                color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e293b"],[1,"#10b981"]],
                zmin=-1, zmax=1,
            )
            fig_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,29,46,0.85)",
                font=dict(family="Inter", color="#94a3b8"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
            )
            st.plotly_chart(fig_h, width="stretch")
    else:
        st.markdown("""
        <div class="ibox">
        📂 <b>File data belum tersedia.</b><br><br>
        File yang dibutuhkan:<br>
        • <code>saham_lq45_energi_TEKNIKAL.csv</code><br>
        • <code>hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv</code><br>
        • <code>hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv</code>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS HARGA
# ════════════════════════════════════════════════════════════════════
with tab2:
    if DATA_OK and len(df_f) > 0:
        kode2 = st.selectbox(
            "Pilih Emiten",
            [e for e in emiten_sel if e in df_f["Kode_Emiten"].unique()],
            key="tab2",
        )
        df_e2 = df_f[df_f["Kode_Emiten"]==kode2].sort_values("Tanggal")
        wn2   = WARNA.get(kode2, "#6366f1")

        # 3 panel
        fig3 = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Harga Close (Rp)","Log Return Harian",
                            "Volatilitas Rolling 20 Hari"],
            vertical_spacing=0.07,
            row_heights=[0.5, 0.25, 0.25],
        )
        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"], y=df_e2["Close"],
            name="Close", line=dict(color=wn2, width=1.5),
            fill="tozeroy", fillcolor=hex_rgba(wn2, 0.07),
            hovertemplate="Rp %{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        colors_bar = [wn2 if v >= 0 else "#ef4444"
                      for v in df_e2["Log_Return"].fillna(0)]
        fig3.add_trace(go.Bar(
            x=df_e2["Tanggal"], y=df_e2["Log_Return"],
            name="Log Return", marker_color=colors_bar, opacity=0.8,
        ), row=2, col=1)

        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"], y=df_e2["Volatilitas"],
            name="Volatilitas", line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy", fillcolor=hex_rgba("#f59e0b", 0.08),
        ), row=3, col=1)
        fig3.add_hline(
            y=df_e2["Volatilitas"].mean(), row=3, col=1,
            line_dash="dot", line_color="#64748b",
            annotation_text=f"Mean={df_e2['Volatilitas'].mean():.4f}",
            annotation_font_color="#94a3b8",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.85)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(t=40, b=30, l=50, r=20),
            height=580,
            showlegend=False,
            title=dict(text=f"{kode2} — {NAMA.get(kode2,'')}",
                       font=dict(size=14, color="#e2e8f0")),
        )
        fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig3, width="stretch")

        # Indikator teknikal
        st.markdown('<div class="sec">📐 Indikator Teknikal</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["RSI"],
                line=dict(color=wn2, width=1.5),
                fill="tozeroy", fillcolor=hex_rgba(wn2, 0.06), name="RSI",
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                               annotation_text="Overbought",
                               annotation_font_color="#ef4444")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981",
                               annotation_text="Oversold",
                               annotation_font_color="#10b981")
            fig_rsi.update_layout(**layout_dark("RSI (14 Hari)", height=270),
                                   yaxis_range=[0,100])
            st.plotly_chart(fig_rsi, width="stretch")

        with cb:
            fig_st = go.Figure()
            fig_st.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["Stoch_K"],
                line=dict(color="#6366f1", width=1.5), name="%K",
            ))
            fig_st.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2["Stoch_D"],
                line=dict(color="#f59e0b", width=1.5, dash="dot"), name="%D",
            ))
            fig_st.add_hline(y=80, line_dash="dash", line_color="#ef4444",
                               annotation_text="Overbought",
                               annotation_font_color="#ef4444")
            fig_st.add_hline(y=20, line_dash="dash", line_color="#10b981",
                               annotation_text="Oversold",
                               annotation_font_color="#10b981")
            fig_st.update_layout(**layout_dark("Stochastic Oscillator", height=270))
            st.plotly_chart(fig_st, width="stretch")

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=df_e2["Tanggal"], y=df_e2["Close"],
            line=dict(color="#475569", width=1), name="Close", opacity=0.5,
        ))
        for ma, wma in [("MA_5","#10b981"),("MA_10","#f59e0b"),("MA_20","#8b5cf6")]:
            fig_ma.add_trace(go.Scatter(
                x=df_e2["Tanggal"], y=df_e2[ma],
                line=dict(color=wma, width=1.5), name=ma,
            ))
        fig_ma.update_layout(**layout_dark("Moving Average (5/10/20 Hari)", height=270))
        st.plotly_chart(fig_ma, width="stretch")
    else:
        st.info("📂 Data belum tersedia.")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — HASIL PREDIKSI
# ════════════════════════════════════════════════════════════════════
with tab3:
    if len(df_pred) > 0:
        em_pred = [e for e in emiten_sel if e in df_pred["Kode_Emiten"].unique()]
        if em_pred:
            kode3 = st.selectbox("Pilih Emiten", em_pred, key="tab3")
            dp3   = df_pred[df_pred["Kode_Emiten"]==kode3].sort_values("Tanggal")
            col_p = next((c for c in dp3.columns if "Pred" in c), None)
            wn3   = WARNA.get(kode3, "#6366f1")

            if col_p:
                # Grafik utama
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=dp3["Tanggal"], y=dp3["Volatilitas_Aktual"],
                    name="Aktual", line=dict(color=wn3, width=2),
                ))
                fig_p.add_trace(go.Scatter(
                    x=dp3["Tanggal"], y=dp3[col_p],
                    name="Prediksi LSTM (t+1)",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ))
                fig_p.update_layout(
                    **layout_dark(f"Prediksi vs Aktual — {kode3} (t+1)", height=360),
                    legend=dict(orientation="h", y=1.08,
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_p, width="stretch")

                # KPI
                err  = (dp3["Volatilitas_Aktual"] - dp3[col_p]).abs()
                rmse = float(np.sqrt((err**2).mean()))
                mae  = float(err.mean())
                mape = float((err / dp3["Volatilitas_Aktual"].replace(0,np.nan)).mean() * 100)

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f"""
                    <div class="kcard green">
                      <div class="icon">🎯</div>
                      <div class="val">{rmse:.5f}</div>
                      <div class="lbl">RMSE</div>
                    </div>""", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""
                    <div class="kcard blue">
                      <div class="icon">📏</div>
                      <div class="val">{mae:.5f}</div>
                      <div class="lbl">MAE</div>
                    </div>""", unsafe_allow_html=True)
                with k3:
                    st.markdown(f"""
                    <div class="kcard purple">
                      <div class="icon">📐</div>
                      <div class="val">{mape:.2f}%</div>
                      <div class="lbl">MAPE</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                cl, cr = st.columns(2)
                with cl:
                    fig_sc = px.scatter(
                        dp3, x="Volatilitas_Aktual", y=col_p,
                        opacity=0.5, color_discrete_sequence=[wn3],
                        title="Scatter: Aktual vs Prediksi",
                        labels={"Volatilitas_Aktual":"Aktual", col_p:"Prediksi"},
                    )
                    mn = min(dp3["Volatilitas_Aktual"].min(), dp3[col_p].min())
                    mx = max(dp3["Volatilitas_Aktual"].max(), dp3[col_p].max())
                    fig_sc.add_trace(go.Scatter(
                        x=[mn,mx], y=[mn,mx],
                        mode="lines", name="Sempurna",
                        line=dict(color="#475569", dash="dash", width=1),
                    ))
                    fig_sc.update_layout(**layout_dark(height=320))
                    st.plotly_chart(fig_sc, width="stretch")

                with cr:
                    err_dist = (dp3["Volatilitas_Aktual"] - dp3[col_p]).rename("Error")
                    fig_e = px.histogram(
                        err_dist, nbins=30,
                        color_discrete_sequence=[wn3],
                        title="Distribusi Error",
                    )
                    fig_e.add_vline(x=0, line_color="#ef4444", line_dash="dash")
                    fig_e.update_layout(**layout_dark(height=320))
                    st.plotly_chart(fig_e, width="stretch")

                st.markdown('<div class="sec">📋 20 Hari Terbaru</div>', unsafe_allow_html=True)
                ds = dp3.tail(20)[["Tanggal","Volatilitas_Aktual",col_p,
                                    "Error_Abs","Error_Pct"]].copy()
                ds.columns = ["Tanggal","Aktual","Prediksi t+1","Error Abs","Error %"]
                st.dataframe(ds.round(6), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="ibox">
        📂 File hasil prediksi belum tersedia.<br>
        Jalankan notebook <code>PREDIKSI_VOLATILITAS_LSTM_TEKNIKAL_v2.ipynb</code>.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════
with tab4:
    if len(df_eval) > 0:
        st.markdown('<div class="sec">📋 Tabel Evaluasi LSTM</div>', unsafe_allow_html=True)
        st.dataframe(df_eval.round(6), use_container_width=True, hide_index=True)

        e1, e2, e3 = st.columns(3)
        for col, metrik, ikon in [(e1,"RMSE","🎯"),(e2,"MAE","📏"),(e3,"MAPE","📐")]:
            if metrik in df_eval.columns:
                with col:
                    fig_b = px.bar(
                        df_eval.sort_values(metrik),
                        x="Emiten", y=metrik,
                        color="Emiten", color_discrete_map=WARNA,
                        text_auto=".4f",
                        title=f"{ikon} {metrik}",
                    )
                    fig_b.update_layout(
                        **layout_dark(height=270),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_b, width="stretch")

        if all(m in df_eval.columns for m in ["RMSE","MAE","MAPE"]):
            st.markdown('<div class="sec">🕸️ Radar Perbandingan</div>', unsafe_allow_html=True)
            dn = df_eval.copy()
            for m in ["RMSE","MAE","MAPE"]:
                mx = dn[m].max()
                dn[m] = dn[m]/mx if mx > 0 else 0
            fig_r = go.Figure()
            cats = ["RMSE","MAE","MAPE"]
            for _, row in dn.iterrows():
                vals = [row[m] for m in cats] + [row["RMSE"]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats+[cats[0]],
                    fill="toself", name=row["Emiten"],
                    line_color=WARNA.get(row["Emiten"],"#888"), opacity=0.7,
                ))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                    gridcolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(26,29,46,0.6)",
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
                font=dict(color="#94a3b8", family="Inter"),
                height=400,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Normalisasi Error (lebih kecil = lebih baik)",
                           font=dict(color="#e2e8f0", size=13)),
                margin=dict(t=50, b=30, l=30, r=30),
            )
            st.plotly_chart(fig_r, width="stretch")
    else:
        st.markdown("""
        <div class="ibox">
        📂 File evaluasi belum tersedia.<br>
        Jalankan notebook LSTM terlebih dahulu.
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  📈 Dashboard Prediksi Volatilitas Return Saham LQ45 Sektor Energi &nbsp;|&nbsp;
  LSTM Baseline &nbsp;|&nbsp; Ilda Nurida &nbsp;•&nbsp; SIMT ITS Surabaya &nbsp;•&nbsp; 2026
</div>
""", unsafe_allow_html=True)
