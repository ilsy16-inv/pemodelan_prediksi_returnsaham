import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import os, warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Volatilitas Saham LQ45 Energi",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body { font-family:'Inter',sans-serif !important; }
.block-container { padding-top:1.2rem !important; }
.hero {
    background:linear-gradient(135deg,#0d1117 0%,#161b27 40%,#1a1033 100%);
    border:1px solid rgba(99,102,241,0.25); border-radius:18px;
    padding:28px 32px; margin-bottom:22px;
    box-shadow:0 0 40px rgba(99,102,241,0.12);
}
.hero-title { color:#f1f5f9; font-size:1.75rem; font-weight:800; margin:0 0 5px; }
.hero-sub   { color:#64748b; font-size:0.88rem; margin:0; }
.badge {
    display:inline-block; background:rgba(99,102,241,0.15);
    border:1px solid rgba(99,102,241,0.3); color:#a5b4fc;
    border-radius:20px; padding:2px 12px; font-size:0.71rem;
    font-weight:600; margin:10px 3px 0 0; letter-spacing:.3px;
}
.kcard {
    background:linear-gradient(145deg,#161b27,#1c2132);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:18px 16px 14px; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.25); transition:.2s;
}
.kcard:hover { transform:translateY(-2px); }
.kcard .ico { font-size:1.5rem; margin-bottom:6px; }
.kcard .val { font-size:1.55rem; font-weight:700; color:#f1f5f9; margin:3px 0; }
.kcard .lbl { font-size:0.67rem; color:#475569; text-transform:uppercase; letter-spacing:1.2px; }
.kcard .sub { font-size:0.76rem; color:#94a3b8; margin-top:3px; }
.kcard.c1 { border-top:3px solid #3b82f6; }
.kcard.c2 { border-top:3px solid #10b981; }
.kcard.c3 { border-top:3px solid #8b5cf6; }
.kcard.c4 { border-top:3px solid #f59e0b; }
.kcard.c5 { border-top:3px solid #ef4444; }
.sec {
    font-size:.95rem; font-weight:600; color:#e2e8f0;
    margin:18px 0 10px; padding-left:10px;
    border-left:3px solid #6366f1;
}
.live-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
    color:#34d399; border-radius:20px; padding:3px 12px;
    font-size:0.73rem; font-weight:600;
}
.live-dot {
    width:7px; height:7px; border-radius:50%;
    background:#10b981; animation:pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:.3; }
}
.ibox {
    background:rgba(59,130,246,0.07);
    border:1px solid rgba(59,130,246,0.18);
    border-radius:10px; padding:14px 16px;
    color:#93c5fd; font-size:.82rem; margin:10px 0;
}
.sbox {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; padding:12px; margin-bottom:10px;
}
.footer {
    text-align:center; padding:18px 0 8px;
    color:#2d3748; font-size:.74rem;
    border-top:1px solid rgba(255,255,255,0.04);
    margin-top:24px;
}
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ── KONSTANTA ──────────────────────────────────────────────────────────────────
EMITEN = {
    "ADRO":{"nama":"Adaro Energy",       "ticker":"ADRO.JK","warna":"#6366f1"},
    "ADMR":{"nama":"Adaro Minerals",     "ticker":"ADMR.JK","warna":"#10b981"},
    "ITMG":{"nama":"Indo Tambangraya",   "ticker":"ITMG.JK","warna":"#f59e0b"},
    "PTBA":{"nama":"Bukit Asam",         "ticker":"PTBA.JK","warna":"#ef4444"},
    "MEDC":{"nama":"Medco Energi",       "ticker":"MEDC.JK","warna":"#8b5cf6"},
    "AKRA":{"nama":"AKR Corporindo",     "ticker":"AKRA.JK","warna":"#06b6d4"},
}
WARNA  = {k: v["warna"] for k,v in EMITEN.items()}
TICKER = {k: v["ticker"] for k,v in EMITEN.items()}

def rgba(hex_c, a):
    h = hex_c.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def ld(title="", h=340, mt=36, mb=28):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=rgba("#1c2132",0.9),
        font=dict(family="Inter",color="#94a3b8",size=11),
        xaxis=dict(gridcolor=rgba("#ffffff",0.04),linecolor=rgba("#ffffff",0.07),showgrid=True),
        yaxis=dict(gridcolor=rgba("#ffffff",0.04),linecolor=rgba("#ffffff",0.07),showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=rgba("#ffffff",0.07)),
        margin=dict(t=mt,b=mb,l=48,r=16),
        height=h,
        **({"title":dict(text=title,font=dict(size=13,color="#e2e8f0"))} if title else {}),
    )

# ── LOAD DATA CSV ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_csv_teknikal():
    for p in ["saham_lq45_energi_TEKNIKAL.csv",
              "saham_lq45_energi_covid2020_TEKNIKAL.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_csv_prediksi():
    for p in ["hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv",
              "hasil_prediksi_lstm_teknikal_t1.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_csv_evaluasi():
    for p in ["hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv",
              "evaluasi_lstm_teknikal_t1.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()

# ── LOAD DATA REALTIME DARI YAHOO FINANCE ─────────────────────────────────────
@st.cache_data(ttl=300)
def load_realtime(kode_list, periode="3mo"):
    """
    Ambil data harga terkini dari Yahoo Finance.
    Cache 5 menit — otomatis update setiap kunjungan baru.
    """
    rows = []
    for kode in kode_list:
        try:
            tk = yf.Ticker(TICKER[kode])
            df = tk.history(period=periode, interval="1d")
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c.replace(" ","_") for c in df.columns]
            df["Kode_Emiten"] = kode
            # Hitung log return & volatilitas 20 hari
            df = df.sort_values("Date")
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
            df["Volatilitas"] = df["Log_Return"].rolling(20, min_periods=5).std()
            rows.append(df[["Date","Kode_Emiten","Open","High","Low","Close",
                             "Volume","Log_Return","Volatilitas"]])
        except Exception:
            pass
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.rename(columns={"Date":"Tanggal"}, inplace=True)
        out["Tanggal"] = pd.to_datetime(out["Tanggal"]).dt.tz_localize(None)
        return out
    return pd.DataFrame()

# ── INISIALISASI DATA ──────────────────────────────────────────────────────────
df_csv  = load_csv_teknikal()
df_pred = load_csv_prediksi()
df_eval = load_csv_evaluasi()
CSV_OK  = len(df_csv) > 0

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 16px">
      <div style="font-size:2rem">📈</div>
      <div style="font-size:.92rem;font-weight:700;color:#e2e8f0">Volatilitas LQ45</div>
      <div style="font-size:.7rem;color:#475569;margin-top:2px">Sektor Energi Indonesia</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("**⚙️ Pilih Emiten**")
    emiten_all  = list(EMITEN.keys())
    emiten_sel  = st.multiselect("Emiten", emiten_all, default=emiten_all,
                                  label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("**🔄 Sumber Data**")
    sumber_mode = st.radio(
        "Mode",
        ["📁 CSV (Hasil Riset)", "🌐 Real-time Yahoo Finance"],
        label_visibility="collapsed",
    )
    if "Real-time" in sumber_mode:
        periode_rt = st.selectbox(
            "Periode Real-time",
            ["1mo","3mo","6mo","1y","2y","5y"],
            index=2,
            label_visibility="collapsed",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if CSV_OK:
        tgl_min = df_csv["Tanggal"].min().date()
        tgl_max = df_csv["Tanggal"].max().date()
    else:
        tgl_min = datetime.date(2020,3,1)
        tgl_max = datetime.date.today()

    if "CSV" in sumber_mode:
        st.markdown('<div class="sbox">', unsafe_allow_html=True)
        st.markdown("**📅 Rentang Tanggal**")
        rentang = st.date_input(
            "Rentang tanggal data",
            value=(tgl_min, tgl_max),
            min_value=tgl_min, max_value=tgl_max,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.77rem;color:#94a3b8;line-height:1.9">
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
    <div style="font-size:.77rem;color:#94a3b8;line-height:1.9">
    👩‍🎓 <b style="color:#e2e8f0">Ilda Nurida</b><br>
    Manajemen Teknologi<br>
    SIMT — ITS Surabaya<br>
    <span style="color:#6366f1;font-weight:700">2026</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── PILIH DATA AKTIF ───────────────────────────────────────────────────────────
IS_REALTIME = "Real-time" in sumber_mode

if IS_REALTIME:
    with st.spinner("🌐 Mengambil data terkini dari Yahoo Finance..."):
        df_rt = load_realtime(emiten_sel if emiten_sel else emiten_all, periode_rt)
    df_aktif = df_rt
    DATA_OK  = len(df_aktif) > 0
else:
    if CSV_OK and emiten_sel and len(rentang) == 2:
        mask = (
            df_csv["Kode_Emiten"].isin(emiten_sel) &
            (df_csv["Tanggal"].dt.date >= rentang[0]) &
            (df_csv["Tanggal"].dt.date <= rentang[1])
        )
        df_aktif = df_csv[mask].copy()
    else:
        df_aktif = pd.DataFrame()
    DATA_OK = len(df_aktif) > 0

# ── HERO ───────────────────────────────────────────────────────────────────────
now_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
live_html = f"""
<span class="live-pill">
  <span class="live-dot"></span>
  {"LIVE • " + now_str if IS_REALTIME else "📁 CSV • " + now_str}
</span>""" if DATA_OK else ""

st.markdown(f"""
<div class="hero">
  <div class="hero-title">📈 Dashboard Prediksi Volatilitas Return Saham</div>
  <div class="hero-sub">LQ45 Sektor Energi &nbsp;•&nbsp; Model LSTM (Baseline Teknikal) &nbsp;•&nbsp; {live_html}</div>
  <div>
    <span class="badge">🏭 6 Emiten Energi</span>
    <span class="badge">⏱ COVID 2020 – Kini</span>
    <span class="badge">🤖 LSTM t+1</span>
    <span class="badge">{'🌐 Real-time' if IS_REALTIME else '📁 CSV Riset'}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🏠  Overview",
    "📉  Harga & Volatilitas",
    "🤖  Prediksi LSTM",
    "📋  Evaluasi Model",
    "🌐  Harga Real-time",
])

# ════════════════════════════ TAB 1 — OVERVIEW ════════════════════════════════
with tab1:
    if DATA_OK and len(df_aktif) > 0:
        df_a = df_aktif
        c1,c2,c3,c4,c5 = st.columns(5)
        cards = [
            ("c1","📰",f"{len(df_a):,}","Total Baris","hari trading"),
            ("c2","🏭",f"{df_a['Kode_Emiten'].nunique()}","Emiten","LQ45 Energi"),
            ("c3","📊",f"{df_a['Volatilitas'].mean():.4f}","Volatilitas Rata2","std dev"),
            ("c4","📅",f"{(df_a['Tanggal'].max()-df_a['Tanggal'].min()).days//365}+","Tahun Data","periode"),
            ("c5","📈",f"{df_a['Log_Return'].std():.4f}","Std Return","log return"),
        ]
        for col,(cls,ico,val,lbl,sub) in zip([c1,c2,c3,c4,c5], cards):
            with col:
                st.markdown(f"""
                <div class="kcard {cls}">
                  <div class="ico">{ico}</div>
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                  <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tren volatilitas bulanan
        st.markdown('<div class="sec">📊 Tren Volatilitas Bulanan per Emiten</div>',
                    unsafe_allow_html=True)
        df_m = df_a.copy()
        df_m["Bulan"] = df_a["Tanggal"].dt.to_period("M").dt.to_timestamp()
        df_m = df_m.groupby(["Bulan","Kode_Emiten"])["Volatilitas"].mean().reset_index()
        fig_area = go.Figure()
        for kode in (emiten_sel or emiten_all):
            dk = df_m[df_m["Kode_Emiten"]==kode]
            if dk.empty: continue
            wn = WARNA.get(kode,"#888")
            fig_area.add_trace(go.Scatter(
                x=dk["Bulan"], y=dk["Volatilitas"], name=kode,
                mode="lines", line=dict(color=wn,width=2),
                fill="tozeroy", fillcolor=rgba(wn,0.07),
                hovertemplate=f"<b>{kode}</b><br>%{{x|%b %Y}}<br>%{{y:.4f}}<extra></extra>",
            ))
        fig_area.update_layout(**ld("Rata-rata Volatilitas per Bulan", h=300))
        st.plotly_chart(fig_area, use_container_width=True)

        cl, cr = st.columns([1.4,0.6])
        with cl:
            st.markdown('<div class="sec">📌 Ringkasan per Emiten</div>',
                        unsafe_allow_html=True)
            tbl = df_a.groupby("Kode_Emiten").agg(
                Hari=("Tanggal","count"),
                Close_Mean=("Close","mean"),
                Vol_Mean=("Volatilitas","mean"),
                Vol_Max=("Volatilitas","max"),
                Return_Mean=("Log_Return","mean"),
            ).round(5).reset_index()
            tbl.columns = ["Emiten","Hari","Close Rata2","Vol Rata2","Vol Max","Return Rata2"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        with cr:
            st.markdown('<div class="sec">🥧 Porsi Data</div>', unsafe_allow_html=True)
            vc = df_a["Kode_Emiten"].value_counts().reset_index()
            vc.columns = ["Emiten","N"]
            fig_pie = px.pie(vc, values="N", names="Emiten",
                             color="Emiten", color_discrete_map=WARNA, hole=0.5)
            fig_pie.update_traces(textposition="outside", textinfo="label+percent")
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter",color="#94a3b8"),
                showlegend=False,
                margin=dict(t=8,b=8,l=8,r=8), height=260,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Heatmap korelasi
        pv = df_a.pivot_table(index="Tanggal",columns="Kode_Emiten",
                               values="Volatilitas").dropna()
        if len(pv.columns) > 1:
            st.markdown('<div class="sec">🔥 Korelasi Volatilitas Antar Emiten</div>',
                        unsafe_allow_html=True)
            corr = pv.corr().round(3)
            fig_hm = px.imshow(corr, text_auto=True,
                                color_continuous_scale=["#ef4444","#1e293b","#10b981"],
                                zmin=-1, zmax=1)
            fig_hm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter",color="#94a3b8"),
                margin=dict(t=8,b=8,l=8,r=8), height=300,
            )
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.markdown("""
        <div class="ibox">📂 Tidak ada data. Pilih sumber data di sidebar kiri.<br>
        Jika pilih CSV, pastikan file <code>saham_lq45_energi_TEKNIKAL.csv</code>
        ada di root repository.</div>""", unsafe_allow_html=True)

# ════════════════════════ TAB 2 — HARGA & VOLATILITAS ═════════════════════════
with tab2:
    if DATA_OK and len(df_aktif) > 0:
        em_ada = [e for e in (emiten_sel or emiten_all)
                  if e in df_aktif["Kode_Emiten"].unique()]
        kode2  = st.selectbox("Pilih Emiten",em_ada,key="t2")
        df_e2  = df_aktif[df_aktif["Kode_Emiten"]==kode2].sort_values("Tanggal")
        wn2    = WARNA.get(kode2,"#6366f1")

        fig3 = make_subplots(
            rows=3,cols=1,shared_xaxes=True,
            subplot_titles=["Harga Close (Rp)","Log Return Harian",
                            f"Volatilitas Rolling 20 Hari — {kode2}"],
            vertical_spacing=0.06, row_heights=[0.5,0.25,0.25],
        )
        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"],y=df_e2["Close"],
            name="Close",line=dict(color=wn2,width=1.5),
            fill="tozeroy",fillcolor=rgba(wn2,0.07),
            hovertemplate="Rp %{y:,.0f}<extra></extra>",
        ),row=1,col=1)
        cbar = [wn2 if v>=0 else "#ef4444" for v in df_e2["Log_Return"].fillna(0)]
        fig3.add_trace(go.Bar(
            x=df_e2["Tanggal"],y=df_e2["Log_Return"],
            name="Log Return",marker_color=cbar,opacity=.8,
        ),row=2,col=1)
        fig3.add_trace(go.Scatter(
            x=df_e2["Tanggal"],y=df_e2["Volatilitas"],
            name="Volatilitas",line=dict(color="#f59e0b",width=1.5),
            fill="tozeroy",fillcolor=rgba("#f59e0b",0.08),
        ),row=3,col=1)
        vm_e2 = df_e2["Volatilitas"].mean()
        fig3.add_hline(y=vm_e2,row=3,col=1,line_dash="dot",line_color="#475569",
                       annotation_text=f"Mean={vm_e2:.4f}",
                       annotation_font_color="#64748b")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=rgba("#1c2132",0.9),
            font=dict(family="Inter",color="#94a3b8"),
            margin=dict(t=36,b=26,l=50,r=18),
            height=560, showlegend=False,
            title=dict(text=f"{kode2} — {EMITEN[kode2]['nama']}",
                       font=dict(size=14,color="#e2e8f0")),
        )
        fig3.update_xaxes(gridcolor=rgba("#ffffff",0.04))
        fig3.update_yaxes(gridcolor=rgba("#ffffff",0.04))
        st.plotly_chart(fig3, use_container_width=True)

        # Indikator teknikal (jika ada kolom RSI)
        if "RSI" in df_e2.columns:
            st.markdown('<div class="sec">📐 Indikator Teknikal</div>',
                        unsafe_allow_html=True)
            ia, ib = st.columns(2)
            with ia:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df_e2["Tanggal"],y=df_e2["RSI"],
                    line=dict(color=wn2,width=1.5),
                    fill="tozeroy",fillcolor=rgba(wn2,0.06),name="RSI",
                ))
                fig_rsi.add_hline(y=70,line_dash="dash",line_color="#ef4444",
                                   annotation_text="Overbought",
                                   annotation_font_color="#ef4444")
                fig_rsi.add_hline(y=30,line_dash="dash",line_color="#10b981",
                                   annotation_text="Oversold",
                                   annotation_font_color="#10b981")
                fig_rsi.update_layout(**ld("RSI (14 Hari)",h=260),yaxis_range=[0,100])
                st.plotly_chart(fig_rsi, use_container_width=True)
            with ib:
                if "Stoch_K" in df_e2.columns:
                    fig_st = go.Figure()
                    fig_st.add_trace(go.Scatter(
                        x=df_e2["Tanggal"],y=df_e2["Stoch_K"],
                        line=dict(color="#6366f1",width=1.5),name="%K",
                    ))
                    fig_st.add_trace(go.Scatter(
                        x=df_e2["Tanggal"],y=df_e2["Stoch_D"],
                        line=dict(color="#f59e0b",width=1.5,dash="dot"),name="%D",
                    ))
                    fig_st.add_hline(y=80,line_dash="dash",line_color="#ef4444",
                                      annotation_text="Overbought",
                                      annotation_font_color="#ef4444")
                    fig_st.add_hline(y=20,line_dash="dash",line_color="#10b981",
                                      annotation_text="Oversold",
                                      annotation_font_color="#10b981")
                    fig_st.update_layout(**ld("Stochastic Oscillator",h=260))
                    st.plotly_chart(fig_st, use_container_width=True)
    else:
        st.info("📂 Data belum tersedia.")

# ══════════════════════════ TAB 3 — PREDIKSI LSTM ═════════════════════════════
with tab3:
    if len(df_pred) > 0:
        em_p  = [e for e in (emiten_sel or emiten_all)
                 if e in df_pred["Kode_Emiten"].unique()]
        kode3 = st.selectbox("Pilih Emiten",em_p,key="t3") if em_p else None
        if kode3:
            dp3   = df_pred[df_pred["Kode_Emiten"]==kode3].sort_values("Tanggal")
            col_p = next((c for c in dp3.columns if "Pred" in c),None)
            wn3   = WARNA.get(kode3,"#6366f1")
            if col_p:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=dp3["Tanggal"],y=dp3["Volatilitas_Aktual"],
                    name="Aktual",line=dict(color=wn3,width=2.2),
                    hovertemplate="Aktual: %{y:.5f}<extra></extra>",
                ))
                fig_pred.add_trace(go.Scatter(
                    x=dp3["Tanggal"],y=dp3[col_p],
                    name="Prediksi LSTM (t+1)",
                    line=dict(color="#f59e0b",width=1.6,dash="dot"),
                    hovertemplate="Prediksi: %{y:.5f}<extra></extra>",
                ))
                fig_pred.update_layout(
                    **ld(f"Prediksi vs Aktual — {kode3} (Horizon t+1)",h=360),
                    legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                err  = (dp3["Volatilitas_Aktual"] - dp3[col_p]).abs()
                rmse = float(np.sqrt((err**2).mean()))
                mae  = float(err.mean())
                mape = float((err/dp3["Volatilitas_Aktual"].replace(0,np.nan)).mean()*100)

                m1,m2,m3 = st.columns(3)
                for col,cls,ico,v,lbl in [
                    (m1,"c2","🎯",f"{rmse:.6f}","RMSE"),
                    (m2,"c1","📏",f"{mae:.6f}","MAE"),
                    (m3,"c3","📐",f"{mape:.3f}%","MAPE"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="kcard {cls}">
                          <div class="ico">{ico}</div>
                          <div class="val">{v}</div>
                          <div class="lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                s1,s2 = st.columns(2)
                with s1:
                    mn = min(dp3["Volatilitas_Aktual"].min(),dp3[col_p].min())
                    mx = max(dp3["Volatilitas_Aktual"].max(),dp3[col_p].max())
                    fig_sc = px.scatter(dp3,x="Volatilitas_Aktual",y=col_p,
                                         opacity=.5,color_discrete_sequence=[wn3],
                                         title="Scatter: Aktual vs Prediksi")
                    fig_sc.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],
                                                 mode="lines",name="Sempurna",
                                                 line=dict(color="#475569",dash="dash")))
                    fig_sc.update_layout(**ld(h=300))
                    st.plotly_chart(fig_sc, use_container_width=True)
                with s2:
                    ed = (dp3["Volatilitas_Aktual"]-dp3[col_p]).rename("Error")
                    fig_eh = px.histogram(ed,nbins=30,
                                           color_discrete_sequence=[wn3],
                                           title="Distribusi Error")
                    fig_eh.add_vline(x=0,line_color="#ef4444",line_dash="dash")
                    fig_eh.update_layout(**ld(h=300))
                    st.plotly_chart(fig_eh, use_container_width=True)

                st.markdown('<div class="sec">📋 20 Hari Prediksi Terbaru</div>',
                            unsafe_allow_html=True)
                ds = dp3.tail(20)[["Tanggal","Volatilitas_Aktual",col_p,
                                    "Error_Abs","Error_Pct"]].copy()
                ds.columns = ["Tanggal","Aktual","Prediksi t+1","Error Abs","Error %"]
                st.dataframe(ds.round(6), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="ibox">📂 File prediksi belum tersedia.<br>
        Jalankan <code>PREDIKSI_VOLATILITAS_LSTM_TEKNIKAL_v2.ipynb</code> lalu
        upload hasil ke folder <code>hasil_lstm_teknikal/</code>.</div>""",
        unsafe_allow_html=True)

# ════════════════════════ TAB 4 — EVALUASI MODEL ══════════════════════════════
with tab4:
    if len(df_eval) > 0:
        st.markdown('<div class="sec">📋 Tabel Evaluasi Lengkap</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_eval.round(6), use_container_width=True, hide_index=True)
        e1,e2,e3 = st.columns(3)
        for col,m,ico in [(e1,"RMSE","🎯"),(e2,"MAE","📏"),(e3,"MAPE","📐")]:
            if m not in df_eval.columns: continue
            with col:
                fig_b = px.bar(df_eval.sort_values(m),x="Emiten",y=m,
                                color="Emiten",color_discrete_map=WARNA,
                                text_auto=".4f",title=f"{ico} {m}")
                fig_b.update_layout(**ld(h=260),showlegend=False)
                st.plotly_chart(fig_b, use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center;color:#64748b;font-size:.77rem'>"
                    f"Rata-rata: <b style='color:#e2e8f0'>{df_eval[m].mean():.5f}"
                    f"{'%' if m=='MAPE' else ''}</b></div>",
                    unsafe_allow_html=True)

        if all(m in df_eval.columns for m in ["RMSE","MAE","MAPE"]):
            st.markdown('<div class="sec">🕸️ Radar Perbandingan</div>',
                        unsafe_allow_html=True)
            dn = df_eval.copy()
            for m in ["RMSE","MAE","MAPE"]:
                mx = dn[m].max()
                dn[m] = dn[m]/mx if mx>0 else 0
            fig_r = go.Figure()
            for _,row in dn.iterrows():
                cats = ["RMSE","MAE","MAPE"]
                vals = [row[m] for m in cats]+[row["RMSE"]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals,theta=cats+[cats[0]],fill="toself",
                    name=row["Emiten"],
                    line_color=WARNA.get(row["Emiten"],"#888"),opacity=.7,
                ))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    radialaxis=dict(visible=True,range=[0,1],
                                    gridcolor=rgba("#ffffff",0.08)),
                    bgcolor=rgba("#1c2132",0.7),
                    angularaxis=dict(gridcolor=rgba("#ffffff",0.08)),
                ),
                font=dict(color="#94a3b8",family="Inter"),
                height=400,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Normalisasi Error Antar Emiten (lebih kecil = lebih baik)",
                           font=dict(color="#e2e8f0",size=13)),
                margin=dict(t=48,b=28,l=28,r=28),
            )
            st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.markdown("""
        <div class="ibox">📂 File evaluasi belum tersedia.<br>
        Jalankan notebook LSTM lalu upload <code>evaluasi_lstm_teknikal_t1.csv</code>.</div>""",
        unsafe_allow_html=True)

# ════════════════════════ TAB 5 — HARGA REAL-TIME ════════════════════════════
with tab5:
    st.markdown("""
    <div class="ibox" style="background:rgba(16,185,129,0.07);
         border-color:rgba(16,185,129,0.25);color:#6ee7b7">
    🌐 <b>Data Real-time dari Yahoo Finance</b><br>
    Harga diambil otomatis setiap 5 menit. Klik <b>Refresh</b> untuk memperbarui.
    Cache otomatis habis dalam 5 menit.
    </div>""", unsafe_allow_html=True)

    col_rt, _ = st.columns([1,3])
    with col_rt:
        if st.button("🔄  Refresh Data Sekarang", type="primary"):
            st.cache_data.clear()
            st.rerun()

    em_rt = emiten_sel if emiten_sel else emiten_all
    with st.spinner("Mengambil data..."):
        df_live = load_realtime(em_rt, "3mo")

    if len(df_live) > 0:
        st.markdown('<div class="sec">💹 Harga Penutupan Terkini</div>',
                    unsafe_allow_html=True)
        cols_rt = st.columns(len(em_rt))
        for col_r, kode in zip(cols_rt, em_rt):
            dk = df_live[df_live["Kode_Emiten"]==kode]
            if dk.empty: continue
            last  = dk.iloc[-1]
            prev  = dk.iloc[-2] if len(dk)>1 else last
            delta = last["Close"] - prev["Close"]
            pct   = delta / prev["Close"] * 100 if prev["Close"]!=0 else 0
            arrow = "▲" if delta>=0 else "▼"
            clr   = "#10b981" if delta>=0 else "#ef4444"
            with col_r:
                st.markdown(f"""
                <div class="kcard" style="border-top:3px solid {WARNA[kode]}">
                  <div class="ico">🏭</div>
                  <div style="font-size:.85rem;font-weight:700;
                       color:{WARNA[kode]}">{kode}</div>
                  <div class="val" style="font-size:1.15rem">
                    Rp {last['Close']:,.0f}
                  </div>
                  <div style="color:{clr};font-size:.8rem;font-weight:600">
                    {arrow} {abs(delta):,.0f} ({pct:+.2f}%)
                  </div>
                  <div class="sub">
                    Vol: {last['Volatilitas']:.4f}
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec">📈 Harga Close 3 Bulan Terakhir</div>',
                    unsafe_allow_html=True)
        fig_live = go.Figure()
        for kode in em_rt:
            dk = df_live[df_live["Kode_Emiten"]==kode]
            if dk.empty: continue
            wn = WARNA.get(kode,"#888")
            fig_live.add_trace(go.Scatter(
                x=dk["Tanggal"],y=dk["Close"],
                name=kode,mode="lines",
                line=dict(color=wn,width=2),
                hovertemplate=f"<b>{kode}</b> Rp%{{y:,.0f}}<extra></extra>",
            ))
        fig_live.update_layout(**ld("Close Price — 3 Bulan Terakhir",h=340))
        st.plotly_chart(fig_live, use_container_width=True)

        st.markdown('<div class="sec">📊 Volatilitas Real-time</div>',
                    unsafe_allow_html=True)
        fig_vrt = go.Figure()
        for kode in em_rt:
            dk = df_live[df_live["Kode_Emiten"]==kode]
            if dk.empty: continue
            wn = WARNA.get(kode,"#888")
            fig_vrt.add_trace(go.Scatter(
                x=dk["Tanggal"],y=dk["Volatilitas"],
                name=kode,mode="lines",
                line=dict(color=wn,width=1.5),
                fill="tozeroy",fillcolor=rgba(wn,0.06),
                hovertemplate=f"<b>{kode}</b> %{{y:.4f}}<extra></extra>",
            ))
        fig_vrt.update_layout(**ld("Volatilitas Rolling 20 Hari — Real-time",h=300))
        st.plotly_chart(fig_vrt, use_container_width=True)
    else:
        st.warning("⚠️ Gagal mengambil data real-time. Cek koneksi internet Streamlit Cloud.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  📈 Dashboard Prediksi Volatilitas Return Saham LQ45 Sektor Energi &nbsp;|&nbsp;
  LSTM Baseline (Teknikal) &nbsp;|&nbsp;
  Ilda Nurida &nbsp;•&nbsp; Manajemen Teknologi SIMT-ITS &nbsp;•&nbsp; 2026
</div>""", unsafe_allow_html=True)
