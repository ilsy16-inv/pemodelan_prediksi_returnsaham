import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ── Konfigurasi Halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Prediksi Volatilitas Saham LQ45 Energi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Custom ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-title {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .sidebar-info {
        background: #f0f4ff;
        border-left: 4px solid #2a5298;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Warna per Emiten ───────────────────────────────────────────────────────────
WARNA = {
    "ADRO": "#1f77b4",
    "ADMR": "#2ca02c",
    "ITMG": "#ff7f0e",
    "PTBA": "#d62728",
    "MEDC": "#9467bd",
    "AKRA": "#8c564b",
}
EMITEN_INFO = {
    "ADRO": "Adaro Energy Indonesia Tbk",
    "ADMR": "Adaro Minerals Indonesia Tbk",
    "ITMG": "Indo Tambangraya Megah Tbk",
    "PTBA": "Bukit Asam Tbk",
    "MEDC": "Medco Energi Internasional Tbk",
    "AKRA": "AKR Corporindo Tbk",
}

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data_teknikal():
    path = "saham_lq45_energi_TEKNIKAL.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    return df

@st.cache_data
def load_hasil_prediksi():
    path = "hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    return df

@st.cache_data
def load_evaluasi():
    path = "hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")

df_teknikal = load_data_teknikal()
df_prediksi = load_hasil_prediksi()
df_evaluasi = load_evaluasi()

DATA_ADA = len(df_teknikal) > 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/TESIS-ITS%202026-blue?style=for-the-badge",
             use_column_width=True)
    st.markdown("## ⚙️ Filter")

    emiten_list = list(EMITEN_INFO.keys())
    if DATA_ADA:
        emiten_list = sorted(df_teknikal["Kode_Emiten"].unique().tolist())

    emiten_dipilih = st.multiselect(
        "Pilih Emiten",
        options=emiten_list,
        default=emiten_list,
        help="Pilih satu atau lebih emiten untuk ditampilkan"
    )

    st.markdown("---")

    if DATA_ADA:
        tgl_min = df_teknikal["Tanggal"].min().date()
        tgl_max = df_teknikal["Tanggal"].max().date()
    else:
        import datetime
        tgl_min = datetime.date(2023, 4, 1)
        tgl_max = datetime.date(2026, 4, 30)

    rentang_tgl = st.date_input(
        "Rentang Tanggal",
        value=(tgl_min, tgl_max),
        min_value=tgl_min,
        max_value=tgl_max,
    )

    st.markdown("---")
    st.markdown("### 📋 Info Model")
    st.markdown("""
    <div class="sidebar-info">
    🤖 <b>Model:</b> LSTM 3 Layer<br>
    ⏱ <b>Lookback:</b> 20 hari<br>
    🔮 <b>Horizon:</b> t+1 (1 hari ke depan)<br>
    📊 <b>Split:</b> 80% Train / 20% Test<br>
    📥 <b>Fitur X:</b> 12 variabel teknikal<br>
    🎯 <b>Target Y:</b> Volatilitas Return
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👩‍🎓 Peneliti")
    st.markdown("""
    <div class="sidebar-info">
    <b>Ilda Nurida</b><br>
    Program Studi Manajemen Teknologi<br>
    SIMT — Institut Teknologi Sepuluh Nopember<br>
    2026
    </div>
    """, unsafe_allow_html=True)

# ── Filter Data ────────────────────────────────────────────────────────────────
if DATA_ADA and emiten_dipilih and len(rentang_tgl) == 2:
    mask = (
        df_teknikal["Kode_Emiten"].isin(emiten_dipilih) &
        (df_teknikal["Tanggal"].dt.date >= rentang_tgl[0]) &
        (df_teknikal["Tanggal"].dt.date <= rentang_tgl[1])
    )
    df_filtered = df_teknikal[mask].copy()
else:
    df_filtered = pd.DataFrame()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-title">
    <h2 style="margin:0; font-size:1.6rem;">
        📈 Dashboard Prediksi Volatilitas Return Saham
    </h2>
    <p style="margin:6px 0 0; opacity:0.85; font-size:0.95rem;">
        Emiten LQ45 Sektor Energi | Model LSTM | Periode 3 Tahun (2023–2026)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📉 Harga & Volatilitas",
    "🤖 Hasil Prediksi LSTM",
    "📋 Evaluasi Model",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📌 Ringkasan Dataset")

    if DATA_ADA and len(df_filtered) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Data</div>
                <div class="metric-value">{len(df_filtered):,}</div>
                <div class="metric-label">hari trading</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Emiten</div>
                <div class="metric-value">{df_filtered["Kode_Emiten"].nunique()}</div>
                <div class="metric-label">LQ45 Sektor Energi</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            vol_mean = df_filtered["Volatilitas"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rata-rata Volatilitas</div>
                <div class="metric-value">{vol_mean:.4f}</div>
                <div class="metric-label">std dev log return</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            periode = f"{df_filtered['Tanggal'].min().strftime('%b %Y')} — {df_filtered['Tanggal'].max().strftime('%b %Y')}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Periode</div>
                <div class="metric-value" style="font-size:1.1rem;">{periode}</div>
                <div class="metric-label">3 tahun</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel jumlah data per emiten
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🏭 Distribusi Data per Emiten")
            tbl = df_filtered.groupby("Kode_Emiten").agg(
                Jumlah_Hari=("Tanggal","count"),
                Vol_Mean=("Volatilitas","mean"),
                Vol_Max=("Volatilitas","max"),
                Return_Mean=("Log_Return","mean"),
            ).round(6).reset_index()
            tbl.columns = ["Emiten","Hari Trading","Vol Mean","Vol Max","Return Mean"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### 📊 Komposisi Data per Emiten")
            fig_pie = px.pie(
                tbl, values="Hari Trading", names="Emiten",
                color="Emiten",
                color_discrete_map=WARNA,
                hole=0.4,
            )
            fig_pie.update_layout(height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Heatmap korelasi volatilitas
        st.markdown("#### 🔥 Heatmap Korelasi Volatilitas Antar Emiten")
        pivot_vol = df_filtered.pivot_table(
            index="Tanggal", columns="Kode_Emiten",
            values="Volatilitas"
        ).dropna()
        if len(pivot_vol.columns) > 1:
            corr = pivot_vol.corr().round(3)
            fig_heat = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
            )
            fig_heat.update_layout(height=350, margin=dict(t=10,b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("📂 Upload file `saham_lq45_energi_TEKNIKAL.csv` untuk menampilkan data.")
        st.markdown("""
        **File yang dibutuhkan:**
        - `saham_lq45_energi_TEKNIKAL.csv` — dari notebook scraping saham
        - `hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv` — dari notebook LSTM
        - `hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv` — dari notebook LSTM
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HARGA & VOLATILITAS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if DATA_ADA and len(df_filtered) > 0:

        emiten_tab2 = st.selectbox(
            "Pilih Emiten",
            options=emiten_dipilih,
            key="tab2_emiten"
        )

        df_e = df_filtered[df_filtered["Kode_Emiten"] == emiten_tab2]
        warna_e = WARNA.get(emiten_tab2, "#1f77b4")

        # Grafik harga + volatilitas berdampingan
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f"Harga Close — {emiten_tab2} ({EMITEN_INFO.get(emiten_tab2,'')})",
                "Log Return Harian",
                f"Volatilitas Return (Rolling 20 Hari)",
            ],
            vertical_spacing=0.08,
            row_heights=[0.45, 0.25, 0.30],
        )

        # Close price
        fig.add_trace(go.Scatter(
            x=df_e["Tanggal"], y=df_e["Close"],
            mode="lines", name="Close Price",
            line=dict(color=warna_e, width=1.5),
            fill="tozeroy", fillcolor=f"rgba{tuple(list(int(warna_e.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.1])}",
        ), row=1, col=1)

        # Log return
        fig.add_trace(go.Bar(
            x=df_e["Tanggal"], y=df_e["Log_Return"],
            name="Log Return",
            marker_color=[warna_e if v >= 0 else "#e74c3c"
                         for v in df_e["Log_Return"].fillna(0)],
        ), row=2, col=1)

        # Volatilitas
        fig.add_trace(go.Scatter(
            x=df_e["Tanggal"], y=df_e["Volatilitas"],
            mode="lines", name="Volatilitas",
            line=dict(color="#e74c3c", width=1.5),
        ), row=3, col=1)

        # Garis rata-rata volatilitas
        vol_mean_e = df_e["Volatilitas"].mean()
        fig.add_hline(
            y=vol_mean_e, row=3, col=1,
            line_dash="dash", line_color="gray",
            annotation_text=f"Mean={vol_mean_e:.4f}",
            annotation_position="right",
        )

        fig.update_layout(
            height=650,
            showlegend=False,
            margin=dict(t=40, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

        # Statistik deskriptif
        st.markdown(f"#### 📋 Statistik Deskriptif — {emiten_tab2}")
        col1, col2, col3, col4, col5 = st.columns(5)
        stats = {
            "Mean": df_e["Volatilitas"].mean(),
            "Std": df_e["Volatilitas"].std(),
            "Min": df_e["Volatilitas"].min(),
            "Median": df_e["Volatilitas"].median(),
            "Max": df_e["Volatilitas"].max(),
        }
        for col, (lbl, val) in zip([col1,col2,col3,col4,col5], stats.items()):
            col.metric(lbl, f"{val:.6f}")

        # Indikator teknikal
        st.markdown(f"#### 📊 Indikator Teknikal — {emiten_tab2}")
        fig_ind = make_subplots(
            rows=2, cols=2,
            subplot_titles=["RSI (14)", "Moving Average",
                            "Stochastic Oscillator", "Volume"],
            vertical_spacing=0.15, horizontal_spacing=0.08,
        )

        # RSI
        fig_ind.add_trace(go.Scatter(
            x=df_e["Tanggal"], y=df_e["RSI"],
            mode="lines", line=dict(color=warna_e, width=1.2), name="RSI"
        ), row=1, col=1)
        fig_ind.add_hline(y=70, row=1, col=1, line_dash="dash",
                          line_color="red", annotation_text="Overbought")
        fig_ind.add_hline(y=30, row=1, col=1, line_dash="dash",
                          line_color="green", annotation_text="Oversold")

        # MA
        for ma, warna_ma in [("MA_5","#1f77b4"),("MA_10","#ff7f0e"),("MA_20","#2ca02c")]:
            fig_ind.add_trace(go.Scatter(
                x=df_e["Tanggal"], y=df_e[ma],
                mode="lines", name=ma,
                line=dict(color=warna_ma, width=1.2),
            ), row=1, col=2)

        # Stochastic
        fig_ind.add_trace(go.Scatter(
            x=df_e["Tanggal"], y=df_e["Stoch_K"],
            mode="lines", name="%K",
            line=dict(color="#1f77b4", width=1.2),
        ), row=2, col=1)
        fig_ind.add_trace(go.Scatter(
            x=df_e["Tanggal"], y=df_e["Stoch_D"],
            mode="lines", name="%D",
            line=dict(color="#ff7f0e", width=1.2, dash="dash"),
        ), row=2, col=1)

        # Volume
        fig_ind.add_trace(go.Bar(
            x=df_e["Tanggal"], y=df_e["Volume"],
            name="Volume", marker_color=warna_e, opacity=0.7,
        ), row=2, col=2)

        fig_ind.update_layout(
            height=500, showlegend=True,
            margin=dict(t=40, b=20),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    else:
        st.info("📂 Data belum tersedia. Jalankan notebook scraping terlebih dahulu.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HASIL PREDIKSI LSTM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df_prediksi) > 0:
        st.markdown("### 🤖 Hasil Prediksi Volatilitas — LSTM (t+1)")

        emiten_tab3 = st.selectbox(
            "Pilih Emiten",
            options=[e for e in emiten_dipilih if e in df_prediksi["Kode_Emiten"].unique()],
            key="tab3_emiten",
        )

        df_pred_e = df_prediksi[df_prediksi["Kode_Emiten"] == emiten_tab3]

        if len(df_pred_e) > 0:
            # Kolom prediksi — cari nama kolom yang tepat
            col_pred = [c for c in df_pred_e.columns if "Pred" in c]
            col_pred = col_pred[0] if col_pred else None

            if col_pred:
                # Grafik prediksi vs aktual
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df_pred_e["Tanggal"],
                    y=df_pred_e["Volatilitas_Aktual"],
                    mode="lines", name="Aktual",
                    line=dict(color=WARNA.get(emiten_tab3,"#1f77b4"), width=2),
                ))
                fig_pred.add_trace(go.Scatter(
                    x=df_pred_e["Tanggal"],
                    y=df_pred_e[col_pred],
                    mode="lines", name="Prediksi LSTM (t+1)",
                    line=dict(color="#e74c3c", width=1.5, dash="dash"),
                ))
                fig_pred.update_layout(
                    title=f"Prediksi vs Aktual Volatilitas — {emiten_tab3}",
                    xaxis_title="Tanggal",
                    yaxis_title="Volatilitas",
                    height=400,
                    plot_bgcolor="white",
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Scatter plot
                col_l, col_r = st.columns(2)
                with col_l:
                    fig_scatter = px.scatter(
                        df_pred_e,
                        x="Volatilitas_Aktual", y=col_pred,
                        title="Scatter: Aktual vs Prediksi",
                        labels={"Volatilitas_Aktual":"Aktual",
                                col_pred:"Prediksi"},
                        color_discrete_sequence=[WARNA.get(emiten_tab3,"#1f77b4")],
                        opacity=0.6,
                    )
                    vmin = df_pred_e["Volatilitas_Aktual"].min()
                    vmax = df_pred_e["Volatilitas_Aktual"].max()
                    fig_scatter.add_trace(go.Scatter(
                        x=[vmin, vmax], y=[vmin, vmax],
                        mode="lines", name="Sempurna",
                        line=dict(color="gray", dash="dash"),
                    ))
                    fig_scatter.update_layout(height=350, plot_bgcolor="white")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with col_r:
                    st.markdown("#### 📊 Statistik Error")
                    error = df_pred_e["Volatilitas_Aktual"] - df_pred_e[col_pred]
                    fig_err = px.histogram(
                        error, nbins=30,
                        title="Distribusi Error Prediksi",
                        color_discrete_sequence=[WARNA.get(emiten_tab3,"#1f77b4")],
                    )
                    fig_err.update_layout(height=350, plot_bgcolor="white")
                    st.plotly_chart(fig_err, use_container_width=True)

                # Tabel 20 baris terakhir
                st.markdown("#### 📋 Data Prediksi Terbaru (20 Hari Terakhir)")
                df_show = df_pred_e.tail(20)[[
                    "Tanggal","Volatilitas_Aktual", col_pred, "Error_Abs","Error_Pct"
                ]].copy()
                df_show.columns = [
                    "Tanggal","Aktual","Prediksi t+1","Error Abs","Error %"
                ]
                df_show = df_show.round(6)
                st.dataframe(df_show, use_container_width=True, hide_index=True)

    else:
        st.info("📂 File hasil prediksi belum tersedia. Jalankan notebook LSTM terlebih dahulu.")
        st.markdown("""
        **File yang dibutuhkan:**
        `hasil_lstm_teknikal/hasil_prediksi_lstm_teknikal_t1.csv`
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(df_evaluasi) > 0:
        st.markdown("### 📋 Evaluasi Model LSTM — Semua Emiten")

        # Tabel evaluasi
        st.dataframe(
            df_evaluasi.round(6),
            use_container_width=True,
            hide_index=True,
        )

        # Grafik perbandingan metrik
        st.markdown("#### 📊 Perbandingan Metrik per Emiten")
        col_l, col_m, col_r = st.columns(3)

        for col, metrik, warna_bar in [
            (col_l, "RMSE", "#1f77b4"),
            (col_m, "MAE",  "#2ca02c"),
            (col_r, "MAPE", "#ff7f0e"),
        ]:
            with col:
                if metrik in df_evaluasi.columns:
                    fig_bar = px.bar(
                        df_evaluasi.sort_values(metrik),
                        x="Emiten", y=metrik,
                        title=metrik,
                        color="Emiten",
                        color_discrete_map=WARNA,
                        text_auto=".4f",
                    )
                    fig_bar.update_layout(
                        height=300, showlegend=False,
                        margin=dict(t=40,b=10),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart perbandingan emiten
        st.markdown("#### 🕸️ Radar Chart Perbandingan Evaluasi")
        if all(m in df_evaluasi.columns for m in ["RMSE","MAE","MAPE"]):
            df_norm = df_evaluasi.copy()
            for m in ["RMSE","MAE","MAPE"]:
                max_val = df_norm[m].max()
                df_norm[m] = df_norm[m] / max_val if max_val > 0 else 0

            fig_radar = go.Figure()
            categories = ["RMSE","MAE","MAPE"]
            for _, row in df_norm.iterrows():
                values = [row[m] for m in categories] + [row["RMSE"]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=row["Emiten"],
                    line_color=WARNA.get(row["Emiten"],"#888"),
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=400,
                title="Normalisasi Metrik Error per Emiten (lebih kecil = lebih baik)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Rata-rata
        st.markdown("#### 📌 Rata-rata Evaluasi Semua Emiten")
        col1, col2, col3 = st.columns(3)
        for col, metrik in [(col1,"RMSE"),(col2,"MAE"),(col3,"MAPE")]:
            if metrik in df_evaluasi.columns:
                val = df_evaluasi[metrik].mean()
                suffix = "%" if metrik == "MAPE" else ""
                col.metric(f"Rata-rata {metrik}", f"{val:.6f}{suffix}")

    else:
        st.info("📂 File evaluasi belum tersedia. Jalankan notebook LSTM terlebih dahulu.")
        st.markdown("""
        **File yang dibutuhkan:**
        `hasil_lstm_teknikal/evaluasi_lstm_teknikal_t1.csv`
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem; padding:10px 0">
    📈 Dashboard Prediksi Volatilitas Return Saham LQ45 Sektor Energi |
    Model: LSTM (Baseline — Teknikal) |
    Peneliti: Ilda Nurida | SIMT-ITS 2026
</div>
""", unsafe_allow_html=True)
