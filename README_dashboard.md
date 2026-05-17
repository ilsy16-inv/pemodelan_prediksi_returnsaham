# Dashboard Prediksi Volatilitas Return Saham LQ45 Energi

## Cara Menjalankan

### 1. Install dependency
```bash
pip install -r requirements.txt
```

### 2. Siapkan file data
Pastikan file berikut ada di folder yang sama dengan dashboard.py:
```
dashboard_volatilitas.py
saham_lq45_energi_TEKNIKAL.csv          ← dari notebook scraping saham
hasil_lstm_teknikal/
    hasil_prediksi_lstm_teknikal_t1.csv  ← dari notebook LSTM
    evaluasi_lstm_teknikal_t1.csv        ← dari notebook LSTM
```

### 3. Jalankan dashboard
```bash
streamlit run dashboard_volatilitas.py
```

Dashboard akan terbuka di browser: http://localhost:8501

## Fitur Dashboard

| Tab | Isi |
|-----|-----|
| Overview | Ringkasan dataset, distribusi emiten, heatmap korelasi |
| Harga & Volatilitas | Grafik close price, log return, volatilitas, indikator teknikal |
| Hasil Prediksi LSTM | Grafik aktual vs prediksi, scatter plot, distribusi error |
| Evaluasi Model | Tabel RMSE/MAE/MAPE, bar chart, radar chart perbandingan |
