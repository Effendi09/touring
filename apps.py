import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import io

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Harga AI", layout="wide")

# 2. Dataset Referensi 
data_csv = """Usia Mobil (Tahun)	Jarak Tempuh (Ribu KM)	Tenaga Kuda (HP)	Harga Jual (Juta Rupiah)
2	15	150	250
3	25	150	230
5	40	120	180
1	10	180	320
7	80	110	130
4	35	140	210
6	60	120	160
2	20	170	280
8	100	100	100
5	50	150	195"""

df = pd.read_csv(io.StringIO(data_csv), sep='\t')

# 3. Model Decision Tree Regression 
X = df[['Usia Mobil (Tahun)', 'Jarak Tempuh (Ribu KM)', 'Tenaga Kuda (HP)']]
y = df['Harga Jual (Juta Rupiah)']
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# 4. Antarmuka Pengguna
st.title("🤖 AI Kalkulator Harga Mobil Bekas")
st.markdown("---")

col_input, col_viz = st.columns([1, 2])

with col_input:
    st.subheader("📥 Masukkan Data")
    with st.form("input_form"):
        u = st.number_input("Usia Mobil (Tahun)", 1, 15, 3)
        j = st.number_input("Jarak Tempuh (Ribu KM)", 0, 200, 25)
        h = st.number_input("Tenaga (HP)", 50, 250, 150)
        submit = st.form_submit_button("Hitung Estimasi")

    if submit:
        prediksi = model.predict([[u, j, h]])[0]
        st.success(f"### Estimasi Harga:\n# Rp {prediksi:.1f} Juta")

with col_viz:
    st.subheader("📈 Visualisasi & Akurasi")
    tab1, tab2 = st.tabs(["Grafik Tren", "Evaluasi Error"])
    
    with tab1:
        # Grafik Usia vs Harga
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='Usia Mobil (Tahun)', y='Harga Jual (Juta Rupiah)', ax=ax, color='teal')
        st.pyplot(fig)
        
    with tab2:
        # Perhitungan Error
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} Juta")
        st.metric("R-Squared (Akurasi)", f"{r2*100:.1f}%")
        st.info("MAE menunjukkan rata-rata selisih harga prediksi dengan harga asli.")

# 5. Tabel Data 
with st.expander("Lihat Detail Dataset Referensi"):
    st.dataframe(df, use_container_width=True)