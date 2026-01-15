import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Judul & Konfigurasi Halaman
st.set_page_config(page_title="AI Price Real-time", layout="wide")
st.title("🚗 Prediksi Harga Mobil Real-time (Decision Tree)")

# 2. Load Data Real-time (TTL 10 menit agar tidak berat tapi tetap update)
@st.cache_data(ttl=600)
def load_data():
    # Membaca data terbaru dari file lokal/repo
    df = pd.read_csv('dataset.csv')
    df = df.dropna()
    # Membersihkan outlier tahun yang tidak masuk akal (misal 1870) jika diperlukan
    df = df[df['tahun'] > 1980] 
    
    le = LabelEncoder()
    df['model_encoded'] = le.fit_transform(df['model'])
    return df, le

try:
    df, le = load_data()

    # 3. Model Training
    X = df[['model_encoded', 'tahun', 'jarak_tempuh', 'ukuran_mesin']]
    y = df['harga']
    model = DecisionTreeRegressor(random_state=42, max_depth=12)
    model.fit(X, y)

    # 4. Layout Dashboard
    col_input, col_viz = st.columns([1, 2])

    with col_input:
        st.subheader("📝 Input Spesifikasi")
        with st.form("input_form"):
            pilihan_model = st.selectbox("Pilih Model", df['model'].unique())
            tahun = st.number_input("Tahun", 1990, 2026, 2020)
            jarak = st.number_input("Jarak Tempuh (KM)", 0, 1000000, 20000)
            mesin = st.slider("Ukuran Mesin (L)", 0.5, 6.0, 1.5)
            
            submit = st.form_submit_button("Prediksi Harga")

        if submit:
            model_val = le.transform([pilihan_model])[0]
            prediksi = model.predict([[model_val, tahun, jarak, mesin]])[0]
            st.success(f"### Estimasi Harga: Rp {prediksi:,.0f}")
            
            # Evaluasi singkat
            y_pred = model.predict(X)
            st.info(f"**Akurasi Model (R²):** {r2_score(y, y_pred)*100:.1f}%")

    with col_viz:
        st.subheader("📊 Visualisasi Data Real-time")
        tab1, tab2 = st.tabs(["Distribusi Harga", "Tren Tahun vs Harga"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df['harga'], kde=True, color="blue", ax=ax)
            ax.set_title("Distribusi Harga Mobil dalam Dataset")
            st.pyplot(fig)
            
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df, x='tahun', y='harga', alpha=0.5, ax=ax2)
            ax2.set_title("Tren Harga Berdasarkan Tahun")
            st.pyplot(fig2)

    st.divider()
    st.subheader("📋 Dataset Saat Ini")
    st.dataframe(df.drop(columns=['model_encoded']), use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")