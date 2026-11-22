import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Judul & Divider (Bab 2.2)
st.title("🛒 Web Analisis Customer Segmentation")
st.subheader("Klasterisasi Daya Belanja & Pendapatan Pelanggan Mall")
st.caption("Dibuat untuk Laporan Praktikum Machine Learning berbasis Web")
st.divider()

# Memuat Dataset dengan Aman (Bab 2.3)
@st.cache_data
def load_data():
    # Membaca dataset Mall Customers
    df = pd.read_csv('Mall_Customers.csv')
    return df

try:
    df_mall = load_data()
    
    # Menggunakan Layout Tabs (Bab 2.7)
    tab1, tab2, tab3 = st.tabs(["📊 Data Pelanggan", "📈 Statistik & Grafik", "🔍 Simulasi Form Input"])

    with tab1:
        st.header("Tabel Dataframe Mall Customers")
        st.caption("Menampilkan data mentah pelanggan mall.")
        st.dataframe(df_mall, use_container_width=True)

    with tab2:
        st.header("Visualisasi Data Pendapatan vs Skor Pengeluaran")
        
        # Komponen Metric (Bab 2.3)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pelanggan", len(df_mall), "Orang")
        
        # Mengambil nama kolom secara fleksibel
        col_income = [col for col in df_mall.columns if 'Income' in col][0]
        col_spending = [col for col in df_mall.columns if 'Spending' in col][0]
        
        col2.metric("Rata-rata Pendapatan", f"${df_mall[col_income].mean():.1f}k")
        col3.metric("Rata-rata Skor Belanja", f"{df_mall[col_spending].mean():.1f}")
        
        # Grafik Bawaan Streamlit (Bab 2.4)
        st.line_chart(df_mall[[col_income, col_spending]].head(20))

    with tab3:
        st.header("Form Input Data Pelanggan Baru")
        st.caption("Gunakan form di bawah untuk mensimulasikan input data pelanggan.")
        
        # Control Flow / Form (Bab 2.8)
        with st.form("form_klaster"):
            usia = st.number_input("Usia (Age)", min_value=15, max_value=100, value=25)
            pendapatan = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
            spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
            
            submit_val = st.form_submit_button("Proses Klaster Pelanggan")
            
            if submit_val:
                st.success(f"Data berhasil diproses! Pendapatan: ${pendapatan}k, Skor Pengeluaran: {spending}")

    # Sidebar Navigasi (Bab 2.7)
    st.sidebar.title("Menu Navigasi")
    menu_pilihan = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Tentang Dataset", "Kontak"])
    if menu_pilihan == "Tentang Dataset":
        st.sidebar.info("Dataset ini digunakan untuk segmentasi pelanggan berdasarkan Annual Income dan Spending Score.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
    st.info("Pastikan file 'Mall_Customers.csv' berada di dalam folder C:\\datamining dan sejajar dengan file app_mall.py.")