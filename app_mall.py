import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Segmentation - Mall Customers",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Kustom untuk tema gelap (Dark Mode) yang modern & kontras jelas
st.markdown(
    """
    <style>
        /* Hero banner gelap beraksen biru-ungu */
        .hero-box {
            background: linear-gradient(125deg, #1e293b 0%, #0f172a 55%, #1e1b4b 100%);
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.3rem;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.16);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.65rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .hero-sub {
            font-size: 0.98rem;
            color: #cbd5e1;
            margin: 0;
        }

        /* Kartu metrik & alur tema gelap */
        .dark-card {
            background: #161b26;
            border: 1px solid #2a3346;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            height: 100%;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .dark-card:hover {
            transform: translateY(-2px);
            border-color: #38bdf8;
        }
        .card-label {
            font-size: 0.84rem;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 0.35rem;
        }
        .card-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }
        .card-note {
            font-size: 0.8rem;
            color: #38bdf8;
            font-weight: 600;
        }
        .step-num {
            display: inline-block;
            background: rgba(99, 102, 241, 0.18);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.35);
            font-weight: 700;
            font-size: 0.78rem;
            padding: 0.2rem 0.65rem;
            border-radius: 8px;
            margin-bottom: 0.55rem;
        }
        .step-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.35rem;
        }
        .step-desc {
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.45;
            margin: 0;
        }

        /* Tombol submit form */
        div.stFormSubmitButton > button {
            background: linear-gradient(90deg, #2563eb, #0284c7);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")


# Sidebar Navigasi
with st.sidebar:
    st.markdown("### 🧭 Navigasi Utama")
    halaman = st.radio(
        "Pilih Bagian",
        ["Beranda & Ringkasan", "Data & Visualisasi", "Simulasi Pelanggan", "Tentang Dataset"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Praktikum Machine Learning / Data Mining")
    st.caption("Dataset: Mall Customers")

try:
    df_mall = load_data()

    col_income = [c for c in df_mall.columns if "Income" in c][0]
    col_spending = [c for c in df_mall.columns if "Spending" in c][0]
    col_age = [c for c in df_mall.columns if "Age" in c][0]

    total_pelanggan = len(df_mall)
    rata_pendapatan = df_mall[col_income].mean()
    rata_spending = df_mall[col_spending].mean()
    rata_usia = df_mall[col_age].mean()

    if halaman == "Beranda & Ringkasan":
        st.markdown(
            """
            <div class="hero-box">
                <div class="hero-badge">✨ Dashboard Segmentasi Pelanggan</div>
                <div class="hero-title">Analisis Customer Segmentation Mall</div>
                <p class="hero-sub">
                    Eksplorasi pola pendapatan tahunan (Annual Income) dan skor pengeluaran (Spending Score)
                    pelanggan mall untuk mendukung analisis klasterisasi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Kartu Ringkasan Utama
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">👥 Jumlah Pelanggan</div>
                    <div class="card-value">{total_pelanggan:,}</div>
                    <div class="card-note">Data {len(df_mall.columns)} kolom</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">💰 Rata-rata Pendapatan</div>
                    <div class="card-value">${rata_pendapatan:.1f}k</div>
                    <div class="card-note">Annual Income</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">🛍️ Rata-rata Skor Belanja</div>
                    <div class="card-value">{rata_spending:.1f}</div>
                    <div class="card-note">Skala 1 – 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">🎂 Rata-rata Usia</div>
                    <div class="card-value">{rata_usia:.1f} th</div>
                    <div class="card-note">Rentang {df_mall[col_age].min()}–{df_mall[col_age].max()} tahun</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🔄 Alur Eksplorasi Data")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">Langkah 1</div>
                    <div class="step-title">Pemuatan Dataset</div>
                    <p class="step-desc">Membaca data pelanggan mall secara aman dengan cache agar akses cepat.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">Langkah 2</div>
                    <div class="step-title">Observasi Fitur</div>
                    <p class="step-desc">Meninjau usia, pendapatan tahunan, dan skor pengeluaran pelanggan.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">Langkah 3</div>
                    <div class="step-title">Visualisasi Awal</div>
                    <p class="step-desc">Membandingkan tren pendapatan dan skor belanja melalui grafik.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">Langkah 4</div>
                    <div class="step-title">Simulasi Input</div>
                    <p class="step-desc">Menguji input data pelanggan baru untuk melihat profilnya.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif halaman == "Data & Visualisasi":
        st.subheader("📊 Tabel Dataframe & Grafik Pelanggan")
        tab1, tab2 = st.tabs(["📋 Tabel Data Pelanggan", "📈 Statistik & Grafik"])

        with tab1:
            st.caption("Menampilkan data mentah pelanggan mall.")
            st.dataframe(df_mall, use_container_width=True)

        with tab2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Pelanggan", f"{total_pelanggan} Orang")
            m2.metric("Rata-rata Pendapatan", f"${rata_pendapatan:.1f}k")
            m3.metric("Rata-rata Skor Belanja", f"{rata_spending:.1f}")

            st.markdown("#### Grafik Pendapatan vs Skor Pengeluaran (20 Data Pertama)")
            st.line_chart(df_mall[[col_income, col_spending]].head(20), use_container_width=True)

    elif halaman == "Simulasi Pelanggan":
        st.subheader("📝 Form Input Data Pelanggan Baru")
        st.caption("Gunakan form di bawah untuk mensimulasikan input data pelanggan.")

        with st.form("form_klaster"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                usia = st.number_input("Usia (Age)", min_value=15, max_value=100, value=25)
            with col_b:
                pendapatan = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
            with col_c:
                spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

            submit_val = st.form_submit_button("Proses Klaster Pelanggan")

            if submit_val:
                st.success(
                    f"Data berhasil diproses! Usia: {usia} tahun | Pendapatan: ${pendapatan}k | Skor Pengeluaran: {spending}"
                )

    elif halaman == "Tentang Dataset":
        st.subheader("ℹ️ Tentang Dataset Mall Customers")
        st.markdown(
            """
            <div class="dark-card">
                <div class="step-title">Ringkasan Dataset</div>
                <p class="step-desc">
                    Dataset <b>Mall_Customers.csv</b> berisi informasi dasar pelanggan seperti jenis kelamin, usia,
                    pendapatan tahunan (dalam ribuan dolar / k$), serta skor pengeluaran (1–100) yang sering
                    digunakan untuk latihan segmentasi pelanggan / klasterisasi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
    st.info("Pastikan file 'Mall_Customers.csv' berada di folder yang sama dengan 'app_mall.py'.")
