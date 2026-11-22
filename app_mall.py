import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Segmentation - Mall Customers",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Kustom tema gelap (Dark Mode) + menu sidebar bertingkat
st.markdown(
    """
    <style>
        .hero-box {
            background: linear-gradient(125deg, #1e293b 0%, #0f172a 55%, #1e1b4b 100%);
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 1.5rem 1.8rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.25rem;
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
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .hero-sub {
            font-size: 0.96rem;
            color: #cbd5e1;
            margin: 0;
        }
        .dark-card {
            background: #161b26;
            border: 1px solid #2a3346;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
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
            font-size: 1.65rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.2rem;
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
            margin-bottom: 0.5rem;
        }
        .step-title {
            font-size: 1rem;
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
        .sidebar-group {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #94a3b8;
            margin: 0.85rem 0 0.25rem 0;
        }
        .sidebar-footer {
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.45;
            margin-top: 0.5rem;
        }
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

MENU_SECTIONS = {
    "Pengantar": [
        "Beranda",
        "Tentang Dataset",
        "Eksplorasi Data (EDA)",
    ],
    "Pengolahan & Model": [
        "Preprocessing",
        "Reduksi Dimensi (PCA)",
        "Modelling & Evaluasi",
    ],
    "Penerapan": [
        "Simulasi Satu Pelanggan",
        "Simulasi Massal (CSV)",
        "Kesimpulan & Saran",
    ],
}
ALL_PAGES = [item for items in MENU_SECTIONS.values() for item in items]

if "halaman_aktif" not in st.session_state:
    st.session_state.halaman_aktif = "Beranda"


@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")


@st.cache_data
def fit_clustering(df, n_clusters=5):
    fitur_num = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[fitur_num].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    df_hasil = df.copy()
    df_hasil["Cluster"] = labels
    df_hasil["PCA1"] = X_pca[:, 0]
    df_hasil["PCA2"] = X_pca[:, 1]

    return scaler, pca, model, X_scaled, df_hasil, sil_score


def buat_grafik_gender(df):
    gender_counts = df["Gender"].value_counts()
    labels = gender_counts.index.tolist()
    values = gender_counts.values.tolist()

    fig, (ax_bar, ax_donut) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_alpha(0.0)

    ax_bar.set_facecolor("none")
    colors_bar = ["#3b82f6", "#ec4899"]
    bars = ax_bar.bar(labels, values, color=colors_bar[: len(labels)], width=0.5)
    ax_bar.set_title("Jumlah Pelanggan", color="#60a5fa", fontsize=12, fontweight="bold", pad=12)
    ax_bar.tick_params(colors="#e2e8f0")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#475569")
    ax_bar.spines["bottom"].set_color("#475569")
    ax_bar.set_ylim(0, max(values) * 1.2)
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            h + (max(values) * 0.03),
            f"{int(h):,}",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontweight="bold",
            fontsize=10,
        )

    ax_donut.set_facecolor("none")
    colors_donut = ["#3b82f6", "#f472b6"]
    _, _, autotexts = ax_donut.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_donut[: len(labels)],
        pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor="#0f172a", linewidth=2),
        textprops=dict(color="#e2e8f0", fontsize=10),
    )
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
    ax_donut.set_title("Proporsi", color="#60a5fa", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig


# Sidebar Bertingkat
with st.sidebar:
    for grup, daftar_menu in MENU_SECTIONS.items():
        st.markdown(f'<div class="sidebar-group">{grup}</div>', unsafe_allow_html=True)
        for item in daftar_menu:
            is_selected = st.session_state.halaman_aktif == item
            button_type = "primary" if is_selected else "secondary"
            if st.button(item, key=f"nav_{item}", use_container_width=True, type=button_type):
                st.session_state.halaman_aktif = item
                st.rerun()

    st.divider()
    st.markdown(
        """
        <div class="sidebar-footer">
            Praktikum Data Mining / Machine Learning<br>
            Dataset: Mall Customers (CSV)
        </div>
        """,
        unsafe_allow_html=True,
    )

halaman = st.session_state.halaman_aktif

try:
    df_mall = load_data()
    col_income = "Annual Income (k$)"
    col_spending = "Spending Score (1-100)"
    col_age = "Age"

    total_pelanggan = len(df_mall)
    total_kolom = len(df_mall.columns)
    rata_pendapatan = df_mall[col_income].mean()
    rata_spending = df_mall[col_spending].mean()
    rata_usia = df_mall[col_age].mean()

    scaler, pca, kmeans_model, X_scaled, df_clustered, sil_k5 = fit_clustering(df_mall, n_clusters=5)

    # 1. BERANDA
    if halaman == "Beranda":
        st.markdown(
            """
            <div class="hero-box">
                <div class="hero-badge">🛍️ Segmentasi Pelanggan Mall</div>
                <div class="hero-title">Analisis Customer Segmentation Mall</div>
                <p class="hero-sub">
                    Aplikasi ini memaparkan alur eksplorasi data, standardisasi fitur, reduksi dimensi PCA,
                    klasterisasi K-Means, serta simulasi segmentasi pada dataset <i>Mall Customers</i>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">Jumlah pelanggan</div>
                    <div class="card-value">{total_pelanggan:,}</div>
                    <div class="card-note">{total_kolom} kolom fitur</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-label">Rata-rata pendapatan</div>
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
                    <div class="card-label">Rata-rata skor belanja</div>
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
                    <div class="card-label">Silhouette Score (K=5)</div>
                    <div class="card-value">{sil_k5:.3f}</div>
                    <div class="card-note">K-Means (Age, Income, Score)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Alur pengerjaan")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="step-num">1. Dataset</div>
                    <div class="step-title">Mall Customers</div>
                    <p class="step-desc">Karakteristik data: {total_pelanggan} baris, {total_kolom} kolom, tanpa nilai kosong.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">2. EDA & Preprocessing</div>
                    <div class="step-title">Penyiapan Fitur</div>
                    <p class="step-desc">Meninjau distribusi usia, pendapatan, skor belanja, dan standardisasi skala.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="step-num">3. Reduksi PCA</div>
                    <div class="step-title">Proyeksi 2 Komponen</div>
                    <p class="step-desc">3 fitur numerik diproyeksikan ke 2 komponen utama ({pca.explained_variance_ratio_.sum()*100:.1f}% varians).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                """
                <div class="dark-card">
                    <div class="step-num">4. Klasterisasi</div>
                    <div class="step-title">K-Means & Simulasi</div>
                    <p class="step-desc">Mengelompokkan profil pelanggan dan menguji input data satuan maupun CSV.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. TENTANG DATASET
    elif halaman == "Tentang Dataset":
        st.title("Tentang Dataset")
        st.caption("Sumber: Mall Customers Dataset (format CSV)")

        t1, t2, t3, t4 = st.tabs(["Ringkasan", "Kamus Kolom", "Lihat Data", "Kualitas Data"])

        with t1:
            n_female = int((df_mall["Gender"] == "Female").sum())
            n_male = int((df_mall["Gender"] == "Male").sum())
            rasio_gender = f"{(n_female / n_male):.2f} : 1" if n_male > 0 else "-"

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="dark-card"><div class="card-label">Baris</div><div class="card-value">{total_pelanggan:,}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="dark-card"><div class="card-label">Kolom</div><div class="card-value">{total_kolom}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="dark-card"><div class="card-label">Pelanggan Female</div><div class="card-value">{n_female:,}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="dark-card"><div class="card-label">Rasio Female : Male</div><div class="card-value">{rasio_gender}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            st.markdown(
                """
                * **Jenis masalah:** segmentasi / klasterisasi pelanggan (*unsupervised learning*)
                * **Fitur utama:** `Age`, `Annual Income (k$)`, dan `Spending Score (1-100)`
                * **Tujuan:** mengelompokkan pelanggan berdasarkan daya beli dan perilaku belanja
                * **Catatan:** skala antar fitur berbeda sehingga *standardization* diperlukan sebelum klasterisasi
                """
            )
            st.pyplot(buat_grafik_gender(df_mall), use_container_width=True)

        with t2:
            kamus = pd.DataFrame(
                [
                    {"Kolom": "CustomerID", "Tipe": str(df_mall["CustomerID"].dtype), "Keterangan": "ID unik pelanggan"},
                    {"Kolom": "Gender", "Tipe": str(df_mall["Gender"].dtype), "Keterangan": "Jenis kelamin (Male / Female)"},
                    {"Kolom": "Age", "Tipe": str(df_mall[col_age].dtype), "Keterangan": "Usia pelanggan (tahun)"},
                    {"Kolom": col_income, "Tipe": str(df_mall[col_income].dtype), "Keterangan": "Pendapatan tahunan (ribuan USD / k$)"},
                    {"Kolom": col_spending, "Tipe": str(df_mall[col_spending].dtype), "Keterangan": "Skor pengeluaran (1–100)"},
                ]
            )
            st.dataframe(kamus, use_container_width=True, hide_index=True)

        with t3:
            st.dataframe(df_mall, use_container_width=True)
            st.markdown("#### Statistik Deskriptif")
            st.dataframe(df_mall.describe().T, use_container_width=True)

        with t4:
            missing_total = int(df_mall.isnull().sum().sum())
            dup_total = int(df_mall.duplicated().sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Nilai Kosong (Missing)", missing_total)
            k2.metric("Baris Duplikat", dup_total)
            k3.metric("Kelengkapan Data", f"{((1 - missing_total / (total_pelanggan * total_kolom)) * 100):.1f}%")

    # 3. EKSPLORASI DATA (EDA)
    elif halaman == "Eksplorasi Data (EDA)":
        st.title("Eksplorasi Data (EDA)")
        st.caption("Analisis distribusi usia, pendapatan tahunan, dan skor pengeluaran pelanggan")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        fig.patch.set_alpha(0.0)
        for ax, col, warna in zip(
            axes,
            [col_age, col_income, col_spending],
            ["#38bdf8", "#818cf8", "#34d399"],
        ):
            ax.set_facecolor("none")
            ax.hist(df_mall[col], bins=12, color=warna, edgecolor="#0f172a", alpha=0.9)
            ax.set_title(col, color="#f1f5f9", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#cbd5e1", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#475569")
            ax.spines["bottom"].set_color("#475569")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("#### Scatter Plot: Annual Income vs Spending Score")
        st.scatter_chart(df_mall, x=col_income, y=col_spending, color="Gender", use_container_width=True)

    # 4. PREPROCESSING
    elif halaman == "Preprocessing":
        st.title("Preprocessing")
        st.caption("Seleksi fitur numerik dan standardisasi menggunakan StandardScaler")

        fitur_dipilih = [col_age, col_income, col_spending]
        df_scaled = pd.DataFrame(X_scaled, columns=[f"{c} (scaled)" for c in fitur_dipilih])

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### Fitur Numerik Asli")
            st.dataframe(df_mall[fitur_dipilih].head(10), use_container_width=True)
        with p2:
            st.markdown("#### Setelah StandardScaler (Mean ≈ 0, Std = 1)")
            st.dataframe(df_scaled.head(10).round(3), use_container_width=True)

    # 5. REDUKSI DIMENSI (PCA)
    elif halaman == "Reduksi Dimensi (PCA)":
        st.title("Reduksi Dimensi (PCA)")
        st.caption("Memproyeksikan 3 fitur numerik (Age, Annual Income, Spending Score) ke 2 komponen utama")

        var1, var2 = pca.explained_variance_ratio_
        r1, r2, r3 = st.columns(3)
        r1.metric("Explained Variance PC1", f"{var1*100:.1f}%")
        r2.metric("Explained Variance PC2", f"{var2*100:.1f}%")
        r3.metric("Total Varians Tercakup", f"{(var1+var2)*100:.1f}%")

        st.scatter_chart(df_clustered, x="PCA1", y="PCA2", color="Cluster", use_container_width=True)

    # 6. MODELLING & EVALUASI
    elif halaman == "Modelling & Evaluasi":
        st.title("Modelling & Evaluasi")
        st.caption("Klasterisasi pelanggan menggunakan algoritma K-Means")

        k_pilih = st.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=8, value=5)
        _, _, _, _, df_eval, sil_eval = fit_clustering(df_mall, n_clusters=k_pilih)

        e1, e2 = st.columns(2)
        e1.metric("Jumlah Cluster (K)", k_pilih)
        e2.metric("Silhouette Score", f"{sil_eval:.4f}")

        st.markdown("#### Rata-rata Karakteristik Tiap Cluster")
        ringkasan_cluster = (
            df_eval.groupby("Cluster")[[col_age, col_income, col_spending]]
            .agg(["count", "mean"])
            .round(2)
        )
        st.dataframe(ringkasan_cluster, use_container_width=True)

    # 7. SIMULASI SATU PELANGGAN
    elif halaman == "Simulasi Satu Pelanggan":
        st.title("Simulasi Satu Pelanggan")
        st.caption("Masukkan profil pelanggan baru untuk melihat penentuan cluster K-Means (K=5)")

        with st.form("form_klaster"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                usia = st.number_input("Usia (Age)", min_value=15, max_value=100, value=28)
            with col_b:
                pendapatan = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=65)
            with col_c:
                spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=75)

            submit_val = st.form_submit_button("Proses Klaster Pelanggan")
            if submit_val:
                input_scaled = scaler.transform([[usia, pendapatan, spending]])
                pred_cluster = int(kmeans_model.predict(input_scaled)[0])
                st.success(
                    f"Pelanggan masuk ke **Cluster {pred_cluster}** | Usia: {usia} th | Pendapatan: ${pendapatan}k | Skor Belanja: {spending}"
                )

    # 8. SIMULASI MASSAL (CSV)
    elif halaman == "Simulasi Massal (CSV)":
        st.title("Simulasi Massal (CSV)")
        st.caption("Unggah file CSV dengan kolom Age, Annual Income (k$), dan Spending Score (1-100)")

        uploaded_csv = st.file_uploader("Pilih file CSV", type=["csv"])
        if uploaded_csv is not None:
            df_up = pd.read_csv(uploaded_csv)
            wajib = [col_age, col_income, col_spending]
            if all(col in df_up.columns for col in wajib):
                scaled_up = scaler.transform(df_up[wajib])
                df_up["Cluster_Prediksi"] = kmeans_model.predict(scaled_up)
                st.dataframe(df_up, use_container_width=True)
            else:
                st.warning(f"Kolom wajib harus ada: {', '.join(wajib)}")
        else:
            st.info("Belum ada file CSV diunggah. Kamu bisa menguji dengan salinan data `Mall_Customers.csv`.")

    # 9. KESIMPULAN & SARAN
    elif halaman == "Kesimpulan & Saran":
        st.title("Kesimpulan & Saran")
        st.markdown(
            f"""
            <div class="dark-card">
                <div class="step-title">Ringkasan Hasil Analisis</div>
                <p class="step-desc">
                    • Dataset <b>Mall_Customers.csv</b> terdiri dari {total_pelanggan} baris dan {total_kolom} kolom tanpa data kosong.<br>
                    • Standardisasi fitur numerik dan proyeksi 2 komponen PCA mencakup {pca.explained_variance_ratio_.sum()*100:.1f}% varians data.<br>
                    • Model <b>K-Means (K=5)</b> memperoleh Silhouette Score sebesar <b>{sil_k5:.4f}</b> dan dapat memisahkan segmen pendapatan serta skor belanja pelanggan.<br>
                    • <b>Saran:</b> pada pengembangan lanjutan, evaluasi rentang nilai $K$ lebih luas serta tambahkan interpretasi bisnis tiap segmen untuk strategi promosi yang lebih tepat sasaran.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
    st.info("Pastikan file 'Mall_Customers.csv' berada di folder yang sama dengan 'app_mall.py'.")
