"""
Aplikasi Demo - Sistem Rekomendasi Buku Perpustakaan Hybrid Library
Pendekatan: Content-Based Filtering (TF-IDF + Cosine Similarity)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
from cbf_recomender import BookRecommender

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

st.set_page_config(
    page_title="Sistem Rekomendasi Buku - Hybrid Library",
    page_icon="img/icon2.png",
    layout="wide",
)

st.set_page_config(
    page_title="Sistem Rekomendasi Buku - Hybrid Library",
    page_icon="img/icon2.png",
    layout="wide",
)

# Custom CSS berdasarkan palet warna
st.markdown("""
    <style>
    /* Background Utama */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Warna Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFE6AA;
    }
    
    /* Judul dan Header */
    h1, h2, h3 {
        color: #DF301C !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD DATA & MODEL
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    df_buku = pd.read_csv('data/data-buku-preprocessed.csv')
    df_user = pd.read_csv("data/pengguna.csv")
    df_trx = pd.read_csv("data/transaksi.csv")
    return df_buku, df_user, df_trx


@st.cache_resource
def load_recommender(_df_buku):
    engine = BookRecommender(_df_buku).fit()
    return engine


df_buku, df_user, df_trx = load_data()
engine = load_recommender(df_buku)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------

st.sidebar.title("Hybrid Library Recommendation System")
st.sidebar.caption("With Content-Based Filtering & Cosine Similarity")
menu = st.sidebar.radio(
    "Navigasi",
    [
        "Statistik Buku",
        "Rekomendasi Berbasis Buku",
        "Rekomendasi Berbasis Riwayat User"
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fakultas cakupan:**\n"
    "- Sains, Teknologi, & Kesehatan\n"
    "- Sosial, Humaniora, & Seni"
)

# ----------------------------------------------------------------------
# HALAMAN 1: BERANDA
# ----------------------------------------------------------------------

if menu == "Statistik Buku":
    st.title("Sistem Rekomendasi Buku Perpustakaan")
    st.markdown(
        "Optimalisasi layanan **Hybrid Library** menggunakan pendekatan "
        "**Content-Based Filtering** berbasis TF-IDF dan Cosine Similarity."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Buku", f"{len(df_buku):,}")
    col2.metric("Total Pengguna", f"{len(df_user):,}")
    col3.metric("Total Transaksi", f"{len(df_trx):,}")
    col4.metric("Jumlah Kategori", df_buku["kategori"].nunique())

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribusi Buku per Fakultas")
        fig1 = px.pie(df_buku, names="fakultas_terkait", hole=0.45)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Distribusi Buku per Jenis Koleksi")
        fig2 = px.pie(df_buku, names="jenis_koleksi", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Jumlah Buku per Kategori")
    kat_count = df_buku["kategori"].value_counts().reset_index()
    kat_count.columns = ["kategori", "jumlah"]
    fig3 = px.bar(kat_count, x="jumlah", y="kategori", orientation="h",
                  color="jumlah", color_continuous_scale="Blues")
    fig3.update_layout(height=550, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Status Peminjaman")
    status_count = df_trx["status"].value_counts().reset_index()
    status_count.columns = ["status", "jumlah"]
    fig4 = px.bar(status_count, x="status", y="jumlah", color="status")
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------
# HALAMAN 2: REKOMENDASI BERBASIS BUKU (ITEM-TO-ITEM)
# ----------------------------------------------------------------------

elif menu == "Rekomendasi Berbasis Buku":
    st.title("Rekomendasi Berbasis Buku")
    st.markdown("Pilih satu buku, sistem akan menampilkan buku lain yang **paling mirip secara konten**.")

    fakultas_filter = st.selectbox(
        "Filter Fakultas (opsional)",
        ["Semua"] + sorted(df_buku["fakultas_terkait"].unique().tolist()),
    )
    df_filtered = df_buku if fakultas_filter == "Semua" else df_buku[df_buku["fakultas_terkait"] == fakultas_filter]

    judul_terpilih = st.selectbox("Pilih Buku", df_filtered["judul"] + " — " + df_filtered["id_buku"])
    id_buku_terpilih = judul_terpilih.split("— ")[-1]

    top_n = st.slider("Jumlah rekomendasi", 3, 20, 10)

    if st.button("Tampilkan Rekomendasi", type="primary"):
        info = engine.get_book_info(id_buku_terpilih)
        st.info(
            f"**Buku acuan:** {info['judul']}  \n"
            f"**Pengarang:** {info['pengarang']}  \n"
            f"**Kategori:** {info['kategori']}  \n"
            f"**Sinopsis:** {info['sinopsis']}"
        )
        
        hasil = engine.recommend_by_item(id_buku_terpilih, top_n=top_n)
        st.subheader(f"Top-{top_n} Buku Serupa")
        hasil_display = hasil.copy()
        hasil_display["similarity_score"] = hasil_display["similarity_score"].round(3)
        st.dataframe(hasil_display, use_container_width=True, hide_index=True)

        fig = px.bar(
            hasil_display, x="similarity_score", y="judul", orientation="h",
            color="kategori", title="Skor Kemiripan Konten"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# HALAMAN 3: REKOMENDASI BERBASIS RIWAYAT USER
# ----------------------------------------------------------------------

elif menu == "Rekomendasi Berbasis Riwayat User":
    st.title("Rekomendasi Berbasis Riwayat Peminjaman User")
    st.markdown("Sistem membangun **profil preferensi** dari agregasi seluruh buku yang pernah dipinjam user.")

    id_user_list = df_trx["id_user"].value_counts()
    id_user_list = id_user_list[id_user_list >= 3].index.tolist()
    user_terpilih = st.selectbox("Pilih User (id_user)", id_user_list)

    info_user = df_user[df_user["id_user"] == user_terpilih].iloc[0]
    st.markdown(
        f"**Nama:** {info_user['nama']}  \n"
        f"**Fakultas:** {info_user['fakultas']}  \n"
        f"**Prodi:** {info_user['prodi']}"
    )

    riwayat = df_trx[df_trx["id_user"] == user_terpilih].merge(
        df_buku[["id_buku", "judul", "pengarang", "kategori"]], on="id_buku"
    )
    st.subheader("Riwayat Peminjaman")
    st.dataframe(
        riwayat[["id_buku", "judul", "kategori", "pengarang", "tanggal_pinjam", "status", "rating"]],
        use_container_width=True, hide_index=True,
    )

    top_n = st.slider("Jumlah rekomendasi", 3, 20, 10, key="user_topn")

    if st.button("Buat Rekomendasi Personal", type="primary"):
        id_buku_list = riwayat["id_buku"].tolist()
        hasil = engine.recommend_by_user_profile(id_buku_list, top_n=top_n)
        st.subheader(f"Top-{top_n} Rekomendasi untuk {info_user['nama']}")
        hasil_display = hasil.copy()
        hasil_display["similarity_score"] = hasil_display["similarity_score"].round(3)
        st.dataframe(hasil_display, use_container_width=True, hide_index=True)

# END OF CONTENT