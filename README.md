# Implementasi Kecerdasan Buatan untuk Optimalisasi Layanan Hybrid Library
### Pendekatan Content-Based Filtering pada Sistem Rekomendasi Buku

Sistem rekomendasi buku untuk perpustakaan kampus swasta, dibangun dengan pendekatan
**Content-Based Filtering (TF-IDF + Cosine Similarity)**, mencakup dua fakultas:

- **Fakultas Sains, Teknologi, & Kesehatan (FSTK)** — Informatika, Teknik Industri, Keperawatan, Farmasi, Profesi Ners
- **Fakultas Sosial, Humaniora, & Seni (FSHS)** — Desain Komunikasi Visual, Desain Interior, Administrasi Bisnis, Psikologi, Ilmu Komunikasi

---

## 1. Struktur Proyek

```
├── app/
│   └── hybrid-library.py           # Aplikasi demo Streamlit
├── src/
│   └── cbf_recomender.py           # Class BookRecommender + fungsi evaluasi
├── data/
│   ├── buku.csv                    # Data master buku
│   ├── pengguna.csv                # Data pengguna (mahasiswa & dosen)
│   ├── transaksi.csv               # Riwayat peminjaman
│   ├── data-buku-preprocessed.csv  # buku.csv + content_soup_clean (hasil preprocessing)
│   └── daftar-buku-paling-banyak-dipinjam.csv  # hasil analisis EDA
├── cbf-book-rec.ipynb              # Notebook: EDA, preprocessing, modeling, evaluasi
├── requirements.txt
└── README.md
```

## 2. Metodologi

### 2.1 Eksplorasi Data (EDA)
Analisis dilakukan terhadap 3 dataset: buku, pengguna, dan transaksi peminjaman.
Beberapa temuan kunci dari notebook:
- Distribusi buku didominasi FSTK, dengan Prodi Informatika paling banyak koleksinya.
- Peminjaman terbanyak justru dari FSHS (dominasi Prodi Psikologi), sementara di FSTK
  didominasi Keperawatan dan Profesi Ners.
- Rata-rata keterlambatan pengembalian ±18 hari; Prodi Informatika paling sering telat.
- Pola peminjaman didominasi kesesuaian fakultas/prodi pengguna — mengonfirmasi relevansi
  pendekatan **Content-Based Filtering** untuk kasus ini, meski ada ±160 mahasiswa yang
  meminjam lintas fakultas (menunjukkan perlunya sistem tetap terbuka untuk eksplorasi).

### 2.2 Preprocessing Teks
- **Content soup**: gabungan atribut buku (judul, kategori, sinopsis, pengarang, tags) jadi satu teks.
- **Cleaning & stemming** Bahasa Indonesia menggunakan **Sastrawi**.
- Hasil akhir disimpan di kolom `content_soup_clean` pada `data-buku-preprocessed.csv`, agar
  tidak perlu diproses ulang setiap kali aplikasi dijalankan (stemming untuk >1000 dokumen relatif berat).

### 2.3 Feature Extraction & Modeling
- **TF-IDF Vectorizer** atas `content_soup_clean`.
- **Cosine Similarity** dihitung antar seluruh pasangan buku → dasar dari:
  - `recommend_by_item()` — rekomendasi item-to-item
  - `recommend_by_user_profile()` — rekomendasi berbasis rata-rata profil riwayat peminjaman user

**Hyperparameter TF-IDF** dipilih melalui grid search kecil terhadap `max_features`,
`ngram_range`, dan `min_df`, dievaluasi dengan metrik kategori (lihat di bawah):

| Kombinasi | max_features | ngram_range | min_df | Hit-Rate Kategori | Avg. Precision Kategori |
|---|---|---|---|---|---|
| Pilihan 1 | 2000 | (1,1) | 1 | 0.64 | 0.638 |
| **Pilihan 2 (dipilih)** | 2000 | (1,2) | 3 | 0.64 | 0.636 |

Kedua kombinasi hampir identik secara angka (selisih dalam rentang noise sampling).
**Pilihan 2** ditetapkan sebagai konfigurasi final karena `ngram_range=(1,2)` lebih tangguh
menangkap istilah teknis dua kata (mis. "kecerdasan buatan", "basis data") yang relevan
untuk domain akademik, tanpa mengorbankan performa dibanding unigram saja.

### 2.4 Evaluasi

Evaluasi menggunakan pendekatan **leave-one-out per user** (kronologis, berbasis `tanggal_pinjam`):
buku terakhir yang dipinjam user disembunyikan sebagai *test*, sisanya sebagai *train* untuk
membangun profil preferensi.

Dua metrik dipakai secara komplementer:
- **`evaluate_precision_at_k`** — Precision@K exact-match: apakah buku yang *benar-benar*
  dipinjam user berikutnya persis muncul di Top-K rekomendasi. Metrik paling ketat.
- **`evaluate_category_relevance_at_k`** — mengukur relevansi topik/kategori:
  - `category_hit_rate` — proporsi user yang mendapat minimal 1 rekomendasi berkategori sama
    dengan buku target.
  - `avg_precision_at_k_kategori` — rata-rata proporsi Top-K yang kategorinya relevan.

Metrik kategori dianggap lebih representatif untuk CBF, karena esensi Content-Based
Filtering adalah kedekatan topik/konten, bukan menebak satu transaksi spesifik dari
ribuan kandidat.

## 3. Cara Menjalankan

```bash
pip install -r requirements.txt

# Jalankan aplikasi demo (jalankan dari root folder proyek)
streamlit run app/hybrid-library.py

# Atau eksplorasi pipeline lengkap (EDA, preprocessing, modeling, evaluasi)
jupyter notebook cbf-book-rec.ipynb
```

## 4. Fitur Aplikasi (Streamlit)

Akses demo sistem di URL: https://hybrid-library-book-recommendation-cbf.streamlit.app/ 

- **Statistik Buku** — ringkasan koleksi: distribusi per fakultas, kategori, jenis koleksi, status peminjaman.
- **Rekomendasi Berbasis Buku** — pilih 1 buku, tampilkan Top-N buku paling mirip secara konten.
- **Rekomendasi Berbasis Riwayat User** — bangun profil dari riwayat peminjaman, tampilkan rekomendasi personal.


## 5. Potensi Pengembangan Lanjutan

- Mengaktifkan kembali halaman evaluasi model di aplikasi untuk demo interaktif.
- Kombinasi dengan **Collaborative Filtering** menjadi pendekatan hybrid.
- Bobot **recency** — riwayat peminjaman terbaru lebih berpengaruh ke profil user dibanding yang lama.
- Investigasi lebih lanjut pola keterlambatan pengembalian per prodi untuk kebijakan sirkulasi buku.
- Penyimpanan model terlatih (`joblib`) agar tidak perlu fit ulang TF-IDF setiap start aplikasi.
