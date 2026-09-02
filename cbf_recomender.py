''' 
Content-Based Filtering Recomendation
Menggunakan Cosine Similarity

Fitur:
1. Rekomendasi item-to-item ("buku X mirip dengan buku Y")
2. Rekomendasi user-to-item ("User X pernah pinjam buku Y")
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self, df_book: pd.DataFrame, content_col: str = 'content_soup_clean'):
        ''' 
        df_book = dataframe buku yang sudah di preprocessing dan sudah punya
                  kolom content_soup_clean
        '''

        self.df_book = df_book.reset_index(drop=True)
        self.content_col = content_col
        self.tfidf = None
        self.tfidf_matrix = None
        self.sim_matrix = None
        self._id_to_idx = {
            id_buku: idx for idx, id_buku in enumerate(self.df_book['id_buku'])
        }

    #------------------------------------------------------------------------------------------------

    def fit(self, max_features=5000, ngram_range=(1, 2), min_df=1):
        ''' 
        Membangun TF-IDF matrix dan cosine similarity matrix
        '''
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df_book[self.content_col])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)
        return self

    #-------------------------------------------------------------------------------------------------
    
    def recommend_by_item(self, id_buku: str, top_n: int = 5, exclude_self: bool = True):
        ''' 
        Rekomendasi item-to-item: buku yang mirip dg buku ber-id X
        berdasarkan content-soup yang dibentuk
        '''
        if id_buku not in self._id_to_idx:
            raise ValueError(f'id_buku "{id_buku}" tidak ditemukan!')

        idx = self._id_to_idx[id_buku]
        scores = list(enumerate(self.sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        if exclude_self:
            scores = [s for s in scores if s[0] != idx]

        top_scores = scores[:top_n]
        result_idx = [i for i, _ in top_scores]
        result_score = [s for _, s in top_scores]

        result = self.df_book.iloc[result_idx][
            ['id_buku', 'judul', 'pengarang', 'kategori', 'fakultas_terkait', 'prodi_terkait']
        ].copy()
        result['similarity_score'] = result_score
        return result.reset_index(drop=True)

    def recommend_by_user_profile(self, id_buku_list: list, top_n: int = 5, exclude_seen: bool=True):
        ''' 
        Rekomendasi berbasis profil user: rata-rata vector similarity
        dari seluruh buku yang pernah dipinjam user, lalu ambil buku dengan skor
        tertinggi yang belum pernah pipinjam user

        content soup dari buku yang pernah dipinjam
        '''
        idx_list = [self._id_to_idx[b] for b in id_buku_list if b in self._id_to_idx]
        if not idx_list:
            return pd.DataFrame()

        # rata_rata baris similarity matrix dari buku-buku yang pernah dipinjam
        avg_scores = self.sim_matrix[idx_list].mean(axis=0)
        scores = list(enumerate(avg_scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        if exclude_seen:
            scores = [s for s in scores if s[0] not in idx_list]

        top_scores = scores[:top_n]
        result_idx = [i for i, _ in top_scores]
        result_score = [s for _, s in top_scores]

        result = self.df_book.iloc[result_idx][
            ['id_buku', 'judul', 'pengarang', 'kategori', 'fakultas_terkait', 'prodi_terkait']
        ].copy()
        result['similarity_score'] = result_score
        return result.reset_index(drop=True)

    #-------------------------------------------------------------------------------------------------------------------------------
    def get_book_info(self, id_buku: str) -> pd.Series:
        idx = self._id_to_idx[id_buku]
        return self.df_book.iloc[idx]

# ==========================================================================
# EVALUASI: Precision@K menggunakan riwayat peminjaman (train/test split)
# ==========================================================================

def evaluate_precision_at_k(recommender: BookRecommender, df_transaksi: pd.DataFrame,
                             k: int = 10, min_riwayat: int = 3, sample_user: int = None,
                             random_state: int = 42):
    """
]   Menyembunyikan 1 buku terakhir dari riwayat peminjaman, gunakan sbg sampel test
    Jika rekomendasi mengandung buku test, maka HIT

    precision at k = jumlah HIT yang didapat / jumlah yang dievaluasi
    """
    df_t = df_transaksi.sort_values("tanggal_pinjam")
    user_groups = df_t.groupby("id_user")["id_buku"].apply(list)
    user_groups = user_groups[user_groups.apply(len) >= min_riwayat]

    if sample_user:
        user_groups = user_groups.sample(
            min(sample_user, len(user_groups)), random_state=random_state
        )

    hits = 0
    total = 0
    for id_user, buku_list in user_groups.items():
        train = buku_list[:-1]
        test = buku_list[-1]
        rec = recommender.recommend_by_user_profile(train, top_n=k)
        if len(rec) == 0:
            continue
        total += 1
        if test in rec["id_buku"].values:
            hits += 1

    precision_at_k = hits / total if total > 0 else 0.0
    return {"precision_at_k": precision_at_k, "hits": hits, "total_evaluated": total, "k": k}


def evaluate_category_relevance_at_k(recommender: BookRecommender, df_transaksi: pd.DataFrame,
                                      k: int = 10, min_riwayat: int = 3, sample_user: int = None,
                                      random_state: int = 42):
    """
    evaluate_category_relevance_at_k = jumlah buku rekomendasi dengan kategori sesuai hide book / total buku direkomendasikan
    kalau konteksnya rata-rata = rata-rata nilai semua rekomendasi yang dievaluasi
    """
    df_t = df_transaksi.sort_values("tanggal_pinjam")
    user_groups = df_t.groupby("id_user")["id_buku"].apply(list)
    user_groups = user_groups[user_groups.apply(len) >= min_riwayat]

    if sample_user:
        user_groups = user_groups.sample(
            min(sample_user, len(user_groups)), random_state=random_state
        )

    precisions = []
    hit_rate_count = 0
    total = 0

    for id_user, buku_list in user_groups.items():
        train = buku_list[:-1]
        test_id = buku_list[-1]
        if test_id not in recommender._id_to_idx:
            continue
        test_kategori = recommender.get_book_info(test_id)["kategori"]

        rec = recommender.recommend_by_user_profile(train, top_n=k)
        if len(rec) == 0:
            continue

        total += 1
        relevan = (rec["kategori"] == test_kategori).sum()
        precisions.append(relevan / k)
        if relevan > 0:
            hit_rate_count += 1

    return {
        "avg_precision_at_k_kategori": float(np.mean(precisions)) if precisions else 0.0,
        "category_hit_rate": hit_rate_count / total if total > 0 else 0.0,
        "total_evaluated": total,
        "k": k,
    }
