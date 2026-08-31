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

    def fit(self, max_features=5000, ngram_range=(1, 3), min_df=1):
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
        return self.df_buku.iloc[idx]

