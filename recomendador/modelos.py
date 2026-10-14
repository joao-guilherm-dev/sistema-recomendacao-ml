"""Modelos de recomendação.

Todos expõem `.fit(R, itens)` e `.pontuar(user_id) -> np.ndarray[n_itens]`.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42


class Popularidade:
    """Baseline: média das notas ponderada pelo número de avaliações."""
    nome = "Popularidade"

    def fit(self, R, itens):
        contagem = np.asarray((R > 0).sum(axis=0)).ravel()
        soma = np.asarray(R.sum(axis=0)).ravel()
        media = np.divide(soma, contagem, out=np.zeros_like(soma, dtype=float), where=contagem > 0)
        self.score = media * np.log1p(contagem)
        return self

    def pontuar(self, u):
        return self.score


class ContentBased:
    """Recomenda itens parecidos (TF-IDF dos gêneros) com os que o usuário gostou."""
    nome = "Content-Based (TF-IDF)"

    def fit(self, R, itens):
        self.R = R
        self.X = TfidfVectorizer().fit_transform(itens["generos"])
        self.sim = cosine_similarity(self.X)
        return self

    def pontuar(self, u):
        linha = self.R[u]
        vistos, notas = linha.indices, linha.data
        if len(vistos) == 0:
            return np.zeros(self.R.shape[1])
        pesos = notas - notas.mean() + 1e-6
        perfil = self.sim[vistos].T @ pesos
        return perfil / (np.abs(pesos).sum() + 1e-9)


class ItemBasedCF:
    """Filtragem colaborativa item-item com similaridade de cosseno."""
    nome = "Item-Based CF (cosseno)"

    def __init__(self, k_vizinhos=30):
        self.k = k_vizinhos

    def fit(self, R, itens):
        self.R = R
        sim = cosine_similarity(R.T)
        np.fill_diagonal(sim, 0)
        for i in range(sim.shape[0]):
            corte = np.argsort(sim[i])[:-self.k]
            sim[i, corte] = 0
        self.sim = sim
        return self

    def pontuar(self, u):
        linha = self.R[u]
        vistos, notas = linha.indices, linha.data
        if len(vistos) == 0:
            return np.zeros(self.R.shape[1])
        return self.sim[:, vistos] @ (notas - 3.0)


class SVDRecomendador:
    """Fatoração de matriz (fatores latentes) com TruncatedSVD."""
    nome = "SVD (fatores latentes)"

    def __init__(self, n_fatores=20):
        self.n = n_fatores

    def fit(self, R, itens):
        contagem = np.asarray((R > 0).sum(axis=1)).ravel()
        medias = np.divide(np.asarray(R.sum(axis=1)).ravel(), contagem,
                           out=np.full(R.shape[0], 3.0), where=contagem > 0)
        R_c = R.tocoo(copy=True)
        R_c.data = R_c.data - medias[R_c.row]
        svd = TruncatedSVD(n_components=self.n, random_state=SEED)
        U = svd.fit_transform(R_c.tocsr())
        self.pred = U @ svd.components_ + medias[:, None]
        return self

    def pontuar(self, u):
        return self.pred[u]


class Hibrido:
    """Combinação ponderada de SVD, Item-CF e Content-Based."""
    nome = "Híbrido (SVD + Item-CF + Conteúdo)"

    def __init__(self, pesos=(0.5, 0.3, 0.2)):
        self.pesos = pesos
        self.modelos = [SVDRecomendador(), ItemBasedCF(), ContentBased()]

    def fit(self, R, itens):
        for m in self.modelos:
            m.fit(R, itens)
        return self

    @staticmethod
    def _normalizar(x):
        desvio = x.std()
        return (x - x.mean()) / desvio if desvio > 0 else x * 0

    def pontuar(self, u):
        return sum(p * self._normalizar(m.pontuar(u)) for p, m in zip(self.pesos, self.modelos))
