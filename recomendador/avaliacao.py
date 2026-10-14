"""Geração de recomendações e métricas de avaliação."""
import numpy as np


def recomendar(modelo, R, u, k=10):
    """Top-K itens que o usuário ainda NÃO viu."""
    scores = np.array(modelo.pontuar(u), dtype=float)
    scores[R[u].indices] = -np.inf
    return np.argsort(scores)[::-1][:k]


def avaliar(modelo, R_treino, teste, k=10, limiar=4):
    """Precision@K, Recall@K e NDCG@K. Item relevante = nota >= limiar no teste."""
    precs, recs, ndcgs = [], [], []
    for u, grupo in teste.groupby("user_id"):
        relevantes = set(grupo.loc[grupo["rating"] >= limiar, "item_id"])
        if not relevantes:
            continue
        top = recomendar(modelo, R_treino, u, k)
        acertos = np.array([i in relevantes for i in top], dtype=float)
        precs.append(acertos.mean())
        recs.append(acertos.sum() / len(relevantes))
        dcg = (acertos / np.log2(np.arange(2, k + 2))).sum()
        idcg = (1 / np.log2(np.arange(2, min(len(relevantes), k) + 2))).sum()
        ndcgs.append(dcg / idcg)
    return {f"Precision@{k}": np.mean(precs), f"Recall@{k}": np.mean(recs), f"NDCG@{k}": np.mean(ndcgs)}
