import numpy as np
import pytest

from recomendador import (ContentBased, Hibrido, ItemBasedCF, Popularidade, SVDRecomendador,
                          avaliar, gerar_dados_sinteticos, matriz_usuario_item, recomendar,
                          separar_treino_teste)


@pytest.fixture(scope="module")
def cenario():
    ratings, itens = gerar_dados_sinteticos(n_usuarios=80, n_itens=60)
    treino, teste = separar_treino_teste(ratings)
    R = matriz_usuario_item(treino, ratings["user_id"].nunique(), len(itens))
    return R, itens, treino, teste


def test_treino_e_teste_nao_se_sobrepoem(cenario):
    _, _, treino, teste = cenario
    pares_treino = set(zip(treino.user_id, treino.item_id))
    pares_teste = set(zip(teste.user_id, teste.item_id))
    assert not pares_treino & pares_teste


@pytest.mark.parametrize("Modelo", [Popularidade, ContentBased, ItemBasedCF, SVDRecomendador, Hibrido])
def test_nao_recomenda_itens_ja_vistos(cenario, Modelo):
    R, itens, _, _ = cenario
    modelo = Modelo().fit(R, itens)
    for u in range(10):
        top = recomendar(modelo, R, u, k=10)
        assert len(top) == 10
        assert not set(top) & set(R[u].indices)


def test_metricas_entre_zero_e_um(cenario):
    R, itens, _, teste = cenario
    metricas = avaliar(Hibrido().fit(R, itens), R, teste, k=10)
    assert all(0 <= v <= 1 for v in metricas.values())


def test_hibrido_supera_aleatorio(cenario):
    R, itens, _, teste = cenario

    class Aleatorio:
        def pontuar(self, u):
            return np.random.default_rng(u).random(R.shape[1])

    h = avaliar(Hibrido().fit(R, itens), R, teste)["NDCG@10"]
    a = avaliar(Aleatorio(), R, teste)["NDCG@10"]
    assert h > a
