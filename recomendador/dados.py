"""Carregamento, geração e divisão dos dados."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEED = 42
GENEROS = ["acao", "comedia", "drama", "ficcao", "terror", "romance", "animacao", "documentario"]


def gerar_dados_sinteticos(n_usuarios=400, n_itens=200, seed=SEED):
    """Gera filmes com gêneros e avaliações (1-5) baseadas em gostos por gênero."""
    rng = np.random.default_rng(seed)

    itens = []
    for i in range(n_itens):
        gs = rng.choice(GENEROS, size=rng.integers(1, 4), replace=False)
        itens.append({"item_id": i, "titulo": f"Filme {i:03d}", "generos": " ".join(gs)})
    itens = pd.DataFrame(itens)

    afinidade = rng.normal(0, 1, size=(n_usuarios, len(GENEROS)))
    genero_idx = {g: k for k, g in enumerate(GENEROS)}
    popularidade = rng.beta(1.2, 4, size=n_itens)

    linhas = []
    for u in range(n_usuarios):
        n_avaliacoes = rng.integers(15, 60)
        escolhidos = rng.choice(n_itens, size=n_avaliacoes, replace=False,
                                p=popularidade / popularidade.sum())
        for i in escolhidos:
            gs = itens.loc[i, "generos"].split()
            gosto = np.mean([afinidade[u, genero_idx[g]] for g in gs])
            nota = np.clip(np.round(3 + gosto * 1.2 + rng.normal(0, 0.6)), 1, 5)
            linhas.append((u, i, nota))

    ratings = pd.DataFrame(linhas, columns=["user_id", "item_id", "rating"])
    return ratings, itens


def carregar_movielens(pasta="ml-latest-small"):
    """Carrega o MovieLens (https://grouplens.org/datasets/movielens/).

    A pasta deve conter `ratings.csv` e `movies.csv`.
    """
    r = pd.read_csv(f"{pasta}/ratings.csv").rename(
        columns={"userId": "user_id", "movieId": "item_id"})[["user_id", "item_id", "rating"]]
    m = pd.read_csv(f"{pasta}/movies.csv").rename(
        columns={"movieId": "item_id", "title": "titulo", "genres": "generos"})
    m["generos"] = m["generos"].str.replace("|", " ", regex=False).str.replace("-", "", regex=False)

    r["user_id"] = r["user_id"].astype("category").cat.codes
    m = m[m["item_id"].isin(r["item_id"])].reset_index(drop=True)
    mapa = {old: new for new, old in enumerate(m["item_id"])}
    r = r[r["item_id"].isin(mapa)].copy()
    r["item_id"] = r["item_id"].map(mapa)
    m["item_id"] = m.index
    return r.reset_index(drop=True), m[["item_id", "titulo", "generos"]]


def separar_treino_teste(ratings, frac_teste=0.2, seed=SEED):
    """Separa uma fração dos ratings de CADA usuário para teste."""
    rng = np.random.default_rng(seed)
    teste_idx = []
    for _, grupo in ratings.groupby("user_id"):
        n = max(1, int(len(grupo) * frac_teste))
        teste_idx.extend(rng.choice(grupo.index, size=n, replace=False))
    teste = ratings.loc[teste_idx]
    treino = ratings.drop(index=teste_idx)
    return treino, teste


def matriz_usuario_item(ratings, n_usuarios, n_itens):
    """Matriz esparsa usuários x itens com as notas."""
    return csr_matrix((ratings["rating"], (ratings["user_id"], ratings["item_id"])),
                      shape=(n_usuarios, n_itens))
