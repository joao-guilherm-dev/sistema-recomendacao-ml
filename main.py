"""Executa a comparação dos modelos de recomendação.

Exemplos:
    python main.py                                   # dados sintéticos
    python main.py --movielens ml-latest-small       # dados reais
    python main.py --usuario 12 --k 5
"""
import argparse

import pandas as pd

from recomendador import (ContentBased, Hibrido, ItemBasedCF, Popularidade, SVDRecomendador,
                          avaliar, carregar_movielens, gerar_dados_sinteticos,
                          matriz_usuario_item, recomendar, separar_treino_teste)


def main():
    ap = argparse.ArgumentParser(description="Sistema de recomendação com ML")
    ap.add_argument("--movielens", help="pasta do MovieLens (ratings.csv e movies.csv)")
    ap.add_argument("--k", type=int, default=10, help="tamanho do ranking (default: 10)")
    ap.add_argument("--usuario", type=int, default=7, help="usuário do exemplo de recomendação")
    args = ap.parse_args()

    ratings, itens = (carregar_movielens(args.movielens) if args.movielens
                      else gerar_dados_sinteticos())

    n_u, n_i = ratings["user_id"].nunique(), len(itens)
    esparsidade = 1 - len(ratings) / (n_u * n_i)
    print(f"{n_u} usuários | {n_i} itens | {len(ratings)} avaliações | esparsidade {esparsidade:.1%}\n")

    treino, teste = separar_treino_teste(ratings)
    R = matriz_usuario_item(treino, n_u, n_i)

    modelos = [Popularidade(), ContentBased(), ItemBasedCF(), SVDRecomendador(), Hibrido()]
    resultados = {}
    for m in modelos:
        m.fit(R, itens)
        resultados[m.nome] = avaliar(m, R, teste, k=args.k)

    print(pd.DataFrame(resultados).T.round(3).sort_values(f"NDCG@{args.k}", ascending=False).to_string())

    melhor = modelos[-1]
    print(f"\nTop {args.k} para o usuário {args.usuario} ({melhor.nome}):")
    for pos, i in enumerate(recomendar(melhor, R, args.usuario, k=args.k), 1):
        print(f"  {pos}. {itens.loc[i, 'titulo']}  [{itens.loc[i, 'generos']}]")


if __name__ == "__main__":
    main()
