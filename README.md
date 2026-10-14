# Sistema de Recomendação com Machine Learning

![CI](https://github.com/joao-guilherm-dev/sistema-recomendacao-ml/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Projeto em Python que implementa e **compara cinco abordagens de recomendação** no mesmo conjunto de dados, com avaliação por métricas de ranking (Precision@K, Recall@K e NDCG@K).

## Modelos

| Modelo | Ideia |
|---|---|
| **Popularidade** | Baseline: média das notas ponderada pelo número de avaliações |
| **Content-Based** | TF-IDF dos gêneros; recomenda itens parecidos com os que o usuário gostou |
| **Item-Based CF** | Filtragem colaborativa item-item com similaridade de cosseno e k vizinhos |
| **SVD** | Fatoração de matriz (fatores latentes) com `TruncatedSVD` |
| **Híbrido** | Combinação ponderada de SVD + Item-CF + Content-Based |

## Estrutura

```
.
├── main.py                  # executa a comparação e mostra recomendações
├── recomendador/
│   ├── dados.py             # dados sintéticos, MovieLens, split treino/teste
│   ├── modelos.py           # os cinco modelos
│   └── avaliacao.py         # recomendar() e métricas
├── tests/                   # testes com pytest
└── .github/workflows/ci.yml # CI (GitHub Actions)
```

## Como executar

```bash
git clone https://github.com/joao-guilherm-dev/sistema-recomendacao-ml.git
cd sistema-recomendacao-ml

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

Opções:

```bash
python main.py --k 5 --usuario 12                 # top 5 para o usuário 12
python main.py --movielens ml-latest-small        # dados reais do MovieLens
```

Para usar o MovieLens, baixe o `ml-latest-small` em
[grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/), extraia na raiz do projeto e passe a pasta em `--movielens`.

## Resultados (dados sintéticos)

400 usuários, 200 itens, 14.762 avaliações (esparsidade de 81,5%), 20% das notas de cada usuário reservadas para teste. Item relevante = nota ≥ 4.

| Modelo | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Híbrido | 0.042 | 0.168 | 0.119 |
| SVD | 0.043 | 0.162 | 0.107 |
| Popularidade | 0.043 | 0.152 | 0.098 |
| Content-Based | 0.035 | 0.160 | 0.095 |
| Item-Based CF | 0.026 | 0.088 | 0.065 |

> Os dados sintéticos têm pouco sinal e a matriz é bem esparsa, então as diferenças entre os modelos são pequenas. Com dados reais (MovieLens) a separação costuma ser mais clara.

## Testes

```bash
pytest -q
```

Os testes verificam, entre outros pontos, que treino e teste não se sobrepõem, que nenhum modelo recomenda itens já vistos e que o híbrido supera um recomendador aleatório.

## Próximos passos

- Ajuste de hiperparâmetros (`n_fatores`, `k_vizinhos`, pesos do híbrido)
- ALS / feedback implícito (`implicit`)
- API com FastAPI e interface com Streamlit

## Autor

**João Guilherme Soares Oliveira** — [@joao-guilherm-dev](https://github.com/joao-guilherm-dev)

## Licença

MIT — veja [LICENSE](LICENSE).
