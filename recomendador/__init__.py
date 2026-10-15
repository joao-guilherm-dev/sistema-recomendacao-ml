"""Pacote de recomendação de filmes."""

from .modelos import (
    ContentBased,
    Hibrido,
    ItemBasedCF,
    Popularidade,
    SVDRecomendador,
)
from .avaliacao import avaliar, recomendar
from .dados import (
    gerar_dados_sinteticos,
    matriz_usuario_item,
    separar_treino_teste,
)
