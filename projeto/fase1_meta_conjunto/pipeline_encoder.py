"""
Codificação one-hot dos pipelines.

Transforma cada combinação (scaler, selector, classifier) em um
vetor binário pipeline_enc que concatena os one-hots de cada
componente, resultando em um vetor de tamanho:
    len(SCALERS) + len(SELECTORS) + len(CLASSIFIERS)
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCALERS, SELECTORS, CLASSIFIERS

# Tamanho total do vetor one-hot
PIPELINE_ENC_DIM = len(SCALERS) + len(SELECTORS) + len(CLASSIFIERS)


def codificar_pipeline(scaler: str, selector: str, classifier: str) -> np.ndarray:
    """
    Codifica um pipeline como vetor one-hot.

    Parâmetros
    ----------
    scaler, selector, classifier : str
        Nomes dos componentes conforme definidos em config.py.

    Retorna
    -------
    np.ndarray de shape (PIPELINE_ENC_DIM,) com valores 0 ou 1.
    """
    vec_sc  = np.zeros(len(SCALERS),      dtype=np.float32)
    vec_sel = np.zeros(len(SELECTORS),    dtype=np.float32)
    vec_clf = np.zeros(len(CLASSIFIERS),  dtype=np.float32)

    if scaler     in SCALERS:      vec_sc[SCALERS.index(scaler)]           = 1.0
    if selector   in SELECTORS:    vec_sel[SELECTORS.index(selector)]      = 1.0
    if classifier in CLASSIFIERS:  vec_clf[CLASSIFIERS.index(classifier)]  = 1.0

    return np.concatenate([vec_sc, vec_sel, vec_clf])


def decodificar_pipeline(vec: np.ndarray) -> dict:
    """
    Converte um vetor one-hot de volta para nomes de componentes.

    Parâmetros
    ----------
    vec : np.ndarray de shape (PIPELINE_ENC_DIM,)

    Retorna
    -------
    dict com chaves: scaler, selector, classifier
    """
    n_sc  = len(SCALERS)
    n_sel = len(SELECTORS)

    idx_sc  = int(np.argmax(vec[:n_sc]))
    idx_sel = int(np.argmax(vec[n_sc: n_sc + n_sel]))
    idx_clf = int(np.argmax(vec[n_sc + n_sel:]))

    return {
        "scaler":     SCALERS[idx_sc],
        "selector":   SELECTORS[idx_sel],
        "classifier": CLASSIFIERS[idx_clf],
    }


def adicionar_codificacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona a coluna 'pipeline_enc' a um DataFrame de resultados.

    O DataFrame deve conter as colunas: scaler, selector, classifier.

    Retorna o DataFrame original com a coluna pipeline_enc adicionada
    (cada célula é um np.ndarray).
    """
    df = df.copy()
    df["pipeline_enc"] = df.apply(
        lambda row: codificar_pipeline(row["scaler"], row["selector"], row["classifier"]),
        axis=1
    )
    return df


def expandir_codificacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande a coluna pipeline_enc em colunas individuais nomeadas
    (útil para salvar em CSV).

    Retorna DataFrame com colunas enc_0 … enc_{PIPELINE_ENC_DIM-1}
    em vez da coluna pipeline_enc.
    """
    enc_matrix = np.stack(df["pipeline_enc"].values)
    colunas    = [f"enc_{i}" for i in range(PIPELINE_ENC_DIM)]
    df_enc     = pd.DataFrame(enc_matrix, columns=colunas, index=df.index)
    return pd.concat([df.drop(columns=["pipeline_enc"]), df_enc], axis=1)
