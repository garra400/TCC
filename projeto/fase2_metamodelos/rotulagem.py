"""
Rotulagem binária de qualidade dos pipelines (Y_qual).

Para cada dataset, calcula o limiar de desempenho (por padrão a mediana
de Y_perf entre todos os pipelines do dataset) e rotula:
  - Y_qual = 1 ("adequado")   se Y_perf >= limiar
  - Y_qual = 0 ("inadequado") se Y_perf <  limiar
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LIMIAR_QUALIDADE


def rotular_qualidade(df: pd.DataFrame,
                       coluna_perf: str = "Y_perf",
                       por_dataset: bool = True) -> pd.DataFrame:
    """
    Adiciona a coluna Y_qual ao meta-conjunto.

    Parâmetros
    ----------
    df : pd.DataFrame
        Meta-conjunto com coluna Y_perf.
    coluna_perf : str
        Nome da coluna de desempenho a usar como referência.
    por_dataset : bool
        Se True, o limiar é calculado separadamente para cada dataset
        (abordagem relativa), tornando o rótulo independente da
        dificuldade intrínseca do dataset.
        Se False, usa o limiar global sobre todo o meta-conjunto.

    Retorna
    -------
    pd.DataFrame com coluna Y_qual adicionada.
    """
    df = df.copy()

    if por_dataset:
        df["Y_qual"] = df.groupby("task_id")[coluna_perf].transform(
            lambda s: (s >= s.quantile(LIMIAR_QUALIDADE)).astype(int)
        )
    else:
        limiar_global = df[coluna_perf].quantile(LIMIAR_QUALIDADE)
        df["Y_qual"]  = (df[coluna_perf] >= limiar_global).astype(int)

    n_adequados   = int(df["Y_qual"].sum())
    n_inadequados = int((df["Y_qual"] == 0).sum())
    balanceamento = n_adequados / len(df) * 100
    print(
        f"Rotulagem concluída (limiar={LIMIAR_QUALIDADE*100:.0f}° percentil, "
        f"por_dataset={por_dataset}):\n"
        f"  Adequados:   {n_adequados:>6} ({balanceamento:.1f}%)\n"
        f"  Inadequados: {n_inadequados:>6} ({100-balanceamento:.1f}%)"
    )
    return df


def preparar_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói a matriz de entrada dos metamodelos concatenando
    X_emb e pipeline_enc por linha.

    Parâmetros
    ----------
    df : pd.DataFrame
        Meta-conjunto com colunas X_emb (np.ndarray), pipeline_enc (np.ndarray)
        e Y_qual (int).

    Retorna
    -------
    X_meta : np.ndarray  shape (n, D2V_EMBED_DIM + PIPELINE_ENC_DIM)
    y_qual : np.ndarray  shape (n,)  valores 0 ou 1
    """
    X_meta = np.stack([
        np.concatenate([row["X_emb"], row["pipeline_enc"]])
        for _, row in df.iterrows()
    ])
    y_qual = df["Y_qual"].values.astype(int)
    return X_meta, y_qual
