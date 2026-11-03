"""Deterministic representation: 55 statistical descriptors and nine zeros.

This module never selects a neural backend automatically. Query labels are required.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from .pipelines import build_pipeline

D2V_EMBED_DIM = 64


def _embedding_estatistico(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Vetor de 55 descritores, nove zeros e normalização L2 (64 posições).

    Abordagem: para cada estatística de feature (média, desvio, etc.),
    calcula a DISTRIBUIÇÃO dessa estatística sobre todas as features do
    dataset (mínimo, percentil 25, média, percentil 75 e máximo).
    A agregação produz uma dimensão fixa independentemente do número de
    atributos; sua utilidade preditiva precisa ser avaliada empiricamente.

    Referência determinística baseada em meta-features estáticas usadas em
    AutoSklearn, TPOT e outros sistemas AutoML (Feurer et al., 2015;
    Reif et al., 2012).
    """
    # Mantém apenas colunas numéricas (descarta datetime, object, etc.)
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return np.zeros(D2V_EMBED_DIM, dtype=np.float32)
    arr = X_num.values.astype(float)
    y_np = y.values.astype(float)
    N, P = arr.shape
    eps = 1e-8

    # ── 1. Estatísticas por feature ──────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        medias = np.nanmean(arr, axis=0)
        desvios = np.nanstd(arr, axis=0) + eps
        minimos = np.nanmin(arr, axis=0)
        maximos = np.nanmax(arr, axis=0)
        medianas = np.nanmedian(arr, axis=0)

        # Skewness e kurtosis por feature
        skewness = scipy_stats.skew(arr, axis=0, nan_policy="omit")
        kurtosis = scipy_stats.kurtosis(arr, axis=0, nan_policy="omit")

        # Correlação linear de cada feature com y
        corr_y = np.array(
            [
                np.corrcoef(arr[:, i], y_np)[0, 1] if np.std(arr[:, i]) > 0 else 0.0
                for i in range(P)
            ]
        )
        corr_y = np.nan_to_num(corr_y)

        # Correlação de rank (Spearman) de cada feature com y
        corr_rank = np.array(
            [
                scipy_stats.spearmanr(arr[:, i], y_np).correlation
                if np.std(arr[:, i]) > 0
                else 0.0
                for i in range(P)
            ]
        )
        corr_rank = np.nan_to_num(corr_rank)

        # Proporção de valores únicos por feature (0=constante, 1=todos diferentes)
        n_unicos = np.array([len(np.unique(arr[:, i])) / N for i in range(P)])

    # ── 2. Distribuição de cada estatística sobre as features ────────────────
    def dist5(v):
        """Retorna [min, p25, média, p75, max] de um vetor — 5 números."""
        v = np.nan_to_num(v)
        return np.array(
            [v.min(), np.percentile(v, 25), v.mean(), np.percentile(v, 75), v.max()]
        )

    # 9 estatísticas × 5 números = 45 meta-features distribucionais
    bloco_dist = np.concatenate(
        [
            dist5(medias),
            dist5(desvios),
            dist5(minimos),
            dist5(maximos),
            dist5(medianas),
            dist5(skewness),
            dist5(kurtosis),
            dist5(corr_y),
            dist5(corr_rank),
        ]
    )  # shape (45,)

    # ── 3. Meta-features globais do dataset ──────────────────────────────────
    n_classes = len(np.unique(y_np))
    contagens = np.bincount(y.values.astype(int))
    prop_maior = contagens.max() / len(y)
    prop_menor = contagens.min() / len(y)
    entropia_y = scipy_stats.entropy(contagens / len(y) + eps)
    prop_zero_var = np.mean(np.std(arr, axis=0) < eps)
    # Correlação média entre features (absoluta) — capta redundância
    if P > 1:
        corr_matrix = np.abs(np.corrcoef(arr.T))
        np.fill_diagonal(corr_matrix, 0)
        corr_entre_feat = corr_matrix.mean()
    else:
        corr_entre_feat = 0.0
    # Proporção de valores únicos médios (proxies para tipo de feature)
    n_unicos_medio = n_unicos.mean()

    globais = np.array(
        [
            np.log1p(N),  # log do número de instâncias
            np.log1p(P),  # log do número de features
            n_classes,  # número de classes
            prop_maior,  # proporção da classe majoritária
            prop_menor,  # proporção da classe minoritária
            entropia_y,  # entropia do alvo (balanceamento)
            prop_zero_var,  # proporção de features constantes
            corr_entre_feat,  # correlação média entre features
            n_unicos_medio,  # proporção média de valores únicos
            P / (N + eps),  # razão features / instâncias
        ]
    )  # shape (10,)

    # ── 4. Concatena e ajusta para D2V_EMBED_DIM ─────────────────────────────
    vetor = np.concatenate([bloco_dist, globais])  # 45 + 10 = 55

    if len(vetor) >= D2V_EMBED_DIM:
        vetor = vetor[:D2V_EMBED_DIM]
    else:
        vetor = np.pad(vetor, (0, D2V_EMBED_DIM - len(vetor)))

    vetor = np.nan_to_num(vetor)
    norma = np.linalg.norm(vetor)
    return vetor / norma if norma > 0 else vetor


def describe_labeled_dataset(x, y):
    """Describe a complete labeled query; classifier CV uses separate preprocessing."""
    x, y = pd.DataFrame(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    if len(x) != len(y) or not len(x) or x.shape[1] == 0 or y.isna().any():
        raise ValueError(
            "A nonempty feature table and one nonmissing label per row are required."
        )
    x = x.drop(columns=x.select_dtypes(include=["datetime64", "datetimetz"]).columns)
    if not x.shape[1]:
        raise ValueError("No usable attributes remain.")
    target = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    preprocess = build_pipeline(DecisionTreeClassifier()).named_steps["preprocess"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded = preprocess.fit_transform(x, target)
        return _embedding_estatistico(pd.DataFrame(encoded), target)
