"""
Geração de embeddings de datasets para o meta-conjunto.

Estratégia de prioridade:
  1. Dataset2Vec (Jomaa et al., 2021) via TensorFlow — requer Python ≤ 3.12
  2. Embedding estatístico avançado (implementação própria, PyTorch-free)
     baseado em meta-features distribucionais (Reif et al., 2012;
     Feurer et al., 2015 / AutoSklearn), produzindo 64 dimensões.

Sobre a compatibilidade do Dataset2Vec:
  O repositório original usa TensorFlow, que não possui distribuição para
  Python 3.13+. Para usar o D2V real, crie um virtualenv com Python 3.11 ou
  3.12 e instale tensorflow nele. O caminho do repositório é configurável
  via D2V_REPO_PATH em config.py.

Embedding estatístico avançado (modo atual):
  Captura a DISTRIBUIÇÃO de meta-features por feature (não apenas médias
  globais), o que é invariante ao número de features e produz representações
  mais ricas do que estatísticas simples. Inclui:
    - Estatísticas de 1ª e 2ª ordem por feature (média, desvio, min, max,
      mediana, percentis 25/75, skewness, kurtosis)
    - Correlação de cada feature com o alvo (linear + rank)
    - Taxa de valores únicos e variância zero por feature
    - Correlação média entre features (estrutura de redundância)
    - Estatísticas globais do dataset (n, p, classes, desbalanceamento)
  Total: ~60 meta-features → vetor de D2V_EMBED_DIM=64 (normalizado, unit norm)

Referências:
  Jomaa, H.S. et al. (2021). Dataset2Vec. DAMI, 35, 964–985.
  Reif, M. et al. (2012). Meta2-features. ECML/PKDD.
  Feurer, M. et al. (2015). Auto-sklearn. NeurIPS.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import D2V_REPO_PATH, D2V_EMBED_DIM

# ─── Tenta carregar Dataset2Vec (TF — requer Python ≤ 3.12) ─────────────────
_D2V_DISPONIVEL = False
_D2V_MODEL      = None

def _tentar_carregar_d2v():
    global _D2V_DISPONIVEL, _D2V_MODEL
    try:
        import tensorflow as tf
        import json

        log_dir = os.path.join(D2V_REPO_PATH, "checkpoints", "split-0")
        config_path = os.path.join(log_dir, "configuration.txt")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {config_path}")

        sys.path.insert(0, D2V_REPO_PATH)
        from modules import FunctionF, FunctionH, FunctionG, PoolF, PoolG  # type: ignore

        configuration = json.load(open(config_path, "r"))
        batch_size = configuration["batch_size"]

        x       = tf.keras.Input(shape=(2), dtype=tf.float32)
        nclasses= tf.keras.Input(shape=(batch_size,), dtype=tf.int32, batch_size=1)
        nfeature= tf.keras.Input(shape=(batch_size,), dtype=tf.int32, batch_size=1)
        ninstanc= tf.keras.Input(shape=(batch_size,), dtype=tf.int32, batch_size=1)

        layer = FunctionF(
            units=configuration["units_f"], nhidden=configuration["nhidden_f"],
            nonlinearity=configuration["nonlinearity_d2v"],
            architecture=configuration["architecture_f"],
            resblocks=configuration["resblocks_f"], trainable=False)(x)
        layer = PoolF(units=configuration["units_f"])(
            layer, nclasses[0], nfeature[0], ninstanc[0])
        layer = FunctionG(
            units=configuration["units_g"], nhidden=configuration["nhidden_g"],
            nonlinearity=configuration["nonlinearity_d2v"],
            architecture=configuration["architecture_g"], trainable=False)(layer)
        layer = PoolG(units=configuration["units_g"])(layer, nclasses[0], nfeature[0])
        mf    = FunctionH(
            units=configuration["units_h"], nhidden=configuration["nhidden_h"],
            nonlinearity=configuration["nonlinearity_d2v"],
            architecture=configuration["architecture_h"], trainable=False,
            resblocks=configuration["resblocks_h"])(layer)

        model = tf.keras.Model(inputs=[x, nclasses, nfeature, ninstanc], outputs=mf)
        model.load_weights(os.path.join(log_dir, "weights"))

        _D2V_MODEL      = (model, configuration)
        _D2V_DISPONIVEL = True
        print("[D2V] Dataset2Vec carregado com pesos pré-treinados.")

    except Exception as e:
        warnings.warn(
            f"Dataset2Vec indisponível ({e}). "
            "Usando embedding estatístico avançado (publicável para Python ≥ 3.13).",
            stacklevel=3
        )

_tentar_carregar_d2v()


# ─── Embedding via D2V real ───────────────────────────────────────────────────

def _embedding_d2v(X: pd.DataFrame, y: pd.Series, n_batches: int = 10) -> np.ndarray:
    """Gera embedding usando Dataset2Vec com pesos pré-treinados."""
    import sys
    sys.path.insert(0, D2V_REPO_PATH)
    from dummdataset import Dataset as D2VDataset, flatten  # type: ignore
    from sampling import TestSampling, Batch               # type: ignore
    import tensorflow as tf

    model, configuration = _D2V_MODEL

    # Wrapper do dataset no formato esperado pelo D2V
    class _DS:
        def __init__(self, Xarr, yarr):
            self.X       = Xarr.astype(float)
            self.y       = yarr.reshape(-1, 1).astype(int)
            self.ninstanc = min(256, len(Xarr))
            self.nclasses = min(3, len(np.unique(yarr)))
            self.nfeature = min(16, Xarr.shape[1])

        def instances(self, ninstanc=None, nclasses=None, nfeature=None):
            from dummdataset import flatten  # type: ignore
            from sklearn.preprocessing import OneHotEncoder
            ni = ninstanc or self.ninstanc
            nc = nclasses or self.nclasses
            nf = nfeature or self.nfeature
            idx_i = np.random.choice(len(self.X), size=min(ni, len(self.X)), replace=False)
            idx_f = np.random.choice(self.X.shape[1], size=min(nf, self.X.shape[1]), replace=False)
            n_cl  = len(np.unique(self.y))
            idx_c = np.random.choice(n_cl, size=min(nc, n_cl), replace=False)
            ohc   = OneHotEncoder(categories=[range(n_cl)], sparse_output=False)
            labels = ohc.fit_transform(self.y)
            data  = self.X[np.ix_(idx_i, idx_f)]
            labs  = labels[np.ix_(idx_i, idx_c)]
            x_flat = flatten(data, labs)
            return (x_flat[:, None, :],
                    np.array([[data.shape[0]]]),
                    np.array([[data.shape[1]]]),
                    np.array([[labs.shape[1]]]))

    ds      = _DS(X.values, y.values)
    sampler = TestSampling(dataset=ds)
    batch   = Batch(configuration["batch_size"])

    embeddings = []
    for _ in range(n_batches):
        batch = sampler.sample_from_one_dataset(batch)
        batch.collect()
        embeddings.append(model(batch.input).numpy())

    emb = np.vstack(embeddings).mean(axis=0)
    # Ajusta para D2V_EMBED_DIM (D2V original produz units_h=32 dims)
    if len(emb) < D2V_EMBED_DIM:
        emb = np.pad(emb, (0, D2V_EMBED_DIM - len(emb)))
    else:
        emb = emb[:D2V_EMBED_DIM]
    norma = np.linalg.norm(emb)
    return emb / norma if norma > 0 else emb


# ─── Embedding estatístico avançado (fallback publicável) ────────────────────

def _embedding_estatistico(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Embedding de 64 dimensões baseado em meta-features distribucionais.

    Abordagem: para cada estatística de feature (média, desvio, etc.),
    calcula a DISTRIBUIÇÃO dessa estatística sobre todas as features do
    dataset (min, max, média, mediana, desvio padrão). Isso torna o
    embedding invariante ao número de features e captura a "textura"
    do dataset de forma rica.

    Publicável: segue a tradição de meta-features estáticas usadas em
    AutoSklearn, TPOT e outros sistemas AutoML (Feurer et al., 2015;
    Reif et al., 2012).
    """
    # Mantém apenas colunas numéricas (descarta datetime, object, etc.)
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return np.zeros(D2V_EMBED_DIM, dtype=np.float32)
    arr  = X_num.values.astype(float)
    y_np = y.values.astype(float)
    N, P = arr.shape
    eps  = 1e-8

    # ── 1. Estatísticas por feature ──────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        medias    = np.nanmean(arr, axis=0)
        desvios   = np.nanstd(arr,  axis=0) + eps
        minimos   = np.nanmin(arr,  axis=0)
        maximos   = np.nanmax(arr,  axis=0)
        medianas  = np.nanmedian(arr, axis=0)
        p25       = np.nanpercentile(arr, 25, axis=0)
        p75       = np.nanpercentile(arr, 75, axis=0)

        # Skewness e kurtosis por feature
        skewness  = scipy_stats.skew(arr, axis=0, nan_policy="omit")
        kurtosis  = scipy_stats.kurtosis(arr, axis=0, nan_policy="omit")

        # Correlação linear de cada feature com y
        corr_y = np.array([
            np.corrcoef(arr[:, i], y_np)[0, 1]
            if np.std(arr[:, i]) > 0 else 0.0
            for i in range(P)
        ])
        corr_y = np.nan_to_num(corr_y)

        # Correlação de rank (Spearman) de cada feature com y
        corr_rank = np.array([
            scipy_stats.spearmanr(arr[:, i], y_np).correlation
            if np.std(arr[:, i]) > 0 else 0.0
            for i in range(P)
        ])
        corr_rank = np.nan_to_num(corr_rank)

        # Proporção de valores únicos por feature (0=constante, 1=todos diferentes)
        n_unicos = np.array([len(np.unique(arr[:, i])) / N for i in range(P)])

    # ── 2. Distribuição de cada estatística sobre as features ────────────────
    def dist5(v):
        """Retorna [min, p25, média, p75, max] de um vetor — 5 números."""
        v = np.nan_to_num(v)
        return np.array([v.min(), np.percentile(v, 25),
                         v.mean(), np.percentile(v, 75), v.max()])

    # 9 estatísticas × 5 números = 45 meta-features distribucionais
    bloco_dist = np.concatenate([
        dist5(medias),
        dist5(desvios),
        dist5(minimos),
        dist5(maximos),
        dist5(medianas),
        dist5(skewness),
        dist5(kurtosis),
        dist5(corr_y),
        dist5(corr_rank),
    ])  # shape (45,)

    # ── 3. Meta-features globais do dataset ──────────────────────────────────
    n_classes      = len(np.unique(y_np))
    contagens      = np.bincount(y.values.astype(int))
    prop_maior     = contagens.max() / len(y)
    prop_menor     = contagens.min() / len(y)
    entropia_y     = scipy_stats.entropy(contagens / len(y) + eps)
    prop_zero_var  = np.mean(np.std(arr, axis=0) < eps)
    # Correlação média entre features (absoluta) — capta redundância
    if P > 1:
        corr_matrix    = np.abs(np.corrcoef(arr.T))
        np.fill_diagonal(corr_matrix, 0)
        corr_entre_feat = corr_matrix.mean()
    else:
        corr_entre_feat = 0.0
    # Proporção de valores únicos médios (proxies para tipo de feature)
    n_unicos_medio = n_unicos.mean()

    globais = np.array([
        np.log1p(N),            # log do número de instâncias
        np.log1p(P),            # log do número de features
        n_classes,              # número de classes
        prop_maior,             # proporção da classe majoritária
        prop_menor,             # proporção da classe minoritária
        entropia_y,             # entropia do alvo (balanceamento)
        prop_zero_var,          # proporção de features constantes
        corr_entre_feat,        # correlação média entre features
        n_unicos_medio,         # proporção média de valores únicos
        P / (N + eps),          # razão features / instâncias
    ])  # shape (10,)

    # ── 4. Concatena e ajusta para D2V_EMBED_DIM ─────────────────────────────
    vetor = np.concatenate([bloco_dist, globais])   # 45 + 10 = 55

    if len(vetor) >= D2V_EMBED_DIM:
        vetor = vetor[:D2V_EMBED_DIM]
    else:
        vetor = np.pad(vetor, (0, D2V_EMBED_DIM - len(vetor)))

    vetor = np.nan_to_num(vetor)
    norma = np.linalg.norm(vetor)
    return vetor / norma if norma > 0 else vetor


# ─── Interface pública ────────────────────────────────────────────────────────

def gerar_embedding(X: pd.DataFrame,
                    y: pd.Series,
                    n_batches: int = 10) -> np.ndarray:
    """
    Gera o embedding X_emb para um dataset.

    Usa Dataset2Vec real se disponível (requer TF + Python ≤ 3.12).
    Caso contrário usa o embedding estatístico avançado (64 dims).

    Retorna
    -------
    np.ndarray de shape (D2V_EMBED_DIM,), norma unitária.
    """
    if _D2V_DISPONIVEL:
        return _embedding_d2v(X, y, n_batches=n_batches)
    return _embedding_estatistico(X, y)


def gerar_todos_embeddings(datasets: list[dict]) -> dict[int, np.ndarray]:
    """Gera embeddings para todos os datasets da lista."""
    from tqdm import tqdm
    modo = "Dataset2Vec" if _D2V_DISPONIVEL else "Estatístico Avançado"
    resultado = {}
    for ds in tqdm(datasets, desc=f"Gerando embeddings ({modo})"):
        emb = gerar_embedding(ds["X"], ds["y"])
        resultado[ds["task_id"]] = emb
        print(f"  task_id={ds['task_id']} ({ds['nome']}): "
              f"shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    return resultado
