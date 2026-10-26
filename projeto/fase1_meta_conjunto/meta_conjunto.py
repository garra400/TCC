"""
Montagem e persistência do meta-conjunto.

Une os resultados de:
  - pipeline_builder  → Y_perf (desempenho dos pipelines)
  - embedding_generator → X_emb (embeddings dos datasets)

e salva o meta-conjunto em dois formatos:
  (a) DataFrame longo (uma linha por dataset-pipeline) — para análise exploratória
  (b) Matrizes X_emb e Y_accuracy (n_datasets × n_pipelines) — para o preditor de ranking

Arquivos gerados:
  data/meta_conjunto/meta_conjunto.pkl   (DataFrame longo)
  data/meta_conjunto/meta_conjunto.csv   (DataFrame longo, CSV expandido)
  data/meta_conjunto/X_emb.npy           (n_datasets × emb_dim)
  data/meta_conjunto/Y_accuracy.npy      (n_datasets × n_pipelines)
  data/meta_conjunto/task_ids.npy        (n_datasets,)
  data/meta_conjunto/pipeline_keys.pkl   (lista de (scaler, selector, classifier))
"""

import os
import pickle
import numpy as np
import pandas as pd
from itertools import product

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import META_DIR, D2V_EMBED_DIM, SCALERS, SELECTORS, CLASSIFIERS
from fase1_meta_conjunto.pipeline_encoder import adicionar_codificacao, expandir_codificacao


def _pipeline_keys_ordenados() -> list[tuple]:
    """Lista ordenada de (scaler, selector, classifier) — 60 combinações."""
    return [(sc, sel, clf)
            for sc, sel, clf in product(SCALERS, SELECTORS, CLASSIFIERS)]


def montar_meta_conjunto(df_pipelines: pd.DataFrame,
                          embeddings: dict[int, np.ndarray]) -> pd.DataFrame:
    """
    Monta o meta-conjunto no formato longo (uma linha por dataset-pipeline).

    Parâmetros
    ----------
    df_pipelines : pd.DataFrame
        Saída de pipeline_builder.executar_todos_pipelines().
    embeddings : dict[int, np.ndarray]
        Saída de embedding_generator.gerar_todos_embeddings().

    Retorna
    -------
    pd.DataFrame com colunas:
        task_id, nome_dataset, scaler, selector, classifier,
        balanced_accuracy_mean, f1_macro_mean, tempo_total_s,
        pipeline_enc (np.ndarray), X_emb (np.ndarray), Y_perf
    """
    ids_validos = set(embeddings.keys())
    df = df_pipelines[df_pipelines["task_id"].isin(ids_validos)].copy()

    df = adicionar_codificacao(df)
    df["X_emb"] = df["task_id"].map(embeddings)
    df["Y_perf"] = df["balanced_accuracy_mean"]

    print(f"Meta-conjunto montado: {len(df)} instâncias, "
          f"{df['task_id'].nunique()} datasets, "
          f"{df.groupby(['scaler','selector','classifier']).ngroups} pipelines únicos.")
    return df


def construir_matrizes(df: pd.DataFrame,
                        embeddings: dict[int, np.ndarray]
                        ) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple]]:
    """
    Constrói as matrizes X_emb e Y_accuracy para o preditor de ranking.

    As matrizes têm uma linha por dataset e uma coluna por pipeline, permitindo
    ao preditor KNN (KDTree) prever a acurácia de todos os pipelines de forma
    multi-output a partir do embedding do dataset.

    Parâmetros
    ----------
    df : pd.DataFrame
        Meta-conjunto longo (saída de montar_meta_conjunto).
    embeddings : dict[int, np.ndarray]
        Mapa task_id → embedding do dataset.

    Retorna
    -------
    X_emb       : np.ndarray  (n_datasets, emb_dim)
    Y_accuracy  : np.ndarray  (n_datasets, n_pipelines)  — NaN onde faltou avaliação
    task_ids    : list[int]   tamanho n_datasets, ordenado
    pipeline_keys : list[tuple]  tamanho n_pipelines, (scaler, selector, classifier)
    """
    task_ids      = sorted(df["task_id"].unique())
    pipeline_keys = _pipeline_keys_ordenados()

    n_datasets  = len(task_ids)
    n_pipelines = len(pipeline_keys)
    emb_dim     = D2V_EMBED_DIM

    X_emb      = np.zeros((n_datasets, emb_dim), dtype=np.float32)
    Y_accuracy = np.full((n_datasets, n_pipelines), np.nan, dtype=np.float32)

    tid_idx  = {tid: i for i, tid in enumerate(task_ids)}
    pipe_idx = {p: j for j, p in enumerate(pipeline_keys)}

    for _, row in df.iterrows():
        i = tid_idx[row["task_id"]]
        j = pipe_idx.get((row["scaler"], row["selector"], row["classifier"]))
        if j is None:
            continue
        if i < len(X_emb) and not isinstance(row["X_emb"], float):
            X_emb[i] = row["X_emb"]
        if not np.isnan(row["Y_perf"]):
            Y_accuracy[i, j] = row["Y_perf"]

    # Sobrescreve X_emb diretamente do dicionário (mais confiável)
    for i, tid in enumerate(task_ids):
        if tid in embeddings:
            X_emb[i] = embeddings[tid]

    n_nan = np.isnan(Y_accuracy).sum()
    print(f"Matrizes construídas: X_emb={X_emb.shape}, Y_accuracy={Y_accuracy.shape} "
          f"({n_nan} células NaN = {n_nan/Y_accuracy.size*100:.1f}%)")
    return X_emb, Y_accuracy, task_ids, pipeline_keys


def salvar_meta_conjunto(df: pd.DataFrame,
                          embeddings: dict[int, np.ndarray] | None = None) -> None:
    """
    Salva o meta-conjunto em pickle, CSV e — se embeddings fornecidos —
    as matrizes numpy para o preditor de ranking.
    """
    pkl_path = os.path.join(META_DIR, "meta_conjunto.pkl")
    csv_path = os.path.join(META_DIR, "meta_conjunto.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Meta-conjunto (longo) salvo em: {pkl_path}")

    df_csv = expandir_codificacao(df.copy())
    emb_matrix = np.stack(df_csv["X_emb"].values)
    emb_cols   = [f"emb_{i}" for i in range(D2V_EMBED_DIM)]
    df_emb     = pd.DataFrame(emb_matrix, columns=emb_cols, index=df_csv.index)
    df_csv     = pd.concat([df_csv.drop(columns=["X_emb"]), df_emb], axis=1)
    df_csv.to_csv(csv_path, index=False)
    print(f"Meta-conjunto (CSV) salvo em: {csv_path}")

    if embeddings is not None:
        X_emb, Y_accuracy, task_ids, pipeline_keys = construir_matrizes(df, embeddings)

        np.save(os.path.join(META_DIR, "X_emb.npy"), X_emb)
        np.save(os.path.join(META_DIR, "Y_accuracy.npy"), Y_accuracy)
        np.save(os.path.join(META_DIR, "task_ids.npy"), np.array(task_ids))
        with open(os.path.join(META_DIR, "pipeline_keys.pkl"), "wb") as f:
            pickle.dump(pipeline_keys, f)
        print(f"Matrizes de ranking salvas em: {META_DIR}")


def carregar_meta_conjunto() -> pd.DataFrame:
    """Carrega o meta-conjunto longo (pickle)."""
    pkl_path = os.path.join(META_DIR, "meta_conjunto.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Meta-conjunto não encontrado em {pkl_path}. "
            "Execute primeiro a Fase 1."
        )
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    print(f"Meta-conjunto carregado: {len(df)} instâncias, "
          f"{df['task_id'].nunique()} datasets.")
    return df


def carregar_matrizes() -> tuple[np.ndarray, np.ndarray, list[int], list[tuple]]:
    """
    Carrega as matrizes X_emb e Y_accuracy salvas pela Fase 1.

    Retorna
    -------
    X_emb, Y_accuracy, task_ids, pipeline_keys
    """
    paths = {
        "X_emb":        os.path.join(META_DIR, "X_emb.npy"),
        "Y_accuracy":   os.path.join(META_DIR, "Y_accuracy.npy"),
        "task_ids":     os.path.join(META_DIR, "task_ids.npy"),
        "pipeline_keys": os.path.join(META_DIR, "pipeline_keys.pkl"),
    }
    for nome, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Arquivo '{nome}' não encontrado em {META_DIR}. "
                "Execute a Fase 1 com embeddings para gerar as matrizes."
            )

    X_emb         = np.load(paths["X_emb"])
    Y_accuracy    = np.load(paths["Y_accuracy"])
    task_ids      = np.load(paths["task_ids"]).tolist()
    with open(paths["pipeline_keys"], "rb") as f:
        pipeline_keys = pickle.load(f)

    print(f"Matrizes carregadas: X_emb={X_emb.shape}, "
          f"Y_accuracy={Y_accuracy.shape}, "
          f"{len(task_ids)} datasets, {len(pipeline_keys)} pipelines.")
    return X_emb, Y_accuracy, task_ids, pipeline_keys
