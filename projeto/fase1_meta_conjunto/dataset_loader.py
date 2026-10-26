"""
Carregamento de datasets do OpenML para construção do meta-conjunto.

Filtros aplicados (nesta ordem, do mais barato ao mais caro):
  1. Metadados OpenML (n_instances via API, sem download) — evita baixar
     datasets que seriam filtrados por MIN/MAX_INSTANCIAS.
  2. MIN_INSTANCIAS — descarta datasets pequenos após download.
  3. MAX_INSTANCIAS — descarta datasets muito grandes (custo CV).

Cache de skip: datasets fora do intervalo [MIN, MAX] são marcados com um
arquivo .skip para não serem re-consultados nas próximas execuções.
"""

import os
import pickle
import openml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OPENML_TASK_IDS, MIN_INSTANCIAS, MAX_INSTANCIAS, DATASETS_DIR


def _n_instancias_openml(task_id: int) -> int | None:
    """
    Retorna o número de instâncias de uma tarefa via metadados OpenML,
    sem baixar o dataset completo. Retorna None se não conseguir.
    """
    try:
        task    = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(
            task.dataset_id,
            download_data=False,
            download_qualities=True,
        )
        quals = dataset.qualities
        if quals and "NumberOfInstances" in quals:
            return int(quals["NumberOfInstances"])
    except Exception:
        pass
    return None


def carregar_dataset_openml(task_id: int) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Baixa e prepara um dataset do OpenML.

    Retorna
    -------
    X : pd.DataFrame  — features numéricas (categóricas codificadas como int)
    y : pd.Series     — alvo codificado como int
    nome : str        — nome do dataset
    """
    task    = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    nome    = dataset.name

    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    # Remove colunas de data/hora (não numéricas e não codificáveis)
    dt_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns
    if len(dt_cols) > 0:
        X = X.drop(columns=dt_cols)

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    X = X.fillna(X.median(numeric_only=True))

    return X, y, nome


def carregar_todos(task_ids: list[int] | None = None,
                   usar_cache: bool = True) -> list[dict]:
    """
    Carrega os datasets válidos de OPENML_TASK_IDS.

    Estratégia de eficiência:
      - Consulta metadados (n_instances) antes de baixar → evita download
        de datasets fora do intervalo [MIN_INSTANCIAS, MAX_INSTANCIAS].
      - Salva arquivo .skip para tasks descartadas pelos metadados,
        evitando re-consulta em execuções futuras.
      - Cache pkl para datasets válidos: leitura instantânea no próximo run.

    Retorna
    -------
    list[dict] com chaves: task_id, nome, X, y
    """
    if task_ids is None:
        task_ids = OPENML_TASK_IDS

    # Remove duplicatas mantendo ordem
    task_ids = list(dict.fromkeys(task_ids))

    datasets = []
    for tid in tqdm(task_ids, desc="Carregando datasets"):
        cache_path = os.path.join(DATASETS_DIR, f"task_{tid}.pkl")
        skip_path  = os.path.join(DATASETS_DIR, f"task_{tid}.skip")

        # ── Cache pkl (dataset válido já baixado) ─────────────────────────
        if usar_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                entrada = pickle.load(f)
            # Verifica filtro de tamanho mesmo para datasets em cache
            # (garante que datasets cacheados antes do MAX_INSTANCIAS sejam filtrados)
            n = len(entrada["X"])
            if n < MIN_INSTANCIAS or n > MAX_INSTANCIAS:
                print(f"  [SKIP-CACHE] task_id={tid} ({entrada['nome']}): "
                      f"{n} inst fora de [{MIN_INSTANCIAS}, {MAX_INSTANCIAS}]")
                os.remove(cache_path)          # remove cache inválido
                open(skip_path, "w").close()   # marca skip
                continue
            datasets.append(entrada)
            continue

        # ── Cache skip (dataset já descartado anteriormente) ──────────────
        if usar_cache and os.path.exists(skip_path):
            continue

        # ── Pré-filtro por metadados (sem download) ────────────────────────
        n_inst = _n_instancias_openml(tid)
        if n_inst is not None:
            if n_inst < MIN_INSTANCIAS or n_inst > MAX_INSTANCIAS:
                motivo = f"n={n_inst} fora de [{MIN_INSTANCIAS}, {MAX_INSTANCIAS}]"
                print(f"  [SKIP-META] task_id={tid}: {motivo}")
                if usar_cache:
                    open(skip_path, "w").close()   # marca como skip
                continue

        # ── Download completo ─────────────────────────────────────────────
        try:
            X, y, nome = carregar_dataset_openml(tid)
        except Exception as e:
            print(f"  [ERRO] task_id={tid}: {e}")
            continue

        # Filtro pós-download (caso metadados estivessem ausentes/errados)
        if len(X) < MIN_INSTANCIAS:
            print(f"  [SKIP] task_id={tid} ({nome}): {len(X)} < MIN ({MIN_INSTANCIAS})")
            if usar_cache:
                open(skip_path, "w").close()
            continue
        if len(X) > MAX_INSTANCIAS:
            print(f"  [SKIP] task_id={tid} ({nome}): {len(X)} > MAX ({MAX_INSTANCIAS})")
            if usar_cache:
                open(skip_path, "w").close()
            continue

        entrada = {"task_id": tid, "nome": nome, "X": X, "y": y}
        if usar_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(entrada, f)

        datasets.append(entrada)

    print(f"\n{len(datasets)} datasets carregados com sucesso.")
    return datasets
