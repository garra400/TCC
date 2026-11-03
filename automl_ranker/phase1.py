"""Raw-data cross-validation: train-only transformations and shared task splits."""

from __future__ import annotations
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
import warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from .evaluation import sha256
from .pipelines import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase1"
CACHE = ROOT / "data/cache"
CONFIG = ROOT / "configs/phase1.json"


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def pipeline_keys(config):
    return list(product(config["scalers"], config["selectors"], config["classifiers"]))


def make_classifier(name, config):
    seed = config["seed"]
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=config["rf_estimators"], random_state=seed, n_jobs=1
        )
    if name == "SVM":
        return SVC(kernel="rbf", C=1, gamma="scale", random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=config["classifier_knn_k"], n_jobs=1)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(config["mlp_hidden"]),
            max_iter=config["mlp_max_iter"],
            random_state=seed,
        )
    if name == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=config["decision_tree_max_depth"], random_state=seed
        )
    raise ValueError(name)


def make_scaler(name):
    return {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "None": lambda: None,
    }[name]()


def make_selector(name, n_features, n_train, config):
    k = min(config["selector_max_features"], max(1, n_features - 1), n_train)
    if name == "PCA":
        return PCA(
            n_components=k, svd_solver=config["pca_solver"], random_state=config["seed"]
        )
    if name == "SelectKBest":
        return SelectKBest(f_classif, k=k)
    if name == "None":
        return None
    raise ValueError(name)


def make_splits(y, config):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < max(
        config["min_class_count"], config["folds"]
    ):
        raise ValueError(
            "Each class must support the requested number of stratified folds."
        )
    cv = StratifiedKFold(config["folds"], shuffle=True, random_state=config["seed"])
    return list(cv.split(np.zeros(len(y)), y))


def frame_fingerprint(x):
    # A coluna é renomeada apenas no fingerprint; a transformação não altera dados.
    frame = x.copy().set_axis(range(x.shape[1]), axis=1)
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(str(x.shape).encode() + values.tobytes()).hexdigest()


def run_task(tid, config, output_dir, signature, cache_dir=None):
    """Worker limitado a uma thread BLAS; cada checkpoint é uma task completa."""
    output = Path(output_dir)
    destination = output / "folds" / f"task_{tid}.csv"
    marker = output / "folds" / f"task_{tid}.json"
    with (Path(cache_dir or CACHE) / f"task_{tid}.pkl").open("rb") as stream:
        dataset = pickle.load(stream)
    x, y = dataset["X"], dataset["y"]
    start = time.perf_counter()
    records, split_records = [], []
    ids = {key: index for index, key in enumerate(pipeline_keys(config))}
    splits = make_splits(y, config)
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(splits):
            split_records.append(
                {"fold": fold, "train": train.tolist(), "test": test.tolist()}
            )
            preprocess = build_pipeline(DecisionTreeClassifier()).named_steps[
                "preprocess"
            ]
            with warnings.catch_warnings(record=True) as preprocessing_warnings:
                warnings.simplefilter("always")
                x_train = preprocess.fit_transform(x.iloc[train], y.iloc[train])
                x_test = preprocess.transform(x.iloc[test])
            for scaler_name in config["scalers"]:
                scaler = make_scaler(scaler_name)
                train_scaled = (
                    scaler.fit_transform(x_train) if scaler is not None else x_train
                )
                test_scaled = scaler.transform(x_test) if scaler is not None else x_test
                for selector_name in config["selectors"]:
                    selection_error = None
                    with warnings.catch_warnings(record=True) as selection_warnings:
                        warnings.simplefilter("always")
                        try:
                            selector = make_selector(
                                selector_name, train_scaled.shape[1], len(train), config
                            )
                            train_selected = (
                                selector.fit_transform(train_scaled, y.iloc[train])
                                if selector is not None
                                else train_scaled
                            )
                            test_selected = (
                                selector.transform(test_scaled)
                                if selector is not None
                                else test_scaled
                            )
                        except Exception as error:
                            selection_error = error
                    for classifier_name in config["classifiers"]:
                        row = {
                            "task_id": tid,
                            "nome_dataset": dataset["nome"],
                            "fold": fold,
                            "pipeline_id": ids[
                                (scaler_name, selector_name, classifier_name)
                            ],
                            "scaler": scaler_name,
                            "selector": selector_name,
                            "classifier": classifier_name,
                            "n_train": len(train),
                            "n_test": len(test),
                            "status": "ok",
                            "error": "",
                            "balanced_accuracy": np.nan,
                            "f1_macro": np.nan,
                        }
                        fit_start = time.perf_counter()
                        try:
                            if selection_error is not None:
                                raise selection_error
                            classifier = make_classifier(classifier_name, config)
                            with warnings.catch_warnings(record=True) as model_warnings:
                                warnings.simplefilter("always")
                                classifier.fit(train_selected, y.iloc[train])
                                prediction = classifier.predict(test_selected)
                            row.update(
                                balanced_accuracy=balanced_accuracy_score(
                                    y.iloc[test], prediction
                                ),
                                f1_macro=f1_score(
                                    y.iloc[test],
                                    prediction,
                                    average="macro",
                                    zero_division=0,
                                ),
                                warnings=" | ".join(
                                    sorted(
                                        set(
                                            str(w.message)
                                            for w in list(preprocessing_warnings)
                                            + list(selection_warnings)
                                            + list(model_warnings)
                                        )
                                    )
                                ),
                            )
                        except Exception as error:
                            row.update(
                                status="erro",
                                error=f"{type(error).__name__}: {error}",
                                warnings="",
                            )
                        row["tempo_s"] = time.perf_counter() - fit_start
                        records.append(row)
    frame = pd.DataFrame(records).sort_values(["pipeline_id", "fold"])
    temporary = destination.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(destination)
    write_json(output / "splits" / f"task_{tid}.json", split_records)
    write_json(
        marker,
        {
            "task_id": tid,
            "signature": signature,
            "folds_sha256": sha256(destination),
            "cache_sha256": sha256(Path(cache_dir or CACHE) / f"task_{tid}.pkl"),
            "raw_sha256": dataset["raw_sha256"],
            "n_records": len(frame),
            "n_errors": int(frame.status.ne("ok").sum()),
            "tempo_s": time.perf_counter() - start,
        },
    )
    return {
        "task_id": tid,
        "errors": int(frame.status.ne("ok").sum()),
        "seconds": time.perf_counter() - start,
    }


def execution_signature(config_path):
    parts = [
        sha256(config_path),
        sha256(__file__),
        sha256(Path(__file__).with_name("representation.py")),
        sha256(Path(__file__).with_name("pipelines.py")),
    ]
    parts += [
        importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn"]
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def run(output=OUTPUT, config_path=CONFIG, workers=None, cache_dir=None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signature = execution_signature(config_path)
    catalog = pd.read_csv(output / "datasets.csv")
    ids = catalog.loc[catalog.elegivel, "task_id"].sort_values().astype(int).tolist()
    for directory in ["folds", "splits"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    pending = []
    for tid in ids:
        marker = output / "folds" / f"task_{tid}.json"
        checkpoint = output / "folds" / f"task_{tid}.csv"
        if marker.exists() and checkpoint.exists():
            info = json.loads(marker.read_text(encoding="utf-8"))
            if info["signature"] != signature or info["folds_sha256"] != sha256(
                checkpoint
            ):
                raise ValueError(
                    f"Checkpoint incompatível: task {tid}. Use outro diretório para outra configuração."
                )
            if info.get("cache_sha256") != sha256(
                Path(cache_dir or CACHE) / f"task_{tid}.pkl"
            ):
                raise ValueError(
                    f"Input data changed for task {tid}; use a new output directory."
                )
        else:
            pending.append(tid)
    start = time.perf_counter()
    # Tarefas mais custosas entram primeiro, reduzindo a cauda ao final.
    size = catalog.set_index("task_id").eval("n_instancias * n_atributos")
    pending.sort(key=lambda tid: size.loc[tid], reverse=True)
    write_json(
        output / "progresso.json",
        {
            "status": "running",
            "total": len(ids),
            "complete": len(ids) - len(pending),
            "signature": signature,
            "workers": workers or config["workers"],
        },
    )
    print(
        f"Fase 1: {len(pending)} tarefas pendentes, {workers or config['workers']} workers, sem subamostragem.",
        flush=True,
    )
    completed = len(ids) - len(pending)
    # O ambiente é aplicado antes de criar os processos Windows.
    os.environ.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )

    def record(result):
        nonlocal completed
        completed += 1
        write_json(
            output / "progresso.json",
            {
                "status": "running",
                "total": len(ids),
                "complete": completed,
                "last_task": result,
                "seconds": time.perf_counter() - start,
                "signature": signature,
            },
        )
        print(
            f"{completed}/{len(ids)} task {result['task_id']}: {result['seconds']:.1f}s, {result['errors']} falhas",
            flush=True,
        )

    count = workers or config["workers"]
    if count < 1:
        raise ValueError("workers must be positive.")
    if count == 1:
        for tid in pending:
            record(
                run_task(tid, config, str(output), signature, str(cache_dir or CACHE))
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=count) as pool:
            futures = {
                pool.submit(
                    run_task,
                    tid,
                    config,
                    str(output),
                    signature,
                    str(cache_dir or CACHE),
                ): tid
                for tid in pending
            }
            for future in concurrent.futures.as_completed(futures):
                record(future.result())
    aggregate(output, config_path)


def aggregate(output=OUTPUT, config_path=CONFIG):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog = pd.read_csv(output / "datasets.csv")
    ids = np.load(output / "task_ids.npy")
    frames = [
        pd.read_csv(output / "folds" / f"task_{tid}.csv", keep_default_na=False)
        for tid in ids
    ]
    folds = pd.concat(frames, ignore_index=True)
    for column in ["balanced_accuracy", "f1_macro", "tempo_s"]:
        folds[column] = pd.to_numeric(folds[column], errors="coerce")
    expected = len(ids) * len(pipeline_keys(config)) * config["folds"]
    if (
        len(folds) != expected
        or folds.duplicated(["task_id", "pipeline_id", "fold"]).any()
    ):
        raise ValueError("Cobertura de folds incorreta.")
    summary = (
        folds.groupby(
            [
                "task_id",
                "nome_dataset",
                "pipeline_id",
                "scaler",
                "selector",
                "classifier",
            ],
            dropna=False,
        )
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            tempo_total_s=("tempo_s", "sum"),
            folds_validos=("balanced_accuracy", "count"),
        )
        .reset_index()
    )
    # Nenhum score parcial é promovido a alvo se algum dos cinco folds falhar.
    summary.loc[summary.folds_validos.ne(config["folds"]), "balanced_accuracy_mean"] = (
        np.nan
    )
    summary.to_csv(output / "pipelines_raw.csv", index=False, lineterminator="\n")
    y = (
        summary.pivot(
            index="task_id", columns="pipeline_id", values="balanced_accuracy_mean"
        )
        .reindex(index=ids, columns=range(len(pipeline_keys(config))))
        .to_numpy()
    )
    np.save(output / "Y_accuracy.npy", y)
    folds[folds.status.ne("ok")].to_csv(
        output / "falhas.csv", index=False, lineterminator="\n"
    )
    convergence = (
        folds.warnings.fillna("").astype(str).str.contains("converg", case=False)
    )
    folds.loc[convergence, ["task_id", "pipeline_id", "fold", "warnings"]].to_csv(
        output / "avisos_convergencia.csv", index=False, lineterminator="\n"
    )
    write_json(
        output / "manifesto.json",
        {
            "status": "complete",
            "config": config,
            "config_sha256": sha256(config_path),
            "code_sha256": sha256(__file__),
            "preprocessing_sha256": sha256(Path(__file__).with_name("pipelines.py")),
            "python": platform.python_version(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scipy", "scikit-learn", "openml"]
            },
            "n_tasks": len(ids),
            "n_groups": catalog.loc[catalog.elegivel, "grupo"].nunique(),
            "n_pipelines": len(pipeline_keys(config)),
            "n_fits": len(folds),
            "failed_fits": int(folds.status.ne("ok").sum()),
            "convergence_warnings": int(convergence.sum()),
            "missing_scores": int(np.isnan(y).sum()),
            "artifacts_sha256": {
                name: sha256(output / name)
                for name in [
                    "datasets.csv",
                    "task_ids.npy",
                    "X_emb.npy",
                    "Y_accuracy.npy",
                    "pipelines_raw.csv",
                ]
            },
            "sampling": "all rows, same folds for all classifiers",
            "preprocessing": "fitted exclusively on each training fold",
        },
    )
    write_json(
        output / "progresso.json",
        {"status": "complete", "total": len(ids), "complete": len(ids)},
    )
    print(
        f"Fase 1 concluída: {len(folds)} ajustes, {int(folds.status.ne('ok').sum())} falhas.",
        flush=True,
    )
