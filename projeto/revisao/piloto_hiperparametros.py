"""Piloto separado: seis tasks pré-definidas, árvores e kNN com 3 valores.

Não expande retroativamente a matriz histórica. Usa dados brutos OpenML com
imputação e codificação ajustadas somente no treino de cada fold.
"""
import json
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openml
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from projeto.revisao.experimentos import PROJECT, sha256
from projeto.revisao.pipeline_bruto import build_pipeline


def run():
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = PROJECT/"resultados/revisao_2026/piloto"
    output.mkdir(parents=True, exist_ok=True)
    records, sources, splits = [], {}, []
    metadata = pd.read_csv(PROJECT/"resultados/revisao_2026/openml_tasks.csv").set_index("task_id")
    for tid in config["pilot_tasks"]:
        info = metadata.loc[tid]
        dataset = openml.datasets.get_dataset(int(info.dataset_id), download_data=False)
        x, y, _, _ = dataset.get_data(dataset_format="dataframe", target=info.target_name)
        if y.isna().any():
            raise ValueError(f"Task {tid}: alvo ausente exige política explícita.")
        path = Path(openml.config.get_cache_directory())/f"datasets/{int(info.dataset_id)}/dataset_{int(info.dataset_id)}.pq"
        sources[str(tid)] = {"dataset_id": int(info.dataset_id), "target": info.target_name,
                             "raw_data_hash": sha256(path) if path.exists() else None}
        if pd.Series(y).value_counts().min() < 5:
            raise ValueError(f"Task {tid}: classe com menos de 5 instâncias.")
        cv = list(StratifiedKFold(5, shuffle=True, random_state=config["seed"]).split(x, y))
        for fold, (train, test) in enumerate(cv):
            splits.append({"task_id": tid, "fold": fold, "train_rows": train.tolist(), "test_rows": test.tolist()})
        for classifier, grid in config["pilot_classifiers"].items():
            parameter, values = next(iter(grid.items()))
            for value in values:
                for fold, (train, test) in enumerate(cv):
                    estimator = (DecisionTreeClassifier(max_depth=value, random_state=config["seed"])
                                 if classifier == "DecisionTree" else KNeighborsClassifier(n_neighbors=value))
                    pipe = build_pipeline(estimator)
                    row = dict(task_id=tid, nome_dataset=dataset.name, classifier=classifier,
                               parameter=parameter, value=json.dumps(value), fold=fold,
                               n_train=len(train), n_test=len(test), status="ok")
                    start = time.perf_counter()
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            pipe.fit(x.iloc[train], y.iloc[train])
                            pred = pipe.predict(x.iloc[test])
                        row.update(balanced_accuracy=balanced_accuracy_score(y.iloc[test], pred),
                                   warnings=" | ".join(str(w.message) for w in caught), error="")
                    except Exception as error:
                        row.update(status="erro", balanced_accuracy=np.nan, warnings="", error=str(error))
                    row["tempo_s"] = time.perf_counter() - start
                    records.append(row)
        print(f"Piloto task {tid} concluído", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/"folds.csv", index=False)
    summary = frame.groupby(["task_id", "nome_dataset", "classifier", "parameter", "value"], dropna=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        folds_validos=("balanced_accuracy", "count"), tempo_s=("tempo_s", "sum")).reset_index()
    summary.to_csv(output/"resumo.csv", index=False)
    (output/"splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (output/"manifesto.json").write_text(json.dumps({"config": config, "config_sha256": sha256(config_path),
        "code_sha256": sha256(__file__), "input_sha256": sources, "python": platform.python_version(),
        "preprocessing_code_sha256": sha256(Path(__file__).with_name("pipeline_bruto.py")),
        "limitations": "Exploratory fixed six-task pilot on raw OpenML data with train-only preprocessing; no claim of benchmark-wide improvement."}, indent=2), encoding="utf-8")
    if frame.status.ne("ok").any():
        raise RuntimeError("Há falhas no piloto. Consulte folds.csv.")
    print(f"Piloto: {len(frame)} folds, {len(summary)} configurações/task.", flush=True)


if __name__ == "__main__":
    run()
