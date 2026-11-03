"""Prepare OpenML tasks and metadata without using pipeline outcomes."""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .evaluation import sha256
from .io import REFERENCE, write_json
from .phase1 import frame_fingerprint, pipeline_keys
from .representation import describe_labeled_dataset


def prepare(config, output, cache, manifest=REFERENCE / "datasets.csv", task_ids=None):
    try:
        import openml
    except ImportError as error:
        raise RuntimeError(
            'Install the OpenML extra first: python -m pip install -e ".[openml]"'
        ) from error
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str((cache / "openml").resolve()))
    metadata = pd.read_csv(manifest)
    if metadata.task_id.duplicated().any():
        raise ValueError("The input manifest must contain unique task IDs.")
    if task_ids:
        missing = set(task_ids) - set(metadata.task_id)
        if missing:
            raise ValueError(f"Task IDs absent from the manifest: {sorted(missing)}")
        metadata = metadata[metadata.task_id.isin(task_ids)].copy()
    if metadata.empty:
        raise ValueError("The task selection is empty.")
    if "grupo" not in metadata:
        metadata["grupo"] = metadata.dataset_id.map(lambda value: f"dataset_{value}")
    rows, embeddings, quality = [], {}, []
    for number, info in enumerate(metadata.itertuples(), 1):
        tid, did = int(info.task_id), int(info.dataset_id)
        row = {
            "task_id": tid,
            "dataset_id": did,
            "target_name": info.target_name,
            "grupo": info.grupo,
            "task_type_id": int(getattr(info, "task_type_id", 1)),
            "openml_task_url": f"https://www.openml.org/t/{tid}",
            "openml_dataset_url": f"https://www.openml.org/d/{did}",
        }
        try:
            dataset = openml.datasets.get_dataset(did, download_data=False)
            x, y, _, _ = dataset.get_data(
                dataset_format="dataframe", target=info.target_name
            )
            x = x.drop(
                columns=x.select_dtypes(include=["datetime64", "datetimetz"]).columns
            ).reset_index(drop=True)
            y = y.reset_index(drop=True)
            if y.isna().any():
                raise ValueError(
                    "Missing query labels require a separate explicit policy."
                )
            counts = y.value_counts()
            counts = counts[counts > 0]
            encoded_y = pd.Series(
                LabelEncoder().fit_transform(y.astype(str)), name="target"
            )
            row.update(
                nome_dataset=dataset.name,
                n_instancias=len(x),
                n_atributos=x.shape[1],
                n_classes=len(counts),
                menor_classe_n=int(counts.min()),
                classe_majoritaria=counts.max() / len(y),
                classe_minoritaria=counts.min() / len(y),
                proporcao_ausentes=float(x.isna().to_numpy().mean()),
                proporcao_constantes=float((x.nunique(dropna=False) <= 1).mean()),
                features_sha256=frame_fingerprint(x),
                metadata_status="ok",
            )
            # Logical raw-table fingerprint, independent of OpenML's cache serialization.
            row["raw_sha256"] = row["features_sha256"]
            eligible = (
                row["task_type_id"] == 1
                and config["min_instances"] <= len(x) <= config["max_instances"]
                and len(counts) >= 2
                and counts.min() >= max(config["min_class_count"], config["folds"])
                and x.shape[1] > 0
            )
            row.update(
                elegivel=bool(eligible),
                motivo_exclusao="" if eligible else "class_support_size_or_attributes",
            )
            if eligible:
                payload = {
                    "X": x,
                    "y": encoded_y,
                    "task_id": tid,
                    "nome": dataset.name,
                    "raw_sha256": row["raw_sha256"],
                    "target_name": info.target_name,
                }
                file = cache / f"task_{tid}.pkl"
                with file.open("wb") as stream:
                    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                row["cache_sha256"] = sha256(file)
                embeddings[tid] = describe_labeled_dataset(x, y)
                n_numeric = x.select_dtypes(include=[np.number]).shape[1]
                names = [
                    str(c)
                    for c in x.columns
                    if re.search(
                        r"(^|_)(id|path|file|filename|image)(_|$)", str(c), re.I
                    )
                ]
                quality.append(
                    {
                        "task_id": tid,
                        "nome_dataset": dataset.name,
                        "n_numericos": n_numeric,
                        "n_categoricos": x.shape[1] - n_numeric,
                        "somente_categoricos": n_numeric == 0,
                        "possivel_identificador_ou_caminho": bool(names),
                        "colunas_sugestivas": ";".join(names),
                    }
                )
        except Exception as error:
            row.update(
                elegivel=False,
                motivo_exclusao="loading_error",
                erro=str(error),
                metadata_status="error",
            )
        rows.append(row)
        print(
            f"OpenML {number}/{len(metadata)}: task {tid}, {row['metadata_status']}",
            flush=True,
        )
    catalog = pd.DataFrame(rows)
    parents = list(range(len(catalog)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for field in ["grupo", "dataset_id", "features_sha256"]:
        if field not in catalog:
            continue
        seen = {}
        for i, value in enumerate(catalog[field]):
            if pd.isna(value):
                continue
            if value in seen:
                parents[find(i)] = find(seen[value])
            else:
                seen[value] = i
    labels = catalog.grupo.tolist()
    catalog["grupo"] = [labels[find(i)] for i in range(len(catalog))]
    catalog.to_csv(output / "datasets.csv", index=False)
    eligible = catalog[catalog.elegivel].sort_values("task_id")
    ids = eligible.task_id.to_numpy(dtype=int)
    if len(ids):
        np.save(output / "task_ids.npy", ids)
        np.save(output / "X_emb.npy", np.stack([embeddings[int(i)] for i in ids]))
    pd.DataFrame(quality).to_csv(output / "qualidade_atributos.csv", index=False)
    write_json(output / "pipeline_keys.json", pipeline_keys(config))
    write_json(
        output / "preparation.json",
        {
            "n_candidates": len(catalog),
            "n_eligible": len(ids),
            "n_groups": eligible.grupo.nunique(),
            "raw_cache": str(cache.resolve()),
            "config": config,
            "group_scope": "Preserved declared groups, merged transitively for repeated dataset IDs or identical raw features.",
        },
    )
    if not len(ids) or catalog.metadata_status.ne("ok").any():
        raise RuntimeError(
            "Preparation incomplete. Inspect datasets.csv before running classifiers."
        )
    return catalog
