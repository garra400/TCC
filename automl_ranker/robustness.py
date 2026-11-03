"""Nested k/weight controls and repeated outer group partitions.

The extended inverse-distance model remains prespecified as the principal k-NN.
Outer scores never select k, weighting, epsilon, seeds or the reporting cohort.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import weightedtau
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import (
    METRICS,
    PRIMARY,
    bootstrap_mean,
    paired,
    ranking_metrics,
    sha256,
    summarize,
)
from .io import load_config, load_reference, write_json

GRID = (1, 3, 5, 7, 10, 15, 20, 30, 50, "all")
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class Variant:
    name: str
    scaled: bool
    weighting: str = "inverse"
    epsilon: float = 1e-8


VARIANTS = (
    Variant("knn_ajustado", False),
    Variant(PRIMARY, True),
    Variant("knn_padronizado_uniforme", True, "uniform", 0),
    Variant("knn_padronizado_eps_1e_4", True, "inverse", 1e-4),
    Variant("knn_padronizado_eps_1e_2", True, "inverse", 1e-2),
)


def predict_neighbors(y_train, distance, indices, k, weighting="inverse", epsilon=1e-8):
    """Predict from a stable neighbor order and report normalized-weight concentration."""
    if weighting not in {"inverse", "uniform"}:
        raise ValueError("Unknown weighting rule.")
    n = len(y_train)
    count = n if k == "all" else min(int(k), n)
    if count < 1 or (weighting == "inverse" and epsilon <= 0):
        raise ValueError("k and inverse-distance epsilon must be positive.")
    selected = indices[:, :count]
    distances = np.take_along_axis(distance, selected, axis=1)
    if weighting == "uniform":
        # Use exactly the same mean and dtype as the global-mean baseline.
        prediction = (
            np.tile(y_train.mean(axis=0), (len(distance), 1))
            if k == "all"
            else y_train[selected].mean(axis=1)
        )
        largest = np.full(len(distance), 1 / count)
        effective = np.full(len(distance), count, dtype=float)
    else:
        weights = 1 / (distances + epsilon)
        weights /= weights.sum(axis=1, keepdims=True)
        prediction = np.einsum("ij,ijk->ik", weights, y_train.astype(float)[selected])
        largest = weights.max(axis=1)
        effective = 1 / np.square(weights).sum(axis=1)
    return prediction, largest, effective


def neighborhood(x_train, x_test, scaled):
    if scaled:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    distances = cdist(x_test, x_train)
    return distances, np.argsort(distances, axis=1, kind="stable")


def choose_on_training(x, y, groups, config, variants=VARIANTS, grid=GRID):
    """One inner held-out prediction per training task for every k/weight variant."""
    cv = GroupKFold(
        n_splits=min(config["inner_folds"], len(set(groups))),
        shuffle=True,
        random_state=config["seed"],
    )
    records = []
    for train, valid in cv.split(x, groups=groups):
        assert not set(groups[train]) & set(groups[valid])
        neighborhoods = {
            scaled: neighborhood(x[train], x[valid], scaled)
            for scaled in {v.scaled for v in variants}
        }
        for variant in variants:
            distances, order = neighborhoods[variant.scaled]
            for k in grid:
                predictions, _, _ = predict_neighbors(
                    y[train], distances, order, k, variant.weighting, variant.epsilon
                )
                if variant.weighting == "uniform" and k == "all":
                    assert np.array_equal(
                        predictions, np.tile(y[train].mean(axis=0), (len(valid), 1))
                    )
                for index, prediction in zip(valid, predictions):
                    real = y[index]
                    tau = (
                        np.nan
                        if np.ptp(real) == 0 or np.ptp(prediction) == 0
                        else weightedtau(
                            real, prediction, rank=True, additive=True
                        ).statistic
                    )
                    records.append(
                        dict(
                            modelo=variant.name,
                            k=str(k),
                            group=groups[index],
                            task_id=int(index),
                            tau=tau,
                            effective_k=len(train)
                            if k == "all"
                            else min(int(k), len(train)),
                        )
                    )
    data = pd.DataFrame(records)
    curve = (
        data.groupby(["modelo", "k", "group"], sort=False)
        .tau.mean()
        .groupby(level=["modelo", "k"], sort=False)
        .agg(["mean", "count"])
        .reset_index()
    )
    curve = curve.rename(
        columns={"mean": "inner_group_mean", "count": "n_groups_defined"}
    )
    bounds = (
        data.groupby(["modelo", "k"], sort=False)
        .effective_k.agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "effective_k_min", "max": "effective_k_max"})
    )
    curve = curve.merge(bounds, on=["modelo", "k"], validate="one_to_one")
    selected = {}
    for variant in variants:
        values = curve[curve.modelo.eq(variant.name)].set_index("k")
        selected[variant.name] = max(
            grid,
            key=lambda k: (
                values.loc[str(k), "inner_group_mean"]
                if np.isfinite(values.loc[str(k), "inner_group_mean"])
                else -np.inf
            ),
        )
    return selected, curve


def means_only(metrics):
    rows = []
    for (protocol, model), frame in metrics.groupby(["protocolo", "modelo"]):
        for metric in METRICS:
            tasks = frame[metric].dropna()
            groups = frame.groupby("grupo")[metric].mean().dropna()
            rows.append(
                dict(
                    protocolo=protocol,
                    modelo=model,
                    metrica=metric,
                    media_tasks=tasks.mean(),
                    desvio_tasks=tasks.std(ddof=0),
                    media_grupos=groups.mean(),
                    n_tasks=len(frame),
                    n_validos=len(tasks),
                    n_grupos_validos=len(groups),
                    n_indefinidos=frame[metric].isna().sum(),
                )
            )
    return pd.DataFrame(rows)


def run_partition(
    x,
    y,
    ids,
    groups,
    keys,
    config,
    outer_seed,
    directory,
    partitions=None,
    all_variants=True,
    full_output=True,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv = GroupKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=outer_seed
    )
    partitions = list(cv.split(x, groups=groups)) if partitions is None else partitions
    variants = VARIANTS if all_variants else VARIANTS[:2]
    rows, pred_rows, selections, curves, weights, splits = [], [], [], [], [], []
    for fold, (train, test) in enumerate(partitions):
        assert not set(groups[train]) & set(groups[test])
        selected, curve = choose_on_training(
            x[train], y[train], groups[train], config, variants
        )
        curve["fold"] = fold
        curve["outer_seed"] = outer_seed
        curve["selected"] = [
            str(selected[m]) == k for m, k in zip(curve.modelo, curve.k)
        ]
        curves.append(curve)
        neighborhoods = {
            scaled: neighborhood(x[train], x[test], scaled)
            for scaled in {v.scaled for v in variants}
        }
        outputs = {}
        for variant in variants:
            k = selected[variant.name]
            distances, order = neighborhoods[variant.scaled]
            prediction, largest, effective = predict_neighbors(
                y[train], distances, order, k, variant.weighting, variant.epsilon
            )
            outputs[variant.name] = prediction
            selections.append(
                dict(
                    outer_seed=outer_seed,
                    fold=fold,
                    modelo=variant.name,
                    selected_k=str(k),
                    n_train=len(train),
                )
            )
            if full_output:
                weights.extend(
                    dict(
                        modelo=variant.name,
                        fold=fold,
                        task_id=int(ids[index]),
                        grupo=groups[index],
                        selected_k=str(k),
                        nearest_distance=float(distances[i, order[i, 0]]),
                        maximum_normalized_weight=float(largest[i]),
                        effective_neighbors=float(effective[i]),
                    )
                    for i, index in enumerate(test)
                )
        # Short-grid controls share the same inner estimates and outer partitions.
        for variant, control in [
            (VARIANTS[0], "knn_sem_padronizacao_grade_curta"),
            (VARIANTS[1], "knn_padronizado_grade_curta"),
        ]:
            values = curve[curve.modelo.eq(variant.name)].set_index("k")
            k = max(
                (1, 3, 5, 7, 10),
                key=lambda n: (
                    values.loc[str(n), "inner_group_mean"]
                    if np.isfinite(values.loc[str(n), "inner_group_mean"])
                    else -np.inf
                ),
            )
            outputs[control] = predict_neighbors(
                y[train], *neighborhoods[variant.scaled], k
            )[0]
            selections.append(
                dict(
                    outer_seed=outer_seed,
                    fold=fold,
                    modelo=control,
                    selected_k=str(k),
                    n_train=len(train),
                )
            )
        outputs["knn_k10_fixo"] = predict_neighbors(
            y[train], *neighborhoods[False], 10
        )[0]
        forest = RandomForestRegressor(
            n_estimators=config["rf_estimators"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            random_state=config["seed"],
            n_jobs=1,
        )
        outputs["random_forest"] = forest.fit(x[train], y[train]).predict(x[test])
        outputs["media_global"] = np.tile(y[train].mean(axis=0), (len(test), 1))
        outputs["aleatorio"] = np.random.default_rng(config["seed"] + fold).random(
            (len(test), y.shape[1])
        )
        all_uniform = predict_neighbors(
            y[train], *neighborhoods[True], "all", "uniform", 0
        )[0]
        assert np.array_equal(all_uniform, outputs["media_global"])
        for model, predictions in outputs.items():
            for index, prediction in zip(test, predictions):
                rows.append(
                    dict(
                        protocolo="cv_grupos",
                        modelo=model,
                        outer_seed=outer_seed,
                        fold=fold,
                        task_id=int(ids[index]),
                        grupo=groups[index],
                        **ranking_metrics(y[index], prediction),
                    )
                )
                if full_output:
                    pred_rows.extend(
                        dict(
                            fold=fold,
                            modelo=model,
                            task_id=int(ids[index]),
                            pipeline_id=j,
                            scaler=key[0],
                            selector=key[1],
                            classifier=key[2],
                            real=y[index, j],
                            predito=prediction[j],
                            observado_treino=True,
                        )
                        for j, key in enumerate(keys)
                    )
        splits.append(
            dict(
                fold=fold,
                train_tasks=ids[train].astype(int).tolist(),
                test_tasks=ids[test].astype(int).tolist(),
                train_groups=sorted(set(groups[train])),
                test_groups=sorted(set(groups[test])),
            )
        )
        print(
            f"Robustness seed={outer_seed} fold={fold + 1}/{len(partitions)} k={selected[VARIANTS[0].name]} k_scaled={selected[PRIMARY]}",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "metricas_por_task.csv", index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(directory / "selecao_k.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(
        directory / "curvas_internas.csv", index=False
    )
    write_json(directory / "splits.json", splits)
    summary = summarize(metrics, config) if full_output else means_only(metrics)
    summary.to_csv(directory / "resumo.csv", index=False)
    if full_output:
        pd.DataFrame(pred_rows).to_csv(directory / "predicoes.csv", index=False)
        pd.DataFrame(weights).to_csv(directory / "concentracao_pesos.csv", index=False)
        paired(metrics, config).to_csv(
            directory / "comparacoes_pareadas.csv", index=False
        )
        weight_pairs = [
            (v.name, PRIMARY) for v in variants if v.name != PRIMARY and v.scaled
        ]
        paired(metrics, config, weight_pairs).to_csv(
            directory / "comparacoes_pesos.csv", index=False
        )
    return metrics, summary, splits


def expanded_groups(reference, ids, original):
    x = np.load(Path(reference) / "X_emb.npy", allow_pickle=False).astype(float)
    similarity = 1 - squareform(pdist(x, "cosine"))
    parent = {int(i): int(i) for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(int(a)), find(int(b))
        parent[max(a, b)] = min(a, b)

    for group in sorted(set(original)):
        members = ids[original == group]
        for tid in members:
            union(members[0], tid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if similarity[i, j] >= 0.999:
                union(ids[i], ids[j])
    return np.array([f"proximidade_{find(int(i))}" for i in ids])


def seed_summary(all_metrics, output):
    stats = pd.concat(
        [means_only(frame).assign(outer_seed=seed) for seed, frame in all_metrics],
        ignore_index=True,
    )
    stats.to_csv(output / "medias_por_semente.csv", index=False)
    variation = (
        stats.groupby(["modelo", "metrica"])
        .media_grupos.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    variation.to_csv(output / "variacao_sementes.csv", index=False)
    pairs = []
    for seed, frame in all_metrics:
        for metric in ["kendall_ponderado", "ndcg_5", "regret_1"]:
            pivot = frame.pivot(
                index=["grupo", "task_id"], columns="modelo", values=metric
            )
            for first, second in [
                ("random_forest", PRIMARY),
                (PRIMARY, "media_global"),
                ("random_forest", "media_global"),
            ]:
                both = pivot[[first, second]].dropna()
                difference = (both[first] - both[second]).groupby(level="grupo").mean()
                pairs.append(
                    dict(
                        outer_seed=seed,
                        metrica=metric,
                        primeiro=first,
                        segundo=second,
                        diferenca_media_grupos=difference.mean(),
                        n_groups=len(difference),
                    )
                )
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(output / "diferencas_por_semente.csv", index=False)
    pair_frame.groupby(["primeiro", "segundo", "metrica"]).diferenca_media_grupos.agg(
        ["mean", "std", "min", "max"]
    ).reset_index().to_csv(output / "variacao_diferencas.csv", index=False)


def compare_external(metrics_path, output, config):
    data = pd.read_csv(metrics_path)
    rows = []
    for metric in METRICS:
        pivot = data.pivot(index="dataset", columns="metodo", values=metric)
        for model in [m for m in pivot if m != "media_global"]:
            both = pivot[[model, "media_global"]].dropna()
            difference = both[model] - both.media_global
            lo, hi = bootstrap_mean(
                difference, config["bootstrap_samples"], config["seed"]
            )
            rows.append(
                dict(
                    modelo=model,
                    metrica=metric,
                    diferenca_media=float(difference.mean()),
                    inferior=lo,
                    superior=hi,
                    n_datasets=len(both),
                )
            )
    pd.DataFrame(rows).to_csv(
        output / "representacoes_menos_media_global.csv", index=False
    )


def run(reference, output, config=None, seeds=SEEDS, external_metrics=None):
    reference, output = Path(reference), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config() if config is None else config
    seeds = tuple(dict.fromkeys([42, *[int(seed) for seed in seeds]]))
    x, y, ids, groups, keys, catalog = load_reference(reference)
    sources = [
        reference / name
        for name in [
            "X_emb.npy",
            "Y_accuracy.npy",
            "task_ids.npy",
            "datasets.csv",
            "pipeline_keys.json",
            "qualidade_atributos.csv",
        ]
    ]
    if external_metrics is not None:
        sources.append(Path(external_metrics))
    plan = {
        "client_date": "2026-11-02",
        "nature": "Additional retrospective analyses; rules fixed before computing these new results, after the reference archive was already explored.",
        "grid": [str(k) for k in GRID],
        "include_all_neighbors": True,
        "principal": "standardized inverse distance, epsilon 1e-8, extended grid chosen only in inner training",
        "variants": [v.__dict__ for v in VARIANTS],
        "outer_seeds": list(seeds),
        "inner_and_model_seed": config["seed"],
        "config": config,
        "all_neighbor_control": "Uniform all-neighbor predictions must equal the global mean exactly; inverse-distance all-neighbor predictions need not.",
        "replication_uncertainty": "Report mean, sample SD and min/max across seeds; repeated tasks/folds are not new independent samples.",
        "stress_rule": "Union of original groups and raw-vector cosine >=0.999, transitively; weights and k are reselected only inside each new training fold.",
        "input_sha256": {p.name: sha256(p) for p in sources},
    }
    path = output / "plano.json"
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != plan:
        raise ValueError(
            "A different plan already exists; use another output directory."
        )
    if not path.exists():
        write_json(path, plan)
    write_json(
        output / "started.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256(__file__),
            "python": platform.python_version(),
        },
    )
    all_metrics = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            folder = output / ("principal" if seed == 42 else f"sementes/seed_{seed}")
            metrics, summary, splits = run_partition(
                x, y, ids, groups, keys, config, seed, folder, full_output=seed == 42
            )
            all_metrics.append((seed, metrics))
        seed_summary(all_metrics, output)
        # Same-score short-grid controls must reproduce the pre-expansion reference.
        expected_path = reference / "expected_metrics.csv"
        if expected_path.exists():
            expected = pd.read_csv(expected_path).set_index(["modelo", "metrica"])
            actual = pd.read_csv(output / "principal/resumo.csv").set_index(
                ["modelo", "metrica"]
            )
            for before, after in [
                (PRIMARY, "knn_padronizado_grade_curta"),
                ("knn_ajustado", "knn_sem_padronizacao_grade_curta"),
                ("random_forest", "random_forest"),
                ("media_global", "media_global"),
            ]:
                for metric in METRICS:
                    np.testing.assert_allclose(
                        actual.loc[(after, metric), "media_grupos"],
                        expected.loc[(before, metric), "media_grupos"],
                        atol=1e-10,
                        rtol=0,
                    )
        stress = expanded_groups(reference, ids, groups)
        run_partition(x, y, ids, stress, keys, config, 42, output / "estresse_113")
        write_json(
            output / "grupos.json",
            dict(
                task_ids=ids.astype(int).tolist(),
                original=groups.tolist(),
                expanded=stress.tolist(),
                n_original=len(set(groups)),
                n_expanded=len(set(stress)),
            ),
        )
        # Metadata exclusions preserve the original outer assignments.
        quality = (
            pd.read_csv(reference / "qualidade_atributos.csv")
            .set_index("task_id")
            .loc[ids]
        )
        first = set(quality.index[quality.somente_categoricos])
        second = set(quality.index[quality.possivel_identificador_ou_caminho])
        primary_splits = json.loads(
            (output / "principal/splits.json").read_text(encoding="utf8")
        )
        for name, excluded in [
            ("sem_categoricas", first),
            ("sem_nomes_sugestivos", second),
            ("sem_ambos", first | second),
        ]:
            keep = ~np.isin(ids, list(excluded))
            small_ids = ids[keep]
            partitions = [
                (
                    np.flatnonzero(np.isin(small_ids, s["train_tasks"])),
                    np.flatnonzero(np.isin(small_ids, s["test_tasks"])),
                )
                for s in primary_splits
            ]
            run_partition(
                x[keep],
                y[keep],
                small_ids,
                groups[keep],
                keys,
                config,
                42,
                output / f"sensibilidade/{name}",
                partitions,
                all_variants=False,
            )
        if external_metrics is not None:
            compare_external(external_metrics, output, config)
    write_json(
        output / "manifesto.json",
        {
            "status": "complete",
            "plan_sha256": sha256(path),
            "script_sha256": sha256(__file__),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "uniform_all_equals_global_mean": True,
            "n_tasks": len(ids),
            "n_original_groups": len(set(groups)),
            "n_expanded_groups": len(set(stress)),
            "n_seeds": len(seeds),
            "classifier_phase1_fits_repeated": 0,
            "files_sha256": {
                str(p.relative_to(output)): sha256(p) for p in output.rglob("*.csv")
            },
        },
    )
    print(f"Robustness analyses completed: {output.resolve()}", flush=True)
