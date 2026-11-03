"""Retrospective protocol, metadata and cost sensitivity analyses."""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .evaluation import (
    METRICS,
    MODELS,
    PRIMARY,
    bootstrap_mean,
    knn_predict,
    ranking_metrics,
    summarize,
    paired,
    select_k,
)
from .io import load_reference, write_json, run_manifest

BASE = META = OUT = None
choose_k = select_k


def protocol_audit(x, y, ids, groups, splits, config):
    directory = OUT / "loo_grupos"
    directory.mkdir(exist_ok=True)
    rows = []
    for scaled in [False, True]:
        model = "knn_k10_padronizado" if scaled else "knn_k10_sem_padronizar"
        protocols = {
            "LOO_tarefa": [
                (np.flatnonzero(ids != tid), np.array([i])) for i, tid in enumerate(ids)
            ],
            "LOGO": [
                (np.flatnonzero(groups != group), np.flatnonzero(groups == group))
                for group in sorted(set(groups))
            ],
            "GroupKFold_5": [
                (
                    np.flatnonzero(np.isin(ids, s["train_tasks"])),
                    np.flatnonzero(np.isin(ids, s["test_tasks"])),
                )
                for s in splits
            ],
        }
        for protocol, partitions in protocols.items():
            for fold, (train, test) in enumerate(partitions):
                a, b = x[train], x[test]
                if scaled:
                    scaler = StandardScaler().fit(a)
                    a, b = scaler.transform(a), scaler.transform(b)
                if protocol != "LOO_tarefa":
                    assert not set(groups[train]) & set(groups[test])
                output = knn_predict(a, y[train], b, 10)
                for index, prediction in zip(test, output):
                    rows.append(
                        {
                            "protocolo": protocol,
                            "modelo": model,
                            "fold": fold,
                            "task_id": int(ids[index]),
                            "grupo": groups[index],
                            "n_treino": len(train),
                            "grupo_presente_no_treino": bool(
                                groups[index] in groups[train]
                            ),
                            **ranking_metrics(y[index], prediction),
                        }
                    )
            print("Controle", model, protocol, "concluído", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(directory / "metricas.csv", index=False)
    summary = summarize(data, config)
    summary.to_csv(directory / "resumo.csv", index=False)
    rows = []
    for model in data.modelo.unique():
        for first, second in [
            ("GroupKFold_5", "LOO_tarefa"),
            ("LOGO", "LOO_tarefa"),
            ("GroupKFold_5", "LOGO"),
        ]:
            values = data[data.modelo.eq(model)].pivot(
                index=["grupo", "task_id"],
                columns="protocolo",
                values="kendall_ponderado",
            )
            common = values[[first, second]].dropna()
            delta = (common[first] - common[second]).groupby(level="grupo").mean()
            lo, hi = bootstrap_mean(delta, config["bootstrap_samples"], config["seed"])
            rows.append(
                {
                    "modelo": model,
                    "primeiro": first,
                    "segundo": second,
                    "diferenca_media_grupos": float(delta.mean()),
                    "ic95_inferior": lo,
                    "ic95_superior": hi,
                    "n_grupos_pareados": len(delta),
                    "n_tarefas_pareadas": len(common),
                }
            )
    pd.DataFrame(rows).to_csv(directory / "comparacoes.csv", index=False)
    original = pd.read_csv(META / "metricas_por_task.csv").query(
        "modelo == 'knn_k10_fixo'"
    )
    check = data.query(
        "modelo == 'knn_k10_sem_padronizar' and protocolo == 'GroupKFold_5'"
    ).merge(original, on=["task_id", "grupo"], suffixes=("_novo", "_anterior"))
    for metric in METRICS:
        np.testing.assert_allclose(
            check[metric + "_novo"],
            check[metric + "_anterior"],
            atol=1e-12,
            equal_nan=True,
        )
    write_json(
        directory / "validacao.json",
        {
            "mesmos_escores": True,
            "mesmos_vetores": True,
            "k_fixo": 10,
            "reproducao_condicao_original_exata": True,
            "tarefas_loo_com_grupo_no_treino": int(
                data.query(
                    "modelo == 'knn_k10_sem_padronizar' and protocolo == 'LOO_tarefa'"
                ).grupo_presente_no_treino.sum()
            ),
        },
    )


def sensitivity(x, y, ids, groups, splits, config, plan):
    for name, excluded in plan["sensibilidade"].items():
        directory = OUT / "sensibilidade" / name
        directory.mkdir(parents=True, exist_ok=True)
        keep = ~np.isin(ids, excluded)
        rows, selections, time_rows = [], [], []
        for spec in splits:
            fold = spec["fold"]
            train = np.flatnonzero(keep & np.isin(ids, spec["train_tasks"]))
            test = np.flatnonzero(keep & np.isin(ids, spec["test_tasks"]))
            assert not set(groups[train]) & set(groups[test])
            assert len(train) > 10 and len(test) > 0
            k, scores = choose_k(x[train], y[train], groups[train], config, False)
            ks, scaled_scores = choose_k(
                x[train], y[train], groups[train], config, True
            )
            for scale, values, chosen in [
                (False, scores, k),
                (True, scaled_scores, ks),
            ]:
                selections.extend(
                    {
                        "fold": fold,
                        "scaled": scale,
                        "k": value,
                        "score_interno": score,
                        "selecionado": value == chosen,
                    }
                    for value, score in values.items()
                )
            scaler = StandardScaler().fit(x[train])
            rf = RandomForestRegressor(
                n_estimators=config["rf_estimators"],
                min_samples_leaf=config["rf_min_samples_leaf"],
                random_state=config["seed"],
                n_jobs=1,
            )
            start = time.perf_counter()
            rf.fit(x[train], y[train])
            time_rows.append(
                {
                    "fold": fold,
                    "n_treino": len(train),
                    "n_teste": len(test),
                    "rf_ajuste_s": time.perf_counter() - start,
                }
            )
            outputs = {
                "knn_ajustado": knn_predict(x[train], y[train], x[test], k),
                PRIMARY: knn_predict(
                    scaler.transform(x[train]), y[train], scaler.transform(x[test]), ks
                ),
                "random_forest": rf.predict(x[test]),
                "media_global": np.tile(y[train].mean(axis=0), (len(test), 1)),
                "aleatorio": np.random.default_rng(config["seed"] + fold).random(
                    (len(test), y.shape[1])
                ),
            }
            for model, output in outputs.items():
                for index, prediction in zip(test, output):
                    rows.append(
                        {
                            "protocolo": "cv_grupos",
                            "modelo": model,
                            "fold": fold,
                            "task_id": int(ids[index]),
                            "grupo": groups[index],
                            **ranking_metrics(y[index], prediction),
                        }
                    )
            print(
                "Sensibilidade",
                name,
                f"fold {fold + 1}/5",
                "k",
                k,
                "k padronizado",
                ks,
                flush=True,
            )
        data = pd.DataFrame(rows)
        assert data.groupby("modelo").task_id.nunique().eq(keep.sum()).all()
        assert not set(data.task_id) & set(excluded)
        data.to_csv(directory / "metricas.csv", index=False)
        summarize(data, config).to_csv(directory / "resumo.csv", index=False)
        paired(data, config).to_csv(directory / "pares.csv", index=False)
        pd.DataFrame(selections).to_csv(directory / "selecao_k.csv", index=False)
        pd.DataFrame(time_rows).to_csv(directory / "tempos_metafit.csv", index=False)
        write_json(
            directory / "amostra.json",
            {
                "n_tarefas": int(keep.sum()),
                "n_grupos": len(set(groups[keep])),
                "excluidas": excluded,
                "mesmas_atribuicoes_externas": True,
                "remocao_em_treino_e_teste": True,
                "novos_ajustes_classificadores_fase1": 0,
            },
        )


def cost(config):
    raw = pd.read_csv(BASE / "pipelines_raw.csv")
    prediction = pd.read_csv(META / "predicoes.csv")
    catalog = pd.read_csv(BASE / "datasets.csv").set_index("task_id")
    timing = raw.set_index(["task_id", "pipeline_id"]).tempo_total_s
    assert timing.notna().all() and (timing > 0).all()
    # A origem do tempo é a soma dos cinco registros de classificação, incluindo métricas.
    fold_files = [
        BASE / "folds" / f"task_{int(tid)}.csv" for tid in sorted(raw.task_id.unique())
    ]
    checked_folds = all(p.exists() for p in fold_files)
    if checked_folds:
        fold_records = pd.concat(
            [pd.read_csv(p) for p in fold_files], ignore_index=True
        )
        sums = (
            fold_records.groupby(["task_id", "pipeline_id"])
            .tempo_s.sum()
            .reindex(timing.index)
        )
        np.testing.assert_allclose(sums, timing, atol=1e-9)
    records = []
    for (model, tid), values in prediction[prediction.modelo.isin(MODELS)].groupby(
        ["modelo", "task_id"]
    ):
        values = values.sort_values("pipeline_id", kind="stable")
        order = np.argsort(-values.predito.to_numpy(), kind="stable")
        costs = timing.loc[[(tid, int(pid)) for pid in values.pipeline_id]].to_numpy()
        records.append(
            {
                "modelo": model,
                "task_id": int(tid),
                "grupo": catalog.loc[tid, "grupo"],
                "top1_s": float(costs[order[:1]].sum()),
                "top5_s": float(costs[order[:5]].sum()),
                "todos_s": float(costs.sum()),
                "ids_top5": ";".join(
                    str(v) for v in values.pipeline_id.to_numpy()[order[:5]]
                ),
            }
        )
    details = pd.DataFrame(records)
    details.to_csv(OUT / "custo_por_tarefa.csv", index=False)
    group = details.groupby(["modelo", "grupo"])[["top1_s", "top5_s", "todos_s"]].mean()
    results = []
    for model, values in group.groupby(level="modelo"):
        means = values.mean()
        results.append(
            {
                "modelo": model,
                "n_grupos": len(values),
                "media_top1_s": means.top1_s,
                "media_top5_s": means.top5_s,
                "media_total_s": means.todos_s,
                "mediana_top5_s": values.top5_s.median(),
                "mediana_total_s": values.todos_s.median(),
                "fracao_tempo_top5": means.top5_s / means.todos_s,
                "reducao_equivalente": 1 - means.top5_s / means.todos_s,
                "razao_serial_total_top5": means.todos_s / means.top5_s,
            }
        )
    pd.DataFrame(results).to_csv(OUT / "custo_resumo.csv", index=False)
    write_json(
        OUT / "escopo_custo.json",
        {
            "somatorio_tempos_por_fold_conferido": checked_folds,
            "inclui": "Ajuste de classificadores, predição e cálculo das métricas em cinco folds, conforme instrumentação da Fase 1.",
            "nao_inclui": [
                "transformações compartilhadas",
                "extração de descritores",
                "metamodelo",
                "execução paralela/espera",
            ],
            "interpretacao": "Estimativa serial retrospectiva para os candidatos previstos no teste; não é medição de economia de tempo de parede de um sistema AutoML completo.",
        },
    )


def proximity(x, y, ids, groups, splits, config):
    from scipy.spatial.distance import pdist, squareform
    from sklearn.model_selection import GroupKFold

    directory = OUT / "proximidade"
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "cosine_threshold": 0.999,
        "equal_vector_tolerance": 1e-8,
        "scope": "Retrospective geometric stress test, not confirmed family identification. New groups, weights and folds cannot be interpreted as paired differences from the primary protocol.",
    }
    freeze_plan(directory / "plan.json", plan)
    # Geometric audit uses the original vectors; meta-model inputs use float32.
    raw_vectors = np.load(BASE / "X_emb.npy", allow_pickle=False).astype(float)
    distance = squareform(pdist(raw_vectors, "euclidean"))
    cosine = 1 - squareform(pdist(raw_vectors, "cosine"))
    records = []
    parent = list(range(len(ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for group in sorted(set(groups)):
        members = np.flatnonzero(groups == group)
        for j in members:
            union(int(members[0]), int(j))
    catalog = pd.read_csv(BASE / "datasets.csv").set_index("task_id").loc[ids]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if cosine[i, j] >= 0.999 or distance[i, j] <= 1e-8:
                records.append(
                    dict(
                        task_a=int(ids[i]),
                        task_b=int(ids[j]),
                        same_group=bool(groups[i] == groups[j]),
                        distance=float(distance[i, j]),
                        cosine=float(cosine[i, j]),
                        same_raw_features=bool(
                            catalog.iloc[i].get("features_sha256", None)
                            == catalog.iloc[j].get("features_sha256", False)
                        ),
                    )
                )
                union(i, j)
    pairs = pd.DataFrame(
        records,
        columns=[
            "task_a",
            "task_b",
            "same_group",
            "distance",
            "cosine",
            "same_raw_features",
        ],
    )
    pairs.to_csv(directory / "pairs.csv", index=False)
    expanded = np.array([f"proximidade_{int(ids[find(i)])}" for i in range(len(ids))])
    # Component identity follows minimum task ID, matching the reference grouping.
    assert len(ids) == len(set(ids)) and np.all(np.diff(ids) >= 0)
    pd.DataFrame(
        dict(task_id=ids, original_group=groups, expanded_group=expanded)
    ).to_csv(directory / "groups.csv", index=False)
    cv = GroupKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=config["seed"]
    )
    stress = [
        dict(fold=f, train_tasks=ids[a].tolist(), test_tasks=ids[b].tolist())
        for f, (a, b) in enumerate(cv.split(x, groups=expanded))
    ]
    write_json(
        directory / "summary.json",
        {
            **plan,
            "n_tasks": len(ids),
            "original_groups": len(set(groups)),
            "expanded_groups": len(set(expanded)),
            "cross_group_pairs": int((~pairs.same_group).sum()),
            "cross_group_tasks": len(
                set(pairs.loc[~pairs.same_group, "task_a"])
                | set(pairs.loc[~pairs.same_group, "task_b"])
            ),
        },
    )
    sensitivity(
        x, y, ids, expanded, stress, config, {"sensibilidade": {"proximidade_0999": []}}
    )
    path = OUT / "sensibilidade/proximidade_0999/amostra.json"
    value = json.loads(path.read_text(encoding="utf8"))
    value.update(
        mesmas_atribuicoes_externas=False, novas_particoes_por_grupos_ampliados=True
    )
    write_json(path, value)


def freeze_plan(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != value:
        raise ValueError(
            "An analysis plan already exists with different rules; use a new output directory."
        )
    write_json(path, value)


def run(input_folder, evaluation_folder, output_folder, config, kind="all"):
    global BASE, META, OUT
    BASE, META, OUT = Path(input_folder), Path(evaluation_folder), Path(output_folder)
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, ids, groups, keys, catalog = load_reference(BASE)
    if not (META / "splits.json").exists():
        raise FileNotFoundError(
            "Run 'automl-rank evaluate' first and pass its directory with --evaluation."
        )
    previous = json.loads((META / "run.json").read_text(encoding="utf8"))
    from .evaluation import sha256

    for file, digest in previous["input_files_sha256"].items():
        if sha256(BASE / file) != digest:
            raise ValueError("Evaluation and analysis inputs do not match.")
    if previous["config"] != config:
        raise ValueError("Use the same meta-configuration as the evaluation.")
    splits = json.loads((META / "splits.json").read_text(encoding="utf8"))
    if kind in {"all", "loo"}:
        protocol_audit(x, y, ids, groups, splits, config)
    if kind in {"all", "sensitivity"}:
        quality = (
            pd.read_csv(BASE / "qualidade_atributos.csv")
            .set_index("task_id")
            .loc[ids]
            .reset_index()
        )
        first = set(quality.loc[quality.somente_categoricos, "task_id"].astype(int))
        second = set(
            quality.loc[quality.possivel_identificador_ou_caminho, "task_id"].astype(
                int
            )
        )
        plan = {
            "nature": "Replication of retrospective metadata sensitivity rules, not independent preregistration.",
            "sensibilidade": {
                "sem_categoricas": sorted(first),
                "sem_nomes_sugestivos": sorted(second),
                "sem_ambos": sorted(first | second),
            },
            "overlap": len(first & second),
        }
        freeze_plan(OUT / "sensitivity_plan.json", plan)
        sensitivity(x, y, ids, groups, splits, config, plan)
    if kind in {"all", "proximity"}:
        proximity(x, y, ids, groups, splits, config)
    if kind in {"all", "cost"}:
        cost(config)
    run_manifest(
        OUT,
        [
            BASE / p
            for p in ["X_emb.npy", "Y_accuracy.npy", "task_ids.npy", "datasets.csv"]
        ],
        config,
        kind=kind,
    )
