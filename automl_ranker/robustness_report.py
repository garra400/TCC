"""Derived contrasts, diagnostics and timing for a completed robustness run."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, false_discovery_control
from sklearn.preprocessing import StandardScaler

from .evaluation import PRIMARY, paired, sha256
from .io import load_reference, write_json


def derive(reference, result, config):
    reference, result = Path(reference), Path(result)
    if not (result / "manifesto.json").exists():
        raise ValueError(
            "Complete the robustness experiment before deriving its report."
        )
    out = result / "relatorio"
    out.mkdir(exist_ok=True)
    x, y, ids, groups, keys, catalog = load_reference(reference)
    primary = result / "principal"
    metrics = pd.read_csv(primary / "metricas_por_task.csv")
    comparisons = [
        (PRIMARY, "knn_padronizado_grade_curta"),
        ("knn_ajustado", "knn_sem_padronizacao_grade_curta"),
        ("knn_ajustado", PRIMARY),
        ("knn_padronizado_uniforme", "media_global"),
        ("knn_padronizado_eps_1e_4", "media_global"),
        ("knn_padronizado_eps_1e_2", "media_global"),
    ]
    paired(metrics, config, comparisons).to_csv(
        out / "comparacoes_grade_e_pesos.csv", index=False
    )
    # Quantify both the original short-grid concern and the extended-grid comparator.
    rows = []
    plan = json.loads((result / "plano.json").read_text(encoding="utf8"))
    for seed in plan["outer_seeds"]:
        folder = primary if seed == 42 else result / f"sementes/seed_{seed}"
        frame = pd.read_csv(folder / "metricas_por_task.csv")
        for metric in ["kendall_ponderado", "ndcg_5", "regret_1"]:
            wide = frame.pivot(
                index=["grupo", "task_id"], columns="modelo", values=metric
            )
            for comparator in [PRIMARY, "knn_padronizado_grade_curta", "knn_ajustado"]:
                valid = wide[["random_forest", comparator]].dropna()
                difference = (
                    (valid.random_forest - valid[comparator])
                    .groupby(level="grupo")
                    .mean()
                )
                rows.append(
                    dict(
                        outer_seed=seed,
                        primeiro="random_forest",
                        segundo=comparator,
                        metrica=metric,
                        diferenca=difference.mean(),
                        n_grupos=len(difference),
                    )
                )
    seeds = pd.DataFrame(rows)
    seeds.to_csv(out / "contrastes_sementes.csv", index=False)
    seeds.groupby(["primeiro", "segundo", "metrica"]).diferenca.agg(
        ["mean", "std", "min", "max", lambda values: int((values > 0).sum())]
    ).reset_index().rename(columns={"<lambda_0>": "n_positive"}).to_csv(
        out / "resumo_contrastes_sementes.csv", index=False
    )
    # Five-feature error analysis follows the principal standardized variant.
    splits = json.loads((primary / "splits.json").read_text(encoding="utf8"))
    distances = {}
    for spec in splits:
        train = np.flatnonzero(np.isin(ids, spec["train_tasks"]))
        test = np.flatnonzero(np.isin(ids, spec["test_tasks"]))
        scaler = StandardScaler().fit(x[train])
        values = cdist(scaler.transform(x[test]), scaler.transform(x[train])).min(
            axis=1
        )
        distances.update({int(ids[i]): float(value) for i, value in zip(test, values)})
    diagnostic = metrics[metrics.modelo.eq(PRIMARY)].merge(
        catalog, on=["task_id", "grupo"], validate="one_to_one"
    )
    diagnostic["distancia_vizinho"] = diagnostic.task_id.map(distances)
    diagnostic["correlacao_nao_positiva"] = (
        diagnostic.kendall_ponderado.le(0) & diagnostic.kendall_ponderado.notna()
    )
    diagnostic["correlacao_indefinida"] = diagnostic.kendall_ponderado.isna()
    diagnostic.to_csv(out / "diagnostico_principal.csv", index=False)
    features = [
        "n_instancias",
        "n_atributos",
        "classe_majoritaria",
        "proporcao_constantes",
        "distancia_vizinho",
    ]
    grouped = diagnostic.groupby("grupo")[["kendall_ponderado", *features]].mean()
    records = []
    for feature in features:
        valid = grouped[[feature, "kendall_ponderado"]].dropna()
        rho, p = spearmanr(valid[feature], valid.kendall_ponderado)
        records.append(
            dict(variavel=feature, spearman=rho, p_value=p, n_grupos=len(valid))
        )
    correlations = pd.DataFrame(records)
    correlations["q_bh"] = np.nan
    valid = correlations.p_value.notna()
    correlations.loc[valid, "q_bh"] = false_discovery_control(
        correlations.loc[valid, "p_value"].to_numpy(), method="bh"
    )
    correlations.to_csv(out / "correlacoes_principal.csv", index=False)
    concentrations = []
    for condition, folder in [
        ("165 grupos", primary),
        ("113 grupos", result / "estresse_113"),
    ]:
        data = pd.read_csv(folder / "concentracao_pesos.csv")
        for model, frame in data.groupby("modelo"):
            concentrations.append(
                dict(
                    condicao=condition,
                    modelo=model,
                    n_tarefas=len(frame),
                    n_peso_099=int(frame.maximum_normalized_weight.ge(0.99).sum()),
                    maximo_peso=frame.maximum_normalized_weight.max(),
                    mediana_peso=frame.maximum_normalized_weight.median(),
                    media_vizinhos_efetivos=frame.effective_neighbors.mean(),
                )
            )
    pd.DataFrame(concentrations).to_csv(out / "concentracao_resumo.csv", index=False)
    # Reuse the same timing routine; raw fold sums are checked when available.
    from . import analyses

    analyses.BASE, analyses.META, analyses.OUT = reference, primary, out
    analyses.cost(config)
    write_json(
        out / "manifesto.json",
        {
            "script_sha256": sha256(__file__),
            "source_run_manifest_sha256": sha256(result / "manifesto.json"),
            "files_sha256": {p.name: sha256(p) for p in out.glob("*.csv")},
        },
    )
    print(f"Derived robustness diagnostics and contrasts: {out.resolve()}", flush=True)
