"""Repete a Fase 2 e separa efeitos dos novos escores e representações.

Executa três condições nas mesmas tasks, grupos e folds: (i) referência com
matrizes históricas; (ii) novos escores com vetor histórico; (iii) novos escores
e descritor recalculado dos dados brutos. A última é a avaliação principal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projeto.revisao.experimentos import (
    PROJECT, METRICS, error_analysis, grouped_evaluation, paired_comparisons,
    sha256, summarize,
)
from projeto.revisao.fase1_corrigida import OUTPUT, write_json


def evaluate_case(name, x, y, ids, groups, keys, catalog, config):
    destination = OUTPUT/"fase2"/name
    destination.mkdir(parents=True, exist_ok=True)
    folds, predictions, selections, splits, distances = grouped_evaluation(x, y, ids, groups, keys, config)
    folds.to_csv(destination/"metricas_por_task.csv", index=False, lineterminator="\n")
    predictions.to_csv(destination/"predicoes.csv", index=False, lineterminator="\n")
    selections.to_csv(destination/"selecao_k_interna.csv", index=False, lineterminator="\n")
    write_json(destination/"splits.json", splits)
    summary = summarize(folds, config)
    summary.to_csv(destination/"resumo.csv", index=False, lineterminator="\n")
    paired_comparisons(folds, config).to_csv(destination/"comparacoes_pareadas.csv", index=False, lineterminator="\n")
    errors, correlations = error_analysis(folds, catalog, distances)
    errors.to_csv(destination/"diagnostico_erros.csv", index=False, lineterminator="\n")
    correlations.to_csv(destination/"correlacoes_erros.csv", index=False, lineterminator="\n")
    return folds, summary, splits


def compare_conditions(cases, config):
    """Diferenças pareadas por grupo, usando os mesmos testes em cada condição."""
    from projeto.revisao.experimentos import bootstrap_mean
    baseline = cases["historico_mesmos_folds"][0]
    records = []
    for name in ["novos_escores_vetor_historico", "reconstrucao_completa"]:
        current = cases[name][0]
        paired = current.merge(baseline, on=["task_id", "grupo", "modelo", "fold"], suffixes=("_novo", "_historico"), validate="one_to_one")
        for model, data in paired.groupby("modelo"):
            for metric in METRICS:
                both = data[["grupo", metric+"_novo", metric+"_historico"]].dropna()
                delta = (both[metric+"_novo"]-both[metric+"_historico"]).groupby(both.grupo).mean()
                lo, hi = bootstrap_mean(delta, config["bootstrap_samples"], config["seed"])
                records.append({"condicao": name, "referencia": "historico_mesmos_folds", "modelo": model,
                                "metrica": metric, "diferenca_media_grupos": delta.mean(),
                                "ic95_inferior": lo, "ic95_superior": hi, "n_grupos_pareados": len(delta)})
    return pd.DataFrame(records)


def run():
    phase1 = json.loads((OUTPUT/"manifesto.json").read_text(encoding="utf-8"))
    if phase1["status"] != "complete":
        raise RuntimeError("A Fase 1 ainda não foi concluída.")
    for name, checksum in phase1["artifacts_sha256"].items():
        if sha256(OUTPUT/name) != checksum:
            raise ValueError(f"Artefato alterado após a execução: {name}")
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog = pd.read_csv(OUTPUT/"datasets.csv")
    ids = np.load(OUTPUT/"task_ids.npy")
    catalog = catalog.set_index("task_id").loc[ids].reset_index()
    groups = catalog.grupo.to_numpy()
    keys = json.loads((OUTPUT/"pipeline_keys.json").read_text(encoding="utf-8"))
    historic = PROJECT/"data/meta_conjunto"
    historic_ids = np.load(historic/"task_ids.npy")
    indices = np.array([np.flatnonzero(historic_ids == tid)[0] for tid in ids])
    old_x = np.load(historic/"X_emb.npy")[indices]
    old_y = np.load(historic/"Y_accuracy.npy")[indices]
    # Mesma precisão das matrizes históricas: evita introduzir uma mudança de
    # desempate por usar float64 apenas na condição reconstruída. As médias de
    # origem em float64 continuam preservadas nos CSVs e NPY da Fase 1.
    old_x, old_y = old_x.astype(np.float32), old_y.astype(np.float32)
    new_x = np.load(OUTPUT/"X_emb.npy").astype(np.float32)
    new_y = np.load(OUTPUT/"Y_accuracy.npy").astype(np.float32)
    cases = {}
    for name, x, y in [("historico_mesmos_folds", old_x, old_y),
                        ("novos_escores_vetor_historico", old_x, new_y),
                        ("reconstrucao_completa", new_x, new_y)]:
        print(f"Avaliação: {name}", flush=True)
        cases[name] = evaluate_case(name, x, y, ids, groups, keys, catalog, config)
    reference_splits = cases["historico_mesmos_folds"][2]
    assert all(case[2] == reference_splits for case in cases.values()), "Comparação exige os mesmos folds."
    combined = []
    for name, (_, summary, _) in cases.items():
        summary = summary.copy()
        summary["condicao"] = name
        combined.append(summary)
    pd.concat(combined).to_csv(OUTPUT/"fase2/resumo_condicoes.csv", index=False, lineterminator="\n")
    compare_conditions(cases, config).to_csv(OUTPUT/"fase2/comparacao_condicoes.csv", index=False, lineterminator="\n")
    # Alterações no escore por task/pipeline, sem imputar resultados ausentes.
    rows = []
    for i, tid in enumerate(ids):
        for j, key in enumerate(keys):
            rows.append({"task_id": int(tid), "pipeline_id": j, "scaler": key[0], "selector": key[1],
                         "classifier": key[2], "historico": old_y[i, j], "corrigido": new_y[i, j],
                         "diferenca": new_y[i, j]-old_y[i, j]})
    changed = pd.DataFrame(rows)
    changed.to_csv(OUTPUT/"comparacao_escores.csv", index=False, lineterminator="\n")
    changed.groupby("classifier").agg(diferenca_media=("diferenca", "mean"),
        diferenca_absoluta_media=("diferenca", lambda x: x.abs().mean()),
        pares_validos=("diferenca", "count")).to_csv(OUTPUT/"comparacao_escores_classificador.csv", lineterminator="\n")
    write_json(OUTPUT/"fase2/manifesto.json", {"config": config, "code_sha256": sha256(__file__),
               "evaluation_code_sha256": sha256(Path(__file__).with_name("experimentos.py")),
               "phase1_manifest_sha256": sha256(OUTPUT/"manifesto.json"),
               "reference_input_sha256": {name: sha256(historic/name) for name in ["X_emb.npy", "Y_accuracy.npy", "task_ids.npy"]},
               "n_tasks": len(ids), "n_groups": len(set(groups)), "conditions": list(cases),
               "comparison_input_dtype": "float32 for vectors and scores in all three conditions; float64 source scores retained",
               "same_outer_splits": True, "caveat": "Retrospective grouped CV; bootstrap intervals descriptive."})
    print(pd.concat(combined).query("metrica == 'kendall_ponderado'")[["condicao", "modelo", "media_grupos"]].to_string(index=False))


if __name__ == "__main__":
    run()
