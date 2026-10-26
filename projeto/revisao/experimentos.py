"""Reproduz LOO histórico e compara modelos com separação por grupos.

Uso a partir da raiz: python -m projeto.revisao.experimentos
Os dados existentes são um benchmark retrospectivo; o programa não os reclassifica
como uma coleta externa nem altera seus resultados de Fase 1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, weightedtau
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projeto"
METRICS = ["spearman", "kendalltau", "kendall_ponderado", "ndcg", "ndcg_5", "regret_1", "regret_5"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fill_targets(y):
    """Imputa somente o treino; retorna máscara de alvos observados no treino."""
    y = np.asarray(y, dtype=float)
    supported = np.isfinite(y).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(np.where(np.isfinite(y), y, np.nan), axis=0)
    medians[~supported] = 0.0
    return np.where(np.isfinite(y), y, medians), supported


def knn_predict(x_train, y_train, x_test, k):
    """Mesma ponderação 1/(distância+1e-8) do preditor histórico."""
    k = min(k, len(x_train))
    distances = cdist(x_test, x_train)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    weights = 1.0 / (np.take_along_axis(distances, indices, axis=1) + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, y_train[indices])


def ranking_metrics(truth, prediction):
    """Mantém NaN para correlações indefinidas, mas calcula NDCG e regret.

    NDCG: ganho linear da acurácia balanceada, empates médios do sklearn.
    Regret: diferença para melhor pipeline observado; empates na predição usam
    ordem de coluna estável. Ausência de score real nunca é imputada no teste.
    """
    valid = np.isfinite(truth) & np.isfinite(prediction)
    t, p = np.asarray(truth)[valid], np.asarray(prediction)[valid]
    result = dict.fromkeys(METRICS, np.nan)
    result.update(n_pipelines_validos=int(valid.sum()), status="menos_de_2_pipelines")
    if len(t) < 2:
        return result
    if np.any(t < 0) or np.any(t > 1):
        raise ValueError("Acurácia balanceada fora de [0,1].")
    result["status"] = "ok"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["spearman"] = float(spearmanr(t, p).statistic)
        result["kendalltau"] = float(kendalltau(t, p, variant="b").statistic)
        result["kendall_ponderado"] = float(weightedtau(t, p, rank=True, additive=True).statistic)
    result["ndcg"] = float(ndcg_score(t[None], p[None]))
    result["ndcg_5"] = float(ndcg_score(t[None], p[None], k=min(5, len(t))))
    order = np.argsort(-p, kind="stable")
    result["regret_1"] = float(t.max() - t[order[0]])
    result["regret_5"] = float(t.max() - t[order[:5]].max())
    if np.ptp(t) == 0:
        result["status"] = "alvo_constante"
    elif np.ptp(p) == 0:
        result["status"] = "predicao_constante"
    return result


def dataset_catalog(ids, raw, cache):
    """Agrupa nomes repetidos ou features processadas idênticas, transitivamente."""
    names = raw.drop_duplicates("task_id").set_index("task_id")["nome_dataset"]
    rows = []
    for tid in ids:
        row = {"task_id": int(tid), "nome_dataset": names.loc[int(tid)],
               "openml_task_url": f"https://www.openml.org/t/{tid}"}
        path = cache / f"task_{tid}.pkl"
        if path.exists():
            # Somente caches locais confiáveis produzidos pelo próprio projeto.
            with path.open("rb") as stream:
                data = pickle.load(stream)
            x, y = data["X"], np.asarray(data["y"])
            _, counts = np.unique(y, return_counts=True)
            # Index e nomes de colunas são excluídos do fingerprint.
            hashes = pd.util.hash_pandas_object(pd.DataFrame(x).set_axis(range(x.shape[1]), axis=1), index=False)
            digest = hashlib.sha256(str(x.shape).encode() + hashes.to_numpy().tobytes()).hexdigest()
            numeric = x.select_dtypes(include=[np.number])
            row.update(n_instancias=len(x), n_atributos=x.shape[1], n_classes=len(counts),
                       menor_classe_n=int(counts.min()),
                       classe_majoritaria=counts.max()/len(y), classe_minoritaria=counts.min()/len(y),
                       proporcao_ausentes=float(x.isna().to_numpy().mean()),
                       proporcao_constantes=float((numeric.nunique(dropna=False) <= 1).mean()),
                       features_sha256=digest, cache_sha256=sha256(path))
        rows.append(row)
    catalog = pd.DataFrame(rows)
    parents = list(range(len(catalog)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen_names, seen_hashes = {}, {}
    for i, row in catalog.iterrows():
        for value, seen in [(str(row.nome_dataset).strip().lower(), seen_names),
                            (row.get("features_sha256"), seen_hashes)]:
            if pd.isna(value):
                continue
            if value in seen:
                parents[find(i)] = find(seen[value])
            else:
                seen[value] = i
    catalog["grupo"] = [f"g{int(ids[find(i)])}" for i in range(len(catalog))]
    return catalog


def eligibility(catalog):
    """Critérios de qualidade definidos sem consultar a qualidade do ranking."""
    return (catalog.task_type_id.eq(1) & catalog.menor_classe_n.ge(5)
            & catalog.n_classes.ge(2) & catalog.n_instancias.between(500, 10000)
            & catalog.metadata_status.eq("ok"))


def select_k(x, y, groups, config, scaled=False):
    """Seleciona k apenas por validação interna por grupos do treino externo."""
    cv = GroupKFold(n_splits=min(config["inner_folds"], len(np.unique(groups))),
                    shuffle=True, random_state=config["seed"])
    scores = {k: [] for k in config["k_values"]}
    for train, test in cv.split(x, groups=groups):
        assert not set(groups[train]) & set(groups[test])
        filled, supported = fill_targets(y[train])
        x_train, x_test = x[train], x[test]
        if scaled:
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        for k in scores:
            predictions = knn_predict(x_train, filled, x_test, k)
            for index, prediction in zip(test, predictions):
                value = ranking_metrics(y[index, supported], prediction[supported])[config["primary_metric"]]
                scores[k].append({"group": groups[index], "score": value})
    means = {}
    for k, values in scores.items():
        means[k] = pd.DataFrame(values).groupby("group").score.mean().mean()
    # Empates escolhem o menor k; nenhum score do teste externo é consultado.
    best = max(sorted(means), key=lambda k: means[k] if np.isfinite(means[k]) else -np.inf)
    return best, means


def bootstrap_mean(values, samples, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(folds, config):
    rows = []
    for (protocol, model), data in folds.groupby(["protocolo", "modelo"]):
        for metric in METRICS:
            values = data[metric].dropna()
            grouped = data.groupby("grupo")[metric].mean().dropna()
            lo, hi = bootstrap_mean(grouped, config["bootstrap_samples"], config["seed"])
            rows.append(dict(protocolo=protocol, modelo=model, metrica=metric,
                             media_tasks=values.mean(), desvio_tasks=values.std(ddof=0),
                             media_grupos=grouped.mean(), ic95_grupos_inferior=lo,
                             ic95_grupos_superior=hi, n_tasks=len(data),
                             n_validos=len(values), n_grupos_validos=len(grouped),
                             n_indefinidos=int(data[metric].isna().sum())))
    return pd.DataFrame(rows)


def legacy_loo(x, y, ids, groups, config):
    # Usa a classe original para reproduzir inclusive desempates da KDTree.
    sys.path.insert(0, str(PROJECT))
    from fase2_metamodelos.preditor_knn import PreditorKNNRanking
    rows = []
    for k in config["k_values"]:
        for i, tid in enumerate(ids):
            train = np.arange(len(ids)) != i
            model = PreditorKNNRanking(k=k).fit(x[train], y[train])
            prediction = model.predict(x[i:i+1])[0]
            rows.append(dict(protocolo="loo_historico", modelo=f"knn_k{k}", fold=i,
                             task_id=int(tid), grupo=groups[i], **ranking_metrics(y[i], prediction)))
        print(f"LOO histórico k={k} concluído", flush=True)
    return pd.DataFrame(rows)


def grouped_evaluation(x, y, ids, groups, keys, config):
    rows, predictions, selections, splits, distances = [], [], [], [], []
    cv = GroupKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["seed"])
    for fold, (train, test) in enumerate(cv.split(x, groups=groups)):
        assert not set(groups[train]) & set(groups[test])
        splits.append({"fold": fold, "train_tasks": ids[train].tolist(),
                       "test_tasks": ids[test].tolist(), "train_groups": sorted(set(groups[train])),
                       "test_groups": sorted(set(groups[test]))})
        filled, supported = fill_targets(y[train])
        k, inner = select_k(x[train], y[train], groups[train], config)
        ks, inner_scaled = select_k(x[train], y[train], groups[train], config, scaled=True)
        for scale, scores in [(False, inner), (True, inner_scaled)]:
            selections.extend({"fold": fold, "scaled": scale, "k": value,
                               "inner_group_mean": score, "selected": value == (ks if scale else k)}
                              for value, score in scores.items())
        scaler = StandardScaler().fit(x[train])
        rf = RandomForestRegressor(n_estimators=config["rf_estimators"],
                                   min_samples_leaf=config["rf_min_samples_leaf"],
                                   random_state=config["seed"], n_jobs=-1)
        rf.fit(x[train], filled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(y[train], axis=0)
        rng = np.random.default_rng(config["seed"] + fold)
        outputs = {
            "knn_k10_fixo": knn_predict(x[train], filled, x[test], 10),
            "knn_ajustado": knn_predict(x[train], filled, x[test], k),
            "knn_padronizado_ajustado": knn_predict(scaler.transform(x[train]), filled, scaler.transform(x[test]), ks),
            "random_forest": rf.predict(x[test]),
            "media_global": np.tile(means, (len(test), 1)),
            "aleatorio": rng.random((len(test), y.shape[1])),
        }
        dist = cdist(x[test], x[train])
        nearest = dist.argmin(axis=1)
        for row, index in enumerate(test):
            distances.append({"task_id": int(ids[index]), "fold": fold,
                              "distancia_vizinho": dist[row, nearest[row]],
                              "vizinho_task_id": int(ids[train[nearest[row]]])})
        for name, output in outputs.items():
            output[:, ~supported] = np.nan
            for index, prediction in zip(test, output):
                rows.append(dict(protocolo="cv_grupos", modelo=name, fold=fold,
                                 task_id=int(ids[index]), grupo=groups[index],
                                 **ranking_metrics(y[index], prediction)))
                predictions.extend({"fold": fold, "modelo": name, "task_id": int(ids[index]),
                                    "pipeline_id": j, "scaler": key[0], "selector": key[1],
                                    "classifier": key[2], "real": y[index, j], "predito": prediction[j],
                                    "observado_treino": bool(supported[j])}
                                   for j, key in enumerate(keys))
        print(f"Grupos fold {fold+1}/{config['outer_folds']}: k={k}; k padronizado={ks}", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(selections),
            splits, pd.DataFrame(distances))


def error_analysis(folds, catalog, distances):
    errors = folds[folds.modelo == "knn_ajustado"].merge(catalog, on=["task_id", "grupo"]).merge(distances, on=["task_id", "fold"])
    errors["correlacao_nao_positiva"] = errors.kendall_ponderado.le(0) & errors.kendall_ponderado.notna()
    errors["correlacao_indefinida"] = errors.kendall_ponderado.isna()
    fields = [f for f in ["n_instancias", "n_atributos", "classe_majoritaria", "proporcao_constantes", "distancia_vizinho"] if f in errors]
    # Unidade inferencial: grupo, evitando contar tasks repetidas como independentes.
    independent = errors.groupby("grupo")[["kendall_ponderado"] + fields].mean()
    records = []
    for field in fields:
        data = independent[[field, "kendall_ponderado"]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p = spearmanr(data[field], data.kendall_ponderado)
        records.append({"variavel": field, "spearman": rho, "p_value": p, "n_grupos": len(data)})
    correlations = pd.DataFrame(records)
    finite = correlations.p_value.notna()
    values = correlations.loc[finite, "p_value"].to_numpy()
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.minimum.accumulate((values[order]*len(values)/np.arange(1,len(values)+1))[::-1])[::-1])
    correlations.loc[finite, "q_bh"] = adjusted
    return errors.sort_values("kendall_ponderado", na_position="first"), correlations


def paired_comparisons(folds, config):
    rows = []
    for metric in METRICS:
        pivot = folds.pivot(index=["grupo", "task_id"], columns="modelo", values=metric)
        for model in ["random_forest", "knn_padronizado_ajustado", "media_global"]:
            pair = pivot[[model, "knn_ajustado"]].dropna()
            delta = (pair[model] - pair.knn_ajustado).groupby(level="grupo").mean()
            lo, hi = bootstrap_mean(delta, config["bootstrap_samples"], config["seed"])
            rows.append({"comparacao": model + " - knn_ajustado", "metrica": metric,
                         "diferenca_media_grupos": delta.mean(), "ic95_inferior": lo,
                         "ic95_superior": hi, "n_grupos_pareados": len(delta)})
    return pd.DataFrame(rows)


def run(config_path, output, include_ineligible=False):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = PROJECT / "data/meta_conjunto"
    x, y = np.load(meta/"X_emb.npy"), np.load(meta/"Y_accuracy.npy")
    ids = np.load(meta/"task_ids.npy")
    raw = pd.read_csv(meta/"pipelines_raw.csv", keep_default_na=False)
    # Ordem de colunas explicitamente definida pela construção histórica.
    from itertools import product
    keys = list(product(["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                        ["PCA", "SelectKBest", "None"],
                        ["RandomForest", "SVM", "kNN", "MLP", "DecisionTree"]))
    if x.shape != (len(ids), 64) or y.shape != (len(ids), len(keys)) or len(set(ids)) != len(ids):
        raise ValueError("Dimensões ou IDs inválidos.")
    if not np.isfinite(x).all():
        raise ValueError("Embeddings não finitos.")
    # Confere alinhamento das matrizes com a fonte em formato longo.
    lookup = raw.set_index(["task_id", "scaler", "selector", "classifier"])
    expected = np.array([[pd.to_numeric(lookup.loc[(int(tid), *key), "balanced_accuracy_mean"], errors="coerce")
                          for key in keys] for tid in ids])
    np.testing.assert_allclose(y, expected, atol=1e-6, equal_nan=True)
    # Manifesto portável permite repetir avaliação sem subir caches pessoais.
    catalog_path = output/"datasets.csv"
    if list((PROJECT/"data/datasets").glob("task_*.pkl")):
        catalog = dataset_catalog(ids, raw, PROJECT/"data/datasets")
    elif catalog_path.exists():
        catalog = pd.read_csv(catalog_path).set_index("task_id").loc[ids].reset_index()
    else:
        raise FileNotFoundError("Forneça os caches locais ou um datasets.csv auditado.")
    metadata_path = PROJECT/"resultados/revisao_2026/openml_tasks.csv"
    metadata = pd.read_csv(metadata_path)
    # O catálogo publicado já pode conter metadados; atualiza a partir da fonte.
    catalog = catalog.drop(columns=[c for c in metadata.columns if c != "task_id" and c in catalog])
    catalog = catalog.merge(metadata, on="task_id", how="left", validate="one_to_one")
    if catalog.metadata_status.ne("ok").any():
        raise ValueError("Metadados OpenML não conferidos para todas as tasks.")
    catalog["elegivel"] = eligibility(catalog)
    catalog["motivo_exclusao"] = np.where(catalog.elegivel, "", "classe_com_menos_de_5_exemplares_ou_tipo_tamanho_invalido")
    catalog.to_csv(catalog_path, index=False)
    groups = catalog.grupo.to_numpy()
    legacy = legacy_loo(x, y, ids, groups, config)
    legacy.to_csv(output/"loo_reproducao.csv", index=False)
    archive_n, archive_missing = len(ids), int(np.isnan(y).sum())
    mask = np.ones(len(catalog), dtype=bool) if include_ineligible else catalog.elegivel.to_numpy()
    x, y, ids, groups = x[mask], y[mask], ids[mask], groups[mask]
    catalog_evaluable = catalog.loc[mask].copy()
    folds, predictions, selections, splits, distances = grouped_evaluation(x, y, ids, groups, keys, config)
    folds.to_csv(output/"metricas_por_task.csv", index=False)
    predictions.to_csv(output/"predicoes.csv", index=False)
    selections.to_csv(output/"selecao_k_interna.csv", index=False)
    (output/"splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = summarize(pd.concat([legacy, folds]), config)
    summary.to_csv(output/"resumo.csv", index=False)
    paired = paired_comparisons(folds, config)
    paired.to_csv(output/"comparacoes_pareadas.csv", index=False)
    errors, correlations = error_analysis(folds, catalog_evaluable, distances)
    errors.to_csv(output/"diagnostico_erros.csv", index=False)
    correlations.to_csv(output/"correlacoes_erros.csv", index=False)
    manifest = {"config": config, "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
                "input_sha256": {f: sha256(meta/f) for f in ["X_emb.npy", "Y_accuracy.npy", "task_ids.npy", "pipelines_raw.csv"]},
                "code_sha256": sha256(__file__), "config_sha256": sha256(config_path),
                "archive_tasks": archive_n, "archive_missing_scores": archive_missing,
                "n_tasks": len(ids), "n_excluded_tasks": archive_n-len(ids),
                "include_ineligible": include_ineligible,
                "n_names": catalog_evaluable.nome_dataset.nunique(),
                "n_groups": catalog_evaluable.grupo.nunique(), "missing_scores": int(np.isnan(y).sum()),
                "holdout_363713_in_training_archive": bool(363713 in catalog.task_id.to_numpy()),
                "metadata_sha256": sha256(metadata_path),
                "dataset_catalog_sha256": sha256(catalog_path),
                "caveat": "Retrospective grouped CV on historical Phase 1. Bootstrap intervals are descriptive; training folds overlap."}
    (output/"manifesto.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    table = summary[(summary.protocolo == "cv_grupos") & summary.metrica.isin(["spearman", "kendall_ponderado", "ndcg_5", "regret_1"])]
    table = table.pivot(index="modelo", columns="metrica", values="media_grupos")
    # Exportação direta evita dependência opcional do pandas em Jinja2.
    header = "Modelo & Spearman & Tau ponderado & NDCG@5 & Regret@1 \\\\"
    latex = [r"\begin{tabular}{lrrrr}", r"\toprule", header, r"\midrule"]
    for model, row in table.iterrows():
        values = [row[m] for m in ["spearman", "kendall_ponderado", "ndcg_5", "regret_1"]]
        latex.append(model.replace("_", r"\_") + " & " + " & ".join(f"{v:.4f}" for v in values) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output/"tabela_modelos.tex").write_text("\n".join(latex), encoding="utf-8")
    text = ["# Resultados da revisão\n", f"{len(ids)} tasks; {manifest['n_names']} nomes; {manifest['n_groups']} grupos.\n",
            "Médias por grupo; menor regret é melhor. Intervalos e contagens em resumo.csv.\n", table.to_string(),
            f"\nCasos k-NN ajustado com tau ponderado <= 0: {int(errors.correlacao_nao_positiva.sum())}; indefinidos: {int(errors.correlacao_indefinida.sum())}.\n",
            "Correlação diagnóstica não demonstra causalidade. CIs por bootstrap dos grupos são descritivos e não incorporam toda a incerteza de retreinamento.\n",
            "O teste é retrospectivo: todos os datasets já haviam sido observados no desenvolvimento anterior. A Fase 1 histórica não foi refeita.\n"]
    (output/"RESULTADOS.md").write_text("\n".join(text), encoding="utf-8")
    print(table.to_string(), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=PROJECT/"resultados/revisao_2026")
    parser.add_argument("--include-ineligible", action="store_true", help="Somente análise de sensibilidade do arquivo completo.")
    args = parser.parse_args()
    run(args.config, args.output, args.include_ineligible)
