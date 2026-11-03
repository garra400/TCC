"""Grouped nested evaluation and ranking metrics used by the experiments."""

import hashlib
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

METRICS = [
    "spearman",
    "kendalltau",
    "kendall_ponderado",
    "ndcg",
    "ndcg_5",
    "regret_1",
    "regret_5",
]

PRIMARY = "knn_padronizado_ajustado"
MODELS = ["aleatorio", "media_global", "knn_ajustado", PRIMARY, "random_forest"]
COMPARISONS = [
    ("random_forest", "media_global"),
    (PRIMARY, "media_global"),
    ("knn_ajustado", "media_global"),
    ("random_forest", PRIMARY),
]


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
    if k < 1 or not len(x_train):
        raise ValueError("k and the training-set size must be positive.")
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
        result["kendall_ponderado"] = float(
            weightedtau(t, p, rank=True, additive=True).statistic
        )
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


def select_k(x, y, groups, config, scaled=False):
    """Seleciona k apenas por validação interna por grupos do treino externo."""
    cv = GroupKFold(
        n_splits=min(config["inner_folds"], len(np.unique(groups))),
        shuffle=True,
        random_state=config["seed"],
    )
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
                real, predicted = y[index, supported], prediction[supported]
                if config["primary_metric"] == "kendall_ponderado":
                    value = (
                        np.nan
                        if real.size < 2 or np.ptp(real) == 0 or np.ptp(predicted) == 0
                        else weightedtau(
                            real, predicted, rank=True, additive=True
                        ).statistic
                    )
                else:
                    value = ranking_metrics(real, predicted)[config["primary_metric"]]
                scores[k].append({"group": groups[index], "score": value})
    means = {}
    for k, values in scores.items():
        means[k] = pd.DataFrame(values).groupby("group").score.mean().mean()
    # Empates escolhem o menor k; nenhum score do teste externo é consultado.
    best = max(
        sorted(means), key=lambda k: means[k] if np.isfinite(means[k]) else -np.inf
    )
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
            lo, hi = bootstrap_mean(
                grouped, config["bootstrap_samples"], config["seed"]
            )
            rows.append(
                dict(
                    protocolo=protocol,
                    modelo=model,
                    metrica=metric,
                    media_tasks=values.mean(),
                    desvio_tasks=values.std(ddof=0),
                    media_grupos=grouped.mean(),
                    ic95_grupos_inferior=lo,
                    ic95_grupos_superior=hi,
                    n_tasks=len(data),
                    n_validos=len(values),
                    n_grupos_validos=len(grouped),
                    n_indefinidos=int(data[metric].isna().sum()),
                )
            )
    return pd.DataFrame(rows)


def grouped_evaluation(x, y, ids, groups, keys, config):
    rows, predictions, selections, splits, distances = [], [], [], [], []
    cv = GroupKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=config["seed"]
    )
    for fold, (train, test) in enumerate(cv.split(x, groups=groups)):
        assert not set(groups[train]) & set(groups[test])
        splits.append(
            {
                "fold": fold,
                "train_tasks": ids[train].tolist(),
                "test_tasks": ids[test].tolist(),
                "train_groups": sorted(set(groups[train])),
                "test_groups": sorted(set(groups[test])),
            }
        )
        filled, supported = fill_targets(y[train])
        k, inner = select_k(x[train], y[train], groups[train], config)
        ks, inner_scaled = select_k(
            x[train], y[train], groups[train], config, scaled=True
        )
        for scale, scores in [(False, inner), (True, inner_scaled)]:
            selections.extend(
                {
                    "fold": fold,
                    "scaled": scale,
                    "k": value,
                    "inner_group_mean": score,
                    "selected": value == (ks if scale else k),
                }
                for value, score in scores.items()
            )
        scaler = StandardScaler().fit(x[train])
        rf = RandomForestRegressor(
            n_estimators=config["rf_estimators"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            random_state=config["seed"],
            n_jobs=config.get("n_jobs", 1),
        )
        rf.fit(x[train], filled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(y[train], axis=0)
        rng = np.random.default_rng(config["seed"] + fold)
        outputs = {
            "knn_k10_fixo": knn_predict(x[train], filled, x[test], 10),
            "knn_ajustado": knn_predict(x[train], filled, x[test], k),
            "knn_padronizado_ajustado": knn_predict(
                scaler.transform(x[train]), filled, scaler.transform(x[test]), ks
            ),
            "random_forest": rf.predict(x[test]),
            "media_global": np.tile(means, (len(test), 1)),
            "aleatorio": rng.random((len(test), y.shape[1])),
        }
        dist = cdist(x[test], x[train])
        nearest = dist.argmin(axis=1)
        for row, index in enumerate(test):
            distances.append(
                {
                    "task_id": int(ids[index]),
                    "fold": fold,
                    "distancia_vizinho": dist[row, nearest[row]],
                    "vizinho_task_id": int(ids[train[nearest[row]]]),
                }
            )
        for name, output in outputs.items():
            output[:, ~supported] = np.nan
            for index, prediction in zip(test, output):
                rows.append(
                    dict(
                        protocolo="cv_grupos",
                        modelo=name,
                        fold=fold,
                        task_id=int(ids[index]),
                        grupo=groups[index],
                        **ranking_metrics(y[index], prediction),
                    )
                )
                predictions.extend(
                    {
                        "fold": fold,
                        "modelo": name,
                        "task_id": int(ids[index]),
                        "pipeline_id": j,
                        "scaler": key[0],
                        "selector": key[1],
                        "classifier": key[2],
                        "real": y[index, j],
                        "predito": prediction[j],
                        "observado_treino": bool(supported[j]),
                    }
                    for j, key in enumerate(keys)
                )
        print(
            f"Grupos fold {fold + 1}/{config['outer_folds']}: k={k}; k padronizado={ks}",
            flush=True,
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(predictions),
        pd.DataFrame(selections),
        splits,
        pd.DataFrame(distances),
    )


def paired(data, config, comparisons=COMPARISONS):
    records = []
    for metric in ["kendall_ponderado", "ndcg_5", "regret_1"]:
        pivot = data.pivot(index=["grupo", "task_id"], columns="modelo", values=metric)
        for first, second in comparisons:
            both = pivot[[first, second]].dropna()
            delta = (both[first] - both[second]).groupby(level="grupo").mean()
            lo, hi = bootstrap_mean(delta, config["bootstrap_samples"], config["seed"])
            records.append(
                {
                    "primeiro": first,
                    "segundo": second,
                    "metrica": metric,
                    "diferenca_media_grupos": float(delta.mean()),
                    "ic95_inferior": lo,
                    "ic95_superior": hi,
                    "n_tarefas_pareadas": len(both),
                    "n_grupos_pareados": len(delta),
                }
            )
    return pd.DataFrame(records)
