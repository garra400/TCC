"""
Avaliação da qualidade do preditor de ranking via Leave-One-Out (LOO).

Metodologia:
  Para cada dataset d_i no meta-conjunto (LOO):
    1. Treina o preditor nos outros n-1 datasets
    2. Prediz as notas de todos os 60 pipelines para d_i
    3. Compara o ranking predito com o ranking real pelas métricas abaixo

Métricas reportadas por fold e agregadas (média ± desvio padrão):

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Spearman (ρ)  — scipy.stats.spearmanr                              │
  │    Correlação de postos de Spearman entre o vetor de scores preditos │
  │    e o vetor real. ρ=1 → ordem perfeita; ρ=0 → sem correlação.      │
  │                                                                      │
  │  Kendall's tau (τ)  — scipy.stats.kendalltau                        │
  │    Fração líquida de pares de pipelines corretamente ordenados.      │
  │    τ=1 → todos os pares na ordem certa; τ=0 → acaso; τ=-1 → inverso.│
  │    *** Nota sobre terminologia ***                                   │
  │    "Coeficiente de concordância de Kendall" pode referir-se a:      │
  │      (a) Kendall's tau (τ) — para comparar DOIS rankings. ← usado   │
  │      (b) Kendall's W      — para concordância entre MÚLTIPLOS       │
  │          avaliadores (e.g., comitê de juízes). Não se aplica aqui.  │
  │    Para comparar ranking predito vs. real, tau (a) é o correto.     │
  │                                                                      │
  │  NDCG  — sklearn.metrics.ndcg_score                                 │
  │    Normalized Discounted Cumulative Gain: penaliza mais erros no     │
  │    topo do ranking (os melhores pipelines erroneamente omitidos      │
  │    custam mais do que erros no fundo da lista).                     │
  │    NDCG=1 → ranking predito idêntico ao real; NDCG=0 → pior caso.  │
  │    Implementação direta do scikit-learn (sklearn.metrics.ndcg_score)│
  └──────────────────────────────────────────────────────────────────────┘

Referências:
  Spearman, C. (1904). The proof and measurement of association.
    The American Journal of Psychology, 15(1), 72–101.
  Kendall, M.G. (1938). A new measure of rank correlation.
    Biometrika, 30(1/2), 81–93.
  Järvelin, K. & Kekäläinen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM TOIS, 20(4), 422–446. (NDCG)
  sklearn.metrics.ndcg_score — https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.ndcg_score.html
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, weightedtau
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import ndcg_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import KNN_K_RANGE, METRICAS_DIR
from fase2_metamodelos.preditor_knn import PreditorKNNRanking


# ─── LOO para KNN ─────────────────────────────────────────────────────────────

def avaliar_loo_knn(X_emb: np.ndarray,
                    Y_accuracy: np.ndarray,
                    task_ids: list[int],
                    pipeline_keys: list[tuple],
                    k_values: list[int] = KNN_K_RANGE,
                    ponderado: bool = True,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Avalia o preditor KNN com LOO para diferentes valores de k.

    Métricas calculadas por fold:
      - Spearman (ρ)  : scipy.stats.spearmanr
      - Kendall (τ)   : scipy.stats.kendalltau   ← concordância entre dois rankings
      - NDCG          : sklearn.metrics.ndcg_score ← métrica de ranking do sklearn

    Retorna
    -------
    pd.DataFrame com colunas:
        k, spearman_media, spearman_std,
        kendalltau_media, kendalltau_std,
        ndcg_media, ndcg_std,
        n_folds_validos
    """
    loo = LeaveOneOut()
    resultados = []

    for k in k_values:
        if verbose:
            print(f"\n  [LOO-KNN] k={k}  ({len(task_ids)} folds)...")

        spearman_list         = []
        kendall_list          = []
        weighted_kendall_list = []
        ndcg_list             = []
        fold_detalhes         = []

        preditor = PreditorKNNRanking(k=k, ponderado=ponderado)

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_emb)):
            X_tr, X_te = X_emb[train_idx], X_emb[test_idx]
            Y_tr, Y_te = Y_accuracy[train_idx], Y_accuracy[test_idx]

            ids_tr = [task_ids[i] for i in train_idx]
            preditor.fit(X_tr, Y_tr, task_ids=ids_tr, pipeline_keys=pipeline_keys)

            Y_pred = preditor.predict(X_te)   # (1, n_pipelines)

            true_scores = Y_te[0]
            pred_scores = Y_pred[0]

            # Filtra pipelines sem avaliação no fold de teste
            valido = ~(np.isnan(true_scores) | np.isnan(pred_scores))
            if valido.sum() < 2:
                continue

            ts = true_scores[valido]
            ps = pred_scores[valido]

            # Descarta folds onde true_scores é constante (dataset trivial —
            # todos os pipelines têm a mesma acurácia): spearmanr retorna NaN
            # nesses casos porque não há variância para ranquear.
            if np.std(ts) < 1e-6:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _  = spearmanr(ts, ps)
                tau, _  = kendalltau(ts, ps)
                wtau, _ = weightedtau(ts, ps)
                # Ignora folds individuais com NaN residual
                if np.isnan(rho) or np.isnan(tau):
                    continue
                # ndcg_score espera arrays 2D (n_samples, n_labels)
                # true_scores precisam ser não-negativos; shifted se necessário
                ts_nn = ts - ts.min() if ts.min() < 0 else ts
                ndcg  = ndcg_score(ts_nn.reshape(1, -1), ps.reshape(1, -1))

            spearman_list.append(rho)
            kendall_list.append(tau)
            ndcg_list.append(ndcg)
            if not np.isnan(wtau):
                weighted_kendall_list.append(wtau)
            fold_detalhes.append({
                "fold":               fold,
                "task_id":            task_ids[test_idx[0]],
                "k":                  k,
                "spearman":           rho,
                "kendalltau":         tau,
                "kendall_ponderado":  float(wtau) if not np.isnan(wtau) else np.nan,
                "ndcg":               ndcg,
                "n_pipelines_validos": int(valido.sum()),
            })

        linha = {
            "k":                        k,
            "spearman_media":           float(np.mean(spearman_list)),
            "spearman_std":             float(np.std(spearman_list)),
            "spearman_min":             float(np.min(spearman_list)),
            "spearman_max":             float(np.max(spearman_list)),
            "kendalltau_media":         float(np.mean(kendall_list)),
            "kendalltau_std":           float(np.std(kendall_list)),
            "kendalltau_min":           float(np.min(kendall_list)),
            "kendalltau_max":           float(np.max(kendall_list)),
            "kendall_ponderado_media":  float(np.mean(weighted_kendall_list)) if weighted_kendall_list else np.nan,
            "kendall_ponderado_std":    float(np.std(weighted_kendall_list)) if weighted_kendall_list else np.nan,
            "kendall_ponderado_min":    float(np.min(weighted_kendall_list)) if weighted_kendall_list else np.nan,
            "kendall_ponderado_max":    float(np.max(weighted_kendall_list)) if weighted_kendall_list else np.nan,
            "ndcg_media":               float(np.mean(ndcg_list)),
            "ndcg_std":                 float(np.std(ndcg_list)),
            "ndcg_min":                 float(np.min(ndcg_list)),
            "ndcg_max":                 float(np.max(ndcg_list)),
            "n_folds_validos":          len(spearman_list),
        }
        resultados.append(linha)

        if verbose:
            kw_min = linha["kendall_ponderado_min"]
            kw_max = linha["kendall_ponderado_max"]
            kw_str = f"~{kw_min:.2f}–{kw_max:.2f}" if not (np.isnan(kw_min) or np.isnan(kw_max)) else "N/A"
            print(f"    Spearman=~{linha['spearman_min']:.2f}–{linha['spearman_max']:.2f}  |  "
                  f"Kendall=~{linha['kendalltau_min']:.2f}–{linha['kendalltau_max']:.2f}  |  "
                  f"KendallW={kw_str}  |  "
                  f"NDCG=~{linha['ndcg_min']:.2f}–{linha['ndcg_max']:.2f}  "
                  f"({linha['n_folds_validos']} folds)")

        pd.DataFrame(fold_detalhes).to_csv(
            os.path.join(METRICAS_DIR, f"loo_knn_k{k}_por_fold.csv"), index=False
        )

    df_res = pd.DataFrame(resultados).sort_values("kendalltau_media", ascending=False)
    caminho = os.path.join(METRICAS_DIR, "loo_knn_resumo.csv")
    df_res.to_csv(caminho, index=False)
    print(f"\n  Resultados KNN-LOO salvos em: {caminho}")
    return df_res


# ─── LOO para MLP ─────────────────────────────────────────────────────────────

def avaliar_loo_mlp(X_emb: np.ndarray,
                    Y_accuracy: np.ndarray,
                    task_ids: list[int],
                    pipeline_keys: list[tuple],
                    verbose: bool = True) -> pd.DataFrame:
    """
    Avalia o preditor MLP (BCE loss) com LOO.
    Métricas: Spearman, Kendall's tau e NDCG (sklearn).
    """
    try:
        from fase2_metamodelos.preditor_mlp import PreditorMLPRanking
    except RuntimeError as e:
        print(f"[AVISO] MLP indisponível: {e}")
        return pd.DataFrame()

    loo = LeaveOneOut()
    n_datasets, n_pipelines = Y_accuracy.shape
    emb_dim = X_emb.shape[1]

    spearman_list         = []
    kendall_list          = []
    weighted_kendall_list = []
    ndcg_list             = []
    fold_detalhes         = []

    if verbose:
        print(f"\n  [LOO-MLP] {n_datasets} folds...")

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_emb)):
        X_tr, X_te = X_emb[train_idx], X_emb[test_idx]
        Y_tr, Y_te = Y_accuracy[train_idx], Y_accuracy[test_idx]

        preditor = PreditorMLPRanking(emb_dim=emb_dim, n_pipelines=n_pipelines)
        preditor.fit(X_tr, Y_tr)

        Y_pred      = preditor.predict(X_te)
        true_scores = Y_te[0]
        pred_scores = Y_pred[0]

        valido = ~(np.isnan(true_scores) | np.isnan(pred_scores))
        if valido.sum() < 2:
            continue

        ts = true_scores[valido]
        ps = pred_scores[valido]

        if np.std(ts) < 1e-6:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _  = spearmanr(ts, ps)
            tau, _  = kendalltau(ts, ps)
            wtau, _ = weightedtau(ts, ps)
            if np.isnan(rho) or np.isnan(tau):
                continue
            ts_nn  = ts - ts.min() if ts.min() < 0 else ts
            ndcg   = ndcg_score(ts_nn.reshape(1, -1), ps.reshape(1, -1))

        spearman_list.append(rho)
        kendall_list.append(tau)
        ndcg_list.append(ndcg)
        if not np.isnan(wtau):
            weighted_kendall_list.append(wtau)
        fold_detalhes.append({
            "fold":              fold,
            "task_id":           task_ids[test_idx[0]],
            "spearman":          rho,
            "kendalltau":        tau,
            "kendall_ponderado": float(wtau) if not np.isnan(wtau) else np.nan,
            "ndcg":              ndcg,
        })

        if verbose and (fold + 1) % 10 == 0:
            print(f"    fold {fold+1}/{n_datasets}: "
                  f"Spearman={rho:.4f}, Kendall={tau:.4f}, NDCG={ndcg:.4f}")

    df_folds = pd.DataFrame(fold_detalhes)

    resumo = pd.DataFrame([{
        "modelo":                     "MLP-BCE",
        "spearman_media":             float(np.mean(spearman_list)),
        "spearman_std":               float(np.std(spearman_list)),
        "spearman_min":               float(np.min(spearman_list)),
        "spearman_max":               float(np.max(spearman_list)),
        "kendalltau_media":           float(np.mean(kendall_list)),
        "kendalltau_std":             float(np.std(kendall_list)),
        "kendalltau_min":             float(np.min(kendall_list)),
        "kendalltau_max":             float(np.max(kendall_list)),
        "kendall_ponderado_media":    float(np.mean(weighted_kendall_list)) if weighted_kendall_list else np.nan,
        "kendall_ponderado_std":      float(np.std(weighted_kendall_list)) if weighted_kendall_list else np.nan,
        "kendall_ponderado_min":      float(np.min(weighted_kendall_list)) if weighted_kendall_list else np.nan,
        "kendall_ponderado_max":      float(np.max(weighted_kendall_list)) if weighted_kendall_list else np.nan,
        "ndcg_media":                 float(np.mean(ndcg_list)),
        "ndcg_std":                   float(np.std(ndcg_list)),
        "ndcg_min":                   float(np.min(ndcg_list)),
        "ndcg_max":                   float(np.max(ndcg_list)),
        "n_folds_validos":            len(spearman_list),
    }])

    df_folds.to_csv(os.path.join(METRICAS_DIR, "loo_mlp_por_fold.csv"), index=False)
    resumo.to_csv(os.path.join(METRICAS_DIR, "loo_mlp_resumo.csv"), index=False)

    if verbose:
        kw_min = resumo["kendall_ponderado_min"].iloc[0]
        kw_max = resumo["kendall_ponderado_max"].iloc[0]
        kw_str = f"~{kw_min:.2f}–{kw_max:.2f}" if not (np.isnan(kw_min) or np.isnan(kw_max)) else "N/A"
        print(f"  MLP: Spearman=~{resumo['spearman_min'].iloc[0]:.2f}–{resumo['spearman_max'].iloc[0]:.2f}  |  "
              f"Kendall=~{resumo['kendalltau_min'].iloc[0]:.2f}–{resumo['kendalltau_max'].iloc[0]:.2f}  |  "
              f"KendallW={kw_str}  |  "
              f"NDCG=~{resumo['ndcg_min'].iloc[0]:.2f}–{resumo['ndcg_max'].iloc[0]:.2f}")
    return df_folds


# ─── Métricas pontuais ────────────────────────────────────────────────────────

def metricas_ranking_par(true_scores: np.ndarray,
                          pred_scores: np.ndarray) -> dict:
    """
    Calcula Spearman, Kendall's tau e NDCG entre dois vetores de scores.

    Kendall's tau (τ):
      Mede a fração de pares de itens cujas ordens relativas coincidem entre
      o ranking predito e o real. Matematicamente:
        τ = (pares concordantes − pares discordantes) / (n*(n-1)/2)
      Esta é uma medida de CONCORDÂNCIA ENTRE DOIS RANKINGS, diferente do
      Coeficiente de Concordância de Kendall (W), que mede concordância entre
      múltiplos avaliadores — não aplicável aqui.

    NDCG (sklearn.metrics.ndcg_score):
      Penaliza erros de ranking mais fortemente quando ocorrem nas primeiras
      posições, refletindo o uso real (o usuário testa os primeiros pipelines).

    Retorna dict com:
      spearman, p_spearman, kendalltau, p_kendall, ndcg
    """
    valido = ~(np.isnan(true_scores) | np.isnan(pred_scores))
    if valido.sum() < 2:
        return {"spearman": np.nan, "p_spearman": np.nan,
                "kendalltau": np.nan, "p_kendall": np.nan,
                "ndcg": np.nan}

    ts = true_scores[valido]
    ps = pred_scores[valido]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_rho     = spearmanr(ts, ps)
        tau, p_tau     = kendalltau(ts, ps)
        wtau, p_wtau   = weightedtau(ts, ps)
        ts_nn = ts - ts.min() if ts.min() < 0 else ts
        ndcg  = ndcg_score(ts_nn.reshape(1, -1), ps.reshape(1, -1))

    return {
        "spearman":              rho,
        "p_spearman":            p_rho,
        "kendalltau":            tau,
        "p_kendall":             p_tau,
        "kendall_ponderado":     float(wtau)   if not np.isnan(wtau)   else np.nan,
        "p_kendall_ponderado":   float(p_wtau) if not np.isnan(p_wtau) else np.nan,
        "ndcg":                  ndcg,
    }


# ─── Correlação estatística: Kendall ponderado × embeddings ──────────────────

def analisar_correlacao_kendall_datasets(
    df_folds: pd.DataFrame,
    X_emb: np.ndarray,
    task_ids: list[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Para cada dataset no LOO, usa o kendall_ponderado calculado por fold e
    investiga correlação estatística (Spearman) com as dimensões do embedding.

    Identifica quais aspectos da representação do dataset predizem a qualidade
    do ranqueamento — útil para entender quando o meta-aprendizado funciona bem.

    Retorna
    -------
    pd.DataFrame com colunas: dimensao, spearman, p_value, abs_spearman
        ordenado por abs_spearman desc (top correlações primeiro).
    """
    if "kendall_ponderado" not in df_folds.columns:
        print("[AVISO] df_folds não contém coluna 'kendall_ponderado'.")
        return pd.DataFrame()

    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    df_valid  = df_folds.dropna(subset=["kendall_ponderado"]).copy()
    df_valid  = df_valid[df_valid["task_id"].isin(id_to_idx)]

    if len(df_valid) < 10:
        print(f"[AVISO] Apenas {len(df_valid)} folds válidos para análise de correlação.")
        return pd.DataFrame()

    emb_indices = [id_to_idx[tid] for tid in df_valid["task_id"]]
    X_sub = X_emb[emb_indices]          # (n_valid, emb_dim)
    y     = df_valid["kendall_ponderado"].values

    resultados = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dim in range(X_sub.shape[1]):
            rho, p = spearmanr(X_sub[:, dim], y)
            resultados.append({
                "dimensao":     dim,
                "spearman":     rho,
                "p_value":      p,
                "abs_spearman": abs(rho),
            })
        # norma L2 como proxy de distintividade do dataset no espaço de embeddings
        normas = np.linalg.norm(X_sub, axis=1)
        rho_n, p_n = spearmanr(normas, y)
        resultados.append({
            "dimensao":     "norma_L2",
            "spearman":     rho_n,
            "p_value":      p_n,
            "abs_spearman": abs(rho_n),
        })

    df_corr = (pd.DataFrame(resultados)
                 .sort_values("abs_spearman", ascending=False)
                 .reset_index(drop=True))

    caminho = os.path.join(METRICAS_DIR, "correlacao_kendall_por_dataset.csv")
    df_corr.to_csv(caminho, index=False)

    if verbose:
        print(f"\n  Correlação Kendall ponderado × embedding ({len(df_valid)} datasets):")
        for _, row in df_corr.head(5).iterrows():
            print(f"    dim={row['dimensao']:>8}: rho={row['spearman']:+.4f}  p={row['p_value']:.4f}")
        print(f"  Análise completa salva em: {caminho}")

    return df_corr
