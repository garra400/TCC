"""
Execução completa da Fase 2 – Predição de Ranking e Avaliação.

Pré-requisito: executar_fase1.py deve ter sido concluído.

Metodologia (migração de classificação para ranking):
  - Antes : metamodelos RF/SVM/MLP classificavam pipelines como
            adequados/inadequados (Y_qual binário, cross-validation k-fold).
  - Agora : preditor KNN multi-output (KDTree) e preditor MLP (BCE loss)
            atribuem notas contínuas a TODOS os 60 pipelines simultaneamente,
            avaliados por Leave-One-Out (LOO) com coeficientes de Spearman e
            Kendall's tau como métricas de qualidade do ranking.

Passos:
  1. Carregar matrizes X_emb e Y_accuracy da Fase 1
  2. Avaliação LOO do preditor KNN para k ∈ {1,3,5,7,10}
  3. Avaliação LOO do preditor MLP com BCE loss  (opcional — requer PyTorch)
  4. Seleção do melhor k (pelo Kendall médio)
  5. Demonstração de recomendação (holdout = último dataset)
  6. Geração de visualizações de ranking

Saídas:
  resultados/metricas/loo_knn_resumo.csv       ← Spearman/Kendall por k
  resultados/metricas/loo_knn_k{k}_por_fold.csv← detalhes por fold
  resultados/metricas/loo_mlp_resumo.csv       ← resultado do MLP
  resultados/metricas/loo_mlp_por_fold.csv     ← detalhes por fold
  resultados/metricas/recomendacoes_holdout.csv← ranking para dataset holdout
  resultados/figuras/ranking_loo_knn.png
  resultados/figuras/ranking_distribuicao_metricas.png
  resultados/figuras/matriz_similaridade_embeddings.png

Uso:
    python executar_fase2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from fase1_meta_conjunto.meta_conjunto  import carregar_matrizes
from fase2_metamodelos.avaliacao_ranking import (avaliar_loo_knn,
                                                  avaliar_loo_mlp,
                                                  metricas_ranking_par,
                                                  analisar_correlacao_kendall_datasets)
from fase2_metamodelos.preditor_knn     import PreditorKNNRanking
from fase2_metamodelos.recomendador     import recomendar_de_embedding
from utils.visualizacao import (plot_ranking_loo_knn,
                                plot_distribuicao_metricas_ranking,
                                plot_kendall_ponderado_por_dataset,
                                plot_correlacao_embeddings_matrizes)
from config import KNN_K_PADRAO, METRICAS_DIR


def main():
    print("=" * 60)
    print("  FASE 2 – Predição de Ranking por Meta-aprendizado")
    print("=" * 60)

    # ── 1. Carregar matrizes da Fase 1 ────────────────────────────────────────
    print("\n[1/6] Carregando matrizes X_emb e Y_accuracy...")
    X_emb, Y_accuracy, task_ids, pipeline_keys = carregar_matrizes()
    n_datasets, n_pipelines = Y_accuracy.shape
    print(f"  {n_datasets} datasets × {n_pipelines} pipelines  "
          f"(NaN: {np.isnan(Y_accuracy).sum()} células)")

    # ── 2. LOO – Preditor KNN ────────────────────────────────────────────────
    print("\n[2/6] Avaliação LOO do preditor KNN (KDTree multi-output)...")
    df_knn = avaliar_loo_knn(
        X_emb, Y_accuracy, task_ids, pipeline_keys, verbose=True
    )

    print("\nResumo KNN-LOO:")
    print(df_knn[["k", "spearman_media", "spearman_std",
                   "kendalltau_media", "kendalltau_std",
                   "kendall_ponderado_media", "kendall_ponderado_std",
                   "ndcg_media", "ndcg_std"]].to_string(index=False))

    # ── 3. LOO – Preditor MLP (BCE) ──────────────────────────────────────────
    print("\n[3/6] Avaliação LOO do preditor MLP (BCE loss)...")
    try:
        df_mlp = avaliar_loo_mlp(
            X_emb, Y_accuracy, task_ids, pipeline_keys, verbose=True
        )
        if not df_mlp.empty:
            mlp_resumo = pd.read_csv(
                os.path.join(METRICAS_DIR, "loo_mlp_resumo.csv")
            )
            print("\nResumo MLP-LOO:")
            print(mlp_resumo.to_string(index=False))
    except Exception as e:
        print(f"  [AVISO] MLP pulado: {e}")
        df_mlp = pd.DataFrame()

    # ── 4. Seleção do melhor k ────────────────────────────────────────────────
    print("\n[4/6] Selecionando melhor k pelo Kendall's tau médio...")
    melhor = df_knn.iloc[0]
    k_otimo = int(melhor["k"])
    print(f"  Melhor k = {k_otimo}  "
          f"(Kendall={melhor['kendalltau_media']:.4f} "
          f"± {melhor['kendalltau_std']:.4f}, "
          f"Spearman={melhor['spearman_media']:.4f} "
          f"± {melhor['spearman_std']:.4f})")

    # ── 5. Demonstração de recomendação (holdout) ────────────────────────────
    print(f"\n[5/6] Demonstração – recomendação para dataset holdout "
          f"(task_id={task_ids[-1]})...")

    # Holdout = último dataset; treina nos demais
    idx_holdout   = len(task_ids) - 1
    train_mask    = np.ones(n_datasets, dtype=bool)
    train_mask[idx_holdout] = False

    X_tr = X_emb[train_mask]
    Y_tr = Y_accuracy[train_mask]
    ids_tr = [task_ids[i] for i in range(n_datasets) if train_mask[i]]

    X_te       = X_emb[idx_holdout]
    Y_te_true  = Y_accuracy[idx_holdout]

    df_rec = recomendar_de_embedding(
        X_te, X_tr, Y_tr, ids_tr, pipeline_keys,
        k=k_otimo, top_n=10
    )

    # Métricas do holdout
    preditor_demo = PreditorKNNRanking(k=k_otimo)
    preditor_demo.fit(X_tr, Y_tr, task_ids=ids_tr, pipeline_keys=pipeline_keys)
    Y_pred_demo   = preditor_demo.predict(X_te.reshape(1, -1))[0]

    m = metricas_ranking_par(Y_te_true, Y_pred_demo)
    print(f"\n  Holdout task_id={task_ids[-1]}:")
    print(f"    Spearman rho       = {m['spearman']:.4f}  (p={m['p_spearman']:.4f})")
    print(f"    Kendall  tau       = {m['kendalltau']:.4f}  (p={m['p_kendall']:.4f})")
    kw_h = m.get("kendall_ponderado", float("nan"))
    p_kw_h = m.get("p_kendall_ponderado", float("nan"))
    print(f"    Kendall ponderado  = {kw_h:.4f}  (p={p_kw_h:.4f})")
    print(f"    NDCG               = {m['ndcg']:.4f}  (sklearn.metrics.ndcg_score)")

    rec_path = os.path.join(METRICAS_DIR, "recomendacoes_holdout.csv")
    df_rec.to_csv(rec_path)
    print(f"\n  Recomendações salvas em: {rec_path}")

    # ── 6. Visualizações e análise de correlação ─────────────────────────────
    print("\n[6/6] Gerando visualizações de ranking e análise de correlação...")
    plot_ranking_loo_knn(df_knn)

    # Lê fold-a-fold do melhor k para distribuição e correlação
    fold_path = os.path.join(
        METRICAS_DIR, f"loo_knn_k{k_otimo}_por_fold.csv"
    )
    if os.path.exists(fold_path):
        df_folds = pd.read_csv(fold_path)
        plot_distribuicao_metricas_ranking(df_folds)
        plot_kendall_ponderado_por_dataset(df_folds)
        analisar_correlacao_kendall_datasets(df_folds, X_emb, task_ids, verbose=True)

    # Similaridade entre embeddings
    banco = {task_ids[i]: X_emb[i] for i in range(n_datasets)}
    plot_correlacao_embeddings_matrizes(banco)

    print("\n" + "=" * 60)
    print("  Fase 2 concluída.")
    print(f"  Melhor preditor: KNN k={k_otimo}")
    print(f"  Kendall médio        (LOO): {melhor['kendalltau_media']:.4f} "
          f"± {melhor['kendalltau_std']:.4f}")
    kw_m = melhor.get("kendall_ponderado_media", float("nan"))
    kw_s = melhor.get("kendall_ponderado_std", float("nan"))
    print(f"  Kendall ponderado    (LOO): {kw_m:.4f} ± {kw_s:.4f}")
    print(f"  Spearman médio       (LOO): {melhor['spearman_media']:.4f} "
          f"± {melhor['spearman_std']:.4f}")
    print(f"  NDCG médio           (LOO): {melhor['ndcg_media']:.4f} "
          f"± {melhor['ndcg_std']:.4f}")
    print("  Métricas em: resultados/metricas/")
    print("  Figuras em:  resultados/figuras/")
    print("=" * 60)


if __name__ == "__main__":
    main()
