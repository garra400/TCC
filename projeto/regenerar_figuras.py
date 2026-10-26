"""
Regenera as figuras de resultados/figuras/ a partir dos artefatos já salvos
em disco (meta-conjunto, matrizes X_emb/Y_accuracy e métricas LOO da Fase 2),
sem reexecutar os experimentos completos.

Motivação: aplicar a formatação numérica em pt-BR (vírgula como separador
decimal) introduzida em utils/visualizacao.py — anotação do orientador
"Corrigir decimais" — a todas as figuras já produzidas.
"""

import os
import pandas as pd

from config import METRICAS_DIR
from fase1_meta_conjunto.meta_conjunto import carregar_meta_conjunto, carregar_matrizes
from utils.visualizacao import (
    plot_distribuicao_yperf,
    plot_heatmap_componentes,
    plot_boxplot_classifier,
    plot_umap_embeddings,
    plot_ranking_loo_knn,
    plot_distribuicao_metricas_ranking,
    plot_kendall_ponderado_por_dataset,
    plot_correlacao_embeddings_matrizes,
)


def main():
    print("=" * 60)
    print("  Regeneração de figuras (formatação numérica em pt-BR)")
    print("=" * 60)

    # ── Fase 1 ────────────────────────────────────────────────────────────────
    print("\n[1/2] Figuras da Fase 1 (meta-conjunto e embeddings)...")
    meta = carregar_meta_conjunto()
    X_emb, Y_accuracy, task_ids, _ = carregar_matrizes()
    rotulos = {tid: str(tid) for tid in task_ids}
    banco = {task_ids[i]: X_emb[i] for i in range(len(task_ids))}

    plot_distribuicao_yperf(meta)
    plot_heatmap_componentes(meta, eixo_x="classifier", eixo_y="scaler")
    plot_heatmap_componentes(meta, eixo_x="classifier", eixo_y="selector")
    plot_boxplot_classifier(meta)
    plot_umap_embeddings(banco, rotulos=rotulos)

    # ── Fase 2 ────────────────────────────────────────────────────────────────
    print("\n[2/2] Figuras da Fase 2 (avaliação LOO do preditor de ranking)...")
    df_knn = pd.read_csv(os.path.join(METRICAS_DIR, "loo_knn_resumo.csv"))
    plot_ranking_loo_knn(df_knn)

    k_otimo = int(df_knn.sort_values("kendalltau_media", ascending=False).iloc[0]["k"])
    fold_path = os.path.join(METRICAS_DIR, f"loo_knn_k{k_otimo}_por_fold.csv")
    if os.path.exists(fold_path):
        df_folds = pd.read_csv(fold_path)
        plot_distribuicao_metricas_ranking(df_folds)
        plot_kendall_ponderado_por_dataset(df_folds)

    plot_correlacao_embeddings_matrizes(banco)

    print("\n" + "=" * 60)
    print("  Figuras regeneradas em: resultados/figuras/")
    print("=" * 60)


if __name__ == "__main__":
    main()
