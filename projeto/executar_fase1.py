"""
Execução completa da Fase 1 – Construção do Meta-conjunto.

Passos:
  1. Carregar datasets do OpenML (≥ 100 datasets após filtragem)
  2. Gerar embeddings via Dataset2Vec (ou fallback estatístico)
  3. Executar todos os 60 pipelines com validação cruzada estratificada
  4. Montar e salvar o meta-conjunto (formato longo + matrizes de ranking)
  5. Gerar visualizações exploratórias

Saídas:
  data/meta_conjunto/meta_conjunto.pkl    ← DataFrame longo
  data/meta_conjunto/meta_conjunto.csv    ← CSV expandido
  data/meta_conjunto/X_emb.npy           ← matriz (n_datasets × emb_dim)
  data/meta_conjunto/Y_accuracy.npy      ← matriz (n_datasets × n_pipelines)
  data/meta_conjunto/task_ids.npy        ← IDs dos datasets
  data/meta_conjunto/pipeline_keys.pkl   ← lista de (scaler, selector, clf)
  resultados/figuras/distribuicao_yperf.png
  resultados/figuras/heatmap_*.png
  resultados/figuras/boxplot_classifier.png
  resultados/figuras/umap_embeddings.png

Uso:
    python executar_fase1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fase1_meta_conjunto.dataset_loader      import carregar_todos
from fase1_meta_conjunto.embedding_generator import gerar_todos_embeddings
from fase1_meta_conjunto.pipeline_builder    import executar_todos_pipelines
from fase1_meta_conjunto.meta_conjunto       import (montar_meta_conjunto,
                                                      salvar_meta_conjunto)
from utils.visualizacao import (plot_distribuicao_yperf,
                                plot_heatmap_componentes,
                                plot_boxplot_classifier,
                                plot_umap_embeddings)


def main():
    print("=" * 60)
    print("  FASE 1 – Construção do Meta-conjunto")
    print("=" * 60)

    # ── 1. Carregar datasets ───────────────────────────────────────────────
    print("\n[1/5] Carregando datasets (OpenML)...")
    datasets = carregar_todos(usar_cache=True)
    print(f"  Total carregado: {len(datasets)} datasets")

    # ── 2. Gerar embeddings ────────────────────────────────────────────────
    print("\n[2/5] Gerando embeddings (Dataset2Vec / fallback)...")
    embeddings = gerar_todos_embeddings(datasets)

    # ── 3. Executar pipelines (60 combinações × n_datasets) ───────────────
    print("\n[3/5] Executando 60 pipelines em todos os datasets...")
    print("  (pode levar várias horas dependendo do número de datasets)")
    df_pipelines = executar_todos_pipelines(datasets, verbose=True)

    raw_path = os.path.join("data", "meta_conjunto", "pipelines_raw.csv")
    df_pipelines.to_csv(raw_path, index=False)
    print(f"  Resultado bruto salvo em: {raw_path}")

    # ── 4. Montar meta-conjunto ────────────────────────────────────────────
    print("\n[4/5] Montando meta-conjunto...")
    meta = montar_meta_conjunto(df_pipelines, embeddings)

    # ── 5. Salvar (formato longo + matrizes de ranking) ───────────────────
    print("\n[5/5] Salvando meta-conjunto e matrizes de ranking...")
    # Passa 'embeddings' para que salvar_meta_conjunto gere X_emb.npy e Y_accuracy.npy
    salvar_meta_conjunto(meta, embeddings=embeddings)

    # ── Visualizações ──────────────────────────────────────────────────────
    print("\n[VIZ] Gerando visualizações exploratórias...")
    rotulos = {ds["task_id"]: ds["nome"] for ds in datasets}
    plot_distribuicao_yperf(meta)
    plot_heatmap_componentes(meta, eixo_x="classifier", eixo_y="scaler")
    plot_heatmap_componentes(meta, eixo_x="classifier", eixo_y="selector")
    plot_boxplot_classifier(meta)
    plot_umap_embeddings(embeddings, rotulos=rotulos)

    print("\n" + "=" * 60)
    print("  Fase 1 concluída.")
    print(f"  Meta-conjunto: {len(meta)} instâncias | "
          f"{meta['task_id'].nunique()} datasets | "
          f"{meta.groupby(['scaler','selector','classifier']).ngroups} pipelines")
    print("=" * 60)


if __name__ == "__main__":
    main()
