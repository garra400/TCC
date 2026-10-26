# -*- coding: utf-8 -*-
"""
Execucao de teste rapido (v2) – valida o fluxo completo de ranking com:
  - 8 datasets do OpenML (IDs validados, cache automatico)
  - 12 combinacoes de pipeline (2 scaler x 2 selector x 3 clf)
  - 3-fold CV (Fase 1)
  - LOO completo (Fase 2) — 8 folds
  - Metricas: Spearman, Kendall tau, NDCG

Tempo estimado: 5–20 minutos dependendo do cache e hardware.
"""

import sys
import os
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# ── Override das configuracoes para teste rapido ─────────────────────────────
import config as cfg
cfg.OPENML_TASK_IDS  = [31, 37, 3, 12, 38, 46, 28, 9952]   # 8 datasets validados
cfg.SCALERS          = ["StandardScaler", "RobustScaler"]    # 2 scalers
cfg.SELECTORS        = ["None", "PCA"]                       # 2 selectors
cfg.CLASSIFIERS      = ["RandomForest", "SVM", "kNN"]        # 3 classifiers
cfg.CV_FOLDS         = 3                                     # 3-fold CV
cfg.MIN_INSTANCIAS   = 500
cfg.MAX_INSTANCIAS   = 100_000
cfg.KNN_K_RANGE      = [1, 3, 5]                             # 3 valores de k

import numpy as np
import pandas as pd

from fase1_meta_conjunto.dataset_loader      import carregar_todos
from fase1_meta_conjunto.embedding_generator import gerar_todos_embeddings
from fase1_meta_conjunto.pipeline_builder    import (executar_todos_pipelines,
                                                      gerar_todas_combinacoes)
from fase1_meta_conjunto.meta_conjunto       import (montar_meta_conjunto,
                                                      salvar_meta_conjunto,
                                                      construir_matrizes)
from fase2_metamodelos.preditor_knn          import PreditorKNNRanking
from fase2_metamodelos.avaliacao_ranking     import (avaliar_loo_knn,
                                                      metricas_ranking_par)
from fase2_metamodelos.recomendador          import recomendar_de_embedding
from utils.visualizacao import (plot_distribuicao_yperf,
                                plot_heatmap_componentes,
                                plot_boxplot_classifier,
                                plot_umap_embeddings,
                                plot_ranking_loo_knn,
                                plot_distribuicao_metricas_ranking,
                                plot_correlacao_embeddings_matrizes)

SEP  = "=" * 58
SEP2 = "-" * 50

print(SEP)
print("  TESTE RAPIDO v2 – Meta-aprendizado por Ranking")
print(SEP)
n_pipelines_total = len(cfg.SCALERS) * len(cfg.SELECTORS) * len(cfg.CLASSIFIERS)
print(f"  Datasets : {len(cfg.OPENML_TASK_IDS)}  |  "
      f"Pipelines : {n_pipelines_total}  |  "
      f"CV : {cfg.CV_FOLDS}-fold  |  LOO : {len(cfg.OPENML_TASK_IDS)} folds")

# ════════════════════════════════════════════════════════════════════════════
# FASE 1 – Construcao do meta-conjunto
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'[FASE 1]':=^58}")

print("\n[1/4] Carregando datasets...")
datasets = carregar_todos(usar_cache=True)
print(f"  {len(datasets)} datasets carregados.")

print("\n[2/4] Gerando embeddings (Dataset2Vec/fallback)...")
embeddings = gerar_todos_embeddings(datasets)

print("\n[3/4] Executando pipelines (CV)...")
combinacoes  = gerar_todas_combinacoes()
df_pipelines = executar_todos_pipelines(datasets, combinacoes=combinacoes, verbose=True)

print("\n[4/4] Montando e salvando meta-conjunto...")
meta = montar_meta_conjunto(df_pipelines, embeddings)
salvar_meta_conjunto(meta, embeddings=embeddings)

# ── Estatisticas Fase 1 ──────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  RESULTADOS FASE 1")
print(SEP2)
print(f"  Instancias (linhas)     : {len(meta)}")
print(f"  Datasets                : {meta['task_id'].nunique()}")
print(f"  Pipelines unicos        : {meta.groupby(['scaler','selector','classifier']).ngroups}")
print(f"  Y_perf medio            : {meta['Y_perf'].mean():.4f}")
print(f"  Y_perf mediana          : {meta['Y_perf'].median():.4f}")
print(f"  Y_perf max / min        : {meta['Y_perf'].max():.4f} / {meta['Y_perf'].min():.4f}")

print("\n  Top-5 pipelines por Y_perf medio:")
top5 = (meta.groupby(["scaler","selector","classifier"])["Y_perf"]
            .mean().sort_values(ascending=False).head(5).reset_index())
print(top5.to_string(index=False))

print("\n  Desempenho por classificador:")
print(meta.groupby("classifier")["Y_perf"]
          .agg(["mean","std"]).round(4)
          .sort_values("mean", ascending=False).to_string())

# ── Visualizacoes Fase 1 ─────────────────────────────────────────────────────
print("\n[VIZ] Gerando figuras da Fase 1...")
rotulos = {ds["task_id"]: ds["nome"] for ds in datasets}
plot_distribuicao_yperf(meta)
plot_heatmap_componentes(meta, eixo_x="classifier", eixo_y="scaler")
plot_boxplot_classifier(meta)
plot_umap_embeddings(embeddings, rotulos=rotulos)

# ════════════════════════════════════════════════════════════════════════════
# FASE 2 – Predicao de ranking (LOO)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'[FASE 2]':=^58}")

# Carrega as matrizes geradas na Fase 1
X_emb, Y_accuracy, task_ids, pipeline_keys = construir_matrizes(meta, embeddings)
n_ds, n_pl = Y_accuracy.shape
print(f"\n  Matrizes: X_emb={X_emb.shape}  Y_accuracy={Y_accuracy.shape}")
print(f"  NaN em Y_accuracy: {np.isnan(Y_accuracy).sum()} celulas "
      f"({np.isnan(Y_accuracy).mean()*100:.1f}%)")

print("\n[LOO-KNN] Avaliando preditor KNN para k in "
      f"{cfg.KNN_K_RANGE}...")
df_loo = avaliar_loo_knn(X_emb, Y_accuracy, task_ids, pipeline_keys, verbose=True)

print(f"\n{SEP2}")
print("  RESULTADOS FASE 2 – LOO com preditor KNN")
print(SEP2)
print(df_loo[["k","spearman_media","spearman_std",
               "kendalltau_media","kendalltau_std",
               "ndcg_media","ndcg_std"]].to_string(index=False))

melhor = df_loo.iloc[0]
k_otimo = int(melhor["k"])
print(f"\n  Melhor k = {k_otimo}")
print(f"  Spearman  : {melhor['spearman_media']:.4f} +/- {melhor['spearman_std']:.4f}")
print(f"  Kendall   : {melhor['kendalltau_media']:.4f} +/- {melhor['kendalltau_std']:.4f}")
print(f"  NDCG      : {melhor['ndcg_media']:.4f} +/- {melhor['ndcg_std']:.4f}")

# ── Demo de recomendacao (holdout = diabetes, task_id=37) ────────────────────
print(f"\n[DEMO] Recomendacao para holdout task_id=37 (diabetes)...")
idx_holdout = task_ids.index(37)
mask_tr     = np.ones(n_ds, dtype=bool)
mask_tr[idx_holdout] = False

X_tr  = X_emb[mask_tr]
Y_tr  = Y_accuracy[mask_tr]
ids_tr = [task_ids[i] for i in range(n_ds) if mask_tr[i]]

X_te      = X_emb[idx_holdout]
Y_te_true = Y_accuracy[idx_holdout]

df_rec = recomendar_de_embedding(
    X_te, X_tr, Y_tr, ids_tr, pipeline_keys,
    k=k_otimo, top_n=5
)

m = metricas_ranking_par(Y_te_true,
    PreditorKNNRanking(k=k_otimo).fit(X_tr, Y_tr).predict(X_te.reshape(1,-1))[0])
print(f"\n  Holdout (diabetes) — metricas do ranking predito:")
print(f"    Spearman rho = {m['spearman']:.4f}  (p={m['p_spearman']:.4f})")
print(f"    Kendall  tau = {m['kendalltau']:.4f}  (p={m['p_kendall']:.4f})")
print(f"    NDCG         = {m['ndcg']:.4f}")

caminho_rec = os.path.join("resultados", "metricas", "recomendacoes_teste.csv")
df_rec.to_csv(caminho_rec)

# ── Visualizacoes Fase 2 ─────────────────────────────────────────────────────
print("\n[VIZ] Gerando figuras da Fase 2...")
plot_ranking_loo_knn(df_loo)

fold_path = os.path.join(cfg.METRICAS_DIR, f"loo_knn_k{k_otimo}_por_fold.csv")
if os.path.exists(fold_path):
    plot_distribuicao_metricas_ranking(pd.read_csv(fold_path))

banco = {task_ids[i]: X_emb[i] for i in range(n_ds)}
plot_correlacao_embeddings_matrizes(banco)

# ════════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TESTE CONCLUIDO COM SUCESSO")
print(SEP)
print(f"  Meta-conjunto  : data/meta_conjunto/meta_conjunto.pkl")
print(f"  X_emb          : data/meta_conjunto/X_emb.npy  {X_emb.shape}")
print(f"  Y_accuracy     : data/meta_conjunto/Y_accuracy.npy  {Y_accuracy.shape}")
print(f"  LOO resumo     : resultados/metricas/loo_knn_resumo.csv")
print(f"  Recomendacoes  : {caminho_rec}")
print(f"  Figuras        : resultados/figuras/")
print(SEP)
