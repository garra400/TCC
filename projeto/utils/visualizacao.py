"""
Visualizações para análise exploratória e documentação dos resultados.

Gera e salva automaticamente figuras em resultados/figuras/.

Funções da Fase 1 (exploratórias):
  plot_distribuicao_yperf        — histograma de Y_perf no meta-conjunto
  plot_heatmap_componentes       — heatmap de acurácia média por componente
  plot_boxplot_classifier        — boxplot de Y_perf por classificador
  plot_umap_embeddings           — projeção 2D dos embeddings via UMAP

Funções da Fase 2 (avaliação de ranking):
  plot_ranking_loo_knn           — Spearman/Kendall por k no LOO
  plot_distribuicao_metricas_ranking — boxplot das métricas por fold
  plot_correlacao_embeddings_matrizes — matriz de similaridade cosseno
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FIGURAS_DIR

sns.set_theme(style="whitegrid", palette="muted")


# ─── Formatação numérica em pt-BR (vírgula como separador decimal) ───────────
# Aplica-se a eixos, anotações de heatmap e rótulos textuais das figuras, de
# acordo com a convenção adotada no texto do TCC (ex.: "0,65" em vez de "0.65").

def _ptbr(valor: float, casas: int = 2) -> str:
    """Formata um número com vírgula como separador decimal (convenção pt-BR)."""
    if pd.isna(valor):
        return ""
    return f"{valor:.{casas}f}".replace(".", ",")


def _eixo_ptbr(eixo, casas: int = 2) -> None:
    """Configura um eixo (ax.xaxis/ax.yaxis) para exibir números com vírgula decimal."""
    eixo.set_major_formatter(FuncFormatter(lambda x, _pos: _ptbr(x, casas)))


def _annot_ptbr(matriz: np.ndarray, casas: int = 3) -> np.ndarray:
    """Gera matriz de rótulos textuais (vírgula decimal) para mapas de calor."""
    return np.vectorize(lambda v: _ptbr(v, casas))(np.asarray(matriz, dtype=float))


_ROTULOS_COMPONENTES = {
    "scaler": "Escalonador",
    "selector": "Seletor de atributos",
    "classifier": "Classificador",
}

_VALORES_COMPONENTES = {
    "None": "Nenhum",
    "RandomForest": "Random Forest",
    "DecisionTree": "Decision Tree",
    "StandardScaler": "StandardScaler",
    "MinMaxScaler": "MinMaxScaler",
    "RobustScaler": "RobustScaler",
    "PCA": "PCA",
    "SelectKBest": "SelectKBest",
    "SVM": "SVM",
    "kNN": "kNN",
    "MLP": "MLP",
}


def _rotulo_componente(nome: str) -> str:
    return _ROTULOS_COMPONENTES.get(nome, nome)


def _valor_componente(valor):
    return _VALORES_COMPONENTES.get(valor, valor)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 – Exploração do meta-conjunto
# ═══════════════════════════════════════════════════════════════════════════════

def plot_distribuicao_yperf(df: pd.DataFrame, salvar: bool = True) -> None:
    """Histograma de Y_perf (balanced accuracy) no meta-conjunto."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Y_perf"].dropna(), bins=40, edgecolor="white", color="#4C72B0")
    mediana = df["Y_perf"].median()
    ax.axvline(mediana, color="red", linestyle="--",
               label=f"Mediana = {_ptbr(mediana, 3)}")
    ax.set_xlabel("Acurácia balanceada (Y_perf)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de desempenho dos pipelines no meta-conjunto")
    _eixo_ptbr(ax.xaxis, 2)
    ax.legend()
    _salvar(fig, "distribuicao_yperf.png", salvar)


def plot_heatmap_componentes(df: pd.DataFrame,
                              eixo_x: str = "classifier",
                              eixo_y: str = "scaler",
                              salvar: bool = True) -> None:
    """Mapa de calor da média de Y_perf cruzando dois componentes do pipeline."""
    pivot = df.pivot_table(values="Y_perf", index=eixo_y,
                           columns=eixo_x, aggfunc="mean")
    pivot_plot = pivot.rename(index=_valor_componente, columns=_valor_componente)
    rotulo_x = _rotulo_componente(eixo_x)
    rotulo_y = _rotulo_componente(eixo_y)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot_plot, annot=_annot_ptbr(pivot.values, 3), fmt="", cmap="YlGnBu", ax=ax,
                linewidths=0.4, cbar_kws={"label": "Acurácia balanceada média"})
    _eixo_ptbr(ax.collections[0].colorbar.ax.yaxis, 2)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel(rotulo_y)
    ax.set_title(f"Desempenho médio: {rotulo_y} × {rotulo_x}")
    _salvar(fig, f"heatmap_{eixo_y}_x_{eixo_x}.png", salvar)


def plot_boxplot_classifier(df: pd.DataFrame, salvar: bool = True) -> None:
    """Boxplot de Y_perf por classificador."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ordem = (df.groupby("classifier")["Y_perf"]
               .median()
               .sort_values(ascending=False)
               .index)
    sns.boxplot(data=df, x="classifier", y="Y_perf", order=ordem,
                palette="Set2", ax=ax)
    ax.set_xlabel("Classificador")
    ax.set_ylabel("Acurácia balanceada")
    ax.set_title("Distribuição de desempenho por classificador")
    _eixo_ptbr(ax.yaxis, 2)
    _salvar(fig, "boxplot_classifier.png", salvar)


def plot_umap_embeddings(banco_embeddings: dict[int, np.ndarray],
                          rotulos: dict[int, str] | None = None,
                          salvar: bool = True) -> None:
    """Projeta embeddings em 2D via UMAP."""
    try:
        import umap
    except ImportError:
        print("[AVISO] umap-learn não instalado. Pulando plot UMAP.")
        return

    ids   = list(banco_embeddings.keys())
    embs  = np.stack(list(banco_embeddings.values()))
    nomes = [rotulos[i] if rotulos and i in rotulos else str(i) for i in ids]

    proj_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embs)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(proj_2d[:, 0], proj_2d[:, 1],
                    c=range(len(ids)), cmap="tab20", s=60, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Índice do conjunto de dados")
    ax.set_title("Projeção UMAP dos embeddings dos conjuntos de dados")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    _eixo_ptbr(ax.xaxis, 1)
    _eixo_ptbr(ax.yaxis, 1)
    _salvar(fig, "umap_embeddings.png", salvar)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 – Avaliação de ranking
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ranking_loo_knn(df_loo: pd.DataFrame, salvar: bool = True) -> None:
    """
    Gráfico de linha dupla: Spearman rho e Kendall tau em função de k.

    Parâmetros
    ----------
    df_loo : pd.DataFrame
        Saída de avaliacao_ranking.avaliar_loo_knn() com colunas
        k, spearman_media, spearman_std, kendalltau_media, kendalltau_std.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = df_loo["k"].values

    ax.errorbar(ks, df_loo["spearman_media"],
                yerr=df_loo["spearman_std"],
                marker="o", capsize=4, label="Spearman (ρ)", color="#4C72B0")

    ax.errorbar(ks, df_loo["kendalltau_media"],
                yerr=df_loo["kendalltau_std"],
                marker="s", capsize=4, label="Kendall (τ)", color="#DD8452")

    ax.set_xlabel("Número de vizinhos (k)")
    ax.set_ylabel("Coeficiente de correlação de postos")
    ax.set_title("Qualidade do ranking KNN-LOO\n"
                 "(Spearman ρ e Kendall τ em função de k)")
    ax.set_xticks(ks)
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    _eixo_ptbr(ax.yaxis, 2)
    _salvar(fig, "ranking_loo_knn.png", salvar)


def plot_kendall_ponderado_por_dataset(df_folds: pd.DataFrame,
                                        salvar: bool = True) -> None:
    """
    Dois painéis:
      Esquerda — histograma do Kendall ponderado por conjunto de dados (LOO).
      Direita  — scatter Kendall padrão × Kendall ponderado por fold,
                 evidenciando o efeito da ponderação no topo do ranking.
    """
    if "kendall_ponderado" not in df_folds.columns:
        print("[AVISO] df_folds não contém 'kendall_ponderado'. Pulando plot.")
        return

    df_v = df_folds.dropna(subset=["kendall_ponderado", "kendalltau"])
    if df_v.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — histograma
    mediana_kw = df_v["kendall_ponderado"].median()
    axes[0].hist(df_v["kendall_ponderado"], bins=20, color="#4C72B0",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(mediana_kw, color="red",
                    linestyle="--", label=f"Mediana = {_ptbr(mediana_kw, 3)}")
    axes[0].set_xlabel("Kendall Tau ponderado (scipy.stats.weightedtau)")
    axes[0].set_ylabel("Número de conjuntos de dados")
    axes[0].set_title("Distribuição do Kendall ponderado por conjunto de dados (LOO)")
    axes[0].legend()
    _eixo_ptbr(axes[0].xaxis, 2)

    # — scatter τ vs τ_ponderado
    axes[1].scatter(df_v["kendalltau"], df_v["kendall_ponderado"],
                    alpha=0.6, color="#DD8452", edgecolors="white", linewidths=0.5)
    lims = [min(df_v["kendalltau"].min(), df_v["kendall_ponderado"].min()) - 0.05,
            max(df_v["kendalltau"].max(), df_v["kendall_ponderado"].max()) + 0.05]
    axes[1].plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="diagonal (igual)")
    axes[1].set_xlabel("Kendall τ padrão")
    axes[1].set_ylabel("Kendall τ ponderado")
    axes[1].set_title("Kendall padrão vs ponderado por fold")
    axes[1].legend()
    _eixo_ptbr(axes[1].xaxis, 2)
    _eixo_ptbr(axes[1].yaxis, 2)

    fig.tight_layout()
    _salvar(fig, "kendall_ponderado_por_dataset.png", salvar)


def plot_distribuicao_metricas_ranking(df_folds: pd.DataFrame,
                                        salvar: bool = True) -> None:
    """
    Boxplot da distribuição de Spearman e Kendall por fold (LOO).

    Mostra a variabilidade do preditor entre diferentes conjuntos de dados de teste.

    Parâmetros
    ----------
    df_folds : pd.DataFrame
        Arquivo loo_knn_k{k}_por_fold.csv com colunas spearman e kendalltau.
    """
    if df_folds.empty or "spearman" not in df_folds.columns:
        print("[AVISO] df_folds vazio ou sem colunas esperadas.")
        return

    df_melt = df_folds[["spearman", "kendalltau"]].rename(columns={
        "spearman": "Spearman (ρ)",
        "kendalltau": "Kendall (τ)",
    }).melt(var_name="Métrica", value_name="Valor")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_melt, x="Métrica", y="Valor",
                palette=["#4C72B0", "#DD8452"], ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Distribuição das métricas de ranking por fold (LOO)")
    ax.set_ylabel("Coeficiente de correlação de postos")
    ax.set_ylim(-1.05, 1.05)
    _eixo_ptbr(ax.yaxis, 2)
    _salvar(fig, "ranking_distribuicao_metricas.png", salvar)


def plot_correlacao_embeddings_matrizes(banco_embeddings: dict[int, np.ndarray],
                                         salvar: bool = True) -> None:
    """
    Matriz de similaridade cosseno entre todos os embeddings do banco.

    Versão atualizada: suporta banco com muitos conjuntos de dados (comprime labels).
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    ids  = list(banco_embeddings.keys())
    embs = np.stack(list(banco_embeddings.values()))
    sim  = cos_sim(embs)

    # Ordena linhas pela média de similaridade (decrescente) e aplica
    # a mesma permutação nas colunas para manter a matriz simétrica visualmente
    order  = np.argsort(sim.mean(axis=1))[::-1]
    sim    = sim[np.ix_(order, order)]
    ids    = [ids[i] for i in order]

    n = len(ids)
    figsize = (max(8, n * 0.12), max(7, n * 0.12))
    fig, ax = plt.subplots(figsize=figsize)

    # Suprime labels se houver muitos conjuntos de dados
    labels = ids if n <= 50 else []
    sns.heatmap(sim,
                xticklabels=labels, yticklabels=labels,
                cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Similaridade cosseno"})
    _eixo_ptbr(ax.collections[0].colorbar.ax.yaxis, 1)
    ax.set_title(f"Matriz de similaridade cosseno ({n} conjuntos de dados)")
    if n > 50:
        ax.set_xlabel("ID da tarefa OpenML")
        ax.set_ylabel("ID da tarefa OpenML")
    _salvar(fig, "matriz_similaridade_embeddings.png", salvar)


# ─── Backward-compat (Fase 2 antiga) ─────────────────────────────────────────

def plot_comparacao_metamodelos(df_cv: pd.DataFrame, salvar: bool = True) -> None:
    """Mantido para compatibilidade; use plot_ranking_loo_knn() na nova versão."""
    if df_cv.empty:
        return
    cols_f1 = [c for c in ["f1_macro", "kendalltau_media"] if c in df_cv.columns]
    if not cols_f1:
        return
    col = cols_f1[0]
    col_std = col.replace("media", "std") if "media" in col else "f1_macro_std"

    fig, ax = plt.subplots(figsize=(7, 4))
    yerr = df_cv[col_std].values if col_std in df_cv.columns else None
    ax.bar(df_cv.get("modelo", df_cv.get("k", range(len(df_cv)))).astype(str),
           df_cv[col], yerr=yerr, capsize=5,
           color=["#4C72B0", "#DD8452", "#55A868"][:len(df_cv)],
           edgecolor="white")
    ax.set_ylim(0, 1)
    ax.set_ylabel(col)
    ax.set_title("Comparação de modelos")
    _eixo_ptbr(ax.yaxis, 2)
    _salvar(fig, "comparacao_metamodelos.png", salvar)


def plot_correlacao_embeddings(banco_embeddings: dict[int, np.ndarray],
                                salvar: bool = True) -> None:
    """Alias de plot_correlacao_embeddings_matrizes() para compatibilidade."""
    plot_correlacao_embeddings_matrizes(banco_embeddings, salvar=salvar)


# ─── Auxiliar ─────────────────────────────────────────────────────────────────

def _salvar(fig: plt.Figure, nome: str, salvar: bool) -> None:
    if salvar:
        caminho = os.path.join(FIGURAS_DIR, nome)
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {caminho}")
    plt.show()
    plt.close(fig)
