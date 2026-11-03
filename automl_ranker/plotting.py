"""Generate research charts from numerical results, without manuscript tooling."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .evaluation import MODELS
from .io import load_reference, write_json

LABELS = [
    "Random ranking",
    "Global mean",
    "Unscaled k-NN",
    "Standardized k-NN",
    r"$\mathit{Random\ Forest}$",
]
COLORS = ["#9ca3af", "#ddaa44", "#4088b5", "#66aaa0", "#24644d"]


def plot(reference, evaluation, output):
    reference, evaluation, output = Path(reference), Path(evaluation), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y, ids, groups, keys, catalog = load_reference(reference)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = pd.read_csv(evaluation / "resumo.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, metric, title in zip(
        axes.flat,
        ["spearman", "kendall_ponderado", "ndcg_5", "regret_1"],
        [
            "Spearman (higher is better)",
            "Weighted Kendall (higher is better)",
            "NDCG@5 (higher is better)",
            "Regret@1 (lower is better)",
        ],
    ):
        values = summary[summary.metrica.eq(metric)].set_index("modelo").loc[MODELS]
        means = values.media_grupos.to_numpy()
        errors = np.vstack(
            [means - values.ic95_grupos_inferior, values.ic95_grupos_superior - means]
        )
        ax.barh(LABELS, means, xerr=errors, color=COLORS, capsize=3)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "model_comparison.png", dpi=170)
    plt.close(fig)
    tasks = pd.read_csv(evaluation / "metricas_por_task.csv")
    bygroup = tasks.groupby(["modelo", "grupo"]).kendall_ponderado.mean()
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    arrays = [bygroup.loc[m].dropna().to_numpy() for m in MODELS]
    ax.boxplot(arrays, orientation="horizontal", tick_labels=LABELS, showfliers=True)
    ax.invert_yaxis()
    ax.axvline(0, color="#aa3333", linestyle="--")
    ax.set_xlabel("Weighted Kendall per group")
    fig.savefig(output / "group_distribution.png", dpi=170)
    plt.close(fig)
    original = np.load(reference / "X_emb.npy", allow_pickle=False).astype(float)
    similarity = 1 - squareform(pdist(original, "cosine"))
    order = np.argsort(similarity.mean(axis=1), kind="stable")
    ordered = similarity[np.ix_(order, order)]
    assert np.allclose(ordered, ordered.T) and np.allclose(np.diag(ordered), 1)
    pd.DataFrame(ordered, index=ids[order], columns=ids[order]).to_csv(
        output / "ordered_similarity.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    im = ax.imshow(ordered, cmap="viridis", vmin=-1 if ordered.min() < 0 else 0, vmax=1)
    ax.set_xlabel("Tasks ordered by mean cosine similarity")
    ax.set_ylabel("Same task order")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.savefig(output / "ordered_similarity.png", dpi=170)
    plt.close(fig)
    orders = {"similarity_task_ids": ids[order].astype(int).tolist()}
    if (reference / "pipelines_raw.csv").exists():
        raw = pd.read_csv(reference / "pipelines_raw.csv").fillna(
            {"scaler": "None", "selector": "None"}
        )
        raw = raw.merge(
            catalog[["task_id", "grupo"]], on="task_id", validate="many_to_one"
        )
        for component in ["scaler", "selector"]:
            cells = (
                raw.groupby(["grupo", component, "classifier"])
                .balanced_accuracy_mean.mean()
                .reset_index()
            )
            matrix = (
                cells.groupby([component, "classifier"])
                .balanced_accuracy_mean.mean()
                .unstack()
            )
            matrix = matrix.loc[
                matrix.mean(axis=1).sort_values(kind="stable").index,
                matrix.mean(axis=0).sort_values(kind="stable").index,
            ]
            assert np.all(np.diff(matrix.mean(axis=1)) >= -1e-12) and np.all(
                np.diff(matrix.mean(axis=0)) >= -1e-12
            )
            matrix.to_csv(output / f"{component}_classifier.csv")
            orders[component] = {
                "rows": matrix.index.to_list(),
                "columns": matrix.columns.to_list(),
            }
            # Balanced accuracy is bounded by [0,1]; keep a common, unclipped scale.
            fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
            im = ax.imshow(
                matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto"
            )
            names = {
                "RandomForest": r"$\mathit{Random\ Forest}$",
                "DecisionTree": "Decision\ntree",
            }
            ax.set_xticks(
                range(len(matrix.columns)), [names.get(n, n) for n in matrix.columns]
            )
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix.iloc[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                    )
            fig.colorbar(im, ax=ax, label="Group-averaged balanced accuracy")
            fig.savefig(output / f"{component}_classifier.png", dpi=170)
            plt.close(fig)
    write_json(output / "ordering.json", orders)
    print(f"Charts and numerical source tables saved to {output.resolve()}")
