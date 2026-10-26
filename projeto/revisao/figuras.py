"""Figuras científicas exportáveis para TCC e artigo; usa os CSVs da revisão."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projeto.revisao.experimentos import PROJECT


def run():
    path = PROJECT/"resultados/revisao_2026"
    output = path/"figuras"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    summary = pd.read_csv(path/"resumo.csv")
    data = summary.query("protocolo == 'cv_grupos' and metrica == 'kendall_ponderado'").sort_values("media_grupos")
    labels = {"aleatorio": "Aleatório", "media_global": "Média global", "knn_ajustado": "k-NN ajustado",
              "knn_k10_fixo": "k-NN k=10", "knn_padronizado_ajustado": "k-NN padronizado", "random_forest": "Random Forest"}
    fig, ax = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    ax.errorbar(data.media_grupos, [labels[k] for k in data.modelo],
                xerr=np.array([data.media_grupos-data.ic95_grupos_inferior,
                               data.ic95_grupos_superior-data.media_grupos]), fmt="o", capsize=4, color="#24597e")
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set_xlabel("Kendall ponderado: média por grupo e IC descritivo de 95%")
    ax.grid(axis="x", alpha=.25)
    fig.savefig(output/"comparacao_modelos.png")
    fig.savefig(output/"comparacao_modelos.pdf")
    plt.close(fig)
    errors = pd.read_csv(path/"diagnostico_erros.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, feature, label in zip(axes, ["n_atributos", "classe_majoritaria", "distancia_vizinho"],
                                  ["Número de atributos", "Proporção da classe majoritária", "Distância ao vizinho de treino"]):
        ax.scatter(errors[feature], errors.kendall_ponderado, s=13, alpha=.6, c="#24597e")
        ax.axhline(0, color="#b34242", linewidth=.8)
        if feature != "classe_majoritaria":
            ax.set_xscale("symlog", linthresh=.0001 if feature == "distancia_vizinho" else 1)
        ax.set_xlabel(label)
        ax.set_ylabel("Kendall ponderado")
        ax.grid(alpha=.15)
    fig.savefig(output/"diagnostico_erros.png")
    fig.savefig(output/"diagnostico_erros.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
