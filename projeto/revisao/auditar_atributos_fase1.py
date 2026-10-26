"""Descreve atributos categóricos e possíveis identificadores, sem excluir tasks.

Indicadores são diagnósticos de escopo, não uma seleção orientada aos resultados.
Não altera dados, folds ou critérios de elegibilidade da execução em andamento.
"""
import pickle
import re

import numpy as np
import pandas as pd

from projeto.revisao.fase1_corrigida import CACHE, OUTPUT


def run():
    ids = np.load(OUTPUT/"task_ids.npy")
    records = []
    for tid in ids:
        with (CACHE/f"task_{tid}.pkl").open("rb") as stream:
            data = pickle.load(stream)
        x = data["X"]
        numeric = x.select_dtypes(include=[np.number]).shape[1]
        identifiers = [str(name) for name in x.columns if re.search(r"(^|_)(id|path|file|filename|image)(_|$)", str(name), re.I)]
        records.append({"task_id": int(tid), "nome_dataset": data["nome"],
                        "n_numericos": numeric, "n_categoricos": x.shape[1]-numeric,
                        "somente_categoricos": numeric == 0,
                        "proporcao_unicos_media": float(x.nunique(dropna=False).mean()/len(x)),
                        "possivel_identificador_ou_caminho": bool(identifiers),
                        "colunas_sugestivas": ";".join(identifiers)})
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT/"qualidade_atributos.csv", index=False, lineterminator="\n")
    print(f"{len(table)} tasks: {int(table.somente_categoricos.sum())} somente categóricas; "
          f"{int(table.possivel_identificador_ou_caminho.sum())} com nomes sugestivos de identificador/caminho.")
    return table


if __name__ == "__main__":
    run()
