"""
Recomendador de pipelines para novos datasets via ranking KNN.

Fluxo:
  1. Gera X_emb do novo dataset via Dataset2Vec (ou fallback)
  2. Aplica o preditor KNN treinado no meta-conjunto: prediz a acurácia
     estimada de todos os 60 pipelines
  3. Ordena pipelines pelo score decrescente → ranking final
  4. Retorna DataFrame com ranking, scores e informações de explicabilidade
     (quais datasets-vizinhos guiaram a recomendação)

Diferença em relação à versão anterior:
  - Antes : metamodelos de classificação (RF/SVM/MLP) prediziam Y_qual (0/1)
  - Agora : preditor KNN de regressão multi-output prediz Y_accuracy contínuo
            para todos os pipelines simultaneamente, sem necessidade de
            concatenar pipeline_enc à entrada — o vetor X_emb do dataset
            é suficiente para indexar o KDTree.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import KNN_K_PADRAO
from fase1_meta_conjunto.embedding_generator import gerar_embedding
from fase2_metamodelos.preditor_knn import PreditorKNNRanking


def recomendar(X_novo: pd.DataFrame,
               y_novo: pd.Series,
               X_emb_treino: np.ndarray,
               Y_accuracy_treino: np.ndarray,
               task_ids_treino: list[int],
               pipeline_keys: list[tuple],
               k: int = KNN_K_PADRAO,
               top_n: int = 10) -> pd.DataFrame:
    """
    Recomenda os melhores pipelines para um novo dataset.

    Parâmetros
    ----------
    X_novo, y_novo    : features e alvo do novo dataset (dados brutos)
    X_emb_treino      : (n_train, emb_dim) embeddings do meta-conjunto
    Y_accuracy_treino : (n_train, n_pipelines) acurácias do meta-conjunto
    task_ids_treino   : lista de task_ids correspondendo às linhas de X_emb_treino
    pipeline_keys     : lista de (scaler, selector, classifier) por coluna
    k                 : número de vizinhos do preditor KNN
    top_n             : quantos pipelines retornar no ranking final

    Retorna
    -------
    pd.DataFrame com colunas:
        rank, scaler, selector, classifier,
        score_predito,   ← acurácia estimada (balanced accuracy)
        vizinhos_ids     ← task_ids dos datasets mais similares
    """
    # 1. Embedding do novo dataset
    print("  [1] Gerando embedding do novo dataset...")
    X_emb_novo = gerar_embedding(X_novo, y_novo)

    # 2. Preditor KNN
    print(f"  [2] Preditor KNN (k={k})...")
    preditor = PreditorKNNRanking(k=k, ponderado=True)
    preditor.fit(X_emb_treino, Y_accuracy_treino,
                 task_ids=task_ids_treino, pipeline_keys=pipeline_keys)

    Y_pred = preditor.predict(X_emb_novo)   # (1, n_pipelines)
    scores = Y_pred[0]

    # 3. Vizinhos para explicabilidade
    dist_viz, ids_viz = preditor.vizinhos(X_emb_novo)
    vizinhos_str = ", ".join(str(tid) for tid in ids_viz)

    # 4. Montar tabela de resultados
    registros = []
    for j, (sc, sel, clf) in enumerate(pipeline_keys):
        registros.append({
            "scaler":      sc,
            "selector":    sel,
            "classifier":  clf,
            "score_predito": float(scores[j]) if not np.isnan(scores[j]) else 0.0,
            "vizinhos_ids": vizinhos_str,
        })

    df = (pd.DataFrame(registros)
            .sort_values("score_predito", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.index += 1
    df.index.name = "rank"

    print(f"\n  Top-{top_n} pipelines recomendados (KNN k={k}):")
    print(df[["scaler", "selector", "classifier", "score_predito"]].to_string())
    return df


def recomendar_de_embedding(X_emb_novo: np.ndarray,
                             X_emb_treino: np.ndarray,
                             Y_accuracy_treino: np.ndarray,
                             task_ids_treino: list[int],
                             pipeline_keys: list[tuple],
                             k: int = KNN_K_PADRAO,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Versão de recomendar() quando o embedding já foi calculado externamente.
    Útil para avaliação em batch ou quando não há acesso aos dados brutos.
    """
    preditor = PreditorKNNRanking(k=k, ponderado=True)
    preditor.fit(X_emb_treino, Y_accuracy_treino,
                 task_ids=task_ids_treino, pipeline_keys=pipeline_keys)

    Y_pred = preditor.predict(X_emb_novo)

    if X_emb_novo.ndim == 1:
        Y_pred = Y_pred.reshape(1, -1)

    resultados = []
    for i in range(len(Y_pred)):
        scores = Y_pred[i]
        registros = []
        for j, (sc, sel, clf) in enumerate(pipeline_keys):
            registros.append({
                "scaler":      sc,
                "selector":    sel,
                "classifier":  clf,
                "score_predito": float(scores[j]) if not np.isnan(scores[j]) else 0.0,
            })
        df = (pd.DataFrame(registros)
                .sort_values("score_predito", ascending=False)
                .head(top_n)
                .reset_index(drop=True))
        df.index += 1
        df.index.name = "rank"
        resultados.append(df)

    return resultados[0] if len(resultados) == 1 else resultados
