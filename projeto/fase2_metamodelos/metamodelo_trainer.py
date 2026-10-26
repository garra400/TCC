"""
Treinamento e avaliação dos metamodelos.

Treina três metamodelos de classificação para prever Y_qual:
  - RandomForestClassifier
  - SVC (kernel RBF)
  - MLPClassifier

Avaliação por validação cruzada estratificada (5-fold) com métricas:
  acurácia balanceada, precisão, recall, F1 (macro).

Os modelos treinados no conjunto completo são salvos em disco para
uso posterior no Recomendador.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline as SKPipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (balanced_accuracy_score, precision_score,
                                     recall_score, f1_score)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CV_FOLDS, RANDOM_STATE, METRICAS_DIR

MODELOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "metamodelos")
os.makedirs(MODELOS_DIR, exist_ok=True)


def _criar_metamodelos() -> dict[str, SKPipeline]:
    """Retorna os metamodelos dentro de Pipelines (com StandardScaler)."""
    return {
        "RandomForest": SKPipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "SVM": SKPipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=RANDOM_STATE))
        ]),
        "MLP": SKPipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                random_state=RANDOM_STATE))
        ]),
    }


def avaliar_metamodelos(X: np.ndarray,
                         y: np.ndarray) -> pd.DataFrame:
    """
    Avalia todos os metamodelos por cross-validation.

    Retorna
    -------
    pd.DataFrame com uma linha por metamodelo e colunas:
        modelo, balanced_accuracy, precisao, recall, f1_macro
    """
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro":   "precision_macro",
        "recall_macro":      "recall_macro",
        "f1_macro":          "f1_macro",
    }

    resultados = []
    for nome, modelo in _criar_metamodelos().items():
        print(f"  Avaliando {nome}...", end=" ")
        scores = cross_validate(modelo, X, y, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
        linha = {
            "modelo":              nome,
            "balanced_accuracy":   round(np.mean(scores["test_balanced_accuracy"]), 4),
            "balanced_acc_std":    round(np.std(scores["test_balanced_accuracy"]),  4),
            "precisao":            round(np.mean(scores["test_precision_macro"]),   4),
            "recall":              round(np.mean(scores["test_recall_macro"]),      4),
            "f1_macro":            round(np.mean(scores["test_f1_macro"]),          4),
            "f1_macro_std":        round(np.std(scores["test_f1_macro"]),           4),
        }
        print(f"F1={linha['f1_macro']:.4f} ± {linha['f1_macro_std']:.4f}")
        resultados.append(linha)

    df_res = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
    caminho = os.path.join(METRICAS_DIR, "metamodelos_cv.csv")
    df_res.to_csv(caminho, index=False)
    print(f"\nResultados salvos em: {caminho}")
    return df_res


def treinar_e_salvar(X: np.ndarray, y: np.ndarray) -> dict[str, SKPipeline]:
    """
    Treina todos os metamodelos no conjunto completo e salva em disco.

    Retorna
    -------
    dict nome → modelo treinado
    """
    modelos = _criar_metamodelos()
    treinados = {}
    for nome, modelo in modelos.items():
        print(f"  Treinando {nome} no conjunto completo...", end=" ")
        modelo.fit(X, y)
        caminho = os.path.join(MODELOS_DIR, f"{nome}.pkl")
        with open(caminho, "wb") as f:
            pickle.dump(modelo, f)
        print(f"salvo em {caminho}")
        treinados[nome] = modelo
    return treinados


def carregar_metamodelos() -> dict[str, SKPipeline]:
    """Carrega todos os metamodelos salvos em disco."""
    treinados = {}
    for nome in ["RandomForest", "SVM", "MLP"]:
        caminho = os.path.join(MODELOS_DIR, f"{nome}.pkl")
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Metamodelo '{nome}' não encontrado. "
                "Execute metamodelo_trainer.treinar_e_salvar() primeiro."
            )
        with open(caminho, "rb") as f:
            treinados[nome] = pickle.load(f)
    return treinados
