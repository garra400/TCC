"""
Construção e avaliação de pipelines de AM.

Cada pipeline combina:
  - Scaler      : StandardScaler | MinMaxScaler | RobustScaler | None
  - Selector    : PCA(n=10)      | SelectKBest(k=10) | None
  - Classifier  : RandomForest   | SVM | kNN | MLP | DecisionTree

Para cada par (dataset, pipeline) é executada validação cruzada
estratificada (CV_FOLDS folds) e registradas as métricas:
  - balanced_accuracy (principal → Y_perf)
  - f1_macro
  - tempo de treinamento (s)
  - tempo de inferência (s)
"""

import time
import warnings
import numpy as np
import pandas as pd
from itertools import product

from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition      import PCA
from sklearn.feature_selection  import SelectKBest, f_classif
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import SVC
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.neural_network     import MLPClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.model_selection    import StratifiedKFold, cross_validate
from sklearn.metrics            import balanced_accuracy_score, f1_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CV_FOLDS, SCALERS, SELECTORS, CLASSIFIERS, RANDOM_STATE, SVM_MAX_AMOSTRAS


# ─── Fábricas de componentes ─────────────────────────────────────────────────
def _make_scaler(nome: str):
    return {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler":   MinMaxScaler(),
        "RobustScaler":   RobustScaler(),
        "None":           None,
    }[nome]


def _make_selector(nome: str, n_features: int):
    k = min(10, max(1, n_features - 1))
    return {
        "PCA":          PCA(n_components=k),
        "SelectKBest":  SelectKBest(f_classif, k=k),
        "None":         None,
    }[nome]


def _make_classifier(nome: str):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM":          SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
        "kNN":          KNeighborsClassifier(n_neighbors=5),
        "MLP":          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                      random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    }[nome]


def construir_pipeline(scaler_nome: str,
                        selector_nome: str,
                        clf_nome: str,
                        n_features: int) -> Pipeline:
    """Constrói um sklearn Pipeline a partir dos nomes dos componentes."""
    passos = []
    sc  = _make_scaler(scaler_nome)
    sel = _make_selector(selector_nome, n_features)
    clf = _make_classifier(clf_nome)
    if sc  is not None: passos.append(("scaler",   sc))
    if sel is not None: passos.append(("selector", sel))
    passos.append(("clf", clf))
    return Pipeline(passos)


def avaliar_pipeline(pipeline: Pipeline,
                     X: pd.DataFrame,
                     y: pd.Series) -> dict:
    """
    Avalia um pipeline via validação cruzada estratificada.

    Retorna dict com:
      balanced_accuracy_mean, f1_macro_mean,
      tempo_treino_s, tempo_inferencia_s
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring={"balanced_accuracy": "balanced_accuracy",
                     "f1_macro":          "f1_macro"},
            return_train_score=False,
            n_jobs=-1,
        )
    t_total = time.perf_counter() - t0

    return {
        "balanced_accuracy_mean": float(np.mean(resultado["test_balanced_accuracy"])),
        "balanced_accuracy_std":  float(np.std(resultado["test_balanced_accuracy"])),
        "f1_macro_mean":          float(np.mean(resultado["test_f1_macro"])),
        "f1_macro_std":           float(np.std(resultado["test_f1_macro"])),
        "tempo_total_s":          round(t_total, 4),
    }


def gerar_todas_combinacoes() -> list[dict]:
    """Retorna todas as combinações (scaler, selector, classifier)."""
    return [
        {"scaler": sc, "selector": sel, "classifier": clf}
        for sc, sel, clf in product(SCALERS, SELECTORS, CLASSIFIERS)
    ]


def executar_todos_pipelines(datasets: list[dict],
                              combinacoes: list[dict] | None = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Executa todas as combinações de pipeline em todos os datasets.

    Parâmetros
    ----------
    datasets    : lista retornada por dataset_loader.carregar_todos()
    combinacoes : lista de dicts; se None usa gerar_todas_combinacoes()
    verbose     : imprime progresso

    Retorna
    -------
    pd.DataFrame com colunas:
        task_id, nome_dataset, scaler, selector, classifier,
        balanced_accuracy_mean, f1_macro_mean, tempo_total_s
    """
    from tqdm import tqdm

    if combinacoes is None:
        combinacoes = gerar_todas_combinacoes()

    registros = []
    total = len(datasets) * len(combinacoes)

    with tqdm(total=total, desc="Avaliando pipelines") as barra:
        for ds in datasets:
            tid  = ds["task_id"]
            nome = ds["nome"]
            X    = ds["X"]
            y    = ds["y"]
            nf   = X.shape[1]

            # Subsample para SVM: custo quadrático torna SVM inviável em n grande.
            # Prática padrão em AutoML (AutoSklearn, TPOT, etc.).
            X_svm = X
            y_svm = y
            if len(X) > SVM_MAX_AMOSTRAS:
                rng   = np.random.default_rng(RANDOM_STATE)
                idx   = rng.choice(len(X), SVM_MAX_AMOSTRAS, replace=False)
                X_svm = X.iloc[idx].reset_index(drop=True)
                y_svm = y.iloc[idx].reset_index(drop=True)

            for comb in combinacoes:
                barra.set_postfix(dataset=nome, pipeline=str(comb))
                # Usa subsample apenas para SVM
                X_eval = X_svm if comb["classifier"] == "SVM" else X
                y_eval = y_svm if comb["classifier"] == "SVM" else y
                try:
                    pipe    = construir_pipeline(comb["scaler"], comb["selector"],
                                                  comb["classifier"], X_eval.shape[1])
                    metricas = avaliar_pipeline(pipe, X_eval, y_eval)
                except Exception as e:
                    if verbose:
                        print(f"\n  [ERRO] task={tid}, {comb}: {e}")
                    metricas = {
                        "balanced_accuracy_mean": np.nan,
                        "balanced_accuracy_std":  np.nan,
                        "f1_macro_mean":          np.nan,
                        "f1_macro_std":           np.nan,
                        "tempo_total_s":          np.nan,
                    }

                registros.append({
                    "task_id":      tid,
                    "nome_dataset": nome,
                    **comb,
                    **metricas,
                })
                barra.update(1)

    return pd.DataFrame(registros)
