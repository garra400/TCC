"""
Preditor de ranking baseado em KNN multi-output (KDTree).

Princípio (conforme orientação do Prof. Davi Perreira):
  Dado o embedding X_emb de um novo dataset, encontra os k datasets mais
  próximos no espaço de embeddings e usa a média ponderada de suas acurácias
  como predição para cada pipeline.

  A formulação é matematicamente equivalente ao KNeighborsRegressor com
  múltiplos targets (um por pipeline), mas usa KDTree diretamente para
  permitir:
    - inspeção dos vizinhos (quais datasets treinaram a recomendação)
    - ponderação por distância sem refitting
    - predição de todos os 60 pipelines em uma única chamada

Referência:
  Perreira, D. (2026, comunicação pessoal):
  "Uma vantagem de usar embeddings é que relações espaciais normalmente
   são válidas [...] um simples k-NN pode predizer a acurácia em cada pipeline."
"""

import numpy as np
from scipy.spatial import KDTree

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import KNN_K_PADRAO


class PreditorKNNRanking:
    """
    Preditor multi-output de acurácia de pipelines via KNN (KDTree).

    Parâmetros
    ----------
    k : int
        Número de vizinhos. Padrão: KNN_K_PADRAO de config.py.
    ponderado : bool
        Se True, usa ponderação pelo inverso da distância (distance-weighted).
        Se False, usa média simples (uniform weights).

    Atributos após fit()
    --------------------
    tree_       : KDTree  — índice espacial dos embeddings de treino
    X_train_    : np.ndarray (n_train, emb_dim)
    Y_train_    : np.ndarray (n_train, n_pipelines)  — com NaN preenchido
    pipeline_keys_ : list[tuple]  — (scaler, selector, classifier) por coluna
    task_ids_   : list[int]  — task_id por linha em X_train_
    """

    def __init__(self, k: int = KNN_K_PADRAO, ponderado: bool = True):
        self.k         = k
        self.ponderado = ponderado

        self.tree_         = None
        self.X_train_      = None
        self.Y_train_      = None
        self.pipeline_keys_= None
        self.task_ids_     = None

    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            task_ids: list[int] | None = None,
            pipeline_keys: list[tuple] | None = None) -> "PreditorKNNRanking":
        """
        Indexa os datasets de treino.

        Parâmetros
        ----------
        X : (n_train, emb_dim)  — embeddings dos datasets de treino
        Y : (n_train, n_pipelines) — acurácias; NaN será preenchido pela mediana
        task_ids : opcional, para inspeção dos vizinhos
        pipeline_keys : opcional, para inspeção das colunas
        """
        self.X_train_       = X.astype(np.float64)
        self.Y_train_       = self._preencher_nan(Y.astype(np.float64))
        self.pipeline_keys_ = pipeline_keys
        self.task_ids_      = task_ids
        self.tree_          = KDTree(self.X_train_)
        return self

    # ── Predição ──────────────────────────────────────────────────────────────

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Prediz a acurácia de todos os pipelines para cada novo dataset.

        Parâmetros
        ----------
        X_new : (n_new, emb_dim)  ou  (emb_dim,) para um único dataset

        Retorna
        -------
        Y_pred : (n_new, n_pipelines)  — acurácia estimada por pipeline
        """
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        k_efetivo = min(self.k, len(self.X_train_))
        dist, ind = self.tree_.query(X_new.astype(np.float64), k=k_efetivo)

        # scipy KDTree retorna arrays 1D quando k=1; garante 2D para uniformidade
        if k_efetivo == 1:
            dist = dist.reshape(-1, 1)
            ind  = ind.reshape(-1, 1)

        if self.ponderado:
            # Ponderação por inverso da distância; evita divisão por zero
            pesos = 1.0 / (dist + 1e-8)
            pesos = pesos / pesos.sum(axis=1, keepdims=True)
            # ind shape: (n_new, k); Y_train[ind] shape: (n_new, k, n_pipelines)
            Y_pred = np.einsum("ij,ijk->ik", pesos, self.Y_train_[ind])
        else:
            Y_pred = self.Y_train_[ind].mean(axis=1)

        return Y_pred

    def predict_ranking(self, X_new: np.ndarray) -> np.ndarray:
        """
        Converte a predição de acurácia em posição de ranking.
        Rank 1 = melhor pipeline (maior acurácia predita).

        Retorna
        -------
        ranks : (n_new, n_pipelines)  int, começando em 1
        """
        Y_pred = self.predict(X_new)
        # argsort(-Y_pred) → índices do melhor para o pior
        # argsort(argsort(...)) → posição de cada pipeline no ranking (0-based)
        return np.argsort(np.argsort(-Y_pred, axis=1), axis=1) + 1

    def vizinhos(self, X_query: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Retorna as distâncias e os task_ids dos k vizinhos mais próximos.

        Útil para explicabilidade: "quais datasets guiaram esta recomendação?"
        """
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        k_efetivo = min(self.k, len(self.X_train_))
        dist, ind = self.tree_.query(X_query.astype(np.float64), k=k_efetivo)
        vizinhos_ids = (
            [self.task_ids_[i] for i in ind[0]] if self.task_ids_ else ind[0].tolist()
        )
        return dist[0], vizinhos_ids

    # ── Utilitário ────────────────────────────────────────────────────────────

    @staticmethod
    def _preencher_nan(Y: np.ndarray) -> np.ndarray:
        """Substitui NaN pela mediana da coluna (pipeline)."""
        Y = Y.copy()
        for j in range(Y.shape[1]):
            col = Y[:, j]
            mask_nan = np.isnan(col)
            if mask_nan.any() and not mask_nan.all():
                Y[mask_nan, j] = np.nanmedian(col)
            elif mask_nan.all():
                Y[:, j] = 0.0
        return Y
