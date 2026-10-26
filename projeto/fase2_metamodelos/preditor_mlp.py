"""
Preditor de ranking baseado em MLP com loss BCE (Binary Cross-Entropy).

Abordagem "pointwise learning-to-rank":
  - Entrada : X_emb do dataset (emb_dim dimensões)
  - Saída   : vetor de 60 scores em [0, 1] via sigmoid — um por pipeline
  - Loss    : BCELoss entre os scores preditos e as acurácias normalizadas
              (que já estão em [0,1] por serem balanced_accuracy)
  - Ranking : pipelines ordenados pelo score predito decrescente

Por que BCE em vez de MSE?
  BCE penaliza mais fortemente predições erradas próximas de 0 ou 1,
  favorecendo a discriminação entre os melhores e piores pipelines —
  comportamento mais alinhado com a tarefa de ranking do que MSE.

Referência ao conceito "burnet cross" (binary cross-entropy for ranking):
  RankNet (Burges et al., 2005) e variantes pointwise usam BCE
  como função de custo para aprender escores de relevância.
"""

import warnings
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (MLP_HIDDEN_SIZES, MLP_EPOCHS, MLP_LR,
                    MLP_DROPOUT, MLP_BATCH_SIZE, RANDOM_STATE)

# Importação opcional do PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    warnings.warn(
        "PyTorch não encontrado. PreditorMLPRanking não estará disponível. "
        "Instale com: pip install torch",
        stacklevel=2
    )
    _TORCH_OK = False


# ─── Arquitetura ─────────────────────────────────────────────────────────────

class _RedeRanking(nn.Module if _TORCH_OK else object):
    """MLP feedforward com ativação sigmoid na saída."""

    def __init__(self, emb_dim: int, n_pipelines: int,
                 hidden_sizes: tuple, dropout: float):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch é necessário para PreditorMLPRanking.")
        super().__init__()
        camadas = []
        entrada = emb_dim
        for h in hidden_sizes:
            camadas += [nn.Linear(entrada, h), nn.ReLU(), nn.Dropout(dropout)]
            entrada = h
        camadas += [nn.Linear(entrada, n_pipelines), nn.Sigmoid()]
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)


# ─── Preditor ────────────────────────────────────────────────────────────────

class PreditorMLPRanking:
    """
    Preditor de ranking via MLP com BCE loss (PyTorch).

    Parâmetros
    ----------
    emb_dim     : dimensão dos embeddings de entrada
    n_pipelines : número de pipelines (targets de saída)
    hidden_sizes: arquitetura das camadas ocultas
    epochs      : épocas de treinamento
    lr          : taxa de aprendizado (Adam)
    dropout     : taxa de dropout (regularização)
    batch_size  : tamanho do mini-batch
    verbose     : exibe loss por época se True
    """

    def __init__(self,
                 emb_dim: int,
                 n_pipelines: int,
                 hidden_sizes: tuple = MLP_HIDDEN_SIZES,
                 epochs: int = MLP_EPOCHS,
                 lr: float = MLP_LR,
                 dropout: float = MLP_DROPOUT,
                 batch_size: int = MLP_BATCH_SIZE,
                 verbose: bool = False):
        if not _TORCH_OK:
            raise RuntimeError(
                "PyTorch não encontrado. Instale com: pip install torch"
            )
        self.emb_dim     = emb_dim
        self.n_pipelines = n_pipelines
        self.hidden_sizes= hidden_sizes
        self.epochs      = epochs
        self.lr          = lr
        self.dropout     = dropout
        self.batch_size  = batch_size
        self.verbose     = verbose
        self.modelo_     = None
        self.historico_  = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PreditorMLPRanking":
        """
        Treina a rede com BCE loss.

        Parâmetros
        ----------
        X : (n_datasets, emb_dim)
        Y : (n_datasets, n_pipelines) — valores em [0,1]; NaN preenchido pela mediana
        """
        torch.manual_seed(RANDOM_STATE)
        Y_limpo = self._preencher_nan(Y.astype(np.float32))

        X_t = torch.tensor(X.astype(np.float32))
        Y_t = torch.tensor(Y_limpo)

        dataset  = TensorDataset(X_t, Y_t)
        loader   = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.modelo_  = _RedeRanking(self.emb_dim, self.n_pipelines,
                                      self.hidden_sizes, self.dropout)
        otimizador    = torch.optim.Adam(self.modelo_.parameters(), lr=self.lr)
        criterio      = nn.BCELoss()

        self.modelo_.train()
        self.historico_ = []

        for epoca in range(1, self.epochs + 1):
            loss_total = 0.0
            for Xb, Yb in loader:
                otimizador.zero_grad()
                pred = self.modelo_(Xb)
                loss = criterio(pred, Yb)
                loss.backward()
                otimizador.step()
                loss_total += loss.item() * len(Xb)

            loss_media = loss_total / len(X)
            self.historico_.append(loss_media)

            if self.verbose and (epoca % 20 == 0 or epoca == 1):
                print(f"  [MLP] época {epoca:>4}/{self.epochs}  BCE={loss_media:.5f}")

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Retorna os scores preditos para todos os pipelines.

        Parâmetros
        ----------
        X_new : (n_new, emb_dim) ou (emb_dim,)

        Retorna
        -------
        Y_pred : (n_new, n_pipelines) float em [0,1]
        """
        if self.modelo_ is None:
            raise RuntimeError("Chame fit() antes de predict().")
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        self.modelo_.eval()
        with torch.no_grad():
            X_t  = torch.tensor(X_new.astype(np.float32))
            pred = self.modelo_(X_t).numpy()
        return pred

    def predict_ranking(self, X_new: np.ndarray) -> np.ndarray:
        """Converte scores em posições de ranking (rank 1 = melhor)."""
        Y_pred = self.predict(X_new)
        return np.argsort(np.argsort(-Y_pred, axis=1), axis=1) + 1

    @staticmethod
    def _preencher_nan(Y: np.ndarray) -> np.ndarray:
        Y = Y.copy()
        for j in range(Y.shape[1]):
            col = Y[:, j]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                Y[mask, j] = float(np.nanmedian(col))
            elif mask.all():
                Y[:, j] = 0.0
        return Y
