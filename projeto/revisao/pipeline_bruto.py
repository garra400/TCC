"""Pré-processamento ajustado dentro da CV, para os novos testes em dados brutos."""
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler


def categorical_strings(x):
    # Substituição determinística: não estima estatísticas usando o teste.
    return x.astype("string").fillna("__missing__").astype(str)


def build_pipeline(estimator):
    categorical = Pipeline([
        ("string", FunctionTransformer(categorical_strings)),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
         make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("preprocess", preprocess), ("scaler", StandardScaler()), ("classifier", estimator)])
