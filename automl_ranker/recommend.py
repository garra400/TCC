"""Recommend candidates for a labeled query without evaluating its classifiers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .evaluation import knn_predict
from .io import load_reference
from .representation import describe_labeled_dataset


def recommend(x_query, y_query, reference, config, model="knn", k=10, top=5):
    if top < 1 or k < 1:
        raise ValueError("top and k must be positive integers.")
    x, y, ids, groups, keys, catalog = load_reference(reference)
    query = describe_labeled_dataset(x_query, y_query).astype(np.float32)[None]
    if model == "knn":
        scaler = StandardScaler().fit(x)
        scores = knn_predict(scaler.transform(x), y, scaler.transform(query), k)[0]
    elif model == "random-forest":
        forest = RandomForestRegressor(
            n_estimators=config["rf_estimators"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            random_state=config["seed"],
            n_jobs=config.get("n_jobs", 1),
        )
        scores = forest.fit(x, y).predict(query)[0]
    elif model == "global-mean":
        scores = y.mean(axis=0)
    else:
        raise ValueError(f"Unknown model: {model}")
    order = np.argsort(-scores, kind="stable")[:top]
    return pd.DataFrame(
        [
            dict(
                rank=i + 1,
                pipeline_id=int(j),
                scaler=keys[j][0],
                selector=keys[j][1],
                classifier=keys[j][2],
                predicted_balanced_accuracy=float(scores[j]),
            )
            for i, j in enumerate(order)
        ]
    )
