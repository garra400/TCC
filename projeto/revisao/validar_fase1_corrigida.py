"""Auditoria de cobertura, independência e integridade dos folds reconstruídos."""
import json
import pickle

import numpy as np
import pandas as pd

from projeto.revisao.experimentos import sha256
from projeto.revisao.fase1_corrigida import CACHE, CONFIG, OUTPUT, execution_signature, make_splits, write_json


def run():
    catalog = pd.read_csv(OUTPUT/"datasets.csv").set_index("task_id")
    ids = np.load(OUTPUT/"task_ids.npy")
    expected_signature = execution_signature(CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    errors = []
    all_folds = []
    n_records = 0
    local_caches_verified = 0
    for tid in ids:
        folds_file = OUTPUT/"folds"/f"task_{tid}.csv"
        metadata = json.loads((OUTPUT/"folds"/f"task_{tid}.json").read_text(encoding="utf-8"))
        splits = json.loads((OUTPUT/"splits"/f"task_{tid}.json").read_text(encoding="utf-8"))
        assert metadata["signature"] == expected_signature, f"Configuração divergente: {tid}"
        assert metadata["folds_sha256"] == sha256(folds_file), f"Checkpoint alterado: {tid}"
        assert metadata["raw_sha256"] == catalog.loc[tid, "raw_sha256"], f"Dados de origem alterados: {tid}"
        cache_file = CACHE/f"task_{tid}.pkl"
        if cache_file.exists():
            assert sha256(cache_file) == catalog.loc[tid, "cache_sha256"], f"Cache bruto alterado: {tid}"
            # Somente o cache local confiável produzido pela preparação do projeto.
            with cache_file.open("rb") as stream:
                dataset = pickle.load(stream)
            expected_splits = [{"fold": fold, "train": train.tolist(), "test": test.tolist()}
                               for fold, (train, test) in enumerate(make_splits(dataset["y"], config))]
            assert splits == expected_splits, f"Folds divergem da estratificação/semente registrada: {tid}"
            local_caches_verified += 1
        folds = pd.read_csv(folds_file)
        all_folds.append(folds)
        assert len(folds) == 300 and not folds.duplicated(["pipeline_id", "fold"]).any()
        n_records += len(folds)
        n = int(catalog.loc[tid, "n_instancias"])
        seen_test = []
        for split in splits:
            train, test, fold = split["train"], split["test"], split["fold"]
            assert not set(train) & set(test), f"Vazamento de linhas: {tid}, fold {fold}"
            assert sorted(train+test) == list(range(n)), f"Subamostragem inesperada: {tid}"
            seen_test.extend(test)
            rows = folds[folds.fold.eq(fold)]
            assert len(rows) == 60
            assert rows.n_train.eq(len(train)).all() and rows.n_test.eq(len(test)).all()
        assert sorted(seen_test) == list(range(n)), f"Cobertura de teste incorreta: {tid}"
        failures = folds[folds.status.ne("ok")]
        if len(failures):
            errors.extend(failures[["task_id", "pipeline_id", "fold", "error"]].to_dict("records"))
        valid = folds.status.eq("ok")
        assert folds.loc[valid, "balanced_accuracy"].between(0, 1).all()
        assert folds.loc[valid, "f1_macro"].between(0, 1).all()
    y = np.load(OUTPUT/"Y_accuracy.npy")
    assert y.shape == (len(ids), 60)
    raw = pd.concat(all_folds, ignore_index=True)
    expected = raw.groupby(["task_id", "pipeline_id"]).balanced_accuracy.agg(["mean", "count"])
    expected.loc[expected["count"].ne(5), "mean"] = np.nan
    expected_matrix = expected["mean"].unstack("pipeline_id").reindex(index=ids, columns=range(60)).to_numpy()
    np.testing.assert_allclose(y, expected_matrix, atol=1e-12, rtol=0, equal_nan=True)
    assert np.isfinite(np.load(OUTPUT/"X_emb.npy")).all()
    result = {"n_tasks": len(ids), "n_pipelines": 60, "n_fits": n_records,
              "same_rows_all_classifiers": True, "disjoint_train_test_rows": True,
              "all_rows_tested_once": True, "checksums_verified": True,
              "matrix_matches_five_fold_scores": True,
              "local_raw_caches_and_seeded_splits_verified": local_caches_verified,
              "note_local_cache": "A auditoria estrutural pode ser repetida sem caches; quando disponíveis, seus hashes e folds estratificados também são conferidos.",
              "failed_fits": len(errors), "missing_pipeline_scores": int(np.isnan(y).sum()),
              "signature": expected_signature}
    write_json(OUTPUT/"validacao.json", result)
    print(json.dumps(result, indent=2))
    if errors:
        print("Há falhas explicitamente registradas; precisam de inspeção antes de concluir a revisão.")
    return result


if __name__ == "__main__":
    run()
