"""Testes de independência, métricas e tratamento dos dados ausentes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from projeto.revisao.experimentos import (
    dataset_catalog, eligibility, fill_targets, grouped_evaluation, knn_predict, ranking_metrics,
)


class RankingTests(unittest.TestCase):
    def test_raw_preprocessing_handles_unseen_category_and_train_median(self):
        from sklearn.tree import DecisionTreeClassifier
        from projeto.revisao.pipeline_bruto import build_pipeline
        train = pd.DataFrame({"n": [1., np.nan, 3.], "cat": ["a", "b", "a"]})
        test = pd.DataFrame({"n": [np.nan], "cat": ["new"]})
        model = build_pipeline(DecisionTreeClassifier(random_state=42)).fit(train, [0, 1, 0])
        transformed = model.named_steps["preprocess"].transform(test)
        np.testing.assert_allclose(transformed, [[2, -1]])
        self.assertEqual(model.predict(test).shape, (1,))
    def test_perfect_and_reversed_rankings(self):
        truth = np.array([.2, .5, .9])
        perfect = ranking_metrics(truth, truth)
        reverse = ranking_metrics(truth, -truth)
        for metric in ["spearman", "kendalltau", "kendall_ponderado", "ndcg", "ndcg_5"]:
            self.assertAlmostEqual(perfect[metric], 1)
        self.assertAlmostEqual(reverse["spearman"], -1)
        self.assertAlmostEqual(reverse["regret_1"], .7)

    def test_ndcg_uses_linear_gain(self):
        scores = ranking_metrics(np.array([1., 0., .5]), np.array([.5, 1., 0.]))
        discount = 1 / np.log2(np.arange(2, 5))
        expected = np.dot([0, 1, .5], discount) / np.dot([1, .5, 0], discount)
        self.assertAlmostEqual(scores["ndcg"], expected)

    def test_missing_test_score_is_excluded(self):
        result = ranking_metrics(np.array([np.nan, .9, .1]), np.array([9, .8, .2]))
        self.assertEqual(result["n_pipelines_validos"], 2)
        self.assertAlmostEqual(result["ndcg"], 1)
        self.assertAlmostEqual(result["regret_1"], 0)

    def test_constant_prediction_is_visible(self):
        result = ranking_metrics(np.array([.2, .5, .9]), np.ones(3))
        self.assertEqual(result["status"], "predicao_constante")
        self.assertTrue(np.isnan(result["spearman"]))
        self.assertTrue(np.isfinite(result["ndcg"]))

    def test_imputation_is_train_only_and_flags_unseen_pipeline(self):
        filled, supported = fill_targets(np.array([[.2, np.nan], [np.nan, np.nan], [.8, np.nan]]))
        self.assertAlmostEqual(filled[1, 0], .5)
        np.testing.assert_array_equal(supported, [True, False])
        self.assertTrue(np.isfinite(filled).all())

    def test_knn_distance_weights_and_single_neighbor(self):
        train = np.array([[0.], [2.]])
        y = np.array([[.2, .8], [.8, .2]])
        np.testing.assert_allclose(knn_predict(train, y, np.array([[0.]]), 1), [[.2, .8]])
        np.testing.assert_allclose(knn_predict(train, y, np.array([[1.]]), 9), [[.5, .5]])

    def test_repeated_names_are_kept_in_same_group(self):
        raw = pd.DataFrame({"task_id": [1, 2, 3], "nome_dataset": ["A", "a", "B"]})
        with tempfile.TemporaryDirectory() as folder:
            catalog = dataset_catalog(np.array([1, 2, 3]), raw, Path(folder))
        self.assertEqual(catalog.grupo[0], catalog.grupo[1])
        self.assertNotEqual(catalog.grupo[0], catalog.grupo[2])

    def test_eligibility_rejects_rare_classes_and_wrong_task_type(self):
        catalog = pd.DataFrame({"task_type_id": [1, 1, 2], "menor_classe_n": [5, 1, 10],
                                "n_classes": [2, 9999, 2], "n_instancias": [500, 10000, 1000],
                                "metadata_status": ["ok"] * 3})
        np.testing.assert_array_equal(eligibility(catalog), [True, False, False])

    def test_outer_test_targets_do_not_change_its_predictions_or_k(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(12, 4)), rng.uniform(size=(12, 3))
        ids, groups = np.arange(12), np.repeat(np.arange(6), 2)
        config = {"outer_folds": 3, "inner_folds": 2, "k_values": [1, 3],
                  "primary_metric": "kendall_ponderado", "seed": 42,
                  "rf_estimators": 3, "rf_min_samples_leaf": 2}
        keys = [("None", "None", str(i)) for i in range(3)]
        first = grouped_evaluation(x, y, ids, groups, keys, config)
        held = first[3][0]["test_tasks"]
        changed = y.copy()
        changed[held] = 1 - changed[held]
        second = grouped_evaluation(x, changed, ids, groups, keys, config)
        for split in first[3]:
            self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        a = first[1].query("fold == 0").predito
        b = second[1].query("fold == 0").predito
        np.testing.assert_allclose(a, b, equal_nan=True)
        pd.testing.assert_frame_equal(first[2].query("fold == 0"), second[2].query("fold == 0"))


if __name__ == "__main__":
    unittest.main()
