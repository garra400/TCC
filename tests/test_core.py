"""Behavioral checks for leakage controls, ranking metrics and recommendation."""

import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from automl_ranker.evaluation import (
    fill_targets,
    grouped_evaluation,
    knn_predict,
    ranking_metrics,
    paired,
)
from automl_ranker.io import ROOT, load_config, load_reference
from automl_ranker.phase1 import make_splits, pipeline_keys
from automl_ranker.pipelines import build_pipeline
from automl_ranker.representation import describe_labeled_dataset
from automl_ranker.recommend import recommend


class RankingTests(unittest.TestCase):
    def test_transformations_use_training_statistics_and_allow_unseen_categories(self):
        train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "cat": ["a", "b", "a"]})
        test = pd.DataFrame({"n": [np.nan], "cat": ["new"]})
        model = build_pipeline(DecisionTreeClassifier(random_state=42)).fit(
            train, [0, 1, 0]
        )
        np.testing.assert_allclose(
            model.named_steps["preprocess"].transform(test), [[2, -1]]
        )

    def test_stratified_folds_never_overlap_and_cover_every_row(self):
        target = np.repeat([0, 1], [70, 30])
        seen = []
        for train, test in make_splits(target, load_config(phase="phase1")):
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(target[test]), {0, 1})
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(target))))

    def test_classes_must_support_the_requested_number_of_folds(self):
        config = load_config(phase="phase1")
        config.update(folds=6, min_class_count=3)
        with self.assertRaises(ValueError):
            make_splits(np.array([0] * 20 + [1] * 5), config)

    def test_perfect_reversed_constant_and_missing_rankings(self):
        truth = np.array([0.2, 0.5, 0.9])
        perfect = ranking_metrics(truth, truth)
        for metric in ["spearman", "kendalltau", "kendall_ponderado", "ndcg", "ndcg_5"]:
            self.assertAlmostEqual(perfect[metric], 1)
        self.assertAlmostEqual(ranking_metrics(truth, -truth)["regret_1"], 0.7)
        constant = ranking_metrics(truth, np.ones(3))
        self.assertTrue(np.isnan(constant["spearman"]))
        self.assertEqual(constant["status"], "predicao_constante")
        self.assertEqual(
            ranking_metrics(np.array([np.nan, 0.9, 0.1]), np.array([9, 0.8, 0.2]))[
                "n_pipelines_validos"
            ],
            2,
        )

    def test_ndcg_has_linear_gain(self):
        metric = ranking_metrics(np.array([1.0, 0.0, 0.5]), np.array([0.5, 1.0, 0.0]))
        discount = 1 / np.log2(np.arange(2, 5))
        expected = np.dot([0, 1, 0.5], discount) / np.dot([1, 0.5, 0], discount)
        self.assertAlmostEqual(metric["ndcg"], expected)

    def test_missing_training_targets_do_not_impute_the_test(self):
        filled, supported = fill_targets(
            np.array([[0.2, np.nan], [np.nan, np.nan], [0.8, np.nan]])
        )
        self.assertAlmostEqual(filled[1, 0], 0.5)
        np.testing.assert_array_equal(supported, [True, False])
        self.assertTrue(np.isfinite(filled).all())

    def test_knn_inverse_distance_and_k_bounds(self):
        x = np.array([[0.0], [2.0]])
        y = np.array([[0.2, 0.8], [0.8, 0.2]])
        np.testing.assert_allclose(
            knn_predict(x, y, np.array([[1.0]]), 9), [[0.5, 0.5]]
        )
        with self.assertRaises(ValueError):
            knn_predict(x, y, x, 0)

    def test_descriptor_is_deterministic_finite_and_normalized_with_nine_zeros(self):
        x = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": ["a", "b", "a", "c"]})
        y = pd.Series([0, 1, 0, 1])
        first = describe_labeled_dataset(x, y)
        np.testing.assert_array_equal(first, describe_labeled_dataset(x, y))
        self.assertEqual(first.shape, (64,))
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(np.linalg.norm(first), 1)
        np.testing.assert_array_equal(first[-9:], np.zeros(9))

    def test_outer_test_outcomes_cannot_change_its_predictions_or_selected_k(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(12, 4)), rng.uniform(size=(12, 3))
        ids, groups = np.arange(12), np.repeat(np.arange(6), 2)
        config = load_config()
        config.update(outer_folds=3, inner_folds=2, k_values=[1, 3], rf_estimators=3)
        keys = [("None", "None", str(i)) for i in range(3)]
        first = grouped_evaluation(x, y, ids, groups, keys, config)
        held = first[3][0]["test_tasks"]
        changed = y.copy()
        changed[held] = 1 - changed[held]
        second = grouped_evaluation(x, changed, ids, groups, keys, config)
        for split in first[3]:
            self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        np.testing.assert_allclose(
            first[1].query("fold==0").predito,
            second[1].query("fold==0").predito,
            equal_nan=True,
        )
        pd.testing.assert_frame_equal(
            first[2].query("fold==0"), second[2].query("fold==0")
        )

    def test_paired_intervals_average_within_group_before_across_groups(self):
        data = []
        for model, values in [
            ("random_forest", [0.5, 0.7, 0.8]),
            ("media_global", [0.2, 0.2, 0.4]),
        ]:
            for tid, group, value in zip([1, 2, 3], ["a", "a", "b"], values):
                data.append(
                    dict(
                        modelo=model,
                        task_id=tid,
                        grupo=group,
                        kendall_ponderado=value,
                        ndcg_5=value,
                        regret_1=value,
                    )
                )
        result = paired(
            pd.DataFrame(data),
            dict(bootstrap_samples=20, seed=42),
            [("random_forest", "media_global")],
        )
        np.testing.assert_allclose(result.diferenca_media_grupos, 0.4)

    def test_reference_integrity_and_manifest_dimensions(self):
        x, y, ids, groups, keys, catalog = load_reference()
        self.assertEqual(x.shape, (193, 64))
        self.assertEqual(y.shape, (193, 60))
        self.assertEqual(len(set(groups)), 165)
        self.assertEqual(len(pipeline_keys(load_config(phase="phase1"))), 60)

    def test_csv_recommendation_is_sorted_and_maps_scores_to_candidates(self):
        frame = pd.read_csv(ROOT / "examples/labeled_query.csv")
        output = recommend(
            frame.drop(columns="target"),
            frame.target,
            ROOT / "data/reference",
            load_config(),
            top=5,
        )
        self.assertEqual(len(output), 5)
        self.assertTrue(output.predicted_balanced_accuracy.is_monotonic_decreasing)
        self.assertEqual(output.pipeline_id.nunique(), 5)


if __name__ == "__main__":
    unittest.main()
