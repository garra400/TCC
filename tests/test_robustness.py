"""Controls for the all-neighbor limit, weight concentration and nested selection."""

import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from automl_ranker.io import ROOT, load_config
from automl_ranker.robustness import neighborhood, predict_neighbors, run_partition


class RobustnessTests(unittest.TestCase):
    def test_all_uniform_neighbors_equal_global_mean_but_inverse_neighbors_do_not(self):
        x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], dtype=np.float32)
        distances, indices = neighborhood(x, np.array([[0.0], [0.1]]), False)
        uniform, maximum, effective = predict_neighbors(
            y, distances, indices, "all", "uniform", 0
        )
        np.testing.assert_array_equal(uniform, np.tile(y.mean(axis=0), (2, 1)))
        np.testing.assert_allclose(maximum, 1 / 3)
        np.testing.assert_allclose(effective, 3)
        inverse, _, _ = predict_neighbors(y, distances, indices, "all")
        self.assertFalse(np.allclose(inverse, uniform))

    def test_larger_epsilon_reduces_near_zero_neighbor_dominance(self):
        y = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
        distances = np.array([[0.0, 0.1]])
        indices = np.array([[0, 1]])
        first, dominant, _ = predict_neighbors(
            y, distances, indices, 2, "inverse", 1e-8
        )
        second, smoothed, _ = predict_neighbors(
            y, distances, indices, 2, "inverse", 1e-2
        )
        self.assertGreater(dominant[0], 0.99)
        self.assertLess(smoothed[0], dominant[0])
        self.assertGreater(np.abs(second - y[0]).sum(), np.abs(first - y[0]).sum())

    def test_outer_targets_do_not_select_extended_k_or_change_predictions(self):
        output = (ROOT / "outputs" / f"robust-test-{uuid.uuid4().hex}").resolve()
        output.mkdir(parents=True)
        self.assertTrue(output.is_relative_to((ROOT / "outputs").resolve()))
        try:
            rng = np.random.default_rng(3)
            x = rng.normal(size=(12, 4)).astype(np.float32)
            y = rng.uniform(size=(12, 3)).astype(np.float32)
            ids = np.arange(12)
            groups = np.repeat(np.arange(6), 2)
            config = load_config()
            config.update(
                inner_folds=2, outer_folds=2, rf_estimators=2, bootstrap_samples=20
            )
            splits = [(np.arange(8), np.arange(8, 12))]
            keys = [("None", "None", str(i)) for i in range(3)]
            run_partition(
                x, y, ids, groups, keys, config, 42, output / "first", partitions=splits
            )
            changed = y.copy()
            changed[8:] = 1 - changed[8:]
            run_partition(
                x,
                changed,
                ids,
                groups,
                keys,
                config,
                42,
                output / "second",
                partitions=splits,
            )
            first = pd.read_csv(output / "first/predicoes.csv")
            second = pd.read_csv(output / "second/predicoes.csv")
            np.testing.assert_allclose(first.predito, second.predito)
            pd.testing.assert_frame_equal(
                pd.read_csv(output / "first/selecao_k.csv"),
                pd.read_csv(output / "second/selecao_k.csv"),
            )
        finally:
            if output.is_relative_to((ROOT / "outputs").resolve()):
                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
