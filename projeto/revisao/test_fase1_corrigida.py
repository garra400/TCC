"""Testes das garantias adicionadas à reconstrução da Fase 1."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from projeto.revisao.fase1_corrigida import CONFIG, make_splits, pipeline_keys, run_task


class PhaseOneTests(unittest.TestCase):
    def test_condition_comparison_weights_groups_and_only_pairs_observed_scores(self):
        from projeto.revisao.avaliar_fase1_corrigida import compare_conditions
        from projeto.revisao.experimentos import METRICS
        base = pd.DataFrame({"task_id": [1, 2, 3], "grupo": ["a", "a", "b"],
                             "modelo": ["knn_ajustado"]*3, "fold": [0, 0, 1]})
        for metric in METRICS:
            base[metric] = [.2, .4, .3]
        complete = base.copy()
        partial = base.copy()
        for metric in METRICS:
            complete[metric] = [.4, .4, .6]
            partial[metric] = [.4, np.nan, .6]
        cases = {"historico_mesmos_folds": (base, None, None),
                 "novos_escores_vetor_historico": (partial, None, None),
                 "reconstrucao_completa": (complete, None, None)}
        result = compare_conditions(cases, {"bootstrap_samples": 50, "seed": 42})
        self.assertTrue(np.allclose(result.query("condicao == 'reconstrucao_completa'").diferenca_media_grupos, .2))
        self.assertTrue(np.allclose(result.query("condicao == 'novos_escores_vetor_historico'").diferenca_media_grupos, .25))
        self.assertTrue(result.n_grupos_pareados.eq(2).all())

    def test_split_keeps_all_rows_and_classes_and_never_overlaps(self):
        config = json.loads(CONFIG.read_text())
        y = np.repeat([0, 1], [70, 30])
        splits = make_splits(y, config)
        seen = []
        for train, test in splits:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train)+len(test), len(y))
            self.assertEqual(set(y[train]), {0, 1})
            self.assertEqual(set(y[test]), {0, 1})
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(y))))

    def test_rare_class_is_rejected_before_any_training(self):
        config = json.loads(CONFIG.read_text())
        with self.assertRaises(ValueError):
            make_splits(np.array([0]*20+[1]*4), config)

    def test_all_classifiers_receive_the_same_complete_folds(self):
        config = json.loads(CONFIG.read_text())
        config.update(scalers=["StandardScaler", "None"], selectors=["SelectKBest", "None"],
                      classifiers=["DecisionTree", "SVM"])
        x, y = make_classification(n_samples=100, n_features=4, random_state=9)
        frame = pd.DataFrame(x)
        frame.columns = frame.columns.astype(str)
        frame.iloc[0, 0] = np.nan
        frame["category"] = ["a", "b"]*50
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"folds").mkdir()
            (root/"splits").mkdir()
            with (root/"task_999.pkl").open("wb") as stream:
                pickle.dump({"X": frame, "y": pd.Series(y), "nome": "synthetic",
                             "raw_sha256": "test"}, stream)
            with patch("projeto.revisao.fase1_corrigida.CACHE", root):
                result = run_task(999, config, root, "test_signature")
            folds = pd.read_csv(root/"folds/task_999.csv")
            splits = json.loads((root/"splits/task_999.json").read_text())
        self.assertEqual(result["errors"], 0)
        self.assertEqual(len(folds), len(pipeline_keys(config))*5)
        self.assertTrue(folds.n_train.eq(80).all())
        self.assertTrue(folds.n_test.eq(20).all())
        self.assertTrue(folds.balanced_accuracy.between(0, 1).all())
        self.assertEqual(len(splits), 5)


if __name__ == "__main__":
    unittest.main()
