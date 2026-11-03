"""Exercise the raw-data workflow and editable CLI without network requests."""

import json
import shutil
import subprocess
import sys
import unittest
import uuid
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from automl_ranker.io import ROOT


class WorkflowTests(unittest.TestCase):
    def test_resume_rejects_changed_raw_labels(self):
        from automl_ranker.io import load_config, write_json
        from automl_ranker.phase1 import run, pipeline_keys
        from automl_ranker.representation import describe_labeled_dataset

        output = (ROOT / "outputs" / f"test-{uuid.uuid4().hex}").resolve()
        cache = output / "cache"
        cache.mkdir(parents=True)
        try:
            x, y = make_classification(n_samples=60, n_features=4, random_state=9)
            x, y = pd.DataFrame(x), pd.Series(y)
            file = cache / "task_1.pkl"
            payload = dict(X=x, y=y, nome="synthetic", raw_sha256="unchanged-features")
            with file.open("wb") as stream:
                pickle.dump(payload, stream)
            pd.DataFrame(
                [
                    dict(
                        task_id=1,
                        nome_dataset="synthetic",
                        grupo="g1",
                        elegivel=True,
                        n_instancias=60,
                        n_atributos=4,
                    )
                ]
            ).to_csv(output / "datasets.csv", index=False)
            np.save(output / "task_ids.npy", np.array([1]))
            np.save(output / "X_emb.npy", describe_labeled_dataset(x, y)[None])
            config = load_config(phase="phase1")
            config.update(
                folds=3,
                scalers=["None"],
                selectors=["None"],
                classifiers=["DecisionTree"],
            )
            path = output / "phase1.json"
            write_json(path, config)
            write_json(output / "pipeline_keys.json", pipeline_keys(config))
            run(output, path, 1, cache)
            # A clean resume reuses the completed task, then a label change invalidates it.
            run(output, path, 1, cache)
            payload["y"] = 1 - y
            with file.open("wb") as stream:
                pickle.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, "Input data changed"):
                run(output, path, 1, cache)
        finally:
            if output.is_relative_to((ROOT / "outputs").resolve()):
                shutil.rmtree(output)

    def test_synthetic_workflow_accepts_a_different_arsenal_and_fold_count(self):
        # mkdir inherits workspace ACLs on Windows, unlike restrictive temp-dir modes.
        output = (ROOT / "outputs" / f"test-{uuid.uuid4().hex}").resolve()
        output.mkdir(parents=True)
        self.assertTrue(output.is_relative_to((ROOT / "outputs").resolve()))
        try:
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "automl_ranker",
                    "demo",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(
                (output / "phase1/manifesto.json").read_text(encoding="utf8")
            )
            self.assertEqual(result["n_tasks"], 12)
            self.assertEqual(result["n_pipelines"], 4)
            self.assertEqual(result["n_fits"], 144)
            self.assertEqual(result["failed_fits"], 0)
            self.assertTrue((output / "evaluation/resumo.csv").exists())
        finally:
            # Restrict recursive deletion to the test's verified workspace directory.
            if output.is_relative_to((ROOT / "outputs").resolve()):
                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
