"""Input validation, reproducible paths and machine-readable run records."""

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import sha256

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference"


def load_config(path=None, phase="meta"):
    path = Path(path or ROOT / f"configs/{phase}.json")
    return json.loads(path.read_text(encoding="utf8"))


def load_reference(folder=REFERENCE, verify=True):
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if verify and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        for name, digest in manifest.get("files_sha256", {}).items():
            if sha256(folder / name) != digest:
                raise ValueError(
                    f"Reference checksum mismatch: {name}. Use a separate directory for modified inputs."
                )
    elif verify and (folder / "manifesto.json").exists():
        manifest = json.loads((folder / "manifesto.json").read_text(encoding="utf8"))
        if manifest.get("status") != "complete":
            raise ValueError("Finish classifier evaluation before loading this matrix.")
        for name, digest in manifest.get("artifacts_sha256", {}).items():
            if sha256(folder / name) != digest:
                raise ValueError(f"Classifier output checksum mismatch: {name}.")
    ids = np.load(folder / "task_ids.npy", allow_pickle=False)
    x = np.load(folder / "X_emb.npy", allow_pickle=False).astype(np.float32)
    y = np.load(folder / "Y_accuracy.npy", allow_pickle=False).astype(np.float32)
    catalog = (
        pd.read_csv(folder / "datasets.csv").set_index("task_id").loc[ids].reset_index()
    )
    keys = json.loads((folder / "pipeline_keys.json").read_text(encoding="utf8"))
    if (
        x.shape != (len(ids), 64)
        or y.shape != (len(ids), len(keys))
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Misaligned task IDs, vectors or pipeline scores.")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (y < 0).any()
        or (y > 1).any()
    ):
        raise ValueError(
            "Reference inputs require finite vectors and complete scores in [0, 1]."
        )
    if catalog.grupo.isna().any():
        raise ValueError("Every task needs an explicit group.")
    return x, y, ids, catalog.grupo.to_numpy(), keys, catalog


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf8"
    )


def run_manifest(folder, inputs, config, **extra):
    write_json(
        Path(folder) / "run.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scipy", "scikit-learn"]
            },
            "input_files_sha256": {str(Path(p).name): sha256(p) for p in inputs},
            "config": config,
            "interval_scope": "Retrospective descriptive group bootstrap; overlapping training folds and exploratory comparisons limit inference.",
            **extra,
        },
    )
