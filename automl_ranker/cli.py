"""Command-line entry points for the algorithm and reproducible experiments."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_limits

from .evaluation import grouped_evaluation, paired, summarize
from .io import ROOT, REFERENCE, load_config, load_reference, run_manifest, write_json


def evaluate(folder, output, config, verify_reference=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y, ids, groups, keys, catalog = load_reference(folder)
    with threadpool_limits(limits=1):
        metrics, predictions, selection, splits, distances = grouped_evaluation(
            x, y, ids, groups, keys, config
        )
    metrics.to_csv(output / "metricas_por_task.csv", index=False)
    predictions.to_csv(output / "predicoes.csv", index=False)
    selection.to_csv(output / "selecao_k_interna.csv", index=False)
    summary = summarize(metrics, config)
    summary.to_csv(output / "resumo.csv", index=False)
    paired(metrics, config).to_csv(output / "comparacoes_pareadas.csv", index=False)
    write_json(output / "splits.json", splits)
    if verify_reference:
        expected = pd.read_csv(Path(folder) / "expected_metrics.csv").sort_values(
            ["modelo", "metrica"]
        )
        actual = summary.sort_values(["modelo", "metrica"])
        assert list(zip(expected.modelo, expected.metrica)) == list(
            zip(actual.modelo, actual.metrica)
        )
        np.testing.assert_allclose(
            actual.media_grupos,
            expected.media_grupos,
            atol=1e-10,
            rtol=0,
            equal_nan=True,
        )
        old_k = pd.read_csv(Path(folder) / "expected_k.csv").sort_values(
            ["fold", "scaled", "k"]
        )
        new_k = selection.sort_values(["fold", "scaled", "k"])
        np.testing.assert_array_equal(new_k.selected, old_k.selected)
        print(
            "All 42 reference means and internally selected k values reproduced.",
            flush=True,
        )
    run_manifest(
        output,
        [
            Path(folder) / name
            for name in [
                "X_emb.npy",
                "Y_accuracy.npy",
                "task_ids.npy",
                "datasets.csv",
                "pipeline_keys.json",
            ]
        ],
        config,
        n_tasks=len(ids),
        n_groups=len(set(groups)),
        n_pipelines=len(keys),
        verified_against_reference=verify_reference,
    )
    print(
        summary.pivot(index="modelo", columns="metrica", values="media_grupos")
        .round(4)
        .to_string(),
        flush=True,
    )
    return summary


def demo(output):
    """Small raw-data -> classifier scores -> meta-evaluation workflow; no internet."""
    from .phase1 import run_task, aggregate, pipeline_keys, frame_fingerprint
    from .representation import describe_labeled_dataset

    output = Path(output)
    phase = output / "phase1"
    cache = output / "cache"
    for p in [phase / "folds", phase / "splits", cache]:
        p.mkdir(parents=True, exist_ok=True)
    config = load_config(phase="phase1")
    config.update(
        folds=3,
        min_instances=40,
        min_class_count=3,
        scalers=["None", "StandardScaler"],
        selectors=["None"],
        classifiers=["DecisionTree", "kNN"],
        workers=1,
    )
    cfg = output / "demo-phase1.json"
    write_json(cfg, config)
    ids = np.arange(12)
    rows = []
    vectors = []
    for tid in ids:
        a, b = make_classification(
            n_samples=90,
            n_features=6,
            n_informative=3,
            n_redundant=1,
            class_sep=0.5 + tid / 8,
            random_state=int(40 + tid),
        )
        x = pd.DataFrame(a, columns=[f"x{i}" for i in range(a.shape[1])])
        y = pd.Series(b)
        raw_hash = frame_fingerprint(x)
        with (cache / f"task_{tid}.pkl").open("wb") as stream:
            pickle.dump(
                {"X": x, "y": y, "nome": f"synthetic-{tid}", "raw_sha256": raw_hash},
                stream,
            )
        rows.append(
            dict(
                task_id=int(tid),
                nome_dataset=f"synthetic-{tid}",
                grupo=f"synthetic-{tid}",
                elegivel=True,
                n_instancias=len(x),
                n_atributos=x.shape[1],
            )
        )
        vectors.append(describe_labeled_dataset(x, y))
        result = run_task(int(tid), config, phase, "synthetic-demo", cache)
        if result["errors"]:
            raise RuntimeError("A demo classifier failed.")
    pd.DataFrame(rows).to_csv(phase / "datasets.csv", index=False)
    np.save(phase / "task_ids.npy", ids)
    np.save(phase / "X_emb.npy", np.stack(vectors))
    write_json(phase / "pipeline_keys.json", pipeline_keys(config))
    aggregate(phase, cfg)
    meta = load_config()
    meta.update(
        outer_folds=3,
        inner_folds=2,
        k_values=[1, 3],
        rf_estimators=10,
        bootstrap_samples=100,
    )
    evaluate(phase, output / "evaluation", meta)
    print(f"Synthetic demonstration completed: {output.resolve()}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Meta-learning for classification pipeline ranking and recommendation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "info", help="Validate and describe the bundled numerical reference"
    )
    p.add_argument("--input", type=Path, default=REFERENCE)
    p = sub.add_parser(
        "demo", help="Run a small synthetic experiment without downloading data"
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    p = sub.add_parser("evaluate", help="Run nested grouped meta-evaluation")
    p.add_argument("--input", type=Path, default=REFERENCE)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    p.add_argument("--config", type=Path)
    p.add_argument("--verify-reference", action="store_true")
    p = sub.add_parser("recommend", help="Rank candidates for a labeled CSV dataset")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--input", type=Path, default=REFERENCE)
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--model", choices=["knn", "random-forest", "global-mean"], default="knn"
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/recommendation.csv")
    p = sub.add_parser(
        "prepare", help="Download raw OpenML tasks and compute task descriptors"
    )
    p.add_argument("--manifest", type=Path, default=REFERENCE / "datasets.csv")
    p.add_argument("--config", type=Path)
    p.add_argument("--task-ids", type=int, nargs="+")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/phase1")
    p.add_argument("--cache", type=Path, default=ROOT / "data/cache")
    p = sub.add_parser(
        "phase1", help="Evaluate configured classifiers on the prepared raw tasks"
    )
    p.add_argument("--input", type=Path, default=ROOT / "outputs/phase1")
    p.add_argument("--config", type=Path)
    p.add_argument("--cache", type=Path, default=ROOT / "data/cache")
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser(
        "analyse", help="Run protocol, sensitivity, proximity and cost analyses"
    )
    p.add_argument("--input", type=Path, default=REFERENCE)
    p.add_argument("--evaluation", type=Path, default=ROOT / "outputs/evaluation")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/analyses")
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--kind",
        choices=["loo", "sensitivity", "proximity", "cost", "all"],
        default="all",
    )
    p = sub.add_parser(
        "dataset2vec", help="Compare representations using the authors' published data"
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/dataset2vec")
    p = sub.add_parser("plot", help="Generate research plots without LaTeX")
    p.add_argument("--input", type=Path, default=REFERENCE)
    p.add_argument("--evaluation", type=Path, default=ROOT / "outputs/evaluation")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args(argv)
    if args.command == "info":
        x, y, ids, groups, keys, _ = load_reference(args.input)
        print(
            json.dumps(
                dict(
                    tasks=len(ids),
                    groups=len(set(groups)),
                    candidates=len(keys),
                    descriptor_shape=list(x.shape),
                    score_shape=list(y.shape),
                ),
                indent=2,
            )
        )
    elif args.command == "demo":
        demo(args.output)
    elif args.command == "evaluate":
        evaluate(
            args.input, args.output, load_config(args.config), args.verify_reference
        )
    elif args.command == "recommend":
        from .recommend import recommend

        frame = pd.read_csv(args.csv)
        if args.target not in frame:
            parser.error(f"Target column not found: {args.target}")
        result = recommend(
            frame.drop(columns=[args.target]),
            frame[args.target],
            args.input,
            load_config(args.config),
            args.model,
            args.k,
            args.top,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))
    elif args.command == "prepare":
        from .download import prepare

        prepare(
            load_config(args.config, "phase1"),
            args.output,
            args.cache,
            args.manifest,
            args.task_ids,
        )
    elif args.command == "phase1":
        from .phase1 import CONFIG, run

        run(args.input, args.config or CONFIG, args.workers, args.cache)
    elif args.command == "analyse":
        from .analyses import run

        with threadpool_limits(limits=1):
            run(
                args.input,
                args.evaluation,
                args.output,
                load_config(args.config),
                args.kind,
            )
    elif args.command == "dataset2vec":
        from .external import compare

        with threadpool_limits(limits=1):
            compare(args.source, args.output)
    elif args.command == "plot":
        from .plotting import plot

        plot(args.input, args.evaluation, args.output)
