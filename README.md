# Meta-Learning for Ranking and Recommending Classification Pipelines in AutoML

Research code for representing labeled tabular datasets, predicting pipeline performance, and recommending classification pipelines. The default recommender is standardized k-NN; Random Forest and global-mean baselines are also available.

[Leia em português](docs/README.pt-BR.md) · [Reproduce the experiments](docs/REPRODUCING.md) · [Modify the code](docs/EXTENDING.md) · [Data and protocol](docs/DATA.md)

## Install

Python **3.14.2** reproduces the recorded environment. The package supports Python 3.11 or newer; exact numerical agreement is checked against the bundled reference with the pinned scientific-library versions.

```bash
git clone --depth 1 --single-branch --branch main https://github.com/garra400/automl-pipeline-ranking.git
cd automl-pipeline-ranking
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

```bash
python -m pip install -r requirements-lock.txt -e .
automl-rank info
```

`requirements-lock.txt` pins the direct runtime versions used for reproduction; it is not a complete transitive lock. For development with compatible versions, use `python -m pip install -e .`. For OpenML downloads in that installation, use `python -m pip install -e ".[openml]"`.

All commands also work as `python -m automl_ranker ...`. If PowerShell does not activate the environment, call `.venv\Scripts\python.exe -m automl_ranker ...` directly.

## Start with a small experiment

```bash
automl-rank demo
```

This offline demonstration generates 12 synthetic tasks, evaluates four candidate pipelines in three stratified folds, and runs grouped meta-evaluation. It checks the complete workflow without downloading datasets. Its results are illustrative and are separate from the reference experiment.

## Reproduce the reference meta-evaluation

```bash
automl-rank evaluate --verify-reference
automl-rank analyse --kind all
automl-rank plot
```

The repository includes small numerical reference inputs: **193 tasks, 165 groups, 64 descriptor positions and 60 candidate pipelines**. The first command recomputes nested grouped evaluation and checks all 42 reference metric means and the internally selected k values. It does not rerun the 57,900 classifier fits needed to build the performance matrix.

Outputs are written under `outputs/`, which Git ignores. They include per-task predictions and metrics, group means, descriptive paired bootstrap intervals, split definitions, k-selection records and run manifests. The additional analyses cover task-level LOO versus group separation, metadata exclusions, a conservative proximity stress test and recorded top-5 execution cost.

## Extended k grid, neighbor weights and partition stability

```bash
automl-rank robustness
```

This additional experiment extends the nested search to **1, 3, 5, 7, 10, 15, 20, 30, 50 and all training neighbors**. It saves inner-validation curves for every outer fold, compares inverse-distance weights with epsilon 1e-8, 1e-4 and 1e-2 against uniform weights, and repeats grouped evaluation with ten fixed outer seeds. Inner and model seeds remain 42. The principal variant remains standardized inverse-distance k-NN with epsilon 1e-8; weighting alternatives are controls rather than choices made from outer-test results.

The original short-grid evaluation remains available with `evaluate --verify-reference` as an audit control. Expanded-grid results are in `outputs/robustness/principal/`; repeated-seed variation, the 113-group stress test, metadata exclusions and normalized weight concentration are saved separately. Repeated seeds measure partition variation, not additional independent datasets.

Using all training neighbors equals the global mean **only for uniform weights**. The experiment checks that equality numerically; inverse-distance weighting with all neighbors is a different estimator. To add the published-representation comparisons against the global mean, first run `dataset2vec`, then:

```bash
automl-rank robustness --external-metrics outputs/dataset2vec/metricas.csv
```

Use a new output directory when changing an existing experiment plan. See [the robustness protocol](docs/REPRODUCING.md#6-extended-grid-and-partition-stability) for the complete reporting rules.

## Recommend pipelines for your data

The query must contain **features and a target column**: the deterministic descriptor uses query labels. It is not a label-free recommender.

```bash
automl-rank recommend --csv examples/labeled_query.csv --target target --top 5
automl-rank recommend --csv examples/labeled_query.csv --target target --model random-forest --top 5
```

Replace the example path and `target` with your CSV and label-column name. The output lists candidate IDs, preprocessing choices, classifiers and predicted balanced accuracy. This command ranks candidates; it does not fit or evaluate the suggested classifiers on the query. Use grouped evaluation to assess recommendation quality.

## Rebuild the performance matrix from OpenML

```bash
automl-rank prepare --task-ids 3 11 12 --output outputs/subset
automl-rank phase1 --input outputs/subset --workers 1
```

To rebuild the complete corpus, omit `--task-ids`. It downloads the 201 preserved candidate tasks, applies the final eligibility rules and evaluates the eligible tasks. The full configuration requires **57,900 classifier fits**, so it is substantially slower than the reference meta-evaluation. Downloads need internet access; the synthetic demo, recommendation example and reference evaluation work offline after installation.

Raw data are cached locally in `data/cache/` and are not committed. Every candidate receives the same task-level train/test indices. Imputation, categorical encoding, scaling and feature selection are fitted exclusively on the training partition. Different configurations require a new output directory; incompatible checkpoints are rejected.

## Compare published Dataset2Vec representations

```bash
git clone https://github.com/hadijomaa/dataset2vec.git ../dataset2vec
git -C ../dataset2vec checkout 42058daf5077f3adb2f40af89a26f54f54ee1cdb
automl-rank dataset2vec --source ../dataset2vec
```

This separate exploratory comparison reuses the authors' published representations, 120 datasets, 256 configurations and five predefined partitions. It does not retrain a neural network or require TensorFlow. These results use a different corpus and candidate space and must not be mixed numerically with the 60-pipeline OpenML evaluation.

## Code layout

```text
automl_ranker/
  representation.py   deterministic dataset descriptors
  pipelines.py        train-only numerical and categorical transformations
  phase1.py           classifier evaluation, aggregation and resumable tasks
  download.py         OpenML loading, eligibility and metadata audit
  evaluation.py       nested grouped CV, baselines, metrics and paired intervals
  recommend.py        recommendation for a new labeled table
  analyses.py         LOO, metadata sensitivity, proximity and execution cost
  external.py         published Dataset2Vec comparison
  plotting.py         research plots and their numerical source tables
  cli.py              command-line interface
configs/              editable classifier and meta-evaluation settings
data/reference/       compact, checksummed numerical inputs
examples/             generated labeled CSV example
tests/                leakage controls, metrics and workflow checks
docs/                 reproduction and extension guides
```

## Test

```bash
python -m unittest discover -s tests -v
```

GitHub Actions checks the code on Linux and Windows. Tests cover train-only preprocessing, disjoint task groups, protection against outer-test outcomes influencing predictions or k, metric semantics, descriptor consistency and a synthetic end-to-end workflow.

## Interpretation and attribution

The evaluation is retrospective and conditional on this task archive, candidate space and grouping rule. Descriptors contain **55 statistical values plus nine zero positions**, normalized to 64 dimensions; they are not learned Dataset2Vec embeddings. Categories and nominal labels are encoded numerically, and the initial groups do not identify every related task family. Bootstrap intervals are descriptive. A higher rank correlation need not produce a material improvement over the global mean in the first recommendation. The timing analysis estimates a recorded serial stage, not full-system wall-clock acceleration.

OpenML datasets retain their original provenance and terms. The optional Dataset2Vec material is retrieved from its authors' repository and is not redistributed here. Cite the implementation using [CITATION.cff](CITATION.cff); method background and data details are linked in [docs/DATA.md](docs/DATA.md).
