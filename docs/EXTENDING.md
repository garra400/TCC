# Modifying the implementation

## Change k, partitions or Random Forest

Copy `configs/meta.json` to another JSON file. Edit `k_values`, `outer_folds`, `inner_folds`, `rf_estimators`, `rf_min_samples_leaf`, `seed` or `bootstrap_samples`, then run:

```bash
automl-rank evaluate --config configs/my-meta.json --output outputs/my-evaluation
automl-rank analyse --config configs/my-meta.json --evaluation outputs/my-evaluation --output outputs/my-analyses
```

The defaults compare scaled/unscaled k-NN, a fixed-k control, Random Forest and simple baselines. To add another metamodel, extend the `outputs` mapping inside `grouped_evaluation` in `automl_ranker/evaluation.py`. Fit it only on `x[train]` and training targets. Add any parameter selection inside the inner group loop, and extend the paired comparisons if needed.

Use `--verify-reference` only with the unchanged numerical inputs and default scientific configuration. Modified experiments have different expected results.

## Change the candidate arsenal

Copy `configs/phase1.json`, edit the existing scaler/selector/classifier lists and parameters, then rebuild descriptors and classifier targets in a separate directory:

```bash
automl-rank prepare --config configs/my-phase1.json --output outputs/my-phase1
automl-rank phase1 --config configs/my-phase1.json --input outputs/my-phase1 --workers 1
automl-rank evaluate --input outputs/my-phase1 --output outputs/my-phase2
```

Existing names are implemented by `make_scaler`, `make_selector` and `make_classifier` in `automl_ranker/phase1.py`. Add support there before using a new name in the JSON. The pipeline order is the Cartesian-product order stored in `pipeline_keys.json`. Matrix columns must retain that exact order. The execution and aggregation support a different number of candidates; the synthetic workflow tests four candidates rather than 60.

The two k parameters serve different purposes: `classifier_knn_k` controls the k-NN classifier inside a candidate pipeline; `k_values` controls the meta-level recommender.

## Change descriptors or preprocessing

`automl_ranker/representation.py` defines the statistical descriptor. `describe_labeled_dataset` encodes the labeled query and produces its vector. It does not switch to TensorFlow automatically. Changing it requires recomputing descriptors for both the training reference and new queries; old and new descriptor spaces cannot be mixed.

`automl_ranker/pipelines.py` defines numerical imputation and categorical encoding for classifier folds. Scalers/selectors are applied by `phase1.py` after these transformations. Classifier preprocessing is fitted on each training fold, independently of the complete-task representation. Modifying it requires rebuilding Y, not merely retraining the metamodel.

The loader currently expects 64 descriptor positions. If you change dimension, update input validation and tests as well as recomputing X. Keep descriptor semantics documented; zero padding does not represent additional learned features.

## Use other OpenML tasks

Provide a CSV with `task_id`, `dataset_id`, `target_name`, and preferably `grupo` and `task_type_id`. If `grupo` is absent, the loader starts from dataset IDs; it also merges repeated raw-feature fingerprints. Declared family labels are preferable when available, because names, IDs and geometry do not guarantee detection of every related dataset version.

```bash
automl-rank prepare --manifest examples/my-task-manifest.csv --output outputs/new-tasks
```

For a local labeled CSV query, use `recommend`; it does not need an OpenML task ID. An example API call:

```python
import pandas as pd
from automl_ranker.io import REFERENCE, load_config
from automl_ranker.recommend import recommend

table = pd.read_csv("examples/labeled_query.csv")
ranking = recommend(table.drop(columns="target"), table["target"], REFERENCE,
                    load_config(), model="knn", k=10, top=5)
print(ranking)
```

## Add figures

`automl_ranker/plotting.py` reads numerical evaluation outputs. It saves PNGs and the numeric tables behind ordered maps. Heatmap rows/columns are sorted by their own means; the square similarity matrix applies one identical task permutation to both axes. No LaTeX installation is required.

## Before contributing

Run the unit tests and the synthetic workflow. For changes intended to preserve the reference algorithm, also run `automl-rank evaluate --verify-reference`. Keep raw datasets, local models, generated outputs and manuscript documents outside version control. Tests explicitly check that modifying an outer test fold's targets cannot change that fold's predictions or selected k.
