# Reproducing the experiments

## 1. Verify the numerical reference

Run `automl-rank info`. Expected shapes are X = 193 × 64 and Y = 193 × 60, with 165 groups. `data/reference/manifest.json` records SHA-256 digests; the loader validates the numerical inputs before evaluation. No serialized trained model is required.

The descriptor file preserves its original precision. Meta-evaluation casts both descriptors and performance scores to float32, matching the reference treatment of ties. The geometric proximity audit uses the original descriptor precision.

## 2. Recompute grouped evaluation

```bash
automl-rank evaluate --verify-reference
```

The defaults use five shuffled outer group partitions and four shuffled inner group partitions, seed 42. Inner selection considers k in {1, 3, 5, 7, 10}, using mean weighted Kendall after averaging within each group. Ties select the smallest k. Standardization is fitted inside the applicable training partition.

The standardized k-NN selects 10, 7, 10, 10, 10; unscaled k-NN selects 5, 10, 10, 10, 10. A fixed-k=10 unscaled variant is retained for protocol controls. Random Forest uses 200 trees, at least two observations per leaf and a fixed seed. Its configuration is distinct from the 100-tree classifier used inside the candidate pipelines.

Output files:

| File | Contents |
| --- | --- |
| `metricas_por_task.csv` | Seven metrics and evaluation status for every task/method |
| `predicoes.csv` | Out-of-fold predicted and observed candidate scores |
| `resumo.csv` | Task summaries, group means, support counts and descriptive intervals |
| `selecao_k_interna.csv` | Candidate k values, inner scores and selections in each outer partition |
| `comparacoes_pareadas.csv` | RF/global mean, both k-NN variants/global mean and RF/standardized k-NN |
| `splits.json` | Explicit train/test task and group assignments |
| `run.json` | Input checksums, configuration and actual execution environment |

Metric columns retain their established names: `spearman`, `kendalltau` (tau-b), `kendall_ponderado` (weighted Kendall), `ndcg` (all candidates), `ndcg_5`, `regret_1` and `regret_5`. Regret is in balanced-accuracy units; multiply by 100 for percentage points. Lower regret is better. Undefined correlations remain missing, rather than being replaced with zero.

The verification flag checks 42 method/metric means, including the fixed-k control, and all internally selected k values against the bundled numerical reference. The tolerance is 1e-10 for group means; versions or tie behavior that differ can fail the check. It is not a guarantee that arbitrary future library versions reproduce identical scores.

## 3. Additional analyses

```bash
automl-rank analyse --kind all
```

Use `--input`, `--evaluation`, `--output` and `--config` to point to another completed experiment. The analysis checks that the evaluation input hashes and configuration match.

- **LOO versus groups:** same vectors and score matrix, k=10 fixed, with/without standardization; task-level LOO, leave-one-group-out and five outer group partitions.
- **Metadata sensitivity:** remove categorical-only tasks, tasks with suggestive identifier/path column names, and their union, from both train and test. Preserve the original outer assignments and rerun inner selection. The reference lists have 30 and 18 members, overlap in 16, and therefore remove 32 unique tasks together.
- **Proximity:** inspect cross-group cosine similarity ≥ 0.999. Merge these pairs with the original groups transitively and rerun evaluation as a deliberately conservative stress test. This produces 113 expanded groups on the reference inputs. Similarity does not confirm a family relationship; changing groups also changes weighting and partitions.
- **Cost:** sum recorded fit/predict/metric times of the first candidate and first five recommended candidates, compared with all candidates. This excludes shared transformations, descriptor extraction, recommender overhead and parallel scheduling. The bundled reference contains aggregated times; if original per-fold logs are present, their sums are also checked. Do not report this as measured full-system wall-clock speedup.

Metadata rules were established retrospectively before the additional sensitivity results, after the primary archive had already been explored. Repeating them here is replication, not a new independent preregistration. Plan files reject reuse with different rules in the same output directory.

## 4. Rebuild classifier scores

```bash
automl-rank prepare --output outputs/rebuilt
automl-rank phase1 --input outputs/rebuilt --workers 1
automl-rank evaluate --input outputs/rebuilt --output outputs/rebuilt-evaluation
automl-rank analyse --input outputs/rebuilt --evaluation outputs/rebuilt-evaluation --output outputs/rebuilt-analyses
```

Use the pinned environment and an internet connection for OpenML. Preparation starts from the 201 preserved task IDs, not from the earlier 243-candidate selection whose individual exclusion reasons were not fully retained. Eligibility requires 500–10,000 rows, at least two classes and sufficient class support for five stratified folds. The reference has eight exclusions and 193 eligible tasks. Task IDs and dataset IDs are different identifiers.

The default arsenal is four scaling choices × three selection choices × five classifiers. Every classifier receives the same rows in each task fold. Transformation statistics and selection use training rows only. Shared transformations are reused only within the same fold. SVM has no exclusive subsample. An aggregate target is missing if any of its required folds fails. Convergence warnings are recorded.

Descriptor extraction sees the complete labeled task, as allowed in the recommendation scenario, but not classifier predictions or test performance. This is separate from the classifier transformations inside cross-validation. Raw data downloads and caches are not redistributed by this repository. Local cache serialization fingerprints can differ across systems; logical table fingerprints and recorded versions identify the inputs.

Changing source code or configuration invalidates classifier checkpoints. Use a new output directory, or repeat preparation/execution with an intentionally separate cache. Evaluation cannot repair missing or failed classifier scores automatically: inspect the preparation and fit logs first.

If an institutional proxy produces a certificate error, configure `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` with the trusted CA bundle provided by the institution. The first applies to Requests and the second is honored by the MinIO client used for OpenML's parquet downloads. TLS verification remains enabled. See [Requests certificate configuration](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification). Offline commands do not need network certificate configuration.

## 5. Dataset2Vec exploratory comparison

Follow the clone and checkout commands in the main README, then run `automl-rank dataset2vec --source ../dataset2vec`. The five published partitions each have 80 training, 16 validation and 24 test datasets. The predictor uses only the 80 training datasets; tests cover all 120 once. Candidate identity/order is checked across all 256 configurations.

The published artifacts contain Dataset2Vec vectors of 32 dimensions, MF1 vectors of 41 and MF2 vectors of 22. They are compared with the deterministic 64-position descriptor under the same fixed-k=10 predictor and train-only standardization. No neural training is performed. Previously published normalization and incomplete auditing of pretraining/families limit interpretation. The published score is not relabeled as the balanced accuracy of the OpenML experiment.
