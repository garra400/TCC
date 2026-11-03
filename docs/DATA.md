# Data and scientific scope

The compact reference is numerical experiment input, not a copy of the raw datasets or a trained model.

| Input | Role |
| --- | --- |
| `X_emb.npy` | Dataset descriptors, one row per eligible task |
| `Y_accuracy.npy` | Mean five-fold balanced accuracy, one column per candidate |
| `task_ids.npy` | Row identities and order |
| `pipeline_keys.json` | Exact candidate identity and column order |
| `datasets.csv` | Task/dataset IDs, target, eligibility, metadata and groups for the 201 preserved tasks |
| `qualidade_atributos.csv` | Metadata indicators for the 193 eligible tasks |
| `pipelines_raw.csv` | Per-task/candidate aggregate performance and recorded timing |
| `expected_metrics.csv`, `expected_k.csv` | Verification values for the reference implementation |
| `sensitivity_plan.json` | Retrospective exclusion sets and their overlap |
| `manifest.json` | Checksums, dimensions and provenance of the bundle |

The vector has 55 active statistical positions and nine zero positions. Nine per-attribute statistics are summarized over attributes by minimum, 25th percentile, mean, 75th percentile and maximum (45 values); ten global descriptors complete the active part. L2 normalization follows finite-value handling. Labels are required and ordinal encoding of categories/nominal labels limits interpretation. The implementation does not claim to train Dataset2Vec.

Reference input arrays preserve their original serialization. The final scientific runs derive from the corrected classifier protocol and the retrospective analyses completed on 2 November 2026. The provenance records distinguish the earlier source-code revision from the current numerical reproduction. No claim is made that all new analyses had already been present in that earlier commit.

The eligible corpus has 193 tasks and 165 initial groups. It comes from 201 retained tasks; the earlier 243-candidate collection does not retain an auditable reason for every one of its 42 removals. The later eight exclusions concern class support. Task IDs are not counts of independent datasets. Grouping reduces related-task leakage but misses some families; the proximity stress test is a sensitivity analysis, not a confirmed family catalog.

OpenML input URLs are recorded per task and dataset. Fetching raw datasets is an explicit command, and their original licensing/provenance remains applicable. The example CSV is synthetic, generated with `make_classification`, seed 2026; it contains no user's dataset.

The optional external comparison uses the authors' Dataset2Vec repository at the documented commit. Dataset2Vec representations, raw UCI tables and hyperparameter evaluation material are not vendored into this repository. Their pretrained representations and split definitions are reused rather than approximated with random weights. The optional comparison is a different experiment from the OpenML evaluation.

Method and implementation sources:

- [OpenML](https://www.openml.org/)
- [Dataset2Vec paper](https://doi.org/10.1007/s10618-021-00737-9)
- [Dataset2Vec authors' code and data](https://github.com/hadijomaa/dataset2vec)
- [SciPy weighted Kendall definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html)
- [scikit-learn NDCG implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
