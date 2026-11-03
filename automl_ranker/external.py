"""Exploratory comparison using the Dataset2Vec authors' published artifacts."""

import hashlib
import json
import warnings
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .evaluation import knn_predict, ranking_metrics, METRICS, bootstrap_mean

SOURCE = OUT = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "natureza": "Sondagem externa exploratória; não é reprodução do algoritmo de HPO do artigo nem comparação com os 60 pipelines OpenML.",
        "fonte": "https://github.com/hadijomaa/dataset2vec",
        "commit": "42058daf5077f3adb2f40af89a26f54f54ee1cdb",
        "espaco": "a",
        "particoes": "cinco partições publicadas de 80 treino, 16 validação e 24 teste; usar somente os 80 para treino do preditor",
        "preditor": "k-NN k=10 fixo, pesos inverso da distância, StandardScaler ajustado só no treino de cada divisão para todas as representações",
        "representacoes": ["estatistico_64", "dataset2vec_32", "mf1_41", "mf2_22"],
        "alvos": "Última coluna publicada (desempenho em escala 0 a 1), sem normalização pelo máximo/mínimo da tarefa; conferir igualdade das 256 configurações entre tarefas",
        "estatistico": "Mesmo extrator determinístico do baseline estatístico, aplicado aos preditores e rótulos publicados completos; sem escores de configuração na extração",
        "limitacoes": "Representações pré-calculadas; sem retreino neural. As representações publicadas já foram normalizadas globalmente pelos autores; padronização no treino é invariante a reescala afim por coluna, mas a proveniência completa do pré-treino não foi reexecutada. Não há auditoria de famílias equivalente aos 165 grupos OpenML. Sem alegação de validação externa independente após desenvolvimento.",
    }
    plan_path = OUT / "plano.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding="utf8")) == plan
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8"
        )
    splits = pd.read_csv(
        SOURCE / "metadataset/searchspace-a/searchspace-a-splits.csv", index_col=0
    )
    names = sorted(set(splits.stack().dropna().to_list()))
    assert len(names) == 120
    sources = [SOURCE / "metadataset/searchspace-a/searchspace-a-splits.csv"]
    embeddings = {}
    for s in range(5):
        file = SOURCE / f"metafeatures/meta-features-split-{s}.csv"
        embeddings[s] = pd.read_csv(file, index_col=0).loc[names]
        assert embeddings[s].shape == (120, 32)
        sources.append(file)
    mf1 = pd.read_csv(SOURCE / "metafeatures/mf1.csv", index_col=0).loc[names]
    mf2 = pd.read_csv(SOURCE / "metafeatures/mf2.csv", index_col=0).loc[names]
    sources += [SOURCE / "metafeatures/mf1.csv", SOURCE / "metafeatures/mf2.csv"]
    extractor = Path(__file__).with_name("representation.py")
    from .representation import _embedding_estatistico

    stats, y, catalog = [], [], []
    configurations = None
    for i, name in enumerate(names):
        folder = SOURCE / "datasets" / name
        xp, yp = folder / f"{name}_py.dat", folder / "labels_py.dat"
        x = pd.read_csv(xp, header=None)
        target = pd.read_csv(yp, header=None).iloc[:, 0]
        assert len(x) == len(target) and not x.isna().any().any()
        assert all(np.issubdtype(t, np.number) for t in x.dtypes)
        target = pd.Series(pd.factorize(target, sort=True)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = _embedding_estatistico(x, target)
        assert z.shape == (64,) and np.isfinite(z).all()
        stats.append(z)
        meta = SOURCE / "metadataset/searchspace-a" / f"{name}.txt"
        table = pd.read_csv(meta, header=None).to_numpy()
        assert table.shape == (256, 11) and np.isfinite(table).all()
        if configurations is None:
            configurations = table[:, :-1].copy()
        else:
            assert np.array_equal(configurations, table[:, :-1]), name
        assert table[:, -1].min() >= 0 and table[:, -1].max() <= 1
        y.append(table[:, -1])
        sources += [xp, yp, meta]
        catalog.append(
            dict(
                dataset=name,
                n_instancias=len(x),
                n_atributos=x.shape[1],
                n_classes=target.nunique(),
            )
        )
        if (i + 1) % 20 == 0:
            print(
                f"Dataset2Vec: {i + 1}/120 descritores e alvos conferidos", flush=True
            )
    stats = pd.DataFrame(np.stack(stats), index=names)
    y = pd.DataFrame(np.stack(y), index=names)
    stats.to_csv(OUT / "vetores_estatisticos.csv")
    pd.DataFrame(catalog).to_csv(OUT / "catalogo.csv", index=False)
    records = []
    for fold in range(5):
        train = splits[f"train-{fold}"].dropna().to_list()
        test = splits[f"test-{fold}"].dropna().to_list()
        valid = splits[f"valid-{fold}"].dropna().to_list()
        assert len(train) == 80 and len(test) == 24 and len(valid) == 16
        assert not set(train) & (set(test) | set(valid)) and not set(test) & set(valid)
        representations = {
            "estatistico_64": stats,
            "dataset2vec_32": embeddings[fold],
            "mf1_41": mf1,
            "mf2_22": mf2,
        }
        for method, rep in representations.items():
            scaler = StandardScaler().fit(rep.loc[train])
            pred = knn_predict(
                scaler.transform(rep.loc[train]),
                y.loc[train].to_numpy(),
                scaler.transform(rep.loc[test]),
                10,
            )
            for name, p in zip(test, pred):
                records.append(
                    {
                        "fold": fold,
                        "dataset": name,
                        "metodo": method,
                        **ranking_metrics(y.loc[name].to_numpy(), p),
                    }
                )
        for name in test:
            records.append(
                {
                    "fold": fold,
                    "dataset": name,
                    "metodo": "media_global",
                    **ranking_metrics(
                        y.loc[name].to_numpy(), y.loc[train].mean().to_numpy()
                    ),
                }
            )
    data = pd.DataFrame(records)
    data.to_csv(OUT / "metricas.csv", index=False)
    assert data.groupby("metodo").dataset.nunique().eq(120).all()
    summary = data.groupby("metodo")[METRICS].mean()
    summary.to_csv(OUT / "resumo.csv")
    pairs = []
    for metric in ["kendall_ponderado", "ndcg_5", "regret_1"]:
        wide = data.pivot(index="dataset", columns="metodo", values=metric)
        delta = (wide.dataset2vec_32 - wide.estatistico_64).dropna()
        lo, hi = bootstrap_mean(delta, 2000, 42)
        pairs.append(
            dict(
                primeiro="dataset2vec_32",
                segundo="estatistico_64",
                metrica=metric,
                diferenca=float(delta.mean()),
                inferior=lo,
                superior=hi,
                n=len(delta),
            )
        )
    pd.DataFrame(pairs).to_csv(OUT / "pares.csv", index=False)
    manifest = {
        **plan,
        "n_datasets": 120,
        "n_configuracoes": 256,
        "dimensoes_conferidas": {
            "dataset2vec": 32,
            "mf1": 41,
            "mf2": 22,
            "estatistico": 64,
        },
        "fontes_sha256": {str(p.relative_to(SOURCE)): sha(p) for p in sources},
        "extrator_sha256": sha(extractor),
        "script_sha256": sha(Path(__file__)),
        "algoritmos_neurais_retreinados": 0,
    }
    (OUT / "manifesto.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(summary.round(6).to_string(), flush=True)


def compare(source, output):
    global SOURCE, OUT
    SOURCE, OUT = Path(source), Path(output)
    if (SOURCE / ".git").exists():
        commit = subprocess.run(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if commit != "42058daf5077f3adb2f40af89a26f54f54ee1cdb":
            raise ValueError(
                "Check out the documented Dataset2Vec commit before reproducing the comparison."
            )
    _compare()
