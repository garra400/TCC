"""Exporta parâmetros efetivos atuais, sem presumir versões históricas iguais."""
import importlib.metadata
import json
import sys
from pathlib import Path

from projeto.revisao.experimentos import PROJECT

sys.path.insert(0, str(PROJECT))
from fase1_meta_conjunto.pipeline_builder import construir_pipeline, gerar_todas_combinacoes


if __name__ == "__main__":
    rows = []
    for index, spec in enumerate(gerar_todas_combinacoes()):
        pipeline = construir_pipeline(spec["scaler"], spec["selector"], spec["classifier"], n_features=20)
        rows.append({"pipeline_id": index, **spec,
                     "params": {name: estimator.get_params(deep=False) for name, estimator in pipeline.steps}})
    result = {"sklearn": importlib.metadata.version("scikit-learn"),
              "n_features_exemplo": 20,
              "nota": "Parâmetros efetivos atuais; seletores dependem de P; não reconstrói versões históricas desconhecidas.",
              "pipelines": rows}
    (PROJECT/"resultados/revisao_2026/pipelines_parametros.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
