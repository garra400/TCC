"""Aguarda a Fase 1 e encadeia validação, Fase 2, relatório e compilação.

Não publica resultados nem substitui números enquanto a reconstrução estiver
incompleta. Usar na raiz: python -u scripts/concluir_reconstrucao.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"projeto/resultados/fase1_corrigida_2026"


def main():
    last = None
    while True:
        progress_file = OUTPUT/"progresso.json"
        if progress_file.exists():
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            completed = progress.get("complete", 0)
            if completed != last:
                print(f"Fase 1: {completed}/{progress.get('total')} tarefas completas", flush=True)
                last = completed
            if progress.get("status") == "complete" and (OUTPUT/"manifesto.json").exists():
                break
        time.sleep(5)
    commands = [
        ("atributos", ["-m", "projeto.revisao.auditar_atributos_fase1"]),
        ("validacao", ["-m", "projeto.revisao.validar_fase1_corrigida"]),
        ("fase2", ["-m", "projeto.revisao.avaliar_fase1_corrigida"]),
        ("figuras", ["-m", "projeto.revisao.figuras", "--input", "projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa"]),
        ("texto", ["-m", "projeto.revisao.relatorio_fase1_corrigida"]),
        ("pdfs", ["scripts/compilar_documentos.py"]),
    ]
    logs = ROOT/"build/reconstrucao"
    logs.mkdir(parents=True, exist_ok=True)
    for name, arguments in commands:
        print(f"Iniciando {name}", flush=True)
        with (logs/f"{name}.txt").open("wb") as stream:
            result = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"Falha em {name}; consulte {logs/name}.txt", flush=True)
            raise SystemExit(result.returncode)
        print(f"Concluído: {name}", flush=True)
    print("Reconstrução, avaliação e PDFs concluídos.", flush=True)


if __name__ == "__main__":
    main()
