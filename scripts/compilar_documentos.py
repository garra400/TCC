"""Compila cópias limpas do TCC/artigo, sem reutilizar auxiliares antigos.

Uso: python scripts/compilar_documentos.py
Saída: documentos/tcc_revisado.pdf e documentos/artigo_revisado.pdf.
Requer MiKTeX/TeX Live com pdflatex, bibtex e makeindex disponíveis.
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/"build/latex"
IGNORE = shutil.ignore_patterns("*.aux", "*.bbl", "*.blg", "*.log", "*.fls", "*.fdb_latexmk",
                               "*.toc", "*.out", "*.acn*", "*.acr*", "*.gl*", "*.idx", "*.ilg", "*.ind",
                               "*.ist", "*.nlo", "*.nls", "*.mw", "__pycache__")


def command(args, cwd, log):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(result.stdout)
    log.flush()
    if result.returncode:
        tail = result.stdout.decode("utf-8", errors="replace")[-2200:]
        raise RuntimeError(f"{args[0]} falhou em {cwd}:\n{tail}")


def compile_one(cwd, name):
    with (STAGE/f"compilacao_{name}.txt").open("wb") as log:
        command(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd, log)
        command(["bibtex", "main"], cwd, log)
        if name == "tcc":
            for extension, output in [("acn", "acr"), ("acn2", "acr2"), ("acn3", "acr3"), ("glo", "gls")]:
                source = cwd/f"main.{extension}"
                if source.exists() and source.stat().st_size:
                    command(["makeindex", "-s", "main.ist", "-o", f"main.{output}", source.name], cwd, log)
            if (cwd/"main.nlo").exists():
                command(["makeindex", "-s", "nomencl.ist", "-o", "main.nls", "main.nlo"], cwd, log)
        for _ in range(3):
            command(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd, log)
    final_log = (cwd/"main.log").read_text(encoding="utf-8", errors="replace")
    checks = ["There were undefined references", "There were undefined citations", "multiply defined"]
    if any(text in final_log for text in checks):
        raise RuntimeError(f"Referências/citações pendentes em {name}; consulte {cwd/'main.log'}.")
    output = ROOT/"documentos"
    output.mkdir(exist_ok=True)
    shutil.copy2(cwd/"main.pdf", output/f"{name}_revisado.pdf")
    print(f"PDF compilado: documentos/{name}_revisado.pdf", flush=True)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    for name in ["main.tex", "configuracoes.tex", "variaveis.tex", "utfpr.cls", "abntex2-alf.bst", "main.bib"]:
        shutil.copy2(ROOT/name, STAGE/name)
    for name in ["capitulos", "PreTexto", "PosTexto", "figuras", "uteis"]:
        shutil.copytree(ROOT/name, STAGE/name, dirs_exist_ok=True, ignore=IGNORE)
    shutil.copytree(ROOT/"artigo", STAGE/"artigo", dirs_exist_ok=True, ignore=IGNORE)
    result = Path("projeto/resultados/revisao_2026")
    (STAGE/result).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/result/"tabela_tcc.tex", STAGE/result/"tabela_tcc.tex")
    shutil.copytree(ROOT/result/"figuras", STAGE/result/"figuras", dirs_exist_ok=True)
    compile_one(STAGE, "tcc")
    compile_one(STAGE/"artigo", "artigo")


if __name__ == "__main__":
    main()
