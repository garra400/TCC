# Guia em português

Este repositório contém o algoritmo, os experimentos, testes e instruções de reprodução. A `main` é dedicada ao código; os documentos acadêmicos permanecem nas branches anteriores.

## Instalação no Windows

Use Python 3.14.2 para reproduzir o ambiente registrado. Na pasta em que deseja clonar o projeto:

```powershell
git clone --depth 1 --single-branch --branch main https://github.com/garra400/automl-pipeline-ranking.git
cd automl-pipeline-ranking
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e .
.venv\Scripts\python.exe -m automl_ranker info
```

As versões das dependências científicas estão fixadas em `requirements-lock.txt`. Dependências transitivas não estão integralmente travadas. O arquivo `requirements.txt` e o `pyproject.toml` também permitem instalação para desenvolvimento com versões compatíveis.

## Executar sem baixar dados

```powershell
.venv\Scripts\python.exe -m automl_ranker demo
.venv\Scripts\python.exe -m automl_ranker evaluate --verify-reference
.venv\Scripts\python.exe -m automl_ranker analyse --kind all
.venv\Scripts\python.exe -m automl_ranker plot
```

A demonstração usa tarefas sintéticas e quatro candidatos para percorrer a construção dos escores e a avaliação. A avaliação de referência usa as matrizes numéricas fornecidas de 193 tarefas, 165 grupos e 60 candidatos. Ela recalcula os metamodelos e confere 42 médias de métricas e os valores de k selecionados internamente, sem repetir os 57.900 ajustes dos classificadores. Os resultados são gravados em `outputs/` e não são enviados ao Git.

`analyse --kind all` executa comparações LOO/grupos, exclusões por metadados, teste de estresse por proximidade e estimativa de custo. Também é possível escolher `loo`, `sensitivity`, `proximity` ou `cost` separadamente. As análises são retrospectivas e os intervalos são descritivos.

Para repetir a grade ampliada, os controles de peso e as dez sementes externas:

```powershell
.venv\Scripts\python.exe -m automl_ranker robustness
```

A grade inclui 1, 3, 5, 7, 10, 15, 20, 30, 50 e todos os vizinhos disponíveis no treino. A escolha é interna ao treino; as alternativas de ponderação não são selecionadas pelos resultados externos. A igualdade com a média global ocorre no caso de todos os vizinhos com pesos uniformes. A avaliação antiga com teto 10 continua disponível como controle em `evaluate --verify-reference`.

## Recomendar para seu CSV

```powershell
.venv\Scripts\python.exe -m automl_ranker recommend --csv examples/labeled_query.csv --target target --top 5
.venv\Scripts\python.exe -m automl_ranker recommend --csv examples/labeled_query.csv --target target --model random-forest --top 5
```

Troque o caminho pelo seu arquivo e `target` pelo nome da coluna-alvo. O arquivo deve ter cabeçalho, atributos nas demais colunas e um rótulo por linha. O algoritmo precisa de uma consulta rotulada para calcular o vetor estatístico. O resultado é uma recomendação; essa chamada não executa os classificadores sugeridos. O vetor é estatístico e determinístico, com 55 descritores e nove zeros, não uma rede Dataset2Vec.

## Refazer a primeira fase

```powershell
.venv\Scripts\python.exe -m automl_ranker prepare --task-ids 3 11 12 --output outputs/subset
.venv\Scripts\python.exe -m automl_ranker phase1 --input outputs/subset --workers 1
```

Sem `--task-ids`, a preparação tenta carregar as 201 tarefas preservadas e aplica os critérios finais de elegibilidade. A configuração completa executa 60 candidatos em cinco partições para cada uma das 193 tarefas elegíveis. Essa etapa precisa de internet e é muito mais demorada. O cache fica em `data/cache/`. Use `--cache` nas duas chamadas caso queira outra localização.

## Modificar

- Metamodelos, k, divisões e semente: `configs/meta.json` e `automl_ranker/evaluation.py`.
- Arsenal, classificadores e orçamento: `configs/phase1.json` e `automl_ranker/phase1.py`.
- Imputação/codificação: `automl_ranker/pipelines.py`.
- Descritores: `automl_ranker/representation.py`.
- Figuras: `automl_ranker/plotting.py`.

Após mudar candidatos ou descritores, reconstrua as matrizes correspondentes e avalie os metamodelos em outro diretório. Não misture as matrizes de referência com uma configuração alterada. Há exemplos completos em [EXTENDING.md](EXTENDING.md) e [REPRODUCING.md](REPRODUCING.md).

## Conferir os testes

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

As entradas numéricas de referência são verificadas por hash. O material de terceiros necessário à sondagem Dataset2Vec é obtido separadamente, conforme o README principal. O repositório do autor precisa estar público para que leitores sem permissão consigam cloná-lo.
