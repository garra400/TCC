> Registro histórico da segunda branch. A reconstrução integral e os resultados atuais estão no [painel da terceira branch](../fase1_corrigida/PAINEL.md).

# Revisão do TCC e artigo

Branch: `revisao/avaliacao-kdmile-2026`. A `main` preserva a versão consolidada
anterior às correções. Data da revisão: **26/10/2026**.

**Viável e implementado no escopo documentado.** A comparação de metamodelos,
a auditoria, os diagnósticos, a revisão textual e o piloto foram executados.
A reconstrução de toda a matriz de escores da Fase 1 permanece uma etapa futura.

## Onde encontrar cada informação

| Informação | Arquivo |
| --- | --- |
| Escolha da versão local, hashes e diferenças | [Auditoria](../auditoria/conclusao.md) |
| Parecer de viabilidade e limites | [VIABILIDADE.md](VIABILIDADE.md) |
| Solicitação, ação, evidência, arquivo e pendência por item | [modificacoes.csv](modificacoes.csv) |
| Campos numéricos para atualização do TCC | [CAMPOS_TCC.json](CAMPOS_TCC.json) |
| Explicação de cada posição do vetor | [EMBEDDING.md](EMBEDDING.md) |
| Resumo da execução principal | [RESULTADOS.md](../../projeto/resultados/revisao_2026/RESULTADOS.md) |
| Configuração e versões | [manifesto.json](../../projeto/resultados/revisao_2026/manifesto.json) |
| Datasets, grupos, links, qualidade e exclusões | [datasets.csv](../../projeto/resultados/revisao_2026/datasets.csv) |
| Escores por task/modelo/fold e estados indefinidos | [metricas_por_task.csv](../../projeto/resultados/revisao_2026/metricas_por_task.csv) |
| Todos os valores reais e preditos por pipeline | [predicoes.csv](../../projeto/resultados/revisao_2026/predicoes.csv) |
| Médias, desvios, intervalos e contagens | [resumo.csv](../../projeto/resultados/revisao_2026/resumo.csv) |
| Diferenças pareadas entre modelos | [comparacoes_pareadas.csv](../../projeto/resultados/revisao_2026/comparacoes_pareadas.csv) |
| Casos de falha e características associadas | [diagnostico_erros.csv](../../projeto/resultados/revisao_2026/diagnostico_erros.csv) |
| Piloto de hiperparâmetros, 180 avaliações | [piloto/resumo.csv](../../projeto/resultados/revisao_2026/piloto/resumo.csv) |
| Análise de sensibilidade com 201 tarefas | [sensibilidade_201/RESULTADOS.md](../../projeto/resultados/revisao_2026/sensibilidade_201/RESULTADOS.md) |
| PDF do TCC revisado | [tcc_revisado.pdf](../../documentos/tcc_revisado.pdf) |
| PDF do artigo revisado | [artigo_revisado.pdf](../../documentos/artigo_revisado.pdf) |

## Achados que mudam a interpretação

O acervo histórico tem 201 tarefas, mas 186 nomes de datasets e 172 grupos
conservadores. A task 363713, antes descrita como holdout, pertence ao acervo.
Oito tarefas têm menos de cinco observações em alguma classe; a avaliação
principal fica com **193 tarefas em 165 grupos**, 52 escores ausentes e 192
tarefas com correlações definidas. O catálogo mantém também as oito exclusões.

Kendall ponderado médio **por grupo**: k-NN ajustado **0,3995**, Random Forest
**0,4309**, média global **0,3690**. A diferença RF menos k-NN é 0,0314, com
intervalo descritivo [0,0073; 0,0564]. O intervalo de média global menos k-NN
inclui zero: não se mantém a afirmação anterior de superioridade clara do k-NN
em relação a essa baseline. NDCG@5 e regret são reportados para evitar confundir
NDCG elevado com top-5 exato. As definições foram conferidas nas documentações
de [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html)
e [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).

A comparação com RF mantém a mesma representação e o mesmo arsenal, mas não
constitui uma ablação de representações. Os intervalos são descritivos devido
à sobreposição dos treinos. A divisão por grupos é retrospectiva, não um teste
externo coletado após o desenvolvimento. A análise não explica causalmente os
17 casos de Kendall ponderado não positivo; registra características e associações
exploratórias com correção para comparações múltiplas.

## O que ainda precisa de trabalho

- **Reconstruir a Fase 1 completa** com dados brutos e tratamento dentro dos folds.
  A matriz histórica usa imputação anterior à CV e subamostra SVM globalmente.
  Isso está declarado no artigo e TCC; o piloto novo corrige o tratamento em seis
  tarefas, sem substituir a matriz inteira.
- **Expandir o arsenal no benchmark completo**, se essa extensão for adotada.
  O piloto é executado, mas não mede ranking de um arsenal expandido em 165 grupos.
  No experimento principal, a manutenção de 60 pipelines tem justificativa de
  comparação controlada entre metamodelos.
- **Revisão acadêmica final, vínculo de orientação e calendário**: a ata alterna
  os termos orientador e coorientador. Não foi alterado o cadastro formal nem
  inventada data de defesa. Os PDFs são versões de trabalho, sem submissão a evento.

## Reprodução

Na raiz do repositório, em um ambiente Python compatível:

```powershell
python -m pip install -r projeto/requirements-revisao.txt
python -m unittest projeto.revisao.test_experimentos -v
python -m projeto.revisao.experimentos
python -m projeto.revisao.experimentos --include-ineligible --output projeto/resultados/revisao_2026/sensibilidade_201
python -m projeto.revisao.piloto_hiperparametros
python -m projeto.revisao.figuras
python -m projeto.revisao.relatorio
python scripts/compilar_documentos.py
```

As comparações de metamodelos podem usar as matrizes e o catálogo versionados,
sem os caches de dados. O piloto requer os dados brutos OpenML, obtidos pelo
cliente OpenML e mantidos em seu cache. A auditoria de metadados pode ser renovada
com `python -m projeto.revisao.auditar_openml`; as fontes e hashes ficam no CSV.
O gerador de relatório reescreve o capítulo de resultados: faça alterações
de conteúdo no gerador se quiser preservá-las na próxima geração.

Os PDFs antigos na raiz e em `artigo/main.pdf` permanecem como registros
históricos. Os PDFs atuais estão em `documentos/`.
