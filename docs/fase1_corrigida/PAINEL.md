# Terceira branch — reconstrução integral da Fase 1

Branch: `revisao/fase1-sem-vazamento-2026`, derivada da revisão anterior.
As alterações da segunda branch estão incluídas; a `main` e a segunda branch
preservam seus próprios estados e resultados.

## O que foi concluído

Foram registradas **57900 avaliações completas** de classificadores:
193 tarefas × 60 pipelines × cinco folds. Imputação, codificação,
escalonamento e seleção usam somente o treino de cada fold. Todos os
classificadores, inclusive SVM, recebem as mesmas linhas de treino e teste.
Houve **0 falhas**, **0 escores agregados ausentes** e
**7703 registros de aviso de convergência**, preservados nas evidências.

A execução usa quatro processos inicialmente e seis após retomada dos checkpoints, com uma thread por processo.
A retomada reaproveita apenas checkpoints completos e compatíveis por hash.
Detalhes em `execucao_operacional.json`. O orçamento de MLP continua fixo em 200
iterações; avisos de convergência não foram ocultados.

## Resultados principais

| Metamodelo | Kendall ponderado histórico nos mesmos folds | Kendall ponderado após reconstrução |
| --- | ---: | ---: |
| k-NN ajustado | 0,3995 | 0,4280 |
| Random Forest | 0,4309 | 0,4768 |
| Média global | 0,3690 | 0,4011 |

A diferença Random Forest menos k-NN na avaliação principal é
**0,0515**, com intervalo descritivo de 95% **[0,0255; 0,0771]**.
O intervalo descritivo sugere vantagem do Random Forest sobre o k-NN neste acervo. Há 177 tarefas em 149 grupos com
correlação definida para o k-NN. O diagnóstico registra 13 casos
não positivos e 16 indefinidos.
Os estados distinguem 14 tarefas com alvo constante e
2 com predição constante e alvo variável; as médias de
correlação usam somente os casos definidos, sem convertê-los em acerto ou zero.

A referência acima usa as mesmas tarefas, grupos e folds. O valor antigo do
LOO por tarefa não é uma comparação direta. Mesmo nesta comparação controlada,
reconstruir escores altera os rankings reais e pode recuperar itens antes
ausentes; isso não isola ganho preditivo sob uma única verdade de referência.
Uma condição adicional mantém os vetores históricos ao trocar somente os escores.
Os CSVs preservam as três condições, seus suportes e diferenças pareadas.

Na Fase 1, a variação média de acurácia balanceada dos classificadores nos pares
tarefa/pipeline com escore disponível nas duas versões foi, em pontos percentuais:
árvore de decisão: $-0,84$ p.p.; MLP: $-1,04$ p.p.; Random Forest: $-1,67$ p.p.; SVM: $-1,02$ p.p.; k-NN: $-1,65$ p.p.. É uma média descritiva por par, sem peso igual entre
grupos; ela difere das métricas de ranqueamento dos metamodelos.

Considerando somente os casos avaliáveis nas duas versões, a mudança de Kendall
ponderado foi **-0,0093** para k-NN (intervalo descritivo
**[-0,0282; 0,0092]**, 149 grupos) e
**0,0059** para Random Forest (intervalo
**[-0,0085; 0,0202]**, 151 grupos).
O intervalo da mudança do k-NN ajustado abrange zero. O intervalo da mudança do Random Forest abrange zero. O aumento das médias gerais, com suportes distintos,
não estabelece melhoria geral da predição.

## Onde conferir e editar

| Informação | Arquivo |
| --- | --- |
| Protocolo e parâmetros da reconstrução | [PROTOCOLO.md](PROTOCOLO.md) |
| Validação final dos resultados e documentos | [VALIDACAO.md](VALIDACAO.md) |
| Solicitações, ações, evidências e pendências | [modificacoes.csv](modificacoes.csv) |
| Campos usados no texto | [CAMPOS_TCC.json](CAMPOS_TCC.json) |
| Catálogo, origem bruta, grupos e exclusões | [datasets.csv](../../projeto/resultados/fase1_corrigida_2026/datasets.csv) |
| Auditoria dos tipos e possíveis identificadores | [qualidade_atributos.csv](../../projeto/resultados/fase1_corrigida_2026/qualidade_atributos.csv) |
| Parâmetros efetivos dos classificadores | [parametros_classificadores.json](../../projeto/resultados/fase1_corrigida_2026/parametros_classificadores.json) |
| Manifesto da Fase 1 | [manifesto.json](../../projeto/resultados/fase1_corrigida_2026/manifesto.json) |
| Auditoria da cobertura e igualdade de amostras | [validacao.json](../../projeto/resultados/fase1_corrigida_2026/validacao.json) |
| Matrizes e médias por tarefa/pipeline | [pipelines_raw.csv](../../projeto/resultados/fase1_corrigida_2026/pipelines_raw.csv) |
| Comparação de três condições | [resumo_condicoes.csv](../../projeto/resultados/fase1_corrigida_2026/fase2/resumo_condicoes.csv) |
| Diferenças pareadas de protocolos | [comparacao_condicoes.csv](../../projeto/resultados/fase1_corrigida_2026/fase2/comparacao_condicoes.csv) |
| Desempenho principal por task/modelo | [metricas_por_task.csv](../../projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa/metricas_por_task.csv) |
| Diagnóstico de erros | [diagnostico_erros.csv](../../projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa/diagnostico_erros.csv) |
| Escores que mudaram desde o histórico | [comparacao_escores.csv](../../projeto/resultados/fase1_corrigida_2026/comparacao_escores.csv) |
| Mudança de escores por classificador | [comparacao_escores_classificador.csv](../../projeto/resultados/fase1_corrigida_2026/comparacao_escores_classificador.csv) |
| Registro dos 14 testes automatizados | [testes_unitarios.txt](../../projeto/resultados/fase1_corrigida_2026/testes_unitarios.txt) |
| TCC revisado | [tcc_revisado.pdf](../../documentos/tcc_revisado.pdf) |
| Artigo revisado | [artigo_revisado.pdf](../../documentos/artigo_revisado.pdf) |

Os diretórios `folds/` e `splits/` preservam ajustes e índices por tarefa.
As fontes LaTeX foram atualizadas em resumo, abstract, introdução, métodos,
resultados, conclusão e artigo. O gerador é `relatorio_fase1_corrigida.py`;
alterações manuais nesses trechos podem ser sobrescritas ao regenerar o relatório.

## Limitações que permanecem

A reconstrução da Fase 1 **foi concluída** e deixa de ser uma pendência.
Permanecem a avaliação retrospectiva, os grupos conservadores, a codificação
ordinal nominal e os rótulos necessários na consulta. A auditoria identifica
30 tarefas apenas categóricas e 18 com
nomes sugestivos de identificadores/caminhos, sem exclusão posterior orientada
pelos resultados. Esses nomes são indicadores de escopo, não prova de vazamento.

Expandir todo o arsenal de hiperparâmetros, testar outras representações e obter
avaliação externa posterior continuam como extensões. O piloto anterior de 180
avaliações permanece separado. Vínculo formal de orientação e data de defesa
dependem de confirmação institucional; os PDFs são versões de trabalho.

## Reprodução

```powershell
python -m pip install -r projeto/requirements-fase1-lock.txt
python -m projeto.revisao.fase1_corrigida --prepare
python -m projeto.revisao.fase1_corrigida --workers 6
python -m projeto.revisao.validar_fase1_corrigida
python -m projeto.revisao.auditar_atributos_fase1
python -m projeto.revisao.avaliar_fase1_corrigida
python -m projeto.revisao.figuras --input projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa
python -m projeto.revisao.relatorio_fase1_corrigida
python scripts/compilar_documentos.py
```

Os dados brutos ficam no cache local OpenML e em diretório ignorado pelo Git.
Para retomar, execute a Fase 1 sem `--prepare`; hashes incompatíveis impedem
misturar resultados de configurações diferentes. A reexecução completa é custosa.
