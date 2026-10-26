# Reconstrução da Fase 1 — terceira branch

Branch: `revisao/fase1-sem-vazamento-2026`, derivada da revisão `9c618f3`.
As alterações e os resultados da segunda branch permanecem disponíveis.

## Alteração executada

Os dados brutos são carregados com o alvo oficial de cada tarefa OpenML.
A auditoria das 201 tarefas confirmou os mesmos 193 casos elegíveis e 165 grupos
da análise anterior. Não houve falha de carregamento, mudança do alvo da tarefa
em relação ao padrão do dataset ou alteração dos grupos usados na avaliação.

O limite de cinco exemplos por classe é verificado antes do treino. Mantêm-se
500–10.000 exemplos, classificação supervisionada e ao menos duas classes.
As oito tarefas excluídas e os motivos permanecem no catálogo.

Em cada tarefa, os **mesmos cinco folds estratificados** (shuffle, semente 42)
são usados nos 60 pipelines. Cada linha integra o teste exatamente uma vez.
**Todos os classificadores recebem todas as mesmas linhas de treino e teste**:
a subamostra exclusiva da SVM foi removida.

Em cada fold:

1. A mediana numérica é estimada somente no treino, preservando colunas vazias.
2. Categorias recebem preenchimento determinístico e codificação ordinal
   ajustada somente no treino; valores desconhecidos no teste recebem -1.
3. O escalonador é ajustado somente no treino.
4. PCA ou SelectKBest é ajustado somente no treino, incluindo os rótulos de
   treino na ANOVA; o teste recebe apenas `transform`.
5. O classificador é ajustado no treino e avaliado no teste.

Transformações idênticas são reaproveitadas entre classificadores dentro do
mesmo fold, sem compartilhar ajustes entre folds. Há 193 × 60 × 5 = **57.900
ajustes de classificadores**; isso não se confunde com 57.900 testes independentes.

## Arsenal e controle computacional

Mantêm-se quatro escalonadores × três seletores × cinco classificadores.
Os hiperparâmetros explícitos permanecem em `config_fase1.json`: RF com 100
árvores; SVM RBF C=1 e gamma=scale; k-NN com cinco vizinhos; MLP (64,32), até
200 iterações; árvore de decisão com profundidade máxima 10. PCA usa solver
automático com semente 42; o seletor usa até dez componentes/atributos, limitado
ao número disponível. A semente explícita controla a parte aleatória da PCA,
não fixada no código histórico.

Quatro processos trabalham em tarefas diferentes; cada processo usa uma thread
BLAS e um único job nos classificadores. Isso evita multiplicação de threads.
Avisos de convergência não são escondidos; escores continuam válidos para o
orçamento fixo de 200 iterações, com a limitação registrada. Se algum fold
falhar, o escore agregado desse pipeline/tarefa fica ausente, sem promover
média de menos de cinco folds a alvo do metamodelo.

## Representação e comparação

A meta-descrição continua determinística: 45 descritores distribucionais,
dez globais, nove zeros e norma L2. Ela descreve a tarefa **rotulada completa**,
como entrada da recomendação no meta-nível, e não usa escores ou predições dos
folds. Seu pré-processamento para extrair essa descrição é separado dos ajustes
dos classificadores. O cenário pressupõe rótulos na consulta; não cobre dados
sem rótulos. As limitações da codificação nominal também permanecem.

A Fase 2 executa três condições com as mesmas tarefas, grupos e folds externos:

| Condição | Escores de pipelines | Vetores de entrada |
| --- | --- | --- |
| Referência controlada | Históricos | Históricos |
| Efeito da reconstrução de escores | Recalculados | Históricos |
| Avaliação principal | Recalculados | Recalculados dos dados brutos |

Em todas as condições, o ajuste de k é interno ao treino por grupos. São usados
os mesmos metamodelos, baselines, métricas e sementes da revisão anterior.
As diferenças entre condições são pareadas por grupo e não equivalem a uma
prova causal isolada da imputação: a reconstrução também altera a amostra da SVM,
a codificação por fold e o controle aleatório da PCA.

As três condições usam vetores e escores com a mesma precisão float32 das matrizes
históricas na entrada da Fase 2. Isso evita mudar o tratamento de empates apenas
por usar float64 na condição nova. Os escores originais reconstruídos em float64
continuam preservados nos CSVs e NPY da Fase 1.

## Artefatos e integridade

Diretório novo: `projeto/resultados/fase1_corrigida_2026/`.

- `datasets.csv`: dados, alvo, elegibilidade, grupos e hashes da origem.
- `folds/task_ID.csv`: cada ajuste, score, parâmetros, amostra, tempo e aviso.
- `splits/task_ID.json`: índices exatos de treino e teste.
- `progresso.json`: estado da execução e quantidade de tarefas concluídas.
- `pipelines_raw.csv`: média dos cinco folds por tarefa/pipeline.
- `X_emb.npy`, `Y_accuracy.npy`, `task_ids.npy`: novas matrizes.
- `manifesto.json`, `validacao.json`: cobertura e integridade verificáveis.
- `fase2/`: predições, métricas e comparação das três condições.

Checkpoints são validados por hash de código/configuração e conteúdo. Os dados
brutos em cache ficam locais, sem envio ao GitHub. A documentação e os números
finais devem ser gerados somente depois de terminar e validar a execução.

Os tempos por linha de fold medem ajuste e predição do classificador, incluindo
o cálculo das métricas. As transformações compartilhadas não são repetidas
nesses tempos; o tempo por tarefa nos checkpoints inclui o conjunto do trabalho.
O campo `tempo_total_s` do resumo soma os tempos dos classificadores nos folds e
não é diretamente comparável ao tempo histórico de uma CV paralela completa.

O desenho segue as recomendações de separação do pré-processamento e prevenção
de vazamento da [documentação oficial do scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
