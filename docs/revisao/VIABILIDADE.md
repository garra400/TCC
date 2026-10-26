# Viabilidade da revisão — 26/10/2026

Base: reunião de 24/09/2026 e três pareceres KDMiLe fornecidos pelo autor.

**Parecer: viável com os artefatos disponíveis.** Há código, 201 caches locais,
matriz 201 × 64, matriz de desempenho 201 × 60 e fontes dos dois documentos.
Comparar metamodelos, corrigir métricas e investigar erros dispensa repetir a
construção inteira do benchmark. Uma validação por grupos substitui a demonstração
isolada. Isso testa generalização entre os grupos observados, sem criar uma
validação externa em dados coletados posteriormente.

## Problemas confirmados antes da implementação

- A task 363713 apresentada como holdout **está** em `task_ids.npy`. O mesmo nome
  de dataset aparece em outra task. A demonstração não sustenta independência.
- São 201 tasks, mas 186 nomes de dataset distintos: LOO por task pode manter
  outra tarefa do mesmo dataset no treino. Agrupar por nome é uma precaução
  conservadora; não certifica independência de versões com nomes diferentes.
- Existem 172 resultados ausentes. Sua imputação deve usar somente treino;
  desempenho ausente no teste não pode ser substituído para calcular métricas.
- São 55 descritores informativos e nove zeros, seguidos de normalização L2;
  não há aprendizado da representação estatística. Ela utiliza os rótulos `y`.
- `weightedtau` padrão usa ponderação hiperbólica **aditiva**, não produto dos
  postos. `ndcg_score` usa ganho linear, não `2**rel - 1`. Kendall padrão é tau-b.
- Escolher k e reportar a mesma avaliação exige distinguir exploração de teste;
  a revisão ajusta k dentro dos folds de treino com grupos separados.
- A Fase 1 histórica imputa mediana antes da CV e subamostra todo o dataset para
  SVM. Os resultados existentes não são uma nova execução com pré-processamento
  corrigido. Corrigir o caminho futuro e registrar essa limitação é necessário.

## Execução autorizada

1. Preservar snapshot na main; desenvolver na branch `revisao/avaliacao-kdmile-2026`.
2. Reproduzir LOO histórico; comparar k-NN, Random Forest multi-saída, média global
   e ranking aleatório em folds por grupos; ajustar k exclusivamente no treino.
3. Registrar cada task, fold, grupo, modelo, predição, métrica, erro e exclusão;
   salvar configuração, versões, hashes, tabelas CSV/LaTeX e interpretação.
4. Explicitar 64 dimensões, pipelines, seleção de dados e limitações; atualizar
   literatura e corrigir afirmações sem evidência experimental.
5. Manter arsenal de 60 no experimento principal para isolar o metamodelo.
   Variações de hiperparâmetros serão um piloto separado, sem misturar resultados
   históricos com resultados de um protocolo diferente.

## Limites

O benchmark histórico continua sujeito aos problemas da Fase 1 e à seleção de
tasks disponível. Uma reconstrução completa requer reavaliação em dados brutos,
com codificação e imputação dentro da CV. Os caches próprios contêm dados
transformados; o cache OpenML também disponibiliza dados brutos, utilizados
posteriormente no piloto. Portanto, o impedimento não é ausência dos dados,
e sim a necessidade de executar novamente todos os escores do benchmark.
Nenhuma conclusão de superioridade sobre a literatura decorre de números obtidos
em tarefas e protocolos diferentes. Data de defesa e mudança formal de orientação
dependem do registro institucional; não serão inventadas a partir da ata ambígua.

## Referências técnicas conferidas

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

## Complemento após a auditoria

Oito tarefas não satisfazem cinco exemplos por classe. A análise principal
executada usa 193 tarefas em 165 grupos; 201 tarefas foram mantidas na
sensibilidade. O piloto foi executado em dados brutos, com 180 avaliações.
As limitações remanescentes e o estado por solicitação estão no PAINEL.md.
