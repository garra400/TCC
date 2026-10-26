# Resultados da revisão

201 tasks; 186 nomes; 172 grupos.

Médias por grupo; menor regret é melhor. Intervalos e contagens em resumo.csv.

metrica                   kendall_ponderado    ndcg_5  regret_1  spearman
modelo
aleatorio                         -0.005100  0.860418  0.080980 -0.022245
knn_ajustado                       0.399285  0.934330  0.034542  0.463210
knn_k10_fixo                       0.401230  0.933231  0.033127  0.467304
knn_padronizado_ajustado           0.412134  0.933163  0.030543  0.476577
media_global                       0.376200  0.940807  0.031575  0.406749
random_forest                      0.440586  0.946246  0.026758  0.497236

Casos k-NN ajustado com tau ponderado <= 0: 19; indefinidos: 3.

Correlação diagnóstica não demonstra causalidade. CIs por bootstrap dos grupos são descritivos e não incorporam toda a incerteza de retreinamento.

O teste é retrospectivo: todos os datasets já haviam sido observados no desenvolvimento anterior. A Fase 1 histórica não foi refeita.
