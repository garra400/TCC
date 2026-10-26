# Resultados da revisão

193 tasks; 179 nomes; 165 grupos.

Médias por grupo; menor regret é melhor. Intervalos e contagens em resumo.csv.

metrica                   kendall_ponderado    ndcg_5  regret_1  spearman
modelo
aleatorio                          0.002431  0.868675  0.084882 -0.012884
knn_ajustado                       0.399525  0.935473  0.037206  0.463712
knn_k10_fixo                       0.399525  0.935473  0.037206  0.463712
knn_padronizado_ajustado           0.413765  0.934307  0.033911  0.473551
media_global                       0.368991  0.939595  0.033407  0.400039
random_forest                      0.430918  0.941430  0.030465  0.482178

Casos k-NN ajustado com tau ponderado <= 0: 17; indefinidos: 1.

Correlação diagnóstica não demonstra causalidade. CIs por bootstrap dos grupos são descritivos e não incorporam toda a incerteza de retreinamento.

O teste é retrospectivo: todos os datasets já haviam sido observados no desenvolvimento anterior. A Fase 1 histórica não foi refeita.
