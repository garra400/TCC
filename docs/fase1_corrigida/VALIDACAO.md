# Validação da reconstrução integral — 26/10/2026

Branch: `revisao/fase1-sem-vazamento-2026`. A alteração completa a Fase 1 antes
pendente e mantém as revisões da segunda branch. O [painel](PAINEL.md) reúne os
arquivos; [modificacoes.csv](modificacoes.csv) relaciona solicitações e evidências.

## Execução e auditoria dos dados

- 193 tarefas elegíveis, 165 grupos, 60 pipelines e cinco folds por tarefa.
- **57.900 ajustes**, **zero falhas** e **zero escores agregados ausentes**.
- Mesmas linhas de treino e teste para todos os classificadores, incluindo SVM.
- Cada linha foi testada exatamente uma vez; não há interseção treino/teste.
- As partições das 193 tarefas correspondem à estratificação de semente 42 regenerada
  a partir dos caches locais. Seus hashes foram conferidos.
- Os 300 registros de cada tarefa cobrem exatamente os 60 pipelines e os cinco
  folds. A matriz final corresponde às médias dos cinco escores por pipeline.
- Hashes de configuração, código e artefatos conferidos. A preparação e os
  checkpoints usam o mesmo código e configuração da execução concluída.
- **7.703 avisos de convergência**, preservados como limitação do orçamento
  fixo de treinamento. Não equivalem a falhas de execução nem a convergência
  garantida; os resultados caracterizam os modelos com esse orçamento.

Evidências: [manifesto.json](../../projeto/resultados/fase1_corrigida_2026/manifesto.json),
[validacao.json](../../projeto/resultados/fase1_corrigida_2026/validacao.json),
[avisos_convergencia.csv](../../projeto/resultados/fase1_corrigida_2026/avisos_convergencia.csv)
e [proveniencia_codigo.json](../../projeto/resultados/fase1_corrigida_2026/proveniencia_codigo.json).

Os dados brutos permanecem locais. Matrizes, escores por fold, índices de partição,
predições, parâmetros e diagnósticos são disponibilizados no repositório.

## Verificação da avaliação dos metamodelos

As três condições usam as mesmas tarefas, grupos, folds externos, precisão de
entrada e configuração de metamodelos. A referência histórica reproduziu
**exatamente as 42 médias por grupo** da revisão anterior, sem diferença numérica;
os índices dos folds também são idênticos aos da revisão anterior.

| Comparação em Kendall ponderado | Diferença pareada por grupo | Intervalo descritivo de 95% | Grupos pareados |
| --- | ---: | --- | ---: |
| k-NN reconstruído − k-NN histórico | −0,0093 | [−0,0282; 0,0092] | 149 |
| Random Forest reconstruído − histórico | 0,0059 | [−0,0085; 0,0202] | 151 |
| Random Forest − k-NN, ambos reconstruídos | 0,0515 | [0,0255; 0,0771] | 149 |

Os intervalos das mudanças de protocolo abrangem zero. As médias gerais sobem
de 0,3995 para 0,4280 no k-NN e de 0,4309 para 0,4768 no Random Forest, mas seus
suportes mudam. O k-NN tem 177 tarefas/149 grupos com correlação definida na
reconstrução, enquanto Random Forest tem 179/151; a referência histórica tinha
192/164. Por isso, essas médias não estabelecem melhoria geral da predição.
Além disso, reconstruir os escores muda os próprios rankings reais de referência.

Na avaliação reconstruída, o intervalo pareado sugere vantagem do Random Forest
sobre k-NN neste acervo. A comparação de k-NN com média global permanece incerta
pelo intervalo adotado. Não se infere superioridade universal.

O diagnóstico do k-NN registra 13 correlações não positivas e 16 indefinidas:
14 tarefas têm escores reais constantes e duas têm predição constante com escores
reais variáveis. NDCG e regret continuam disponíveis nas 193 tarefas.

Na Fase 1, as médias descritivas de acurácia balanceada dos classificadores
caíram entre 0,84 e 1,67 ponto percentual nos pares tarefa/pipeline comparáveis.
Essas médias por par não têm a ponderação por grupos das métricas da Fase 2.

## Testes automatizados

**14 testes passaram**, executados com:

```powershell
python -m unittest projeto.revisao.test_experimentos projeto.revisao.test_fase1_corrigida -v
```

Cobrem isolamento dos alvos do teste externo, seleção interna de k, imputação
no treino, categorias desconhecidas, separação de grupos, suporte de classes,
amostras completas compartilhadas entre classificadores, métricas, resultados
ausentes e comparação pareada com peso igual por grupo.

[Registro da execução](../../projeto/resultados/fase1_corrigida_2026/testes_unitarios.txt).
A auditoria dos 57.900 ajustes complementa esses testes com os dados reais.

## Documentos

- [TCC revisado](../../documentos/tcc_revisado.pdf): **42 páginas**.
- [Artigo revisado](../../documentos/artigo_revisado.pdf): **6 páginas**.
- Compilação concluída sem citações/referências indefinidas ou rótulos duplicados.
- Conferência visual de capa, resumos, comparação controlada, tabelas de
  metamodelos, figuras e diagnóstico: conteúdo legível e dentro das páginas.
- Resumos, introdução, métodos, resultados e conclusões refletem a reconstrução.
  O texto distingue alteração da estimativa, mudança de suporte e comparação de
  metamodelos, sem transformar aumento de uma média em ganho geral comprovado.

Os avisos residuais de caixa do LaTeX pertencem a elementos do template; não
impediram a compilação. Campos institucionais ainda sujeitos a confirmação estão
registrados em R16. Os PDFs são versões de trabalho para revisão acadêmica.

## Limites da evidência

A avaliação é retrospectiva. Permanecem agrupamento conservador, codificação
ordinal, rótulos necessários na consulta, orçamento fixo e arsenal de 60
configurações. A auditoria registra 30 tarefas somente categóricas e 18 com nomes
sugestivos de identificadores/caminhos; não houve exclusão orientada pelos
resultados. O piloto anterior de 180 avaliações não equivale a ampliar o arsenal
completo. Intervalos de bootstrap são descritivos devido aos treinos sobrepostos.
