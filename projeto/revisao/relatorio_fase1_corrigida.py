"""Atualiza o TCC somente com resultados completos e validados da terceira branch."""
import json
import re
from pathlib import Path

import pandas as pd

from projeto.revisao.experimentos import ROOT
from projeto.revisao.fase1_corrigida import OUTPUT, write_json


def number(value, digits=4):
    return f"{float(value):.{digits}f}".replace(".", "{,}")


def render(text, fields):
    for name, value in fields.items():
        text = text.replace("@@"+name+"@@", str(value))
    missing = re.findall(r"@@[A-Z_0-9]+@@", text)
    if missing:
        raise ValueError(f"Campos sem resultado: {missing}")
    return text.strip()+"\n"


def run():
    manifest = json.loads((OUTPUT/"manifesto.json").read_text(encoding="utf-8"))
    validation = json.loads((OUTPUT/"validacao.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["failed_fits"] != validation["failed_fits"]:
        raise ValueError("Execução incompleta ou sem validação correspondente.")
    primary = OUTPUT/"fase2/reconstrucao_completa"
    summary = pd.read_csv(primary/"resumo.csv")
    cases = pd.read_csv(OUTPUT/"fase2/resumo_condicoes.csv")
    differences = pd.read_csv(OUTPUT/"fase2/comparacao_condicoes.csv")
    paired = pd.read_csv(primary/"comparacoes_pareadas.csv")
    errors = pd.read_csv(primary/"diagnostico_erros.csv")
    correlations = pd.read_csv(primary/"correlacoes_erros.csv")
    catalog = pd.read_csv(OUTPUT/"datasets.csv")
    quality = pd.read_csv(OUTPUT/"qualidade_atributos.csv")
    selected_k = pd.read_csv(primary/"selecao_k_interna.csv")
    labels = {"aleatorio": "Aleatório", "media_global": "Média global", "knn_ajustado": "k-NN ajustado",
              "knn_padronizado_ajustado": "k-NN padronizado", "random_forest": "Random Forest"}
    def score(model, metric, condition="reconstrucao_completa"):
        return float(cases[(cases.condicao == condition) & (cases.modelo == model) & (cases.metrica == metric)].iloc[0].media_grupos)
    fields = {"N_TASKS": manifest["n_tasks"], "N_GROUPS": manifest["n_groups"],
              "N_FITS": manifest["n_fits"], "N_PIPELINE_SCORES": manifest["n_tasks"]*60,
              "N_FAILURES": manifest["failed_fits"], "N_MISSING": manifest["missing_scores"],
              "N_WARNINGS": manifest["convergence_warnings"],
              "N_ERRORS_KNN": int(errors.correlacao_nao_positiva.sum()),
              "N_UNDEFINED": int(errors.correlacao_indefinida.sum()),
              "N_NAMES": catalog.loc[catalog.elegivel, "nome_dataset"].nunique()}
    fields.update(N_CATEGORICAL_ONLY=int(quality.somente_categoricos.sum()),
                  N_POSSIBLE_IDENTIFIERS=int(quality.possivel_identificador_ou_caminho.sum()))
    for short, model in [("KNN", "knn_ajustado"), ("RF", "random_forest"), ("GLOBAL", "media_global"), ("RANDOM", "aleatorio")]:
        for metric, abbrev in [("kendall_ponderado", "TAU"), ("spearman", "RHO"), ("regret_1", "REGRET"), ("ndcg_5", "NDCG")]:
            fields[f"{abbrev}_{short}"] = number(score(model, metric))
        fields[f"OLD_TAU_{short}"] = number(score(model, "kendall_ponderado", "historico_mesmos_folds"))
    rf_knn = paired[(paired.comparacao == "random_forest - knn_ajustado") & (paired.metrica == "kendall_ponderado")].iloc[0]
    fields.update(RF_KNN_DELTA=number(rf_knn.diferenca_media_grupos), RF_KNN_LO=number(rf_knn.ic95_inferior), RF_KNN_HI=number(rf_knn.ic95_superior))
    if rf_knn.ic95_inferior > 0:
        fields["MODEL_INTERPRETATION"] = "O intervalo descritivo sugere vantagem do Random Forest sobre o k-NN neste acervo."
    elif rf_knn.ic95_superior < 0:
        fields["MODEL_INTERPRETATION"] = "O intervalo descritivo sugere vantagem do k-NN sobre o Random Forest neste acervo."
    else:
        fields["MODEL_INTERPRETATION"] = "O intervalo descritivo não estabelece vantagem entre Random Forest e k-NN neste acervo."
    primary_count = summary[(summary.modelo == "knn_ajustado") & (summary.metrica == "kendall_ponderado")].iloc[0]
    fields.update(N_VALID=int(primary_count.n_validos), N_VALID_GROUPS=int(primary_count.n_grupos_validos))
    ks = selected_k[selected_k.selected & ~selected_k.scaled].sort_values("fold").k.astype(int).tolist()
    fields["K_VALUES"] = ", ".join(str(k) for k in ks)
    corr_text = []
    for row in correlations.itertuples():
        if pd.notna(row.q_bh) and row.q_bh < .05:
            corr_text.append(row.variavel.replace("_", r"\_")+f" ($\\rho={number(row.spearman)}$, $q={number(row.q_bh)}$)")
    fields["CORRELATIONS"] = ("Após ajuste Benjamini--Hochberg das cinco comparações, foram observadas associações para: "+"; ".join(corr_text)+"."
                               if corr_text else "Nenhuma das cinco associações exploratórias permaneceu abaixo de $q=0{,}05$ após ajuste Benjamini--Hochberg.")
    # Tabelas pequenas geradas a partir das mesmas linhas que alimentam o texto.
    table = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Método & Spearman & $\tau_w$ & NDCG@5 & Regret@1 \\", r"\midrule"]
    comparison = [r"\begin{tabular}{lrrr}", r"\toprule", r"Método & $\tau_w$ histórico & $\tau_w$ reconstruído & Diferença pareada \\", r"\midrule"]
    for model, label in labels.items():
        table.append(label+" & "+" & ".join("$"+number(score(model, metric))+"$" for metric in ["spearman", "kendall_ponderado", "ndcg_5", "regret_1"])+r" \\")
        delta = differences[(differences.condicao == "reconstrucao_completa") & (differences.modelo == model) & (differences.metrica == "kendall_ponderado")].iloc[0]
        comparison.append(label+" & "+" & ".join("$"+number(v)+"$" for v in [score(model, "kendall_ponderado", "historico_mesmos_folds"), score(model, "kendall_ponderado"), delta.diferenca_media_grupos])+r" \\")
    for lines, name in [(table, "tabela_tcc.tex"), (comparison, "tabela_condicoes.tex")]:
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (OUTPUT/name).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    chapter = r'''
\chapter{Experimentos e Resultados}\label{cap:experimentos}

Este capítulo apresenta a reconstrução integral dos escores da Fase 1 e a repetição da avaliação dos metamodelos. A comparação usa as mesmas tarefas, grupos e partições da referência histórica revisada; por isso, permite avaliar a mudança sem confundi-la com a alteração anterior de LOO por tarefa para validação por grupos. A avaliação continua retrospectiva sobre um acervo previamente explorado.

\section{Reconstrução Integral da Fase 1}

A auditoria dos dados brutos preservou @@N_TASKS@@ tarefas elegíveis, @@N_NAMES@@ nomes e @@N_GROUPS@@ grupos. As oito exclusões por suporte insuficiente de classe permanecem documentadas. Não houve falha de carregamento ou mudança entre o alvo oficial da tarefa e o alvo padrão disponível no dataset. A identificação de grupos foi preservada, permitindo usar os mesmos folds externos na comparação.

Foram executados @@N_FITS@@ ajustes de classificadores, correspondentes a @@N_TASKS@@ tarefas, 60 pipelines e cinco folds estratificados por tarefa. Para todos os classificadores, inclusive SVM, foram usadas as mesmas linhas de treino e teste. A mediana numérica, o codificador categórico, os escalonadores e os seletores foram ajustados exclusivamente no treino de cada fold. O teste recebeu apenas as transformações já ajustadas. Transformações idênticas foram reutilizadas entre classificadores dentro do mesmo fold, sem compartilhar ajustes entre folds.

O processo registrou @@N_FAILURES@@ falhas de ajuste, @@N_MISSING@@ escores agregados ausentes e @@N_WARNINGS@@ avisos de convergência. Avisos de convergência são preservados como limitação do orçamento fixo, sobretudo do MLP com até 200 iterações; não são convertidos em convergência bem-sucedida. Um escore médio só é disponibilizado se os cinco folds foram avaliados. A matriz reconstruída possui @@N_PIPELINE_SCORES@@ posições de desempenho.

A validação dos artefatos verificou ausência de sobreposição entre treino e teste, cobertura de todas as linhas uma vez no teste e igualdade de amostras entre os 60 pipelines. Configuração, versões, hashes e índices de cada fold foram preservados em arquivos separados. Isso atende à reconstrução antes pendente; os escores históricos permanecem disponíveis para comparação, sem substituição silenciosa.

\section{Comparação Controlada das Condições}

Três condições foram avaliadas nas mesmas tarefas, grupos e partições externas: (i) escores e vetores históricos; (ii) escores reconstruídos com vetores históricos; e (iii) escores reconstruídos com vetores recalculados dos dados brutos. O segundo caso mantém a representação fixa para examinar o efeito dos novos alvos; o terceiro constitui a avaliação principal. As entradas das três condições usam precisão float32, como as matrizes históricas, preservando a mesma resolução numérica para tratar empates; os escores reconstruídos de origem permanecem em float64 nos artefatos da Fase 1. Em todos os casos, o ajuste de k e da padronização ocorre somente no treino interno.

A \autoref{tab:condicoesFaseUm} apresenta a comparação de Kendall ponderado. As médias de cada condição usam os grupos com métrica definida; a diferença é calculada de forma pareada, apenas com tarefas e grupos avaliáveis em ambas. Assim, a diferença pareada não precisa coincidir exatamente com a subtração de duas médias calculadas com suportes distintos. Os intervalos descritivos completos e as três condições estão nos CSVs.

\begin{table}[htbp]
\centering
\caption{Comparação da referência revisada com a reconstrução integral}
\label{tab:condicoesFaseUm}
\resizebox{\textwidth}{!}{\input{projeto/resultados/fase1_corrigida_2026/tabela_condicoes.tex}}
\fonte{Autoria própria (2026).}
\end{table}

O k-NN apresentou $\tau_w=@@OLD_TAU_KNN@@$ com escores históricos e $\tau_w=@@TAU_KNN@@$ após reconstrução completa; Random Forest passou de $@@OLD_TAU_RF@@$ para $@@TAU_RF@@$. Esses resultados caracterizam a mudança no benchmark sob partições equivalentes. Não isolam causalmente apenas a imputação: também mudam a codificação por fold, a amostra da SVM, a semente explícita da PCA e, na condição principal, a recomputação dos vetores. A análise com vetor histórico fixo ajuda a distinguir o efeito dos escores daquele da representação. Os próprios rankings reais são recalculados e a cobertura pode mudar quando pipelines antes ausentes passam a ter escore; portanto, a diferença não isola ganho de capacidade preditiva sob uma única verdade de referência e um conjunto idêntico de itens avaliáveis.

O antigo valor de aproximadamente $0{,}5222$ para k-NN no LOO por tarefa não é a referência controlada desta seção. Ele usava outro protocolo e potencialmente mantinha tarefas relacionadas no treino; compará-lo diretamente aos resultados atuais atribuiria à Fase 1 diferenças de desenho da avaliação.

\section{Avaliação Principal dos Metamodelos}

A \autoref{tab:modelosRevisao} resume a avaliação dos metamodelos sobre a matriz reconstruída. O k-NN teve correlações definidas em @@N_VALID@@ tarefas de @@N_VALID_GROUPS@@ grupos. Os valores de k selecionados nos cinco treinos externos foram @@K_VALUES@@. Casos de correlação indefinida continuam contabilizados nos arquivos; NDCG e regret são preservados quando calculáveis.

\begin{table}[htbp]
\centering
\caption{Metamodelos sobre escores reconstruídos: médias por grupo}
\label{tab:modelosRevisao}
\resizebox{\textwidth}{!}{\input{projeto/resultados/fase1_corrigida_2026/tabela_tcc.tex}}
\fonte{Autoria própria (2026).}
\end{table}

A diferença pareada Random Forest menos k-NN em Kendall ponderado foi $@@RF_KNN_DELTA@@$, com intervalo descritivo de 95\% $[@@RF_KNN_LO@@;@@RF_KNN_HI@@]$. @@MODEL_INTERPRETATION@@ A conclusão é restrita ao acervo e ao protocolo; o Random Forest tem hiperparâmetros fixos e não houve comparação exaustiva de metamodelos. A \autoref{fig:modelosRevisao} mostra os intervalos das médias por grupo.

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\textwidth]{projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa/figuras/comparacao_modelos.pdf}
\caption{Kendall ponderado após reconstrução da Fase 1}
\label{fig:modelosRevisao}
\fonte{Autoria própria (2026).}
\end{figure}

O regret@1 foi $@@REGRET_KNN@@$ para k-NN, $@@REGRET_RF@@$ para Random Forest e $@@REGRET_GLOBAL@@$ para média global. A baseline aleatória apresentou NDCG@5 de $@@NDCG_RANDOM@@$. Isso reforça a necessidade de interpretar NDCG com as correlações e o regret: relevâncias próximas podem gerar NDCG alto sem recuperação exata do top-5. Kendall ponderado enfatiza inversões de maior peso, enquanto NDCG acumula relevância com desconto logarítmico.

\section{Diagnóstico de Falhas}

O k-NN ajustado apresentou @@N_ERRORS_KNN@@ tarefas com Kendall ponderado não positivo e @@N_UNDEFINED@@ com correlação indefinida. O diagnóstico liga cada task a tamanho, dimensionalidade, balanceamento, atributos constantes, distância ao vizinho de treino, predições e estado de avaliação. @@CORRELATIONS@@ Essas associações não demonstram causalidade nem explicam isoladamente a variabilidade.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa/figuras/diagnostico_erros.pdf}
\caption{Diagnóstico do k-NN com escores reconstruídos}
\label{fig:errosRevisao}
\fonte{Autoria própria (2026).}
\end{figure}

\section{Discussão e Limitações}\label{sec:discussao}

A reconstrução corrige os dois problemas identificados na produção dos escores: estimar imputação antes da CV e usar amostra exclusiva da SVM. O controle também inclui codificação e seleção ajustadas no treino. A inclusão desses cuidados não implica aumento necessário das métricas; o objetivo é estimar desempenho sob condições comparáveis.

Permanecem como limites: avaliação retrospectiva, agrupamento conservador que pode não detectar todas as famílias, codificação ordinal de variáveis nominais, correlações de rótulos multiclasses dependentes de sua codificação e orçamento fixo de treinamento. A auditoria registra @@N_CATEGORICAL_ONLY@@ tarefas somente categóricas e @@N_POSSIBLE_IDENTIFIERS@@ com nomes sugestivos de identificadores ou caminhos. Esses indicadores exigem interpretação de escopo e não demonstram vazamento por si só; não foram usados para excluir tarefas após observar resultados. A representação descreve um conjunto rotulado completo e não cobre consulta sem rótulos. Intervalos de bootstrap dos grupos são descritivos, pois treinos de folds se sobrepõem. A comparação de metamodelos com o mesmo vetor não substitui uma ablação de representações.

O piloto anterior de 180 avaliações de hiperparâmetros permanece como evidência exploratória separada. O arsenal principal foi mantido em 60 candidatos para comparar protocolos e metamodelos sob espaço controlado. Expandir todo o arsenal, otimizar os metamodelos e avaliar conjuntos externos coletados após o desenvolvimento são próximos passos, sem tratar resultados de outros benchmarks como comparações numéricas equivalentes \cite{Gijsbers2024,Baratchi2024}.
'''
    (ROOT/"capitulos/cap-experimentos-resultados.tex").write_text(render(chapter, fields), encoding="utf-8", newline="\n")
    conclusion = r'''
\chapter{Conclusão}\label{cap:conclusoeseperspectivas}

Este trabalho desenvolveu e avaliou uma infraestrutura de meta-aprendizado para ranquear pipelines de classificação com descritores estatísticos de conjuntos rotulados. A auditoria das tarefas e a reconstrução da Fase 1 permitiram revisar resultados e conclusões com maior controle experimental.

\section{Síntese dos Resultados}\label{sec:sintese-conclusao}

O acervo histórico continha 201 tarefas. Os critérios de qualidade delimitaram @@N_TASKS@@ tarefas em @@N_GROUPS@@ grupos, sobre as quais foram executados @@N_FITS@@ ajustes dos 60 pipelines em cinco folds. Todas as etapas ajustáveis de pré-processamento usaram somente treino, e a SVM recebeu as mesmas amostras dos demais classificadores. Os avisos e as falhas foram documentados, sem substituir escores ausentes por resultados de teste.

Na avaliação principal, o Kendall ponderado médio por grupo foi $@@TAU_KNN@@$ para k-NN, $@@TAU_RF@@$ para Random Forest e $@@TAU_GLOBAL@@$ para média global. A diferença Random Forest menos k-NN foi $@@RF_KNN_DELTA@@$, com intervalo descritivo $[@@RF_KNN_LO@@;@@RF_KNN_HI@@]$. @@MODEL_INTERPRETATION@@ O diagnóstico encontrou @@N_ERRORS_KNN@@ tarefas com Kendall ponderado não positivo e @@N_UNDEFINED@@ com correlação indefinida.

A comparação com escores históricos usou as mesmas tarefas e partições, distinguindo a reconstrução de escores da recomputação da representação. Os valores de LOO por tarefa divulgados inicialmente não constituem uma comparação direta com a validação revisada por grupos. A melhoria do protocolo deve ser distinguida de qualquer alteração numérica de desempenho.

\section{Atendimento dos Objetivos}\label{sec:objetivosAtendidos}

O primeiro objetivo foi atendido pela construção e reconstrução auditável do meta-conjunto, com dados, configurações, escores e falhas rastreáveis. O segundo foi atendido pela implementação e avaliação de preditores de ranking com ajuste interno e grupos externos separados. O ganho depende da métrica e do comparador; não se conclui adequação uniforme para todos os problemas.

\section{Limitações e Perspectivas}\label{sec:limitacoes}

A representação determinística exige rótulos na consulta e usa codificações nominais que limitam a interpretação de algumas correlações. A avaliação continua retrospectiva, os grupos podem não identificar todas as versões relacionadas, os hiperparâmetros dos classificadores e do Random Forest meta foram mantidos fixos e o orçamento de MLP pode produzir avisos de convergência. Os intervalos de bootstrap são descritivos devido à sobreposição dos treinos.

A reconstrução integral da Fase 1 foi concluída nesta etapa; deixa de ser uma pendência do trabalho. As próximas etapas são ampliar o arsenal de hiperparâmetros, comparar representações sob o mesmo protocolo, avaliar orçamento e convergência e validar em dados externos coletados após o desenvolvimento. O piloto anterior de hiperparâmetros permanece uma análise exploratória separada.
'''
    (ROOT/"capitulos/cap-conclusoes.tex").write_text(render(conclusion, fields), encoding="utf-8", newline="\n")
    abstract_pt = r'''
\begin{resumoutfpr}
A seleção de pipelines de classificação depende das características dos dados e das etapas de pré-processamento. Este trabalho investiga seu ranqueamento por meta-aprendizado com representação estatística determinística de conjuntos rotulados. O acervo histórico contém 201 tarefas OpenML e 60 pipelines; a auditoria delimita @@N_TASKS@@ tarefas elegíveis em @@N_GROUPS@@ grupos. A Fase 1 foi integralmente reconstruída em @@N_FITS@@ ajustes, com imputação, codificação, escalonamento e seleção estimados somente no treino de cada fold. Todos os classificadores, inclusive SVM, recebem as mesmas amostras. A representação usa 55 descritores, nove zeros e normalização L2. Cinco divisões externas por grupos e ajuste interno de k comparam k-NN, Random Forest, média global e ranking aleatório. O Kendall ponderado médio por grupo foi $@@TAU_KNN@@$ para k-NN, $@@TAU_RF@@$ para Random Forest e $@@TAU_GLOBAL@@$ para média global. A diferença Random Forest menos k-NN foi $@@RF_KNN_DELTA@@$, com intervalo descritivo de 95\% $[@@RF_KNN_LO@@;@@RF_KNN_HI@@]$. A comparação controlada mantém tarefas e folds ao substituir escores históricos e recalcular a representação. O diagnóstico registra @@N_ERRORS_KNN@@ tarefas com correlação ponderada não positiva e @@N_UNDEFINED@@ indefinidas. Os artefatos incluem predições, estados, parâmetros, índices e hashes. A avaliação permanece retrospectiva e seus intervalos são descritivos; a expansão do arsenal e a validação externa são próximas etapas.
\end{resumoutfpr}
'''
    (ROOT/"PreTexto/resumo.tex").write_text(render(abstract_pt, fields), encoding="utf-8", newline="\n")
    english = {key: (str(value).replace("{,}", ".")) for key, value in fields.items()}
    abstract_en = r'''
\begin{abstractutfpr}
Classification pipeline selection depends on dataset characteristics and preprocessing choices. This work investigates pipeline ranking through meta-learning with deterministic statistical descriptors of labeled datasets. An archive of 201 OpenML tasks and 60 pipelines yields @@N_TASKS@@ eligible tasks in @@N_GROUPS@@ groups after auditing. Phase 1 was fully reconstructed in @@N_FITS@@ model fits, with imputation, encoding, scaling and feature selection fitted exclusively on each training fold. All classifiers, including SVM, use identical training and test rows. Each dataset vector comprises 55 descriptors, nine padded zeros and L2 normalization. Five outer group partitions and inner neighborhood selection compare k-NN, Random Forest, global-mean and random-ranking baselines. Group-averaged weighted Kendall is @@TAU_KNN@@ for k-NN, @@TAU_RF@@ for Random Forest and @@TAU_GLOBAL@@ for the global mean. The Random Forest minus k-NN difference is @@RF_KNN_DELTA@@, with a descriptive 95\% interval [@@RF_KNN_LO@@, @@RF_KNN_HI@@]. Controlled comparisons retain tasks and partitions while replacing historical scores and recomputing descriptors. Error analysis identifies @@N_ERRORS_KNN@@ tasks with nonpositive weighted correlation and @@N_UNDEFINED@@ undefined cases. Artifacts include predictions, statuses, parameters, row indices and checksums. The evaluation remains retrospective and its intervals descriptive; expanding the pipeline arsenal and external validation remain future work.
\end{abstractutfpr}
'''
    (ROOT/"PreTexto/abstract.tex").write_text(render(abstract_en, english), encoding="utf-8", newline="\n")
    methods = r'''
\chapter{Materiais e Métodos}\label{cap:materialemetodos}

Este capítulo descreve os dados, a reconstrução dos escores de pipelines e a avaliação dos metamodelos. O protocolo distingue os dados brutos usados na validação dos classificadores da meta-descrição da tarefa rotulada. A reconstrução integral substitui a produção histórica de escores na avaliação principal; os artefatos anteriores são preservados para comparação controlada.

\section{Materiais e Fonte dos Dados}\label{sec:materiais}

Os dados provêm de 201 tarefas OpenML \cite{Vanschoren2014} do acervo original. Os metadados oficiais identificam tarefa, versão do dataset e alvo; os arquivos brutos ficam no cache local OpenML, e seus hashes integram o catálogo. Utiliza-se o alvo especificado pela tarefa, confirmado como igual ao padrão do dataset nesta coleta. Não houve falha de carregamento.

Os critérios de inclusão são classificação supervisionada, 500 a 10.000 exemplos, ao menos duas classes, pelo menos cinco observações por classe e atributos tabulares disponíveis após remover colunas de data/hora. Os critérios não usam desempenho dos classificadores ou do ranqueamento. Permanecem @@N_TASKS@@ tarefas elegíveis em @@N_GROUPS@@ grupos; o catálogo registra as oito exclusões por suporte de classe insuficiente, nomes, links, tamanho, atributos, classes, balanceamento e proporção de valores ausentes.

Tarefas distintas podem representar o mesmo dataset ou versões relacionadas. Os grupos anteriores, formados por nomes repetidos ou atributos processados idênticos, são preservados e conferidos contra IDs e fingerprints dos atributos brutos. A atribuição de grupos não mudou nesta reconstrução. Essa regra é conservadora, mas não assegura detectar todas as famílias com nomes e transformações diferentes. A tarefa 363713 já integrava o acervo e não é tratada como holdout externo.

\subsection{Ambiente Computacional}\label{subsec:recursos}

A execução usa Python 3.14.2, NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1 e scikit-learn 1.8.0. O computador local dispõe de processador Intel Core i5-9400F, com seis núcleos lógicos. A reconstrução utiliza quatro processos em tarefas diferentes, com uma thread BLAS e um job por classificador em cada processo. As versões efetivas, configurações e hashes constam dos manifestos; a limitação de paralelismo controla a utilização de recursos sem subamostrar observações.

\section{Fase 1: Escores em Dados Brutos}

Para cada tarefa, uma validação estratificada de cinco folds usa embaralhamento e semente 42. Cada observação participa do teste uma única vez e do treino nos demais folds. Os índices são gerados uma vez por tarefa e compartilhados por todos os pipelines. Não existe subamostra exclusiva da SVM: os cinco classificadores recebem as mesmas linhas de treino e teste em cada fold.

O tratamento numérico estima a mediana exclusivamente no treino, preservando atributos vazios. Nos atributos categóricos, valores ausentes são convertidos deterministicamente em um marcador; um codificador ordinal é ajustado no treino, atribuindo -1 a categorias desconhecidas no teste. Escalonadores e seletores também são ajustados apenas no treino. O teste recebe somente as transformações ajustadas, sem usar seus rótulos para selecionar atributos ou estimar parâmetros.

As transformações comuns são calculadas uma vez por fold e combinação e reutilizadas pelos classificadores que recebem exatamente a mesma entrada. Essa otimização não compartilha ajuste entre folds e não muda os algoritmos. A auditoria verifica cobertura de linhas, ausência de interseção treino/teste e amostras equivalentes entre classificadores.

\subsection{Arsenal de 60 Pipelines}

O arsenal cruza quatro escalonadores (StandardScaler, MinMaxScaler, RobustScaler e ausência), três seletores (PCA, SelectKBest por ANOVA e ausência) e cinco classificadores. Os seletores usam $q=\min(10,\max(1,P-1),N_{treino})$ componentes ou atributos, respeitando dimensões disponíveis. PCA usa solver automático e semente 42, controlando a parte aleatória não fixada no código histórico.

Os classificadores são: (i) Random Forest com 100 árvores; (ii) SVM RBF com $C=1$, gamma scale e sem limite de subamostra; (iii) k-NN com cinco vizinhos uniformes; (iv) MLP com camadas 64 e 32, otimizador Adam e até 200 iterações; e (v) árvore de decisão com profundidade máxima 10. Usa-se semente 42 nos componentes estocásticos. Demais parâmetros são os padrões da versão registrada da biblioteca, exportados para auditoria. Não se otimizam os hiperparâmetros desse arsenal nesta comparação.

\subsection{Métricas, Avisos e Resultados Ausentes}

São calculados acurácia balanceada e F1 macro no teste de cada fold. O alvo do metamodelo é a média de acurácia balanceada dos cinco folds, não uma média de subconjunto de folds bem-sucedidos. Se qualquer ajuste falhar, o alvo desse par tarefa/pipeline fica ausente. Médias, desvios amostrais, tempos, estados, erros e avisos são registrados separadamente.

Avisos de convergência, especialmente no MLP, são preservados como evidência do orçamento finito de treinamento. Eles não são ocultados nem descritos como convergência garantida. Os escores resultantes caracterizam os modelos com o orçamento fixado. No total, a execução contempla @@N_FITS@@ ajustes para @@N_PIPELINE_SCORES@@ posições de escore na matriz final.

\section{Representação Estatística das Tarefas}\label{sec:representacaoReconstruida}

A representação é um vetor determinístico, sem treinamento neural ou otimização de função objetivo. Para cada atributo numérico após codificação da tarefa, calculam-se média, desvio padrão populacional acrescido de $10^{-8}$, mínimo, máximo, mediana, assimetria, curtose de Fisher e correlações de Pearson e Spearman com o alvo codificado. Cada um dos nove descritores é agregado entre atributos por mínimo, percentil 25, média, percentil 75 e máximo: $9\times5=45$ componentes.

Dez propriedades globais completam o vetor: $\log(1+N)$, $\log(1+P)$, número de classes, proporções majoritária e minoritária, entropia das proporções acrescidas de $10^{-8}$, proporção de atributos constantes, média da correlação absoluta entre atributos com diagonal zerada, média das proporções de valores únicos e $P/(N+10^{-8})$. A média da matriz de correlação inclui os zeros da diagonal no denominador $P^2$. Concatenam-se 55 descritores e nove zeros; não há truncamento com dimensão 64. Valores não finitos recebem \texttt{nan\_to\_num} e aplica-se norma L2, preservando vetor nulo.

A meta-descrição utiliza a tarefa rotulada completa, disponível no cenário de recomendação, e é calculada separadamente do ajuste dos classificadores em folds. Ela não usa predições ou escores de teste. Portanto, o estudo não cobre consulta sem rótulos e não trata essa extração global de descritores como uma transformação do classificador dentro da CV. As correlações com alvos nominais multiclasses dependem de codificação arbitrária. O vetor estatístico é distinto de uma representação neural aprendida como Dataset2Vec \cite{Jomaa2021}.

\section{Fase 2: Predição e Avaliação por Grupos}

O k-NN estima a acurácia dos 60 pipelines pela média dos vizinhos ponderada pelo inverso da distância euclidiana:
\begin{equation}\label{eq:knn}
\hat A(P_j)=\frac{\sum_i w_i A_{i,j}}{\sum_i w_i},\qquad
w_i=\frac{1}{\lVert\mathbf{x}_{novo}-\mathbf{x}_i\rVert_2+10^{-8}}.
\end{equation}

O comparador Random Forest meta é um regressor multi-saída nativo com 200 árvores, mínimo de duas observações por folha, max features 1,0 e semente 42. As baselines são média global de escores por pipeline apenas no treino e ranking aleatório por escores uniformes com semente 42 acrescida do fold. Avalia-se também k-NN com padronização de dimensões ajustada exclusivamente no treino.

Usa-se GroupKFold externo com cinco divisões, shuffle e semente 42. Dentro do treino externo, quatro divisões por grupos selecionam $k\in\{1,3,5,7,10\}$ pela média de Kendall ponderado por grupo, escolhendo o menor k em empate. Teste externo não participa da escolha de k, padronização ou imputação de alvos. As divisões têm grupos disjuntos em todos os níveis.

Resultados ausentes no treino meta são imputados pela mediana de cada pipeline. Colunas inteiramente ausentes no treino são excluídas da avaliação correspondente no teste. Escores reais ausentes no teste nunca são imputados para calcular qualidade do ranking. Correlações indefinidas permanecem como NaN com estado explícito, mantendo NDCG e regret quando calculáveis.

\subsection{Métricas de Ranqueamento}

Spearman mede associação de postos. Kendall tau-b trata empates, com
\begin{equation}\label{eq:kendall}
\tau_b=\frac{C-D}{\sqrt{(C+D+T_x)(C+D+T_y)}},
\end{equation}
onde $C,D$ são pares concordantes e discordantes e $T_x,T_y$ contam empates exclusivos. Kendall ponderado padrão da SciPy usa pesos hiperbólicos aditivos $1/(r_i+1)+1/(r_j+1)$ e média de duas ordenações lexicográficas decrescentes dos escores real e predito; sua implementação não fornece p-valor.

NDCG usa ganho linear de acurácia balanceada e desconto logarítmico, conforme a implementação do scikit-learn:
\begin{equation}\label{eq:ndcg}
\operatorname{NDCG}@K=\frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K},\qquad
\operatorname{DCG}@K=\sum_{i=1}^K\frac{rel_i}{\log_2(i+1)}.
\end{equation}
Reportam-se NDCG completo e NDCG@5. Regret@1 é a diferença para a melhor acurácia real disponível ao recomendar o primeiro pipeline; Regret@5 considera o melhor entre os cinco primeiros. Empates recebem tratamento médio em NDCG e ordem estável de colunas em regret. As métricas são complementares: NDCG elevado não equivale a recuperação exata do top-5.

\subsection{Comparação Controlada e Diagnóstico}

Executam-se três condições com as mesmas tarefas, grupos e folds externos: matrizes históricas; escores reconstruídos com vetores históricos; e escores e vetores reconstruídos. As entradas usam precisão float32 em todas as condições, como no armazenamento histórico, evitando mudar a resolução dos empates somente na condição nova. Os valores originais reconstruídos em float64 são preservados. A comparação pareada examina as mesmas tarefas/grupos avaliáveis nas duas condições. Ela distingue parte do efeito da representação, mas não isola causalmente somente a imputação, pois também mudam a codificação por fold, a amostra da SVM e o controle aleatório da PCA.

Calculam-se médias dentro dos grupos e depois entre grupos. Intervalos descritivos de 95\% usam 2.000 reamostragens de grupos, com semente 42; a sobreposição dos treinos limita a interpretação inferencial. O diagnóstico usa cinco associações de Spearman por grupo (tamanho, atributos, proporção majoritária, atributos constantes e distância ao vizinho), com ajuste Benjamini--Hochberg e sem atribuição causal.

\section{Reprodutibilidade e Delimitação}

Arquivos separados preservam parâmetros, versões, hashes de dados e código, predições, métricas, índices e estados de cada ajuste. Checkpoints por tarefa permitem retomar sem misturar execuções de configurações distintas. Dados brutos permanecem no cache local; as evidências e matrizes agregadas são versionadas.

O estudo continua retrospectivo, com consulta rotulada e arsenal fixo. Não houve busca exaustiva de hiperparâmetros ou coleta de teste externa após o desenvolvimento. O piloto anterior de 180 avaliações permanece separado. Essas limitações delimitam as conclusões, mesmo após concluir a reconstrução integral da Fase 1.
'''
    (ROOT/"capitulos/cap-metodologia.tex").write_text(render(methods, fields), encoding="utf-8", newline="\n")
    article_file = ROOT/"artigo/main.tex"
    article = article_file.read_text(encoding="utf-8")
    article_abstract = render(abstract_en, english).replace("abstractutfpr", "abstract")
    article = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda _: article_abstract.strip(), article, flags=re.S)
    intro_pos = article.index(r"\section{Introdução}")
    article = article[:intro_pos] + render(r'''
\section{Introdução}\label{sec:introduction}

A recomendação de pipelines depende da representação dos dados, dos algoritmos candidatos e do protocolo usado para medir desempenho. Em AutoML, distinguir espaço de busca, estratégia e avaliação permite controlar a origem de ganhos e limitações \cite{Baratchi2024}. Este trabalho investiga o ranqueamento de 60 pipelines por meta-aprendizado a partir de descritores estatísticos de conjuntos rotulados.

Uma auditoria do acervo de 201 tarefas OpenML identificou tarefas relacionadas e oito casos sem suporte suficiente de classe para cinco folds. A revisão delimitou @@N_TASKS@@ tarefas em @@N_GROUPS@@ grupos. Nesta etapa, os escores dos classificadores foram integralmente reconstruídos: imputação, codificação, escalonamento e seleção são ajustados apenas no treino, e SVM recebe as mesmas linhas dos demais algoritmos. Isso conclui a reconstrução antes pendente, preservando os resultados históricos como referência.

As contribuições são: (i) disponibilizar os escores reconstruídos com índices e estados de cada fold; (ii) comparar protocolos nas mesmas tarefas e grupos, distinguindo escores e representações; e (iii) reavaliar k-NN, Random Forest e baselines com ajuste interno e métricas complementares. O estudo continua retrospectivo e não trata reconstrução metodológica como garantia de aumento de métricas.

\section{Trabalhos Relacionados}

Três eixos orientam a discussão: representação, recomendação e avaliação. No primeiro, Dataset2Vec aprende representações de conjuntos \cite{Jomaa2021}, enquanto o vetor deste estudo concatena estatísticas fixas sem treino neural. TabPFN \cite{Hollmann2025} exemplifica reutilização de conhecimento entre problemas tabulares, mas sua tarefa preditiva difere do ranqueamento do arsenal aqui avaliado.

No eixo de recomendação, o enquadramento de \citeN{Baratchi2024} separa espaço de busca, estratégia e avaliação. Manter os 60 candidatos controla a comparação entre metamodelos e protocolos, limitando a extrapolação a configurações não observadas. No eixo de avaliação, AMLB \cite{Gijsbers2024} orienta comparações que consideram desempenho, recursos e falhas. Aqui, isso motiva preservar avisos, escores ausentes e condições equivalentes.

Os pressupostos de No Free Lunch não garantem ganho de um recomendador em dados reais; a relação entre os teoremas e vieses indutivos exige cuidado \cite{Goldblum2024}. A evidência do presente estudo é obtida no próprio protocolo, sem converter números de tarefas diferentes em comparação direta de superioridade.

\section{Materiais e Métodos}

O procedimento compreende auditoria, reconstrução dos classificadores, representação de tarefas e avaliação por grupos. Escores históricos e reconstruídos permanecem em diretórios distintos.

\subsection{Dados e Arsenal}

São usadas tarefas de classificação supervisionada do OpenML \cite{Vanschoren2014}, com alvo oficial, 500--10.000 exemplos, ao menos duas classes e cinco exemplos por classe. Após remover colunas de data/hora, exigem-se atributos tabulares disponíveis. A regra não usa desempenho de recomendação. O catálogo inclui links, grupos, dimensionalidade, balanceamento e hashes. Preservam-se @@N_TASKS@@ tarefas, @@N_NAMES@@ nomes e @@N_GROUPS@@ grupos; os grupos anteriores foram conferidos com IDs e fingerprints brutos, sem mudança nesta coleta.

O arsenal cruza StandardScaler, MinMaxScaler, RobustScaler e ausência; PCA, SelectKBest por ANOVA e ausência; e cinco classificadores. Os parâmetros explícitos são: Random Forest com 100 árvores; SVM RBF, C=1 e gamma scale; k-NN com cinco vizinhos; MLP (64,32), Adam e até 200 iterações; árvore com profundidade máxima 10. Seletores usam até dez componentes/atributos, limitados à dimensão disponível; PCA usa solver automático e semente 42. A semente controla a aleatoriedade dos demais componentes estocásticos. Demais parâmetros seguem a versão registrada da biblioteca.

\subsection{Reconstrução Integral da Fase 1}

Cada tarefa recebe cinco folds estratificados, com shuffle e semente 42. Os mesmos índices são usados em todos os pipelines, e cada linha é testada uma vez. Não há subamostra exclusiva da SVM. A mediana numérica é estimada no treino, preservando colunas vazias. Categorias recebem um marcador determinístico para ausência e codificador ordinal ajustado no treino; categorias desconhecidas recebem -1. Escalonadores e seletores usam apenas treino, e o teste recebe somente transformações já ajustadas.

Transformações comuns são reaproveitadas entre classificadores dentro do mesmo fold, sem compartilhar ajuste entre folds. A execução utiliza quatro processos em tarefas diferentes, uma thread BLAS e um job por classificador. O ambiente é Python 3.14.2 e scikit-learn 1.8.0, com demais versões nos manifestos.

São calculados acurácia balanceada e F1 macro em cada teste. O alvo do metamodelo é a média dos cinco folds de acurácia balanceada; falha em qualquer fold torna esse alvo ausente. Avisos de convergência são preservados como limitação do orçamento finito, sem serem descritos como convergência garantida. Foram executados @@N_FITS@@ ajustes, com @@N_FAILURES@@ falhas, @@N_MISSING@@ escores agregados ausentes e @@N_WARNINGS@@ avisos de convergência.

\subsection{Vetor Estatístico}

Para cada atributo numérico codificado, calculam-se média, desvio padrão populacional acrescido de $10^{-8}$, mínimo, máximo, mediana, assimetria, curtose de Fisher e correlações de Pearson e Spearman com o alvo. Cada descritor é agregado por mínimo, percentil 25, média, percentil 75 e máximo entre atributos: 45 entradas.

Dez propriedades globais são concatenadas: $\log(1+N)$, $\log(1+P)$, número de classes, proporções majoritária e minoritária, entropia das proporções acrescidas de $10^{-8}$, proporção de atributos constantes, média da correlação absoluta entre atributos com diagonal zerada e denominador $P^2$, média da proporção de valores únicos e $P/(N+10^{-8})$. Acrescentam-se nove zeros, aplica-se \texttt{nan\_to\_num} e norma L2; vetor nulo permanece nulo.

A meta-descrição usa a tarefa rotulada completa, separadamente da validação dos classificadores, sem usar predições ou escores dos folds. O cenário requer rótulos na consulta e não cobre dados sem rótulos. Correlações com códigos nominais multiclasses dependem de ordenação arbitrária. Não há treinamento da representação.

\subsection{Metamodelos, Partições e Métricas}

O k-NN estima as 60 saídas por média dos vizinhos ponderada por $1/(d+10^{-8})$, com distância euclidiana. O Random Forest meta é um regressor multi-saída nativo com 200 árvores, mínimo de duas observações por folha, max features 1,0 e semente 42. As baselines são média global por pipeline somente no treino e escores uniformes aleatórios com semente 42 acrescida do fold. Avalia-se também k-NN com padronização ajustada no treino.

GroupKFold externo usa cinco divisões, shuffle e semente 42. Quatro divisões internas por grupos selecionam $k\in\{1,3,5,7,10\}$ pela média de Kendall ponderado por grupo, com menor k em empate. Teste externo não participa da seleção, padronização ou imputação de alvos. Medianas de alvos são estimadas no treino; colunas sem observação no treino ficam fora da avaliação correspondente. Escores reais ausentes no teste nunca são imputados para avaliar ranking.

Spearman e Kendall tau-b avaliam ordenação global. Kendall ponderado padrão usa pesos hiperbólicos aditivos e média de duas ordenações lexicográficas real/predito, sem p-valor definido. NDCG usa ganho linear da acurácia e desconto logarítmico; reportam-se NDCG completo e NDCG@5. Regret@1 é a perda em relação à melhor acurácia disponível ao recomendar o primeiro pipeline, e Regret@5 usa o melhor entre os cinco primeiros. NDCG trata empates pela média; regret usa ordem estável de colunas.

As médias são calculadas dentro de grupo e depois entre grupos. Intervalos descritivos de 95\% usam 2.000 reamostragens de grupos com semente 42. A sobreposição de treinos limita inferência. O diagnóstico usa cinco associações exploratórias por grupo com ajuste Benjamini--Hochberg, sem interpretação causal.

\subsection{Condições de Comparação}

As mesmas tarefas, grupos e folds avaliam: (i) matrizes históricas; (ii) escores reconstruídos e vetores históricos; e (iii) escores e vetores reconstruídos, a condição principal. As entradas usam precisão float32 em todas as condições, compatível com as matrizes históricas; os escores reconstruídos de origem ficam preservados em float64. A diferença pareada usa casos avaliáveis em ambas as condições. Comparar escores históricos e novos também altera os próprios rankings reais usados como alvo; por isso, a diferença não isola ganho de capacidade preditiva sob uma única verdade de referência. A comparação com vetor fixo ajuda a separar representação e escores, mantendo essa limitação.

\section{Resultados e Discussão}

Esta seção compara a referência revisada com a reconstrução e apresenta metamodelos e falhas. Os números antigos de LOO por tarefa não são usados como referência direta, pois o protocolo era distinto.

\subsection{Comparação Controlada}

A Tabela~\ref{tab:condicoes} apresenta Kendall ponderado nas mesmas tarefas e partições por grupos. O k-NN passa de $@@OLD_TAU_KNN@@$ com escores históricos para $@@TAU_KNN@@$ na reconstrução completa; Random Forest passa de $@@OLD_TAU_RF@@$ para $@@TAU_RF@@$. Os intervalos pareados e a condição de vetores históricos com novos escores estão nos CSVs. Médias e diferenças pareadas podem usar suportes distintos quando uma correlação é indefinida.

\begin{table}[htbp]
\centering
\caption{Referência revisada e reconstrução da Fase 1.}
\label{tab:condicoes}
\resizebox{\linewidth}{!}{\input{../projeto/resultados/fase1_corrigida_2026/tabela_condicoes.tex}}
\end{table}

\subsection{Avaliação Principal}

A Tabela~\ref{tab:modelos} apresenta os metamodelos sobre a matriz reconstruída. O k-NN possui correlações definidas em @@N_VALID@@ tarefas de @@N_VALID_GROUPS@@ grupos; os k escolhidos nos treinos externos foram @@K_VALUES@@. A diferença Random Forest menos k-NN é $@@RF_KNN_DELTA@@$, com intervalo descritivo $[@@RF_KNN_LO@@;@@RF_KNN_HI@@]$. @@MODEL_INTERPRETATION@@ A Figura~\ref{fig:modelos} mostra os intervalos por grupo.

\begin{table}[htbp]
\centering
\caption{Metamodelos após reconstrução: médias por grupo.}
\label{tab:modelos}
\resizebox{\linewidth}{!}{\input{../projeto/resultados/fase1_corrigida_2026/tabela_tcc.tex}}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\linewidth]{../projeto/resultados/fase1_corrigida_2026/fase2/reconstrucao_completa/figuras/comparacao_modelos.pdf}
\caption{Kendall ponderado e intervalos descritivos após reconstrução.}
\label{fig:modelos}
\end{figure}

O regret@1 é $@@REGRET_KNN@@$ para k-NN, $@@REGRET_RF@@$ para Random Forest e $@@REGRET_GLOBAL@@$ para média global. A baseline aleatória já apresenta NDCG@5 $@@NDCG_RANDOM@@$, ilustrando por que NDCG alto não demonstra top-5 correto. Correlações, relevância e perda da primeira recomendação são interpretadas conjuntamente.

O diagnóstico registra @@N_ERRORS_KNN@@ tarefas com Kendall ponderado não positivo e @@N_UNDEFINED@@ indefinidas. @@CORRELATIONS@@ Essas associações não demonstram causalidade ou explicação exclusiva da instabilidade.

\subsection{Limitações}

A reconstrução conclui o tratamento antes pendente na Fase 1, sem tornar a avaliação externa ou prospectiva. Permanecem grupos potencialmente incompletos, codificação nominal ordinal, rótulos necessários na consulta, hiperparâmetros fixos e orçamento de MLP sujeito a avisos de convergência. A auditoria identifica @@N_CATEGORICAL_ONLY@@ tarefas somente categóricas e @@N_POSSIBLE_IDENTIFIERS@@ com nomes sugestivos de identificadores/caminhos, sem usá-los para seleção após observar resultados. Os intervalos são descritivos devido à sobreposição dos treinos. Comparar dois preditores sobre um vetor não constitui ablação completa de representações. O piloto anterior de 180 avaliações permanece separado e não equivale a expandir todo o arsenal.

\section{Conclusão}

O estudo reconstruiu a avaliação dos 60 pipelines em dados brutos, com tratamento ajustado no treino e amostras equivalentes para todos os classificadores. Os escores, índices, parâmetros, estados e hashes permitem inspeção da execução. A reavaliação de metamodelos e a comparação controlada delimitam os efeitos observados sem confundir melhoria do protocolo com aumento obrigatório de métricas.

As próximas etapas são ampliar o arsenal, estudar orçamento e convergência, comparar representações e validar em conjuntos externos coletados após o desenvolvimento. A reconstrução completa da Fase 1 deixa de ser uma pendência, preservando os limites retrospectivos do benchmark.

\begin{ack}
Os autores agradecem à Universidade Tecnológica Federal do Paraná pelo apoio institucional e à orientação do TCC que deu origem a este estudo.
\end{ack}
\begin{received}
\end{received}
\bibliographystyle{kdmile}
\bibliography{references}
\end{document}
''', fields)
    article_file.write_text(article, encoding="utf-8", newline="\n")
    intro_file = ROOT/"capitulos/cap-introducao.tex"
    intro = intro_file.read_text(encoding="utf-8")
    replacement = render(r'''Diante desse cenário, este trabalho investiga o ranqueamento e a recomendação de pipelines de classificação por meta-aprendizado. A proposta compreende: (i) auditar um acervo histórico de 201 tarefas OpenML e reconstruir os escores de 60 pipelines sobre @@N_TASKS@@ tarefas elegíveis, com pré-processamento ajustado somente no treino e amostras equivalentes para todos os classificadores; (ii) representar cada tarefa rotulada por um vetor estatístico determinístico de 55 descritores e nove zeros; e (iii) comparar k-NN e Random Forest multi-saída com baselines, usando validação por grupos, ajuste interno e comparação controlada com os escores históricos. A avaliação inclui @@N_GROUPS@@ grupos e permanece retrospectiva, sem inferir superioridade universal a partir das métricas do benchmark.''', fields)
    intro, count = re.subn(r"Diante desse cenário,.*?(?=\n\n)", lambda _: replacement.strip(), intro, count=1, flags=re.S)
    if count != 1:
        raise ValueError("Não foi localizado o parágrafo de apresentação da proposta na introdução.")
    intro = intro.replace("com reprodução LOO histórica e validação principal por grupos", "com comparação controlada e validação principal por grupos")
    intro_file.write_text(intro, encoding="utf-8", newline="\n")
    from projeto.revisao.fase1_corrigida import CONFIG, make_classifier
    current_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUTPUT/"parametros_classificadores.json", {
        "base_classifiers": {name: make_classifier(name, current_config).get_params()
                             for name in current_config["classifiers"]},
        "note": "Parâmetros efetivos da reconstrução; seletores e pré-processamento descritos na configuração e código."})
    documentation = ROOT/"docs/fase1_corrigida"
    documentation.mkdir(parents=True, exist_ok=True)
    write_json(documentation/"CAMPOS_TCC.json", fields)
    print("TCC atualizado a partir das saídas reconstruídas e validadas.")
    return fields


if __name__ == "__main__":
    run()
