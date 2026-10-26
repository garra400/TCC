"""Gera campos, tabela e texto de resultados a partir dos CSVs executados."""
import json
from pathlib import Path

import pandas as pd

from projeto.revisao.experimentos import PROJECT, ROOT


def run():
    path = PROJECT/"resultados/revisao_2026"
    manifest = json.loads((path/"manifesto.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(path/"resumo.csv")
    pairs = pd.read_csv(path/"comparacoes_pareadas.csv")
    errors = pd.read_csv(path/"diagnostico_erros.csv")
    catalog = pd.read_csv(path/"datasets.csv")
    pilot = pd.read_csv(path/"piloto/resumo.csv", keep_default_na=False)
    labels = {"aleatorio": "Aleatório", "media_global": "Média global", "knn_ajustado": "k-NN ajustado",
              "knn_padronizado_ajustado": "k-NN padronizado", "random_forest": "Random Forest"}
    metrics = ["spearman", "kendall_ponderado", "ndcg_5", "regret_1"]
    means = summary.query("protocolo == 'cv_grupos'").pivot(index="modelo", columns="metrica", values="media_grupos")
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Método & Spearman & $\tau_w$ & NDCG@5 & Regret@1 \\", r"\midrule"]
    for name in labels:
        tex.append(labels[name] + " & " + " & ".join(f"${means.loc[name, m]:.4f}$" for m in metrics) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (path/"tabela_tcc.tex").write_text("\n".join(tex), encoding="utf-8")

    def value(model, metric):
        return float(means.loc[model, metric])

    def num(value, digits=4):
        return f"{value:.{digits}f}".replace(".", "{,}")

    delta = pairs.query("comparacao == 'random_forest - knn_ajustado' and metrica == 'kendall_ponderado'").iloc[0]
    global_delta = pairs.query("comparacao == 'media_global - knn_ajustado' and metrica == 'kendall_ponderado'").iloc[0]
    primary = summary.query("protocolo == 'cv_grupos' and modelo == 'knn_ajustado' and metrica == 'kendall_ponderado'").iloc[0]
    worst = errors.dropna(subset=["kendall_ponderado"]).nsmallest(3, "kendall_ponderado")
    examples = "; ".join(f"task {int(row.task_id)} ($\\tau_w={num(row.kendall_ponderado)}$)" for row in worst.itertuples())
    pilot_ranges = pilot.groupby(["task_id", "classifier"]).balanced_accuracy_mean.agg(["min", "max"])
    pilot_ranges["amplitude"] = pilot_ranges["max"] - pilot_ranges["min"]
    pilot_ranges.to_csv(path/"piloto/amplitudes.csv")
    chapter = rf"""% Gerado por python -m projeto.revisao.relatorio. Fontes: resultados/revisao_2026/*.csv.
\chapter{{Experimentos e Resultados}}\label{{cap:experimentos}}

Este capítulo distingue a reprodução do benchmark histórico da avaliação revisada. Primeiro é auditada a unidade experimental; depois são comparados metamodelos em grupos separados e investigados os erros. Finalmente, um piloto em dados brutos avalia a sensibilidade a hiperparâmetros. As matrizes históricas não foram reconstruídas nesta etapa.

\section{{Auditoria do Meta-conjunto}}

O acervo contém 201 tarefas OpenML, 186 nomes de conjuntos e 60 pipelines, totalizando 12.060 células, das quais 172 estão ausentes. Tarefas OpenML distintas não garantem conjuntos independentes. A combinação de nomes repetidos e fingerprints de atributos processados identifica 172 grupos conservadores. Todos os tipos de tarefa foram conferidos em metadados oficiais como classificação supervisionada; isso, por si só, não assegura adequação estatística do alvo.

O requisito de pelo menos cinco exemplos em cada classe exclui oito tarefas: 2075, 3486, 362281, 363078, 363094, 363393, 363554 e 363559. A exclusão não utiliza métricas de recomendação. A avaliação principal considera {manifest['n_tasks']} tarefas, {manifest['n_names']} nomes e {manifest['n_groups']} grupos, com {manifest['missing_scores']} escores ausentes. Os critérios e motivos por tarefa constam do catálogo CSV. A análise de sensibilidade com todas as 201 tarefas foi mantida em diretório separado.

A tarefa 363713, anteriormente descrita como holdout, consta da matriz histórica e compartilha o nome do conjunto com outra tarefa. Por isso, a demonstração pontual foi retirada da evidência experimental. Cada task da análise principal passa a receber uma predição de teste em uma partição com grupos separados, conforme o protocolo do capítulo de métodos.

\section{{Reprodução dos Resultados Históricos}}

A reprodução mantém o preditor original KDTree, as matrizes e os cinco valores de $k$. Para $k=10$, o benchmark histórico apresenta Spearman médio de aproximadamente $0{{,}}5801$, Kendall de $0{{,}}4767$, Kendall ponderado de $0{{,}}5222$ e NDCG de $0{{,}}9844$ sobre 198 tarefas com correlações definidas. Esses números são resultados exploratórios por tarefa, com possibilidade de parentes no treino e seleção de $k$ na própria avaliação; não são a estimativa principal revisada.

Não se atribui toda diferença entre a reprodução e a nova avaliação ao agrupamento: também mudam o número de tarefas de treino por fold, os filtros de qualidade e a unidade usada na agregação. Uma comparação causal exigiria ablações adicionais sob todas as demais condições constantes.

\section{{Comparação em Validação por Grupos}}

A \autoref{{tab:modelosRevisao}} apresenta médias calculadas primeiro dentro de cada grupo e depois entre grupos, evitando maior peso de tarefas repetidas. Para as correlações, {int(primary.n_validos)} tarefas em {int(primary.n_grupos_validos)} grupos têm valores definidos; a tarefa restante possui alvo constante entre os pipelines observados. NDCG e regret permanecem contabilizados quando definidos, mesmo se a correlação não existe.

\begin{{table}}[htbp]
\centering
\caption{{Avaliação revisada: médias por grupo em cinco partições externas}}
\label{{tab:modelosRevisao}}
\resizebox{{\textwidth}}{{!}}{{\input{{projeto/resultados/revisao_2026/tabela_tcc.tex}}}}
\fonte{{Autoria própria (2026), com base nos CSVs da execução revisada.}}
\end{{table}}

O k-NN ajustado alcança $\tau_w={num(value('knn_ajustado','kendall_ponderado'))}$ e o Random Forest, ${num(value('random_forest','kendall_ponderado'))}$. A diferença pareada média Random Forest menos k-NN é ${num(delta.diferenca_media_grupos)}$, com intervalo descritivo de 95\% $[{num(delta.ic95_inferior)}; {num(delta.ic95_superior)}]$, estimado por reamostragem dos grupos. A diferença média global menos k-NN é ${num(global_delta.diferenca_media_grupos)}$, com intervalo $[{num(global_delta.ic95_inferior)}; {num(global_delta.ic95_superior)}]$. Portanto, a vantagem do k-NN sobre a média global não fica estabelecida por esse intervalo. Os resultados sugerem vantagem do Random Forest neste benchmark, sem provar superioridade universal.

O ajuste interno escolheu $k=10$ em todos os folds externos, de modo que o k-NN fixo em 10 e o ajustado coincidiram nesta execução. A padronização das dimensões foi ajustada exclusivamente no treino. O Random Forest foi utilizado com hiperparâmetros fixos; a comparação não constitui uma busca exaustiva de metamodelos. A \autoref{{fig:modelosRevisao}} mostra a dispersão inferencial descritiva das médias.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{projeto/resultados/revisao_2026/figuras/comparacao_modelos.pdf}}
\caption{{Kendall ponderado por modelo com intervalos descritivos por grupo}}
\label{{fig:modelosRevisao}}
\fonte{{Autoria própria (2026).}}
\end{{figure}}

\section{{Diagnóstico de Erros e Variabilidade}}

O k-NN ajustado apresenta {int(errors.correlacao_nao_positiva.sum())} tarefas com Kendall ponderado não positivo e {int(errors.correlacao_indefinida.sum())} com correlação indefinida. Entre os casos de pior ordenação estão {examples}. O arquivo de diagnóstico associa a cada caso dimensão, tamanho, balanceamento, distância ao vizinho de treino, métricas e estado da avaliação, permitindo inspeção individual.

As correlações exploratórias foram calculadas por grupo para cinco variáveis, com correção Benjamini--Hochberg. O número de atributos tem associação positiva com a qualidade ($\rho\approx0{{,}}2064$, $q\approx0{{,}}0400$). As demais variáveis investigadas, incluindo desbalanceamento e distância ao vizinho, não alcançam $q<0{{,}}05$ nesta análise. Assim, a inspeção não sustenta atribuir a instabilidade exclusivamente a outliers ou desbalanceamento. A \autoref{{fig:errosRevisao}} ilustra as relações observadas, sem interpretação causal.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{projeto/resultados/revisao_2026/figuras/diagnostico_erros.pdf}}
\caption{{Diagnóstico do k-NN: dimensionalidade, balanceamento e distância ao treino}}
\label{{fig:errosRevisao}}
\fonte{{Autoria própria (2026).}}
\end{{figure}}

\section{{Piloto de Variação de Hiperparâmetros}}

Um experimento separado usa dados brutos de seis tarefas previamente especificadas (11, 15, 37, 4538, 9914 e 361198). Comparam-se profundidades 5, 10 e ilimitada para árvore de decisão e vizinhanças 3, 5 e 11 para k-NN, mantendo StandardScaler e ausência de seletor. Imputação e codificação categórica são ajustadas dentro de cada fold. São 36 combinações de tarefa, classificador e valor, avaliadas nas mesmas cinco divisões por tarefa, totalizando 180 avaliações de treino/teste. Os arquivos preservam índices de linhas, parâmetros, tempos, avisos e resultados por fold.

As amplitudes de acurácia balanceada entre configurações mostram que a escolha de hiperparâmetros pode alterar o desempenho, mas a seleção do melhor valor usando esses próprios folds é exploratória. As seis tarefas foram escolhidas por identificação e diversidade aparente de tamanho e domínio, sem consultar o desempenho de recomendação; não constituem amostra probabilística. O piloto não testa a capacidade de ranqueamento de um arsenal expandido em todos os grupos. Manter os 60 pipelines na avaliação principal controla o espaço de candidatos ao comparar metamodelos. A expansão completa requer nova matriz experimental.

\section{{Interpretação e Limitações}}\label{{sec:discussao}}

Kendall ponderado avalia concordância de pares com maior peso em itens bem ranqueados; NDCG acumula relevância com desconto logarítmico. Com ganho linear e acurácias próximas, NDCG pode ser alto mesmo com erros na ordenação: a baseline aleatória já alcança NDCG@5 de ${num(value('aleatorio','ndcg_5'))}$. Logo, NDCG elevado não demonstra acerto dos cinco melhores pipelines. O regret@1 explicita a perda da primeira recomendação, e NDCG@5 limita a avaliação às primeiras posições, complementando as correlações globais.

A representação de 55 descritores e nove zeros oferece uma referência simples e auditável, mas não isola por si só o efeito da representação. Comparar k-NN e Random Forest com o mesmo vetor evidencia dependência do metamodelo; demonstrar ganho da representação exigiria comparar outras representações sob o mesmo protocolo. Tampouco se convertem valores de trabalhos com tarefas diferentes em uma tabela de superioridade: referências recentes orientam o desenho experimental, não fornecem resultados diretamente comparáveis \cite{{Gijsbers2024,Baratchi2024}}.

Permanecem limitações de origem dos escores: a Fase 1 histórica imputou dados antes da CV e aplicou subamostragem global apenas à SVM. Sua reconstrução completa com tratamento dentro dos folds não foi executada. O agrupamento por nomes e atributos processados reduz repetições observáveis, mas pode não detectar parentes com nomes e transformações diferentes. Os testes são retrospectivos sobre um acervo previamente explorado, sem amostra externa coletada após o desenvolvimento. Os intervalos por bootstrap não incorporam toda a dependência decorrente de folds de treino sobrepostos. Essas restrições delimitam o alcance das conclusões e os requisitos de uma próxima coleta.
"""
    (ROOT/"capitulos/cap-experimentos-resultados.tex").write_text(chapter, encoding="utf-8")
    fields = {"arquivo_base_tasks": manifest["archive_tasks"], "tasks_elegiveis": manifest["n_tasks"],
              "grupos": manifest["n_groups"], "tau_knn": value("knn_ajustado", "kendall_ponderado"),
              "tau_rf": value("random_forest", "kendall_ponderado"),
              "falhas_knn": int(errors.correlacao_nao_positiva.sum()), "indefinidos": int(errors.correlacao_indefinida.sum()),
              "pilot_folds": 180, "source_manifest": "projeto/resultados/revisao_2026/manifesto.json"}
    (ROOT/"docs/revisao/CAMPOS_TCC.json").write_text(json.dumps(fields, indent=2), encoding="utf-8")
    print("Tabela, campos e capítulo de resultados atualizados.")


if __name__ == "__main__":
    run()
