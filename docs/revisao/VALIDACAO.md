# Validação da entrega — 26/10/2026

## Código e experimentos

- **10 testes automatizados aprovados**, incluindo alteração dos alvos do teste
  externo sem alteração das predições ou do k selecionado naquele fold;
  separação dos grupos; rejeição de classes raras; ganho linear em NDCG;
  ausência de imputação no alvo de teste; correlação indefinida visível;
  categoria desconhecida e mediana estimada somente no treino.
- Os hashes do código principal e dos quatro arquivos de entrada conferem com
  os manifestos das duas avaliações. Não há interseção de grupos entre treino
  e teste externo. Cada task aparece uma única vez no teste externo.
- Os hashes também foram conferidos nos blobs efetivamente preparados para o
  Git. `.gitattributes` preserva os bytes dos CSVs e fixa LF no código, evitando
  divergências artificiais de hash entre checkouts Windows e Linux. Os CSVs
  históricos não têm mudanças além da preservação de suas quebras de linha.
- Avaliação principal: **193 tasks**, **165 grupos**, seis métodos, **1.158 registros
  de métricas por task/modelo** e 69.480 pares real/predito por pipeline.
- Sensibilidade: **201 tasks**, **172 grupos**, seis métodos, 1.206 registros
  por task/modelo. Resultados e manifestos permanecem em diretório separado.
- Reprodução histórica: **20 médias verificadas**, correspondentes a quatro
  métricas para cinco valores de k, com diferença menor que **1e-7** frente aos
  CSVs originais. A comparação respeita a exclusão histórica de correlações
  indefinidas; NDCG desses casos continua registrado na nova saída detalhada.
- Piloto bruto: **180 avaliações bem-sucedidas**, em 36 combinações de tarefa,
  classificador e hiperparâmetro. Resultados, índices, tempos e avisos registrados.
- `git diff --check` sem erros de espaço em branco.

## Documentos

TCC compilado com **48 páginas** e artigo com **6 páginas**, usando cópias de
fontes e diretórios de auxiliares separados dos arquivos históricos. BibTeX e
índices/siglas foram processados. Os logs finais não têm citações ou referências
indefinidas nem erro fatal. O texto extraído dos dois PDFs não contém caracteres
de substituição ou referências `??`.

A página de resultados e a primeira página do artigo foram inspecionadas
visualmente. O rodapé fixo de 2025 do template foi substituído por indicação de
manuscrito revisado de outubro de 2026. Os PDFs são versões de trabalho; não houve
submissão a evento ou alteração de cadastro institucional.

Permanecem pequenos avisos de composição do LaTeX, sobretudo em pré-textuais e
lista de siglas. Os logs locais ficam em `build/latex/`, ignorado pelo Git.

## Limite da validação

As verificações acima confirmam execução e coerência dos artefatos. Não tornam
a matriz histórica equivalente a uma reconstrução completa da Fase 1 nem a CV
retrospectiva equivalente a um teste externo posterior. As pendências científicas
estão no [painel](PAINEL.md) e na [matriz de alterações](modificacoes.csv).
