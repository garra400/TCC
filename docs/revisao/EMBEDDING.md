# Contrato do vetor estatístico

`projeto/fase1_meta_conjunto/embedding_generator.py::_embedding_estatistico`
é a referência da representação histórica. Não há treino, loss, pesos aprendidos
nem otimização neural. O vetor exige um dataset **rotulado**, com atributos já
codificados e alvo inteiro; não é uma recomendação para dados sem rótulos.

Cada um dos nove descritores por coluna é agregado, nesta ordem, por mínimo,
percentil 25, média, percentil 75 e máximo. Portanto são 9 × 5 = 45 entradas:

| Índices Python | Descritor por atributo |
| --- | --- |
| 0–4 | Média (`nanmean`) |
| 5–9 | Desvio padrão populacional (`nanstd + 1e-8`) |
| 10–14 | Mínimo |
| 15–19 | Máximo |
| 20–24 | Mediana |
| 25–29 | Assimetria (`scipy.stats.skew`, padrões da biblioteca) |
| 30–34 | Curtose de Fisher (`scipy.stats.kurtosis`, padrões da biblioteca) |
| 35–39 | Correlação de Pearson com o alvo codificado |
| 40–44 | Correlação de Spearman com o alvo codificado |

Os percentis por atributo P25/P75 são calculados no código, mas **não entram**
nos nove blocos. A mediana não é um dos cinco agregadores; a média é.

| Índice | Propriedade global |
| --- | --- |
| 45 | `log1p(n_instancias)` |
| 46 | `log1p(n_atributos)` |
| 47 | Número de classes |
| 48 | Proporção da classe majoritária |
| 49 | Proporção da classe minoritária |
| 50 | Entropia das proporções de classe, com `1e-8` antes do log |
| 51 | Proporção de atributos com desvio padrão menor que `1e-8` |
| 52 | Média da matriz de correlação absoluta entre atributos, diagonal zerada |
| 53 | Média da proporção de valores únicos por atributo |
| 54 | `n_atributos / (n_instancias + 1e-8)` |
| 55–63 | Nove zeros |

O bloco 52 inclui os zeros da diagonal no denominador P². Correlações entre
atributos constantes podem contaminar essa média com NaN, posteriormente
convertido em zero. A escolha não será alterada silenciosamente ao reproduzir
o benchmark. Correlações com rótulos multiclasses codificados dependem da
codificação nominal arbitrária e limitam a interpretação.

Concatenação 45+10; truncamento apenas se o comprimento configurado for menor
que 55, ou padding se for maior. Com dimensão 64 não há truncamento.
Aplica-se `nan_to_num`, seguido de normalização L2; vetor nulo permanece nulo.
Os índices acima são anteriores à normalização, mas a ordem é preservada.
Os nove zeros não acrescentam informação. A ablação de padronização na revisão
aprende suas escalas somente no treino de cada fold.

O caminho público histórico tenta usar Dataset2Vec se houver checkpoint e
TensorFlow. Para reproduzir as matrizes publicadas, use os `.npy` versionados;
não misture silenciosamente embeddings neurais com a representação estatística.
