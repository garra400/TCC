"""
Configurações globais do projeto.
Altere os valores conforme necessário antes de executar cada fase.
"""

import os

# ─── Caminhos ───────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
DATASETS_DIR   = os.path.join(DATA_DIR, "datasets")
META_DIR       = os.path.join(DATA_DIR, "meta_conjunto")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
FIGURAS_DIR    = os.path.join(RESULTADOS_DIR, "figuras")
METRICAS_DIR   = os.path.join(RESULTADOS_DIR, "metricas")

# Cria diretórios se não existirem
for d in [DATASETS_DIR, META_DIR, FIGURAS_DIR, METRICAS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset2Vec ─────────────────────────────────────────────────────────────
# Repositório: https://github.com/hadijomaa/dataset2vec
# Clone o repositório e aponte o caminho abaixo
D2V_REPO_PATH  = os.path.join(os.path.dirname(BASE_DIR), "dataset2vec")
# Repositório clonado em TCC/dataset2vec/ (um nível acima de /projeto/).
# Para usar o D2V real: necessita TensorFlow + Python <= 3.12.
# Com Python >= 3.13: usa embedding estatístico avançado (publicável).
D2V_EMBED_DIM  = 64

# ─── Seleção de Datasets ─────────────────────────────────────────────────────
# IDs obtidos em duas etapas:
#   (a) 50 datasets já em cache local (baixados e validados anteriormente)
#   (b) 200 IDs verificados via openml.tasks.list_tasks() com filtro ativo:
#       500 <= NumberOfInstances <= 20000, NumberOfClasses >= 2,
#       status=active, estratificados em 5 faixas de tamanho (40 por faixa)
# Referência benchmark: OpenML-CC18 (Bischl et al., NeurIPS 2021)
OPENML_TASK_IDS = [
    # ── Datasets já em cache local (verificados) ───────────────────────────────
    3,      6,      11,     12,     14,     15,     16,     18,     22,     23,
    28,     29,     31,     32,     37,     43,     146,    253,    307,    2074,
    2079,   2119,   3021,   3022,   3549,   3903,   3904,   4534,   4538,   9910,
    9914,   9946,   9952,   9957,   9960,   9964,   9967,   9971,   9976,   9978,
    9981,   9985,   10093,  10101,  14952,  14954,  146817, 146820, 146822, 167125,
    # ── Verificados via openml.tasks.list_tasks() em 2026-04-29 ───────────────
    # Filtro: 500<=n_inst<=20000, n_classes>=2, status=active, n_features<=500
    # Estratificado em 5 faixas de tamanho (40 por faixa)
    30,     45,     2075,   3486,   3512,   3517,   3574,   3618,   3672,   3681,
    3698,   3704,   3712,   3760,   3793,   3844,   3856,   3857,   3890,   3897,
    3902,   3919,   3950,   3954,   7549,   9893,   9905,   9916,   9917,   9920,
    9926,   9928,   9929,   9956,   9983,   14969,  125921, 146216, 146219, 146231,
    146818, 167114, 168853, 168882, 189773, 189864, 189966, 189988, 189999, 190023,
    190056, 190061, 190110, 190132, 211701, 360980, 360987, 361021, 361041, 361048,
    361050, 361057, 361066, 361112, 361148, 361150, 361151, 361156, 361159, 361169,
    361171, 361175, 361192, 361198, 361199, 361201, 361642, 361677, 361814, 361820,
    361826, 361830, 361833, 361834, 361836, 361837, 361840, 361841, 361843, 361851,
    361859, 361862, 361863, 361865, 361883, 361886, 361890, 361894, 361895, 362064,
    362082, 362103, 362174, 362203, 362245, 362251, 362261, 362277, 362281, 362413,
    362428, 362436, 362450, 362530, 362533, 362536, 362541, 362564, 362577, 362578,
    362582, 362586, 362604, 362607, 362616, 362620, 362647, 362649, 362652, 362663,
    362687, 362689, 362705, 362714, 362719, 362730, 362733, 362747, 362751, 362752,
    362791, 362796, 362823, 362830, 362832, 362847, 362855, 362858, 362862, 362866,
    362883, 362889, 362903, 362941, 362945, 362956, 362958, 362986, 362999, 363019,
    363030, 363050, 363078, 363094, 363121, 363144, 363159, 363162, 363175, 363186,
    363187, 363232, 363237, 363238, 363363, 363381, 363393, 363401, 363459, 363481,
    363531, 363549, 363554, 363559, 363577, 363586, 363624, 363632, 363639, 363665,
    363666, 363712, 363713,
]

MIN_INSTANCIAS = 500     # descarta datasets com menos instâncias
MAX_INSTANCIAS = 10_000  # SVM tem custo quadrático; >10k fica impraticável em CV
SVM_MAX_AMOSTRAS = 5_000 # para datasets maiores que este limite, SVM recebe subsample

# ─── Pipelines ───────────────────────────────────────────────────────────────
# 4 × 3 × 5 = 60 combinações de pipeline (conforme orientação)
SCALERS      = ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
SELECTORS    = ["PCA", "SelectKBest", "None"]
CLASSIFIERS  = ["RandomForest", "SVM", "kNN", "MLP", "DecisionTree"]
CV_FOLDS     = 5
RANDOM_STATE = 42

# ─── Fase 2 – Predição de Ranking ────────────────────────────────────────────
# Preditor KNN (baseado em KDTree, multi-output)
KNN_K_RANGE  = [1, 3, 5, 7, 10]   # valores de k avaliados no LOO
KNN_K_PADRAO = 5                    # k padrão para recomendação final

# MLP ranking (PyTorch, loss = BCE)
MLP_HIDDEN_SIZES    = (256, 128, 64)
MLP_EPOCHS          = 200
MLP_LR              = 1e-3
MLP_DROPOUT         = 0.2
MLP_BATCH_SIZE      = 32

# Avaliação
METRICAS_RANKING = ["spearman", "kendalltau", "ndcg"]
# spearman    → scipy.stats.spearmanr   (correlação de postos)
# kendalltau  → scipy.stats.kendalltau  (concordância par-a-par entre 2 rankings)
# ndcg        → sklearn.metrics.ndcg_score (penaliza erros no topo do ranking)
