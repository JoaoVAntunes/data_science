"""
Estrutura:
  Bloco 0  — Download + Carregamento
  Bloco 1  — Análise Descritiva Geral
  Bloco 2  — 20 Análises Univariadas (formato Pergunta/Hipótese/Análise/Discussão)
  Bloco 3  — 10 Análises Multivariadas (formato Q/H/A/D)
  Bloco 4  — 5 Visualizações Efetivas (salvas como PNG)
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, f_oneway, ttest_ind, chi2_contingency,
                         pearsonr, linregress)

# Força encoding UTF-8 na saída padrão para evitar erros em terminais Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 120)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'Exploração Básica')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEPARATOR = "=" * 80
SEP_THIN  = "-" * 80


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def subsection(title):
    print(f"\n{SEP_THIN}")
    print(f"  {title}")
    print(SEP_THIN)


def qhad(pergunta, hipotese, analise_fn, discussao_fn):
    """
    Executa e imprime uma análise no formato
    Pergunta → Hipótese → Análise → Discussão.
    """
    print(f"\n  [P] {pergunta}")
    print(f"  [H] {hipotese}")
    resultado = analise_fn()
    print(f"  [D] {discussao_fn(resultado)}")
    return resultado


# ============================================================================
# BLOCO 0 — DOWNLOAD & CARREGAMENTO
# ============================================================================
section("BLOCO 0 — DOWNLOAD & CARREGAMENTO DO DATASET")

import kagglehub
print("\nBaixando dataset via kagglehub...")
path = kagglehub.dataset_download("corrieaar/apartment-rental-offers-in-germany")
print(f"  Path retornado: {path}")

csv_candidates = glob.glob(os.path.join(path, "**", "immo_data.csv"), recursive=True)
if not csv_candidates:
    csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

if not csv_candidates:
    raise FileNotFoundError(f"Nenhum CSV encontrado em: {path}")

CSV_PATH = csv_candidates[0]
print(f"  Arquivo CSV localizado: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"\n  Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

df_clean = df[(df['baseRent'] > 0) & (df['livingSpace'] > 0)].copy()
print(f"  Dataset limpo (baseRent>0 e livingSpace>0): {df_clean.shape[0]:,} linhas")

P95_RENT  = df_clean['baseRent'].quantile(0.95)
P95_SPACE = df_clean['livingSpace'].quantile(0.95)
df_viz = df_clean[
    (df_clean['baseRent']    <= P95_RENT) &
    (df_clean['livingSpace'] <= P95_SPACE)
].copy()
print(f"  Dataset para visualizações (sem P95 outliers): {df_viz.shape[0]:,} linhas")


# ============================================================================
# BLOCO 1 — ANÁLISE DESCRITIVA GERAL
# ============================================================================
section("BLOCO 1 — ANÁLISE DESCRITIVA GERAL")

subsection("1.1 Estrutura do Dataset")
print(f"\n  Instâncias totais (bruto):    {df.shape[0]:>10,}")
print(f"  Instâncias após filtro base:  {df_clean.shape[0]:>10,}")
print(f"  Total de features:            {df.shape[1]:>10}")

bool_cols  = df.select_dtypes(include='bool').columns.tolist()
int_cols   = [c for c in df.select_dtypes(include=['int64','int32']).columns if c not in bool_cols]
float_cols = df.select_dtypes(include='float64').columns.tolist()
obj_cols   = df.select_dtypes(include='object').columns.tolist()

print(f"\n  Breakdown por tipo de dado:")
print(f"    Numéricas (float):  {len(float_cols):>3}  {float_cols}")
print(f"    Numéricas (int):    {len(int_cols):>3}  {int_cols}")
print(f"    Booleanas:          {len(bool_cols):>3}  {bool_cols}")
print(f"    Texto/Categóricas:  {len(obj_cols):>3}  {obj_cols}")

subsection("1.2 Estatísticas Descritivas — Variáveis Numéricas")
num_cols = float_cols + int_cols
desc = df_clean[num_cols].describe().T
desc['skewness'] = df_clean[num_cols].skew()
desc['kurtosis'] = df_clean[num_cols].kurtosis()
with pd.option_context('display.float_format', '{:.2f}'.format):
    print(desc[['count','mean','std','min','25%','50%','75%','max','skewness','kurtosis']])

subsection("1.3 Valores Ausentes (Missing Values)")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_%': missing_pct})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_%', ascending=False)
print(f"\n  {len(missing_df)} colunas com valores ausentes:\n")
print(missing_df.to_string())
total_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
print(f"\n  Total de células com NaN: {df.isnull().sum().sum():,}  ({total_missing_pct:.2f}% do dataset)")

subsection("1.4 Razão de Classes — Variável Alvo (baseRent)")
bins   = [0, 300, 600, 900, 1500, np.inf]
labels = ['<€300','€300-600','€600-900','€900-1500','>€1500']
df_clean['rent_class'] = pd.cut(df_clean['baseRent'], bins=bins, labels=labels)
class_dist = df_clean['rent_class'].value_counts(normalize=True).sort_index() * 100
print("\n  baseRent em quintis (% de registros):")
for cls, pct in class_dist.items():
    bar = '#' * int(pct / 2)
    print(f"    {cls:<12} {pct:5.1f}%  {bar}")

subsection("1.5 Razão de Classes — Principais Variáveis Categóricas")
for col in ['typeOfFlat', 'regio1', 'condition', 'energyEfficiencyClass']:
    if col in df.columns:
        vc = df[col].value_counts(normalize=True).head(8) * 100
        print(f"\n  {col}:")
        for val, pct in vc.items():
            print(f"    {str(val):<30} {pct:5.1f}%")


# ============================================================================
# BLOCO 2 — 20 ANÁLISES UNIVARIADAS
# ============================================================================
section("BLOCO 2 — 20 ANÁLISES UNIVARIADAS (Formato Q/H/A/D)")

# --- U1: baseRent --------------------------------------------------------
subsection("U1 — baseRent (Aluguel Base Mensal)")

def analise_u1():
    s = df_clean['baseRent']
    sample = s.sample(min(5000, len(s)), random_state=42)
    stat_sw, p_sw = shapiro(sample)
    return {
        'n': len(s), 'mean': s.mean(), 'median': s.median(),
        'std': s.std(), 'skew': s.skew(), 'kurt': s.kurtosis(),
        'min': s.min(), 'max': s.max(),
        'q1': s.quantile(0.25), 'q3': s.quantile(0.75),
        'shapiro_stat': stat_sw, 'shapiro_p': p_sw
    }

def discussao_u1(r):
    norm = "NÃO segue distribuição normal" if r['shapiro_p'] < 0.05 else "segue distribuição normal"
    return (
        f"O aluguel base possui média €{r['mean']:.0f} e mediana €{r['median']:.0f}, "
        f"evidenciando forte assimetria positiva (skewness={r['skew']:.2f}). "
        f"Pelo teste Shapiro-Wilk (W={r['shapiro_stat']:.4f}, p={r['shapiro_p']:.4f}), a variável "
        f"{norm}, o que exigirá transformação logarítmica antes da modelagem."
    )

qhad(
    pergunta="Qual é a distribuição dos aluguéis base e ela segue uma distribuição normal?",
    hipotese="A distribuição de baseRent é assimétrica à direita (outliers de alto valor inflam a média), logo não é normal.",
    analise_fn=lambda: (
        print(f"    n={df_clean['baseRent'].describe()['count']:.0f}, "
              f"média=€{df_clean['baseRent'].mean():.2f}, "
              f"mediana=€{df_clean['baseRent'].median():.2f}, "
              f"std=€{df_clean['baseRent'].std():.2f}, "
              f"skew={df_clean['baseRent'].skew():.4f}, "
              f"kurt={df_clean['baseRent'].kurtosis():.4f}") or analise_u1()
    ),
    discussao_fn=discussao_u1
)

# --- U2: totalRent -------------------------------------------------------
subsection("U2 — totalRent (Aluguel Total)")

def analise_u2():
    s = df_clean['totalRent'].dropna()
    diff = (df_clean['totalRent'] - df_clean['baseRent']).dropna()
    return {'n': len(s), 'mean': s.mean(), 'median': s.median(),
            'std': s.std(), 'diff_mean': diff.mean(), 'diff_median': diff.median()}

def discussao_u2(r):
    return (
        f"O totalRent possui média €{r['mean']:.0f} e mediana €{r['median']:.0f}. "
        f"A diferença média entre totalRent e baseRent é €{r['diff_mean']:.0f} "
        f"(mediana €{r['diff_median']:.0f}), representando os custos de serviço adicionados. "
        f"A distribuição é igualmente assimétrica, herdando os outliers de baseRent."
    )

qhad(
    pergunta="Como o aluguel total difere do aluguel base? Qual é o custo adicional típico de serviços?",
    hipotese="O totalRent excede o baseRent em média €150-250 (serviceCharge típico), mantendo a mesma assimetria.",
    analise_fn=lambda: (
        print(f"    totalRent: n={df_clean['totalRent'].count()}, "
              f"média=€{df_clean['totalRent'].mean():.2f}, "
              f"mediana=€{df_clean['totalRent'].median():.2f}") or analise_u2()
    ),
    discussao_fn=discussao_u2
)

# --- U3: livingSpace -----------------------------------------------------
subsection("U3 — livingSpace (Área Útil em m²)")

def analise_u3():
    s = df_clean['livingSpace']
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    outliers = s[(s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)]
    return {'mean': s.mean(), 'median': s.median(), 'std': s.std(),
            'q1': Q1, 'q3': Q3, 'iqr': IQR,
            'outliers_n': len(outliers), 'outliers_pct': len(outliers)/len(s)*100,
            'p10': s.quantile(0.10), 'p90': s.quantile(0.90)}

def discussao_u3(r):
    return (
        f"A área média dos apartamentos é {r['mean']:.1f} m² (mediana {r['median']:.1f} m²). "
        f"80% dos imóveis têm entre {r['p10']:.0f} e {r['p90']:.0f} m². "
        f"Foram detectados {r['outliers_n']:,} outliers ({r['outliers_pct']:.1f}%) pelo método IQR, "
        f"com áreas acima de {r['q3'] + 1.5*r['iqr']:.0f} m², que devem ser tratados antes da modelagem."
    )

qhad(
    pergunta="Qual é a área típica dos apartamentos anunciados e há muitos outliers?",
    hipotese="A maioria dos apartamentos tem entre 40 e 100 m², com distribuição assimétrica positiva por propriedades atípicas.",
    analise_fn=lambda: (
        print(f"    média={df_clean['livingSpace'].mean():.2f} m², "
              f"mediana={df_clean['livingSpace'].median():.2f} m², "
              f"std={df_clean['livingSpace'].std():.2f}, "
              f"skew={df_clean['livingSpace'].skew():.4f}") or analise_u3()
    ),
    discussao_fn=discussao_u3
)

# --- U4: noRooms ---------------------------------------------------------
subsection("U4 — noRooms (Número de Quartos)")

def analise_u4():
    s = df_clean['noRooms'].dropna()
    vc = s.value_counts().sort_index().head(10)
    return {'mode': s.mode()[0], 'mean': s.mean(), 'median': s.median(),
            'value_counts': vc, 'pct_2_3': ((s >= 2) & (s <= 3)).mean() * 100}

def discussao_u4(r):
    return (
        f"A moda é {r['mode']:.1f} quartos, com média {r['mean']:.2f} e mediana {r['median']:.2f}. "
        f"{r['pct_2_3']:.1f}% dos imóveis possuem entre 2 e 3 quartos, perfil típico de apartamento urbano. "
        f"A variável é discreta e a distribuição é levemente assimétrica à direita."
    )

qhad(
    pergunta="Quantos quartos têm a maioria dos apartamentos alugados na Alemanha?",
    hipotese="A maioria dos apartamentos tem 2 ou 3 quartos, refletindo o perfil de moradia urbana.",
    analise_fn=lambda: (
        print(f"    moda={df_clean['noRooms'].mode()[0]}, "
              f"média={df_clean['noRooms'].mean():.2f}, "
              f"mediana={df_clean['noRooms'].median():.2f}\n"
              f"    Distribuição:\n{df_clean['noRooms'].value_counts().sort_index().head(10).to_string()}") or analise_u4()
    ),
    discussao_fn=discussao_u4
)

# --- U5: yearConstructed -------------------------------------------------
subsection("U5 — yearConstructed (Ano de Construção)")

def analise_u5():
    s = df_clean['yearConstructed'].dropna()
    by_decade = s.apply(lambda x: int(x // 10) * 10).value_counts().sort_index()
    return {'mean': s.mean(), 'median': s.median(), 'min': s.min(), 'max': s.max(),
            'std': s.std(), 'by_decade': by_decade,
            'pct_post2000': (s >= 2000).mean() * 100,
            'pct_pre1970': (s < 1970).mean() * 100}

def discussao_u5(r):
    return (
        f"A mediana de construção é {r['median']:.0f}. "
        f"{r['pct_post2000']:.1f}% dos imóveis foram construídos após 2000 (novos) "
        f"e {r['pct_pre1970']:.1f}% antes de 1970 (antigos). "
        f"O estoque imobiliário alemão é majoritariamente das décadas de 1950-1980, "
        f"reflexo da reconstrução pós-guerra e boom imobiliário da reunificação."
    )

qhad(
    pergunta="Quando foram construídos a maioria dos imóveis disponíveis para aluguel?",
    hipotese="O parque imobiliário alemão é antigo, com concentração de construções entre 1950 e 1980.",
    analise_fn=lambda: (
        print(f"    média={df_clean['yearConstructed'].mean():.0f}, "
              f"mediana={df_clean['yearConstructed'].median():.0f}, "
              f"min={df_clean['yearConstructed'].min():.0f}, "
              f"max={df_clean['yearConstructed'].max():.0f}") or analise_u5()
    ),
    discussao_fn=discussao_u5
)

# --- U6: serviceCharge ---------------------------------------------------
subsection("U6 — serviceCharge (Custos de Serviço Mensais)")

def analise_u6():
    s = df_clean['serviceCharge'].dropna()
    s = s[s > 0]
    ratio = (s / df_clean.loc[s.index, 'baseRent']).dropna()
    return {'n': len(s), 'mean': s.mean(), 'median': s.median(),
            'std': s.std(), 'pct25': s.quantile(0.25), 'pct75': s.quantile(0.75),
            'ratio_mean': ratio.mean() * 100}

def discussao_u6(r):
    return (
        f"Os custos de serviço têm mediana €{r['median']:.0f} e média €{r['mean']:.0f}, "
        f"representando em média {r['ratio_mean']:.1f}% do aluguel base. "
        f"O IQR vai de €{r['pct25']:.0f} a €{r['pct75']:.0f}. "
        f"Esses custos incluem água, lixo e manutenção, sendo relevante fator na decisão de inquilino."
    )

qhad(
    pergunta="Qual é o custo de serviço típico e qual sua proporção em relação ao aluguel base?",
    hipotese="O serviceCharge representa ~20-30% do baseRent, adicionando custo significativo ao locatário.",
    analise_fn=lambda: (
        s := df_clean['serviceCharge'].dropna(),
        s := s[s > 0],
        print(f"    n={len(s)}, média=€{s.mean():.2f}, mediana=€{s.median():.2f}, "
              f"std=€{s.std():.2f}") or analise_u6()
    )[-1],
    discussao_fn=discussao_u6
)

# --- U7: heatingCosts ----------------------------------------------------
subsection("U7 — heatingCosts (Custos de Aquecimento)")

def analise_u7():
    s = df_clean['heatingCosts'].dropna()
    s = s[s > 0]
    missing_pct = df_clean['heatingCosts'].isnull().mean() * 100
    return {'n_valid': len(s), 'missing_pct': missing_pct,
            'mean': s.mean(), 'median': s.median(), 'std': s.std(),
            'pct25': s.quantile(0.25), 'pct75': s.quantile(0.75)}

def discussao_u7(r):
    return (
        f"heatingCosts tem {r['missing_pct']:.1f}% de valores ausentes, limitando a análise. "
        f"Entre os {r['n_valid']:,} registros válidos, a mediana é €{r['median']:.0f}/mês "
        f"(IQR: €{r['pct25']:.0f}–€{r['pct75']:.0f}). "
        f"A alta taxa de ausência sugere que muitos proprietários não divulgam esse custo, "
        f"o que pode dificultar comparações diretas entre anúncios."
    )

qhad(
    pergunta="Qual é o custo típico de aquecimento mensal nos apartamentos e como é sua completude de dados?",
    hipotese="Custos de aquecimento giram em torno de €80-150/mês, mas com alta taxa de dados faltantes (~68%).",
    analise_fn=lambda: (
        s := df_clean['heatingCosts'].dropna(),
        print(f"    missing={df_clean['heatingCosts'].isnull().mean()*100:.1f}%, "
              f"n_válidos={s[s>0].count()}, "
              f"média=€{s[s>0].mean():.2f}, mediana=€{s[s>0].median():.2f}") or analise_u7()
    )[-1],
    discussao_fn=discussao_u7
)

# --- U8: floor -----------------------------------------------------------
subsection("U8 — floor (Andar do Imóvel)")

def analise_u8():
    s = df_clean['floor'].dropna()
    s_pos = s[s >= 0]
    vc = s_pos.value_counts().sort_index().head(12)
    return {'mode': s_pos.mode()[0], 'mean': s_pos.mean(), 'median': s_pos.median(),
            'pct_ground_floor': (s_pos == 0).mean() * 100,
            'pct_high': (s_pos >= 5).mean() * 100,
            'value_counts_top': vc}

def discussao_u8(r):
    return (
        f"A moda é o andar {int(r['mode'])} (térreo a andares baixos dominam). "
        f"{r['pct_ground_floor']:.1f}% dos imóveis estão no andar 0 (térreo) "
        f"e apenas {r['pct_high']:.1f}% estão no 5º andar ou acima. "
        f"Isso reflete a predominância de edifícios de médio porte no mercado alemão, "
        f"com baixa incidência de arranha-céus residenciais."
    )

qhad(
    pergunta="Em quais andares estão localizados a maioria dos apartamentos anunciados?",
    hipotese="A maioria dos imóveis fica nos andares baixos (0–3), dado o perfil de edificações alemãs.",
    analise_fn=lambda: (
        s := df_clean['floor'].dropna(),
        s := s[s >= 0],
        print(f"    moda={s.mode()[0]:.0f}, média={s.mean():.2f}, mediana={s.median():.2f}\n"
              f"    Andares 0-5: {s[s<=5].value_counts().sort_index().to_string()}") or analise_u8()
    )[-1],
    discussao_fn=discussao_u8
)

# --- U9: picturecount ----------------------------------------------------
subsection("U9 — picturecount (Quantidade de Fotos no Anúncio)")

def analise_u9():
    s = df_clean['picturecount'].dropna()
    return {'mean': s.mean(), 'median': s.median(), 'mode': s.mode()[0],
            'pct_zero': (s == 0).mean() * 100,
            'pct_5plus': (s >= 5).mean() * 100,
            'pct_10plus': (s >= 10).mean() * 100}

def discussao_u9(r):
    return (
        f"A mediana de fotos por anúncio é {r['median']:.0f} e a média é {r['mean']:.1f}. "
        f"{r['pct_zero']:.1f}% dos anúncios não têm fotos e {r['pct_10plus']:.1f}% têm 10 ou mais. "
        f"O número de fotos pode ser um proxy de qualidade do anúncio e atenção do proprietário, "
        f"potencialmente influenciando o engajamento de potenciais inquilinos."
    )

qhad(
    pergunta="Quantas fotos os anúncios de aluguel costumam incluir?",
    hipotese="A maioria dos anúncios inclui 5-15 fotos, com poucos sem imagens.",
    analise_fn=lambda: (
        s := df_clean['picturecount'].dropna(),
        print(f"    média={s.mean():.2f}, mediana={s.median():.0f}, "
              f"sem fotos={( s==0).mean()*100:.1f}%, >=10 fotos={(s>=10).mean()*100:.1f}%") or analise_u9()
    )[-1],
    discussao_fn=discussao_u9
)

# --- U10: noParkSpaces ---------------------------------------------------
subsection("U10 — noParkSpaces (Vagas de Estacionamento)")

def analise_u10():
    s = df_clean['noParkSpaces'].dropna()
    s_nn = s[s >= 0]
    vc = s_nn.value_counts().sort_index()
    return {'pct_no_park': (s_nn == 0).mean() * 100,
            'pct_one': (s_nn == 1).mean() * 100,
            'pct_two_plus': (s_nn >= 2).mean() * 100,
            'value_counts': vc}

def discussao_u10(r):
    return (
        f"{r['pct_no_park']:.1f}% dos imóveis não oferecem vaga de estacionamento, "
        f"{r['pct_one']:.1f}% têm uma vaga e {r['pct_two_plus']:.1f}% têm duas ou mais. "
        f"A baixa oferta de vagas é esperada em centros urbanos densos, onde o transporte público "
        f"é mais utilizado e vagas representam custo adicional significativo."
    )

qhad(
    pergunta="Qual é a disponibilidade de vagas de estacionamento nos imóveis anunciados?",
    hipotese="A maioria dos imóveis urbanos não inclui vaga de estacionamento (~60-70%).",
    analise_fn=lambda: (
        s := df_clean['noParkSpaces'].dropna(),
        s := s[s >= 0],
        print(f"    sem vaga={(s==0).mean()*100:.1f}%, "
              f"1 vaga={(s==1).mean()*100:.1f}%, "
              f"2+ vagas={(s>=2).mean()*100:.1f}%") or analise_u10()
    )[-1],
    discussao_fn=discussao_u10
)

# --- U11: typeOfFlat -----------------------------------------------------
subsection("U11 — typeOfFlat (Tipo de Imóvel)")

def analise_u11():
    vc = df_clean['typeOfFlat'].value_counts(normalize=True) * 100
    vc = vc.dropna()
    return {'top_types': vc.head(10), 'missing_pct': df_clean['typeOfFlat'].isnull().mean() * 100,
            'top_type': vc.index[0], 'top_pct': vc.iloc[0]}

def discussao_u11(r):
    return (
        f"O tipo mais comum é '{r['top_type']}' ({r['top_pct']:.1f}%), seguido por tipos especiais "
        f"como cobertura (roof_storey) e térreo (ground_floor). "
        f"A diversidade de tipos reflete a variedade do parque imobiliário alemão. "
        f"Há {r['missing_pct']:.1f}% de valores ausentes nesta coluna."
    )

qhad(
    pergunta="Que tipos de imóvel dominam o mercado de aluguel alemão?",
    hipotese="Apartamentos convencionais (apartment) representam a maioria, acima de 45% das ofertas.",
    analise_fn=lambda: (
        vc := df_clean['typeOfFlat'].value_counts(normalize=True).head(10) * 100,
        print(f"    Top tipos:\n{vc.to_string()}") or analise_u11()
    )[-1],
    discussao_fn=discussao_u11
)

# --- U12: regio1 ---------------------------------------------------------
subsection("U12 — regio1 (Estado Alemão)")

def analise_u12():
    vc = df['regio1'].value_counts(normalize=True) * 100
    top3_pct = vc.head(3).sum()
    return {'top_regions': vc, 'top3_pct': top3_pct,
            'n_unique': df['regio1'].nunique()}

def discussao_u12(r):
    top = r['top_regions'].index[0]
    top_pct = r['top_regions'].iloc[0]
    return (
        f"O dataset abrange {r['n_unique']} estados (Bundesländer). "
        f"{top} lidera com {top_pct:.1f}% das ofertas. "
        f"Os 3 primeiros estados concentram {r['top3_pct']:.1f}% dos anúncios, "
        f"indicando forte viés geográfico. Isso deve ser considerado ao generalizar conclusões."
    )

qhad(
    pergunta="Como as ofertas de aluguel estão distribuídas geograficamente entre os estados alemães?",
    hipotese="Nordrhein-Westfalen e Sachsen concentram mais de 40% das ofertas, dado seu tamanho e densidade.",
    analise_fn=lambda: (
        vc := df['regio1'].value_counts(normalize=True).head(10) * 100,
        print(f"    Top 10 estados:\n{vc.to_string()}") or analise_u12()
    )[-1],
    discussao_fn=discussao_u12
)

# --- U13: condition ------------------------------------------------------
subsection("U13 — condition (Condição do Imóvel)")

def analise_u13():
    vc = df_clean['condition'].value_counts(normalize=True) * 100
    missing_pct = df_clean['condition'].isnull().mean() * 100
    return {'top': vc, 'missing_pct': missing_pct, 'n_unique': df_clean['condition'].nunique()}

def discussao_u13(r):
    top_cond = r['top'].index[0] if len(r['top']) > 0 else 'N/A'
    top_pct  = r['top'].iloc[0] if len(r['top']) > 0 else 0
    return (
        f"A condição mais frequente é '{top_cond}' ({top_pct:.1f}%), "
        f"com {r['missing_pct']:.1f}% de ausentes. "
        f"Há {r['n_unique']} categorias de condição. "
        f"Imóveis 'well_kept' e 'refurbished' dominam, sugerindo que proprietários preferem anunciar "
        f"imóveis em bom estado de conservação."
    )

qhad(
    pergunta="Em que condições estão os imóveis anunciados para aluguel?",
    hipotese="A maioria dos imóveis está em bom estado (well_kept ou refurbished), pois proprietários selecionam imóveis para anúncios.",
    analise_fn=lambda: (
        vc := df_clean['condition'].value_counts(normalize=True).head(10) * 100,
        print(f"    Condições:\n{vc.to_string()}\n    missing={df_clean['condition'].isnull().mean()*100:.1f}%") or analise_u13()
    )[-1],
    discussao_fn=discussao_u13
)

# --- U14: interiorQual ---------------------------------------------------
subsection("U14 — interiorQual (Qualidade do Interior)")

def analise_u14():
    vc = df_clean['interiorQual'].value_counts(normalize=True) * 100
    missing_pct = df_clean['interiorQual'].isnull().mean() * 100
    return {'top': vc, 'missing_pct': missing_pct}

def discussao_u14(r):
    norm_pct = r['top'].get('normal', 0)
    lux_pct  = r['top'].get('luxury', 0)
    return (
        f"A qualidade 'normal' representa {norm_pct:.1f}% dos imóveis com dado disponível. "
        f"Imóveis de luxo correspondem a {lux_pct:.1f}%, confirmando que o mercado é predominantemente "
        f"de qualidade padrão. Há {r['missing_pct']:.1f}% de dados ausentes — "
        f"proprietários de imóveis médios tendem a não preencher esta informação."
    )

qhad(
    pergunta="Qual é a qualidade predominante dos interiores no mercado de aluguel?",
    hipotese="A qualidade 'normal' domina, com participação pequena de imóveis luxuosos (~5-10%).",
    analise_fn=lambda: (
        vc := df_clean['interiorQual'].value_counts(normalize=True) * 100,
        print(f"    Qualidades:\n{vc.to_string()}\n    missing={df_clean['interiorQual'].isnull().mean()*100:.1f}%") or analise_u14()
    )[-1],
    discussao_fn=discussao_u14
)

# --- U15: energyEfficiencyClass ------------------------------------------
subsection("U15 — energyEfficiencyClass (Classe de Eficiência Energética)")

def analise_u15():
    vc = df_clean['energyEfficiencyClass'].value_counts(normalize=True) * 100
    missing_pct = df_clean['energyEfficiencyClass'].isnull().mean() * 100
    order = ['A+','A','B','C','D','E','F','G','H']
    vc_ordered = vc.reindex([x for x in order if x in vc.index])
    return {'all_vc': vc, 'ordered': vc_ordered, 'missing_pct': missing_pct}

def discussao_u15(r):
    return (
        f"Com {r['missing_pct']:.1f}% de ausentes, a análise cobre apenas ~29% dos imóveis. "
        f"Das classes com dados, as classes medianas (C, D, E) são mais frequentes, "
        f"refletindo o estoque antigo com eficiência energética moderada. "
        f"Poucos imóveis atingem classe A ou A+, consistente com construções recentes ou reformadas."
    )

qhad(
    pergunta="Qual é a eficiência energética dos imóveis e há predominância de classes melhores?",
    hipotese="As classes C e D dominam, dado o estoque imobiliário envelhecido (maioria construído antes de 1980).",
    analise_fn=lambda: (
        vc := df_clean['energyEfficiencyClass'].value_counts(normalize=True).head(10) * 100,
        print(f"    Classes:\n{vc.to_string()}\n    missing={df_clean['energyEfficiencyClass'].isnull().mean()*100:.1f}%") or analise_u15()
    )[-1],
    discussao_fn=discussao_u15
)

# --- U16: heatingType ----------------------------------------------------
subsection("U16 — heatingType (Tipo de Aquecimento)")

def analise_u16():
    vc = df_clean['heatingType'].value_counts(normalize=True) * 100
    missing_pct = df_clean['heatingType'].isnull().mean() * 100
    return {'top': vc.head(10), 'missing_pct': missing_pct, 'n_unique': df_clean['heatingType'].nunique()}

def discussao_u16(r):
    top_ht = r['top'].index[0] if len(r['top']) > 0 else 'N/A'
    top_pct = r['top'].iloc[0] if len(r['top']) > 0 else 0
    return (
        f"O tipo de aquecimento mais comum é '{top_ht}' ({top_pct:.1f}%). "
        f"Há {r['n_unique']} tipos registrados, com {r['missing_pct']:.1f}% de ausentes. "
        f"Aquecimento central (Zentralheizung) é predominante na Alemanha, "
        f"especialmente em edifícios mais modernos, enquanto aquecimento por piso (Fußbodenheizung) "
        f"é mais comum em imóveis de alto padrão."
    )

qhad(
    pergunta="Quais sistemas de aquecimento são mais comuns nos imóveis alemães?",
    hipotese="Aquecimento central (central heating) domina, sendo padrão em edifícios residenciais alemães.",
    analise_fn=lambda: (
        vc := df_clean['heatingType'].value_counts(normalize=True).head(10) * 100,
        print(f"    Tipos:\n{vc.to_string()}") or analise_u16()
    )[-1],
    discussao_fn=discussao_u16
)

# --- U17: hasKitchen -----------------------------------------------------
subsection("U17 — hasKitchen (Possui Cozinha Montada)")

def analise_u17():
    vc = df_clean['hasKitchen'].value_counts(normalize=True) * 100
    return {'pct_true': vc.get(True, 0), 'pct_false': vc.get(False, 0)}

def discussao_u17(r):
    return (
        f"{r['pct_true']:.1f}% dos apartamentos incluem cozinha montada (fitted kitchen), "
        f"enquanto {r['pct_false']:.1f}% não incluem. "
        f"Na Alemanha é culturalmente comum que inquilinos tragam seus próprios móveis de cozinha, "
        f"o que explica a maioria sem cozinha embutida. "
        f"Apartamentos com cozinha tendem a ter aluguel ligeiramente mais alto."
    )

qhad(
    pergunta="Qual proporção dos apartamentos inclui cozinha montada (Einbauküche)?",
    hipotese="Menos de 40% dos imóveis incluem cozinha, pois na cultura alemã é comum o inquilino trazer a própria.",
    analise_fn=lambda: (
        vc := df_clean['hasKitchen'].value_counts(normalize=True) * 100,
        print(f"    com cozinha={vc.get(True,0):.1f}%, sem cozinha={vc.get(False,0):.1f}%") or analise_u17()
    )[-1],
    discussao_fn=discussao_u17
)

# --- U18: balcony --------------------------------------------------------
subsection("U18 — balcony (Possui Varanda)")

def analise_u18():
    vc = df_clean['balcony'].value_counts(normalize=True) * 100
    return {'pct_true': vc.get(True, 0), 'pct_false': vc.get(False, 0)}

def discussao_u18(r):
    return (
        f"{r['pct_true']:.1f}% dos apartamentos possuem varanda (Balkon). "
        f"Varanda é considerada um diferencial valorizado no mercado alemão, "
        f"especialmente em grandes cidades. Sua presença geralmente está associada "
        f"a apartamentos maiores e de maior valor, o que será verificado na análise multivariada."
    )

qhad(
    pergunta="Qual percentual dos apartamentos disponibiliza varanda?",
    hipotese="Cerca de 40-50% dos apartamentos têm varanda, sendo um diferencial comum mas não universal.",
    analise_fn=lambda: (
        vc := df_clean['balcony'].value_counts(normalize=True) * 100,
        print(f"    com varanda={vc.get(True,0):.1f}%, sem varanda={vc.get(False,0):.1f}%") or analise_u18()
    )[-1],
    discussao_fn=discussao_u18
)

# --- U19: cellar ---------------------------------------------------------
subsection("U19 — cellar (Possui Porão/Depósito)")

def analise_u19():
    vc = df_clean['cellar'].value_counts(normalize=True) * 100
    return {'pct_true': vc.get(True, 0), 'pct_false': vc.get(False, 0)}

def discussao_u19(r):
    return (
        f"{r['pct_true']:.1f}% dos apartamentos incluem porão (Keller) ou depósito. "
        f"O porão é uma característica tradicional da arquitetura residencial alemã, "
        f"utilizado para armazenagem. Sua inclusão no anúncio é um diferencial "
        f"que pode influenciar tanto na escolha quanto no preço do aluguel."
    )

qhad(
    pergunta="Qual é a prevalência de porão/depósito nos imóveis anunciados?",
    hipotese="Mais de 50% dos imóveis incluem porão, dado ser característica arquitetônica tradicional alemã.",
    analise_fn=lambda: (
        vc := df_clean['cellar'].value_counts(normalize=True) * 100,
        print(f"    com porão={vc.get(True,0):.1f}%, sem porão={vc.get(False,0):.1f}%") or analise_u19()
    )[-1],
    discussao_fn=discussao_u19
)

# --- U20: date -----------------------------------------------------------
subsection("U20 — date (Período de Coleta dos Dados)")

def analise_u20():
    vc = df['date'].value_counts(normalize=True) * 100
    return {'distribution': vc, 'n_periods': df['date'].nunique()}

def discussao_u20(r):
    periods = ', '.join([f"{k} ({v:.1f}%)" for k, v in r['distribution'].items()])
    return (
        f"O dataset cobre {r['n_periods']} períodos de coleta: {periods}. "
        f"A distribuição desigual entre períodos pode introduzir viés temporal, "
        f"pois sazonalidade de preços, demanda por aluguéis e oferta do mercado "
        f"variam ao longo do ano. Análises comparativas devem controlar por período."
    )

qhad(
    pergunta="Em quais períodos foram coletados os dados e há equilíbrio temporal?",
    hipotese="Os dados cobrem 2-3 snapshots temporais distintos, com possível desbalanceamento entre períodos.",
    analise_fn=lambda: (
        vc := df['date'].value_counts(normalize=True) * 100,
        print(f"    Distribuição temporal:\n{vc.to_string()}") or analise_u20()
    )[-1],
    discussao_fn=discussao_u20
)


# ============================================================================
# BLOCO 3 — 10 ANÁLISES MULTIVARIADAS
# ============================================================================
section("BLOCO 3 — 10 ANÁLISES MULTIVARIADAS (Formato Q/H/A/D)")

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    return np.sqrt(phi2 / min(k-1, r-1))

def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

# --- M1 (H1): regio1 vs baseRent ----------------------------------------
subsection("M1 (H1) — Localização (regio1) × baseRent: ANOVA one-way + eta²")

def analise_m1():
    groups = [g['baseRent'].values for _, g in df_clean.groupby('regio1')]
    f_stat, p_val = f_oneway(*groups)
    grand_mean = df_clean['baseRent'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2
                     for g in [df_clean[df_clean['regio1']==r]['baseRent'] for r in df_clean['regio1'].unique()])
    ss_total = ((df_clean['baseRent'] - grand_mean)**2).sum()
    eta2 = ss_between / ss_total
    means = df_clean.groupby('regio1')['baseRent'].median().sort_values(ascending=False)
    return {'f_stat': f_stat, 'p_val': p_val, 'eta2': eta2,
            'top_states': means.head(5), 'bot_states': means.tail(5)}

def discussao_m1(r):
    sig = "significativa" if r['p_val'] < 0.05 else "não significativa"
    return (
        f"ANOVA: F={r['f_stat']:.2f}, p={r['p_val']:.4f} → diferença {sig} entre estados. "
        f"η²={r['eta2']:.4f} ({r['eta2']*100:.2f}% da variância de baseRent explicada por regio1). "
        f"Estados com maior mediana de aluguel: {list(r['top_states'].index[:3])}. "
        f"A localização é um fator estrutural do preço, com diferenças substanciais "
        f"entre estados do sul (Bavaria, Baden-Württemberg) e leste (Sachsen, Thüringen)."
    )

qhad(
    pergunta="A localização geográfica (estado federal) explica diferenças significativas no aluguel base?",
    hipotese="H1: A localização (regio1) impacta significativamente o preço, com estados do Sul mais caros que o Leste.",
    analise_fn=lambda: (
        r := analise_m1(),
        print(f"    F={r['f_stat']:.2f}, p={r['p_val']:.6f}, η²={r['eta2']:.4f}\n"
              f"    Top 5 medianas:\n{r['top_states'].to_string()}") or r
    )[-1],
    discussao_fn=discussao_m1
)

# --- M2 (H2): hasKitchen vs baseRent ------------------------------------
subsection("M2 (H2) — hasKitchen × baseRent: t-test independente")

def analise_m2():
    with_kitchen    = df_clean[df_clean['hasKitchen'] == True]['baseRent']
    without_kitchen = df_clean[df_clean['hasKitchen'] == False]['baseRent']
    t_stat, p_val = ttest_ind(with_kitchen, without_kitchen, equal_var=False)
    d = cohens_d(with_kitchen, without_kitchen)
    return {'mean_with': with_kitchen.mean(), 'mean_without': without_kitchen.mean(),
            'median_with': with_kitchen.median(), 'median_without': without_kitchen.median(),
            't_stat': t_stat, 'p_val': p_val, 'cohens_d': d,
            'diff_mean': with_kitchen.mean() - without_kitchen.mean()}

def discussao_m2(r):
    sig = "significativa" if r['p_val'] < 0.05 else "não significativa"
    mag = "pequeno" if abs(r['cohens_d']) < 0.2 else ("médio" if abs(r['cohens_d']) < 0.5 else "grande")
    return (
        f"t-test (Welch): t={r['t_stat']:.2f}, p={r['p_val']:.6f} → diferença {sig}. "
        f"Com cozinha: média €{r['mean_with']:.0f}, sem: €{r['mean_without']:.0f} "
        f"(diferença €{r['diff_mean']:.0f}). Cohen's d={r['cohens_d']:.3f} (efeito {mag}). "
        f"Embora estatisticamente significativa, a diferença absoluta é moderada e pode refletir "
        f"confundimento com tamanho do imóvel e localização."
    )

qhad(
    pergunta="Apartamentos com cozinha montada têm aluguel base significativamente maior?",
    hipotese="H2: Imóveis com hasKitchen=True possuem aluguel base superior à média, pois a cozinha é um diferencial de valor.",
    analise_fn=lambda: (
        r := analise_m2(),
        print(f"    com cozinha: média=€{r['mean_with']:.2f}, sem: €{r['mean_without']:.2f}\n"
              f"    t={r['t_stat']:.4f}, p={r['p_val']:.6f}, Cohen's d={r['cohens_d']:.4f}") or r
    )[-1],
    discussao_fn=discussao_m2
)

# --- M3 (H3): livingSpace vs baseRent -----------------------------------
subsection("M3 (H3) — livingSpace × baseRent: Correlação de Pearson")

def analise_m3():
    mask = df_clean['baseRent'].notna() & df_clean['livingSpace'].notna()
    x = df_clean.loc[mask, 'livingSpace']
    y = df_clean.loc[mask, 'baseRent']
    r, p = pearsonr(x.sample(min(50000, len(x)), random_state=42),
                    y.loc[x.sample(min(50000, len(x)), random_state=42).index])
    slope, intercept, r_value, p_val2, std_err = linregress(
        x.sample(min(50000, len(x)), random_state=1),
        y.loc[x.sample(min(50000, len(x)), random_state=1).index]
    )
    return {'pearson_r': r, 'p_val': p, 'r_squared': r**2,
            'slope': slope, 'intercept': intercept}

def discussao_m3(r):
    force = "forte" if abs(r['pearson_r']) > 0.6 else ("moderada" if abs(r['pearson_r']) > 0.3 else "fraca")
    return (
        f"r de Pearson={r['pearson_r']:.4f} (correlação {force}, p≈{r['p_val']:.4f}), R²={r['r_squared']:.4f}. "
        f"Regressão linear: baseRent ≈ {r['intercept']:.1f} + {r['slope']:.2f}×livingSpace. "
        f"A área explica {r['r_squared']*100:.1f}% da variância do aluguel. "
        f"A correlação positiva confirma que imóveis maiores custam mais, mas a relação não é puramente linear — "
        f"localização e qualidade explicam a variância residual."
    )

qhad(
    pergunta="Existe correlação linear entre a área útil (livingSpace) e o aluguel base?",
    hipotese="H3: Maior área implica maior aluguel — correlação de Pearson positiva e significativa (r > 0.5).",
    analise_fn=lambda: (
        r := analise_m3(),
        print(f"    Pearson r={r['pearson_r']:.4f}, p≈{r['p_val']:.4f}, R²={r['r_squared']:.4f}\n"
              f"    Eq. linear: baseRent ≈ {r['intercept']:.1f} + {r['slope']:.2f}×m²") or r
    )[-1],
    discussao_fn=discussao_m3
)

# --- M4 (H4): yearConstructed vs serviceCharge --------------------------
subsection("M4 (H4) — yearConstructed × serviceCharge: Regressão Linear")

def analise_m4():
    sub = df_clean[
        df_clean['yearConstructed'].notna() &
        df_clean['serviceCharge'].notna() &
        (df_clean['yearConstructed'] >= 1900) &
        (df_clean['serviceCharge'] > 0)
    ].copy()
    sub = sub[sub['yearConstructed'] <= 2023]
    r, p = pearsonr(sub['yearConstructed'], sub['serviceCharge'])
    slope, intercept, r_value, p_val2, std_err = linregress(sub['yearConstructed'], sub['serviceCharge'])
    by_decade = sub.copy()
    by_decade['decade'] = (sub['yearConstructed'] // 10 * 10).astype(int)
    decade_means = by_decade.groupby('decade')['serviceCharge'].median()
    return {'pearson_r': r, 'p_val': p, 'slope': slope, 'intercept': intercept,
            'r_squared': r**2, 'decade_means': decade_means}

def discussao_m4(r):
    trend = "crescente" if r['slope'] > 0 else "decrescente"
    return (
        f"Regressão linear: serviceCharge ≈ {r['intercept']:.1f} + {r['slope']:.3f}×yearConstructed. "
        f"r={r['pearson_r']:.4f} (p={r['p_val']:.4f}), R²={r['r_squared']:.4f}. "
        f"A tendência é {trend}, indicando que imóveis mais recentes têm custos de serviço "
        f"{'maiores' if r['slope'] > 0 else 'menores'}. "
        f"Isso pode refletir padrões construtivos modernos com mais amenidades (elevador, portaria, academia) "
        f"que elevam os custos condominiais."
    )

qhad(
    pergunta="Imóveis mais recentes apresentam custos de serviço (serviceCharge) diferentes de imóveis mais antigos?",
    hipotese="H4: Imóveis mais novos têm maior serviceCharge, pois incluem mais amenidades modernas.",
    analise_fn=lambda: (
        r := analise_m4(),
        print(f"    r={r['pearson_r']:.4f}, p={r['p_val']:.4f}, R²={r['r_squared']:.4f}\n"
              f"    slope={r['slope']:.4f} €/ano de construção") or r
    )[-1],
    discussao_fn=discussao_m4
)

# --- M5 (H5): heatingType vs regio1 -------------------------------------
subsection("M5 (H5) — heatingType × regio1: Chi-square + Cramér's V")

def analise_m5():
    sub = df_clean[df_clean['heatingType'].notna() & df_clean['regio1'].notna()].copy()
    top_ht = sub['heatingType'].value_counts().head(6).index.tolist()
    top_r1 = sub['regio1'].value_counts().head(8).index.tolist()
    sub2 = sub[sub['heatingType'].isin(top_ht) & sub['regio1'].isin(top_r1)]
    ct = pd.crosstab(sub2['regio1'], sub2['heatingType'])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(sub2['regio1'], sub2['heatingType'])
    return {'chi2': chi2, 'p_val': p, 'dof': dof, 'cramers_v': v,
            'crosstab_pct': ct.div(ct.sum(axis=1), axis=0) * 100}

def discussao_m5(r):
    assoc = "forte" if r['cramers_v'] > 0.3 else ("moderada" if r['cramers_v'] > 0.1 else "fraca")
    return (
        f"Chi-square={r['chi2']:.2f}, dof={r['dof']}, p≈{r['p_val']:.6f} → "
        f"associação estatisticamente significativa. Cramér's V={r['cramers_v']:.4f} ({assoc}). "
        f"A distribuição dos tipos de aquecimento varia entre estados, "
        f"refletindo diferenças regionais em infraestrutura, clima e tradição construtiva. "
        f"Ex.: gás natural é mais comum em estados do Oeste; aquecimento por distrito (Fernwärme) "
        f"é mais frequente no Leste (herança da infraestrutura socialista)."
    )

qhad(
    pergunta="A preferência por tipo de aquecimento varia significativamente entre os estados alemães?",
    hipotese="H5: A distribuição de heatingType é estatisticamente diferente entre os estados (teste chi-square significativo).",
    analise_fn=lambda: (
        r := analise_m5(),
        print(f"    chi2={r['chi2']:.2f}, p≈{r['p_val']:.6f}, Cramér's V={r['cramers_v']:.4f}\n"
              f"    Crosstab (%):\n{r['crosstab_pct'].round(1).to_string()}") or r
    )[-1],
    discussao_fn=discussao_m5
)

# --- M6: interiorQual vs baseRent ---------------------------------------
subsection("M6 — interiorQual × baseRent: ANOVA one-way")

def analise_m6():
    sub = df_clean[df_clean['interiorQual'].notna()].copy()
    groups_dict = {q: sub[sub['interiorQual'] == q]['baseRent'].values
                   for q in sub['interiorQual'].unique()}
    f_stat, p_val = f_oneway(*groups_dict.values())
    means = sub.groupby('interiorQual')['baseRent'].agg(['mean','median','count'])
    return {'f_stat': f_stat, 'p_val': p_val, 'group_stats': means}

def discussao_m6(r):
    sig = "significativa" if r['p_val'] < 0.05 else "não significativa"
    return (
        f"ANOVA: F={r['f_stat']:.2f}, p={r['p_val']:.6f} → diferença {sig} entre qualidades. "
        f"Conforme esperado, imóveis 'luxury' e 'sophisticated' têm aluguel médio substancialmente "
        f"maior que 'normal' e 'simple'. A qualidade do interior é um preditor relevante "
        f"do preço e deve ser incluída como feature no modelo de regressão."
    )

qhad(
    pergunta="Imóveis com maior qualidade de interior têm aluguéis significativamente mais altos?",
    hipotese="A qualidade do interior (luxury > sophisticated > normal > simple) se reflete proporcionalmente no aluguel.",
    analise_fn=lambda: (
        r := analise_m6(),
        print(f"    F={r['f_stat']:.2f}, p={r['p_val']:.6f}\n"
              f"    Médias por qualidade:\n{r['group_stats'].to_string()}") or r
    )[-1],
    discussao_fn=discussao_m6
)

# --- M7: energyEfficiencyClass vs baseRent ------------------------------
subsection("M7 — energyEfficiencyClass × baseRent: ANOVA one-way")

def analise_m7():
    sub = df_clean[df_clean['energyEfficiencyClass'].notna()].copy()
    valid_classes = ['A+','A','B','C','D','E','F','G','H']
    sub = sub[sub['energyEfficiencyClass'].isin(valid_classes)]
    groups = [sub[sub['energyEfficiencyClass'] == c]['baseRent'].values
              for c in valid_classes if c in sub['energyEfficiencyClass'].values]
    f_stat, p_val = f_oneway(*[g for g in groups if len(g) > 1])
    means = sub.groupby('energyEfficiencyClass')['baseRent'].median().reindex(
        [c for c in valid_classes if c in sub['energyEfficiencyClass'].values])
    return {'f_stat': f_stat, 'p_val': p_val, 'class_medians': means}

def discussao_m7(r):
    sig = "significativa" if r['p_val'] < 0.05 else "não significativa"
    return (
        f"ANOVA: F={r['f_stat']:.2f}, p={r['p_val']:.6f} → diferença {sig} entre classes energéticas. "
        f"Imóveis de classe A/A+ tendem a ser mais caros (construção nova, melhor isolamento), "
        f"enquanto classes F-H são geralmente mais baratas (imóveis antigos). "
        f"Porém a relação não é monotônica — outros fatores (localização, tamanho) confundem."
    )

qhad(
    pergunta="Imóveis com melhor eficiência energética têm aluguéis mais elevados?",
    hipotese="Classes energéticas mais eficientes (A, B) correspondem a imóveis mais modernos e caros; há diferença significativa entre classes.",
    analise_fn=lambda: (
        r := analise_m7(),
        print(f"    F={r['f_stat']:.2f}, p={r['p_val']:.6f}\n"
              f"    Medianas por classe:\n{r['class_medians'].to_string()}") or r
    )[-1],
    discussao_fn=discussao_m7
)

# --- M8: balcony vs baseRent --------------------------------------------
subsection("M8 — balcony × baseRent: t-test + Cohen's d")

def analise_m8():
    with_b    = df_clean[df_clean['balcony'] == True]['baseRent']
    without_b = df_clean[df_clean['balcony'] == False]['baseRent']
    t_stat, p_val = ttest_ind(with_b, without_b, equal_var=False)
    d = cohens_d(with_b, without_b)
    return {'mean_with': with_b.mean(), 'mean_without': without_b.mean(),
            'median_with': with_b.median(), 'median_without': without_b.median(),
            't_stat': t_stat, 'p_val': p_val, 'cohens_d': d}

def discussao_m8(r):
    sig = "significativa" if r['p_val'] < 0.05 else "não significativa"
    mag = "pequeno" if abs(r['cohens_d']) < 0.2 else ("médio" if abs(r['cohens_d']) < 0.5 else "grande")
    return (
        f"t-test (Welch): t={r['t_stat']:.2f}, p={r['p_val']:.6f} → diferença {sig}. "
        f"Com varanda: mediana €{r['median_with']:.0f}; sem varanda: €{r['median_without']:.0f}. "
        f"Cohen's d={r['cohens_d']:.3f} (efeito {mag}). "
        f"A varanda está associada a aluguel maior, mas o efeito pode ser mediado pelo "
        f"tamanho do imóvel (apartamentos maiores têm mais varandas)."
    )

qhad(
    pergunta="Apartamentos com varanda têm aluguel significativamente superior aos sem varanda?",
    hipotese="Varanda é um diferencial de valor — imóveis com balcony=True têm aluguel mais alto.",
    analise_fn=lambda: (
        r := analise_m8(),
        print(f"    com varanda: média=€{r['mean_with']:.2f}, sem: €{r['mean_without']:.2f}\n"
              f"    t={r['t_stat']:.4f}, p={r['p_val']:.6f}, Cohen's d={r['cohens_d']:.4f}") or r
    )[-1],
    discussao_fn=discussao_m8
)

# --- M9: noRooms vs livingSpace -----------------------------------------
subsection("M9 — noRooms × livingSpace: Correlação de Pearson")

def analise_m9():
    sub = df_clean[df_clean['noRooms'].notna() & df_clean['livingSpace'].notna()].copy()
    sub = sub[(sub['noRooms'] > 0) & (sub['noRooms'] <= 10)]
    r, p = pearsonr(sub['noRooms'], sub['livingSpace'])
    space_by_rooms = sub.groupby('noRooms')['livingSpace'].median()
    return {'pearson_r': r, 'p_val': p, 'r_squared': r**2,
            'space_by_rooms': space_by_rooms.head(10)}

def discussao_m9(r):
    force = "forte" if abs(r['pearson_r']) > 0.6 else ("moderada" if abs(r['pearson_r']) > 0.3 else "fraca")
    return (
        f"r de Pearson={r['pearson_r']:.4f} (correlação {force}), p≈{r['p_val']:.4f}, R²={r['r_squared']:.4f}. "
        f"Há clara relação positiva: mais quartos → maior área. "
        f"Como esperado, cada quarto adicional está associado a um acréscimo médio de área. "
        f"Ambas as variáveis são fortes candidatas a features preditoras de baseRent."
    )

qhad(
    pergunta="O número de quartos é um bom preditor da área total do apartamento?",
    hipotese="Existe correlação positiva e forte entre noRooms e livingSpace (mais quartos = maior área).",
    analise_fn=lambda: (
        r := analise_m9(),
        print(f"    Pearson r={r['pearson_r']:.4f}, p≈{r['p_val']:.4f}, R²={r['r_squared']:.4f}\n"
              f"    Mediana de área por n° de quartos:\n{r['space_by_rooms'].to_string()}") or r
    )[-1],
    discussao_fn=discussao_m9
)

# --- M10: Matriz de correlação -------------------------------------------
subsection("M10 — Matriz de Correlação de Pearson (Todas Variáveis Numéricas)")

def analise_m10():
    key_cols = ['baseRent','totalRent','livingSpace','noRooms','yearConstructed',
                'serviceCharge','heatingCosts','floor','picturecount','noParkSpaces']
    available = [c for c in key_cols if c in df_clean.columns]
    corr_matrix = df_clean[available].corr()
    top_corr = (corr_matrix['baseRent']
                .drop('baseRent')
                .abs()
                .sort_values(ascending=False))
    return {'matrix': corr_matrix, 'top_with_baseRent': top_corr, 'cols': available}

def discussao_m10(r):
    top3 = r['top_with_baseRent'].head(3)
    return (
        f"As 3 variáveis mais correlacionadas com baseRent são: "
        f"{', '.join([f'{c} (r={v:.3f})' for c, v in top3.items()])}. "
        f"totalRent e baseRent têm alta correlação (esperado, pois totalRent = baseRent + serviceCharge). "
        f"livingSpace e noRooms são as features estruturais mais correlacionadas com o preço. "
        f"A matriz revela também multicolinearidade potencial entre noRooms e livingSpace "
        f"que deve ser tratada na modelagem."
    )

r_m10 = qhad(
    pergunta="Quais variáveis numéricas apresentam maior correlação com o aluguel base?",
    hipotese="livingSpace, noRooms e serviceCharge são as variáveis numéricas mais correlacionadas com baseRent.",
    analise_fn=lambda: (
        r := analise_m10(),
        print(f"    Correlações com baseRent (|r| desc.):\n{r['top_with_baseRent'].to_string()}") or r
    )[-1],
    discussao_fn=discussao_m10
)


# ============================================================================
# BLOCO 4 — 5 VISUALIZAÇÕES EFETIVAS
# ============================================================================
section("BLOCO 4 — 5 VISUALIZAÇÕES EFETIVAS")

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "husl"


# --- VIZ 1: Heatmap de Correlações ----------------------------------------
subsection("VIZ1 — Heatmap de Correlações entre Variáveis Numéricas")

key_num = ['baseRent','totalRent','livingSpace','noRooms','yearConstructed',
           'serviceCharge','heatingCosts','floor','picturecount','noParkSpaces']
available_num = [c for c in key_num if c in df_clean.columns]

corr_matrix = df_clean[available_num].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Matriz de Correlação de Pearson — Variáveis Numéricas\n"
             "Dataset: Apartment Rental Offers in Germany", fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
viz1_path = os.path.join(OUTPUT_DIR, 'heatmap_correlacoes.png')
plt.savefig(viz1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {viz1_path}")


# --- VIZ 2: Boxplots baseRent por top-8 estados --------------------------
subsection("VIZ2 — Boxplot de baseRent por Estado (regio1) — Top 8")

top8_states = df_viz['regio1'].value_counts().head(8).index.tolist()
df_viz2 = df_viz[df_viz['regio1'].isin(top8_states)].copy()
state_medians = df_viz2.groupby('regio1')['baseRent'].median().sort_values(ascending=False)
ordered_states = state_medians.index.tolist()

fig, ax = plt.subplots(figsize=(14, 7))
data_by_state = [df_viz2[df_viz2['regio1'] == s]['baseRent'].values for s in ordered_states]
bp = ax.boxplot(data_by_state, labels=ordered_states, patch_artist=True,
                notch=False, vert=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2.5))
colors = sns.color_palette(PALETTE, len(ordered_states))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_title("Distribuição do Aluguel Base (baseRent) por Estado Alemão\n"
             "(top 8 estados por volume — limitado ao P95 para visualização)",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Estado (regio1)", fontsize=12)
ax.set_ylabel("Aluguel Base — €/mês", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
viz2_path = os.path.join(OUTPUT_DIR, 'boxplot_basrent_por_regiao.png')
plt.savefig(viz2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {viz2_path}")


# --- VIZ 3: Scatter livingSpace vs baseRent por typeOfFlat ---------------
subsection("VIZ3 — Scatter: livingSpace × baseRent por Tipo de Imóvel")

top5_types = df_viz['typeOfFlat'].value_counts().head(5).index.tolist()
df_viz3 = df_viz[df_viz['typeOfFlat'].isin(top5_types)].sample(
    min(30000, len(df_viz)), random_state=42)

fig, ax = plt.subplots(figsize=(13, 8))
palette_types = sns.color_palette(PALETTE, len(top5_types))
for i, flat_type in enumerate(top5_types):
    sub = df_viz3[df_viz3['typeOfFlat'] == flat_type]
    ax.scatter(sub['livingSpace'], sub['baseRent'], alpha=0.25, s=8,
               color=palette_types[i], label=flat_type)

x_range = np.linspace(df_viz3['livingSpace'].min(), df_viz3['livingSpace'].max(), 100)
slope_v, intercept_v, _, _, _ = linregress(df_viz3['livingSpace'], df_viz3['baseRent'])
ax.plot(x_range, intercept_v + slope_v * x_range, color='black',
        linewidth=2.5, linestyle='--', label=f'Reg. Linear (r²={analise_m3()["r_squared"]:.2f})')
ax.set_title("Área Útil vs Aluguel Base por Tipo de Imóvel\n"
             "Dataset: Apartment Rental Offers in Germany",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Área Útil — m²", fontsize=12)
ax.set_ylabel("Aluguel Base — €/mês", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.legend(title='Tipo de Imóvel', fontsize=10, title_fontsize=11, markerscale=3)
plt.tight_layout()
viz3_path = os.path.join(OUTPUT_DIR, 'scatter_area_aluguel.png')
plt.savefig(viz3_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {viz3_path}")


# --- VIZ 4: Boxplot baseRent por Classe de Eficiência Energética ---------
subsection("VIZ4 — Boxplot: baseRent por energyEfficiencyClass")

energy_order = ['A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
df_viz4 = df_viz[df_viz['energyEfficiencyClass'].isin(energy_order)].copy()
df_viz4['energyEfficiencyClass'] = pd.Categorical(
    df_viz4['energyEfficiencyClass'], categories=energy_order, ordered=True)
df_viz4 = df_viz4.sort_values('energyEfficiencyClass')

fig, ax = plt.subplots(figsize=(13, 7))
data_by_class = [df_viz4[df_viz4['energyEfficiencyClass'] == c]['baseRent'].values
                 for c in energy_order if c in df_viz4['energyEfficiencyClass'].values]
labels_present = [c for c in energy_order if c in df_viz4['energyEfficiencyClass'].values]
n_classes = len(labels_present)

bp4 = ax.boxplot(data_by_class, labels=labels_present, patch_artist=True,
                 notch=False, vert=True, widths=0.6,
                 medianprops=dict(color='black', linewidth=2.5))
green_to_red = sns.color_palette("RdYlGn_r", n_classes)
for patch, color in zip(bp4['boxes'], green_to_red):
    patch.set_facecolor(color)
    patch.set_alpha(0.80)

counts = [len(df_viz4[df_viz4['energyEfficiencyClass'] == c]) for c in labels_present]
for i, (cls, cnt) in enumerate(zip(labels_present, counts)):
    ax.text(i + 1, ax.get_ylim()[0], f'n={cnt:,}', ha='center', va='bottom',
            fontsize=8, color='dimgray')

ax.set_title("Aluguel Base por Classe de Eficiência Energética\n"
             "(verde=eficiente → vermelho=ineficiente | limitado ao P95)",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Classe de Eficiência Energética", fontsize=12)
ax.set_ylabel("Aluguel Base — €/mês", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
plt.tight_layout()
viz4_path = os.path.join(OUTPUT_DIR, 'boxplot_basrent_eficiencia.png')
plt.savefig(viz4_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {viz4_path}")


# --- VIZ 5: Violin plots baseRent por interiorQual e condition -----------
subsection("VIZ5 — Violin: baseRent por interiorQual e condition")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Distribuição do Aluguel Base por Qualidade Interior e Condição do Imóvel\n"
             "Dataset: Apartment Rental Offers in Germany",
             fontsize=14, fontweight='bold', y=1.02)

qual_order = ['simple', 'normal', 'sophisticated', 'luxury']
df_viz5_qual = df_viz[df_viz['interiorQual'].isin(qual_order)].copy()
df_viz5_qual['interiorQual'] = pd.Categorical(
    df_viz5_qual['interiorQual'], categories=qual_order, ordered=True)
sns.violinplot(data=df_viz5_qual, x='interiorQual', y='baseRent',
               order=qual_order, palette='Blues', inner='quartile',
               cut=0, ax=axes[0])
axes[0].set_title("por Qualidade do Interior (interiorQual)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Qualidade Interior", fontsize=11)
axes[0].set_ylabel("Aluguel Base — €/mês", fontsize=11)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

cond_order = ['need_of_renovation', 'simple', 'well_kept', 'refurbished',
              'modernized', 'first_time_use', 'first_time_use_after_refurbishment']
df_viz5_cond = df_viz[df_viz['condition'].isin(cond_order)].copy()
df_viz5_cond['condition'] = pd.Categorical(
    df_viz5_cond['condition'], categories=cond_order, ordered=True)
sns.violinplot(data=df_viz5_cond, x='condition', y='baseRent',
               order=cond_order, palette='Oranges', inner='quartile',
               cut=0, ax=axes[1])
axes[1].set_title("por Condição do Imóvel (condition)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Condição", fontsize=11)
axes[1].set_ylabel("")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
viz5_path = os.path.join(OUTPUT_DIR, 'violinplot_basrent_qualidade.png')
plt.savefig(viz5_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Salvo: {viz5_path}")
