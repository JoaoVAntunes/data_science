import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARREGAMENTO E SANITY CHECK
# ============================================================================
df = pd.read_csv('immo_data.csv')

print("="*80)
print("EXPLORAÇÃO BÁSICA - DATASET APARTMENT RENTAL OFFERS IN GERMANY")
print("="*80)

print("\n1. ESTRUTURA DO DATASET")
print(f"Shape do dataset: {df.shape}")
print(f"   Linhas: {df.shape[0]}")
print(f"   Colunas: {df.shape[1]}")

print("\nTipos de dados (dtypes):")
print(df.dtypes)

print("\n" + "="*80)
print("2. PRIMEIRAS E ÚLTIMAS LINHAS")
print("="*80)
print("\nPrimeiras 5 linhas:")
print(df.head())
print("\nÚltimas 5 linhas:")
print(df.tail())

print("\n" + "="*80)
print("3. ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS NUMÉRICAS")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("4. VALORES AUSENTES (MISSING VALUES)")
print("="*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percentage': missing_pct
}).sort_values('Missing_Count', ascending=False)
print(missing_df)

print(f"\nTotal de células com valores ausentes: {df.isnull().sum().sum()}")
print(f"Percentual de dados faltantes no dataset: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

print("\n" + "="*80)
print("5. IDENTIFICAÇÃO DE INCONSISTÊNCIAS")
print("="*80)

print("\nRegistros com livingSpace negativo:")
neg_living = df[df['livingSpace'] < 0]
print(f"Total: {len(neg_living)}")

print("\nRegistros com baseRent <= 0:")
neg_rent = df[df['baseRent'] <= 0]
print(f"Total: {len(neg_rent)}")

print("\nRegistros com totalRent <= 0:")
neg_total = df[df['totalRent'] <= 0]
print(f"Total: {len(neg_total)}")

print("\nRegistros com serviceCharge negativo:")
neg_service = df[df['serviceCharge'] < 0]
print(f"Total: {len(neg_service)}")

print("\n" + "="*80)
print("6. VERIFICAÇÃO DAS RESTRIÇÕES DO PROJETO")
print("="*80)
num_rows, num_cols = df.shape
print(f"\nRestrição 1 - Linhas: {num_rows} >= 500? {'✓ SIM' if num_rows >= 500 else '✗ NÃO'}")
print(f"Restrição 2 - Colunas: {num_cols} >= 30? {'✓ SIM' if num_cols >= 30 else '✗ NÃO'}")

if num_rows >= 500 and num_cols >= 30:
    print("\n✓ Dataset ATENDE às restrições mínimas do projeto (Checkpoint 1).")
else:
    print("\n✗ Dataset NÃO ATENDE às restrições mínimas.")

print("\n" + "="*80)
print("7. ANÁLISE DA VARIÁVEL ALVO (baseRent)")
print("="*80)

baserent_clean = df[df['baseRent'] > 0]['baseRent']
print(f"\nbaseRent (filtrando valores > 0):")
print(f"   Contagem: {len(baserent_clean)}")
print(f"   Média: €{baserent_clean.mean():.2f}")
print(f"   Mediana: €{baserent_clean.median():.2f}")
print(f"   Desvio Padrão: €{baserent_clean.std():.2f}")
print(f"   Mínimo: €{baserent_clean.min():.2f}")
print(f"   Máximo: €{baserent_clean.max():.2f}")
print(f"   Q1 (25%): €{baserent_clean.quantile(0.25):.2f}")
print(f"   Q3 (75%): €{baserent_clean.quantile(0.75):.2f}")
print(f"   IQR: €{baserent_clean.quantile(0.75) - baserent_clean.quantile(0.25):.2f}")
print(f"   Assimetria (Skewness): {stats.skew(baserent_clean):.4f}")
print(f"   Curtose (Kurtosis): {stats.kurtosis(baserent_clean):.4f}")

print("\n" + "="*80)
print("8. ANÁLISE DA VARIÁVEL SECUNDÁRIA (livingSpace)")
print("="*80)

livingspace_clean = df[df['livingSpace'] > 0]['livingSpace']
print(f"\nlivingSpace (filtrando valores > 0):")
print(f"   Contagem: {len(livingspace_clean)}")
print(f"   Média: {livingspace_clean.mean():.2f} m²")
print(f"   Mediana: {livingspace_clean.median():.2f} m²")
print(f"   Desvio Padrão: {livingspace_clean.std():.2f} m²")
print(f"   Mínimo: {livingspace_clean.min():.2f} m²")
print(f"   Máximo: {livingspace_clean.max():.2f} m²")
print(f"   Q1 (25%): {livingspace_clean.quantile(0.25):.2f} m²")
print(f"   Q3 (75%): {livingspace_clean.quantile(0.75):.2f} m²")
print(f"   Assimetria (Skewness): {stats.skew(livingspace_clean):.4f}")

print("\n" + "="*80)
print("9. ANÁLISE DE VARIÁVEIS CATEGÓRICAS PRINCIPAIS")
print("="*80)

print("\nDistribuição por regio1 (Top 10):")
regio1_dist = df['regio1'].value_counts().head(10)
for idx, (region, count) in enumerate(regio1_dist.items(), 1):
    pct = (count / len(df) * 100)
    print(f"   {idx}. {region}: {count} ({pct:.1f}%)")

print("\nDistribuição de hasKitchen:")
kitchen_dist = df['hasKitchen'].value_counts()
print(f"   Com cozinha montada: {kitchen_dist[True]} ({kitchen_dist[True]/len(df)*100:.1f}%)")
print(f"   Sem cozinha montada: {kitchen_dist[False]} ({kitchen_dist[False]/len(df)*100:.1f}%)")

print("\nDistribuição de typeOfFlat (Top 10):")
if 'typeOfFlat' in df.columns:
    typeflat_dist = df['typeOfFlat'].value_counts().head(10)
    for idx, (flat_type, count) in enumerate(typeflat_dist.items(), 1):
        pct = (count / len(df) * 100)
        print(f"   {idx}. {flat_type}: {count} ({pct:.1f}%)")


print("\n" + "="*80)
print("10. ANÁLISE DE CORRELAÇÃO - VARIÁVEIS NUMÉRICAS")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nVariáveis numéricas disponíveis ({len(numeric_cols)}):")
print(numeric_cols)

# Correlações com baseRent
print(f"\nCorrelações com baseRent (variável alvo):")
correlations = df[numeric_cols].corr()['baseRent'].sort_values(ascending=False)
print(correlations.head(10))

print("\n" + "="*80)
print("11. DETECÇÃO DE OUTLIERS - MÉTODO IQR")
print("="*80)

def detect_outliers_iqr(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Outliers em baseRent
baserent_filtered = df[df['baseRent'] > 0]['baseRent']
outliers_count, lower, upper = detect_outliers_iqr(baserent_filtered, 'baseRent')
print(f"\nbaseRent:")
print(f"   Limite inferior: €{lower:.2f}")
print(f"   Limite superior: €{upper:.2f}")
print(f"   Outliers detectados: {outliers_count} ({outliers_count/len(baserent_filtered)*100:.1f}%)")

# Outliers em livingSpace
livingspace_filtered = df[df['livingSpace'] > 0]['livingSpace']
outliers_count, lower, upper = detect_outliers_iqr(livingspace_filtered, 'livingSpace')
print(f"\nlivingSpace:")
print(f"   Limite inferior: {lower:.2f} m²")
print(f"   Limite superior: {upper:.2f} m²")
print(f"   Outliers detectados: {outliers_count} ({outliers_count/len(livingspace_filtered)*100:.1f}%)")

print("\n" + "="*80)
print("12. RESUMO FINAL")
print("="*80)
print(f"Total de registros: {len(df):,}")
print(f"Total de colunas: {df.shape[1]}")
print(f"Dados válidos (sem NaN) em baseRent: {df['baseRent'].notna().sum()}")
print(f"Registros com baseRent > 0: {len(df[df['baseRent'] > 0])}")
print(f"Percentual de dados faltantes: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
print("\nStatus: ✓ Dataset pronto para Passo 2 (Limpeza e Filtragem de Dados)")

# ============================================================================
# VISUALIZAÇÕES - DASHBOARD COM MATPLOTLIB
# ============================================================================
print("\n" + "="*80)
print("GERANDO VISUALIZAÇÕES...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Dashboard de Exploração Básica - Dataset Imóveis Alemanha', fontsize=16, fontweight='bold')

# 1. Histograma de baseRent
baserent_valid = df[df['baseRent'] > 0]['baseRent']
axes[0, 0].hist(baserent_valid, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribuição de baseRent', fontweight='bold')
axes[0, 0].set_xlabel('baseRent (€)')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].axvline(baserent_valid.mean(), color='red', linestyle='--', label=f'Média: €{baserent_valid.mean():.0f}')
axes[0, 0].axvline(baserent_valid.median(), color='green', linestyle='--', label=f'Mediana: €{baserent_valid.median():.0f}')
axes[0, 0].legend()

# 2. Boxplot de baseRent
axes[0, 1].boxplot(baserent_valid)
axes[0, 1].set_title('Boxplot de baseRent', fontweight='bold')
axes[0, 1].set_ylabel('baseRent (€)')

# 3. Histograma de livingSpace
livingspace_valid = df[df['livingSpace'] > 0]['livingSpace']
axes[0, 2].hist(livingspace_valid, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Distribuição de livingSpace', fontweight='bold')
axes[0, 2].set_xlabel('livingSpace (m²)')
axes[0, 2].set_ylabel('Frequência')
axes[0, 2].axvline(livingspace_valid.mean(), color='red', linestyle='--', label=f'Média: {livingspace_valid.mean():.0f} m²')
axes[0, 2].legend()

# 4. Boxplot de livingSpace
axes[1, 0].boxplot(livingspace_valid)
axes[1, 0].set_title('Boxplot de livingSpace', fontweight='bold')
axes[1, 0].set_ylabel('livingSpace (m²)')

# 5. Top 10 Regiões
top_regio1 = df['regio1'].value_counts().head(10)
axes[1, 1].barh(range(len(top_regio1)), top_regio1.values, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_yticks(range(len(top_regio1)))
axes[1, 1].set_yticklabels(top_regio1.index)
axes[1, 1].set_title('Top 10 Regiões (regio1)', fontweight='bold')
axes[1, 1].set_xlabel('Contagem')
axes[1, 1].invert_yaxis()

# 6. Scatter plot: livingSpace vs baseRent
valid_scatter = df[(df['livingSpace'] > 0) & (df['baseRent'] > 0)]
axes[1, 2].scatter(valid_scatter['livingSpace'], valid_scatter['baseRent'], alpha=0.3, color='red', s=10)
axes[1, 2].set_title('livingSpace vs baseRent', fontweight='bold')
axes[1, 2].set_xlabel('livingSpace (m²)')
axes[1, 2].set_ylabel('baseRent (€)')

plt.tight_layout()
plt.savefig('dashboard_exploracao_basica.png', dpi=300, bbox_inches='tight')
print("\n✓ Dashboard salvo como 'dashboard_exploracao_basica.png'")

print("\n" + "="*80)
print("FIM DA EXPLORAÇÃO BÁSICA")
print("="*80)
