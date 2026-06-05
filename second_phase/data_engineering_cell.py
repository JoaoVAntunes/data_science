# ============================================================================
# DATA ENGINEERING AND DATA CLEANSING DIAGNOSTIC CELL
# Análise completa de engenharia e limpeza de dados
# Dataset: Apartamentos de aluguel na Alemanha (immo_data.csv)
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, boxcox, yeojohnson
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\joaov\Workspace\4 ano\Data Science\data_science\Dataset\immo_data.csv")

# Criar cópia do dataframe para não modificar o original
df_analysis = df.copy()

print("\n" + "="*100)
print("DATA ENGINEERING AND DATA CLEANSING DIAGNOSTIC REPORT")
print("="*100 + "\n")

# ============================================================================
# BLOCO 1: DIAGNÓSTICO DE MISSING VALUES
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 1 — DIAGNÓSTICO DE MISSING VALUES")
print("▓"*100 + "\n")

# Calcular missing values por coluna
missing_count = df_analysis.isnull().sum()
missing_pct = (missing_count / len(df_analysis) * 100).round(2)

# Classificar colunas por faixa de missing
def classify_missing(pct):
    if pct == 0:
        return "0% (Complete)"
    elif pct <= 10:
        return "≤10%"
    elif pct <= 30:
        return "10-30%"
    elif pct <= 70:
        return "30-70%"
    else:
        return ">70%"

# Gerar recomendação automática
def recommend_missing_action(pct):
    if pct == 0:
        return "KEEP"
    elif pct <= 10:
        return "KEEP"
    elif pct <= 30:
        return "IMPUTE"
    elif pct <= 70:
        return "REVIEW"
    else:
        return "DROP"

missing_diagnosis = pd.DataFrame({
    'Column': df_analysis.columns,
    'Missing_Count': missing_count.values,
    'Missing_%': missing_pct.values,
    'Category': [classify_missing(pct) for pct in missing_pct.values],
    'Recommendation': [recommend_missing_action(pct) for pct in missing_pct.values]
}).sort_values('Missing_%', ascending=False)

print(missing_diagnosis.to_string(index=False))

# Resumo por categoria
print("\n📊 RESUMO POR CATEGORIA:")
category_summary = missing_diagnosis['Category'].value_counts().sort_index()
for cat, count in category_summary.items():
    print(f"  • {cat}: {count} coluna(s)")

# ============================================================================
# BLOCO 2: DIAGNÓSTICO DE OUTLIERS
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 2 — DIAGNÓSTICO DE OUTLIERS (IQR METHOD)")
print("▓"*100 + "\n")

# Obter apenas variáveis numéricas
numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()

outlier_diagnosis = []

for col in numeric_cols:
    Q1 = df_analysis[col].quantile(0.25)
    Q3 = df_analysis[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Contar outliers
    outliers = df_analysis[(df_analysis[col] < lower_bound) | (df_analysis[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = round((outlier_count / len(df_analysis) * 100), 2)
    
    # Detectar possíveis erros de preenchimento (valores redondos grandes)
    suspicious_values = df_analysis[col][(df_analysis[col] % 1000 == 0) & (df_analysis[col] != 0)].count()
    
    outlier_diagnosis.append({
        'Column': col,
        'Outlier_Count': outlier_count,
        'Outlier_%': outlier_pct,
        'Lower_Bound': round(lower_bound, 2),
        'Upper_Bound': round(upper_bound, 2),
        'Min_Value': round(df_analysis[col].min(), 2),
        'Max_Value': round(df_analysis[col].max(), 2),
        'Suspicious_Values': suspicious_values
    })

outlier_df = pd.DataFrame(outlier_diagnosis).sort_values('Outlier_Count', ascending=False)

print(outlier_df.to_string(index=False))

# Ranking das variáveis com mais outliers
print("\n📊 RANKING DE VARIÁVEIS COM MAIS OUTLIERS:")
top_outliers = outlier_df.nlargest(5, 'Outlier_Count')
for idx, row in top_outliers.iterrows():
    print(f"  {row['Column']:25s} | Outliers: {row['Outlier_Count']:6d} ({row['Outlier_%']:6.2f}%) | Suspicious: {row['Suspicious_Values']}")

# Destacar possíveis erros
suspicious = outlier_df[outlier_df['Suspicious_Values'] > 0]
if len(suspicious) > 0:
    print("\n⚠️  POSSÍVEIS ERROS DE PREENCHIMENTO (valores redondos):")
    for idx, row in suspicious.iterrows():
        print(f"  • {row['Column']}: {row['Suspicious_Values']} valores redondos múltiplos de 1000")

# ============================================================================
# BLOCO 3: DIAGNÓSTICO DE SKEWNESS
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 3 — DIAGNÓSTICO DE SKEWNESS E TRANSFORMAÇÕES RECOMENDADAS")
print("▓"*100 + "\n")

def classify_skewness(skew_value):
    """Classificar o nível de assimetria"""
    abs_skew = abs(skew_value)
    if abs_skew < 0.5:
        return "Aproximadamente Normal"
    elif abs_skew < 1.0:
        return "Moderadamente Assimétrica"
    else:
        return "Fortemente Assimétrica"

def recommend_transformation(skew_value, min_val, max_val):
    """Recomendar transformação baseada em skewness e range de dados"""
    abs_skew = abs(skew_value)
    
    # Se os dados têm valores negativos, não usar log
    if min_val <= 0:
        if abs_skew < 0.5:
            return "NONE"
        else:
            return "YEO-JOHNSON"
    
    # Se os dados são todos positivos
    if abs_skew < 0.5:
        return "NONE"
    elif abs_skew < 1.0:
        return "LOG1P"
    else:
        # Para skewness forte, experimentar Box-Cox ou Yeo-Johnson
        return "BOX-COX"

skewness_diagnosis = []

for col in numeric_cols:
    col_data = df_analysis[col].dropna()
    
    if len(col_data) > 0:
        skew_value = skew(col_data)
        classification = classify_skewness(skew_value)
        recommendation = recommend_transformation(skew_value, df_analysis[col].min(), df_analysis[col].max())
        
        skewness_diagnosis.append({
            'Column': col,
            'Skewness': round(skew_value, 4),
            'Abs_Skewness': round(abs(skew_value), 4),
            'Classification': classification,
            'Recommendation': recommendation
        })

skewness_df = pd.DataFrame(skewness_diagnosis).sort_values('Abs_Skewness', ascending=False)

print(skewness_df[['Column', 'Skewness', 'Classification', 'Recommendation']].to_string(index=False))

# Resumo de transformações recomendadas
print("\n📊 RESUMO DE TRANSFORMAÇÕES RECOMENDADAS:")
transform_counts = skewness_df['Recommendation'].value_counts()
for transform, count in transform_counts.items():
    print(f"  • {transform:12s}: {count:2d} variável(is)")

# ============================================================================
# BLOCO 4: DIAGNÓSTICO DE VARIÁVEIS CATEGÓRICAS
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 4 — DIAGNÓSTICO DE VARIÁVEIS CATEGÓRICAS")
print("▓"*100 + "\n")

# Obter variáveis categóricas (object dtype)
categorical_cols = df_analysis.select_dtypes(include=['object']).columns.tolist()

def classify_cardinality(unique_count):
    """Classificar cardinalidade"""
    if unique_count <= 10:
        return "Baixa (≤10)"
    elif unique_count <= 30:
        return "Média (11-30)"
    else:
        return "Alta (>30)"

def recommend_encoding(unique_count, dominant_pct):
    """Recomendar tipo de encoding"""
    if unique_count <= 10:
        return "One-Hot Encoding"
    elif unique_count <= 30:
        if dominant_pct > 80:
            return "Target Encoding"
        else:
            return "Ordinal Encoding"
    else:
        if dominant_pct > 90:
            return "Drop (High Cardinality)"
        else:
            return "Target Encoding"

categorical_diagnosis = []

for col in categorical_cols:
    unique_count = df_analysis[col].nunique()
    value_counts = df_analysis[col].value_counts()
    dominant_pct = (value_counts.iloc[0] / len(df_analysis) * 100).round(2)
    cardinality = classify_cardinality(unique_count)
    encoding = recommend_encoding(unique_count, dominant_pct)
    missing_pct_cat = (df_analysis[col].isnull().sum() / len(df_analysis) * 100).round(2)
    
    categorical_diagnosis.append({
        'Column': col,
        'Unique_Values': unique_count,
        'Cardinality': cardinality,
        'Dominant_Category_%': dominant_pct,
        'Missing_%': missing_pct_cat,
        'Encoding_Recommendation': encoding
    })

categorical_df = pd.DataFrame(categorical_diagnosis).sort_values('Unique_Values', ascending=False)

print(categorical_df.to_string(index=False))

# Detalhes das categorias mais frequentes
print("\n📊 DISTRIBUIÇÃO DAS CATEGORIAS MAIS FREQUENTES:")
for col in categorical_cols:
    print(f"\n  {col}:")
    top_categories = df_analysis[col].value_counts().head(3)
    for cat, count in top_categories.items():
        pct = (count / len(df_analysis) * 100)
        print(f"    • {str(cat):20s}: {count:6d} ({pct:6.2f}%)")

# ============================================================================
# BLOCO 5: DIAGNÓSTICO DE ESCALONAMENTO (SCALING)
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 5 — DIAGNÓSTICO DE ESCALONAMENTO (SCALING)")
print("▓"*100 + "\n")

def recommend_scaler(has_outliers, amplitude, std_dev, mean):
    """
    Recomendar tipo de scaler baseado em características dos dados:
    - StandardScaler: dados com distribuição aproximadamente normal, sem muitos outliers
    - MinMaxScaler: dados sem outliers, quando intervalo fixo é importante
    - RobustScaler: dados com muitos outliers, pois usa mediana e IQR
    """
    if has_outliers and amplitude > std_dev * 6:
        return "RobustScaler (Recomendado: muitos outliers)"
    elif amplitude > std_dev * 5:
        return "RobustScaler (Amplitude alta)"
    else:
        return "StandardScaler (Recomendado)"

scaling_diagnosis = []

for col in numeric_cols:
    col_data = df_analysis[col].dropna()
    
    if len(col_data) > 0:
        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()
        amplitude = max_val - min_val
        
        # Contar outliers
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        has_outliers = outlier_count > 0
        
        scaler_recommendation = recommend_scaler(has_outliers, amplitude, std_val, mean_val)
        
        scaling_diagnosis.append({
            'Column': col,
            'Min': round(min_val, 2),
            'Max': round(max_val, 2),
            'Mean': round(mean_val, 2),
            'Std': round(std_val, 2),
            'Amplitude': round(amplitude, 2),
            'Outliers': outlier_count,
            'Scaler_Recommendation': scaler_recommendation
        })

scaling_df = pd.DataFrame(scaling_diagnosis)

print(scaling_df.to_string(index=False))

# Resumo de recomendações
print("\n📊 RESUMO DE RECOMENDAÇÕES DE SCALING:")
scaler_counts = scaling_df['Scaler_Recommendation'].value_counts()
for scaler, count in scaler_counts.items():
    print(f"  • {count} variável(is): {scaler}")

# ============================================================================
# BLOCO 6: VERIFICAÇÃO DE CLASS IMBALANCE (REGRESSÃO)
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 6 — ANÁLISE DE DISTRIBUIÇÃO DA VARIÁVEL ALVO (baseRent)")
print("▓"*100 + "\n")

if 'baseRent' in df_analysis.columns:
    target = df_analysis['baseRent'].dropna()
    
    print("📌 INFORMAÇÕES SOBRE O PROBLEMA:")
    print(f"  • Tipo de Problema: REGRESSÃO (previsão contínua)")
    print(f"  • Variável Alvo: baseRent")
    print(f"  • Número de Amostras: {len(target)}")
    print(f"  • Valores Únicos: {target.nunique()}")
    print(f"  • Range: [{target.min():.2f}, {target.max():.2f}]")
    print(f"  • Média: {target.mean():.2f}")
    print(f"  • Mediana: {target.median():.2f}")
    print(f"  • Desvio Padrão: {target.std():.2f}")
    
    print("\n📝 POR QUE SMOTE E BALANCEAMENTO NÃO SÃO NECESSÁRIOS:")
    print("  ✓ SMOTE (Synthetic Minority Over-sampling Technique) é para CLASSIFICAÇÃO")
    print("    → Balanceia classes discretas (ex: 0 e 1, ou Low/Medium/High)")
    print("    → Em REGRESSÃO, a variável alvo é CONTÍNUA, não há \"classes\"")
    print("")
    print("  ✓ Balanceamento de classes é conceito de CLASSIFICAÇÃO")
    print("    → Em REGRESSÃO, trabalha-se com distribuições contínuas de valores")
    print("    → Cada amostra tem um valor único contínuo")
    print("")
    print("  ✓ Para REGRESSÃO, o foco é em:")
    print("    → Normalização/escalonamento dos dados")
    print("    → Tratamento de outliers (que afetam modelos sensíveis)")
    print("    → Transformações para melhorar distribuição (ex: log, Box-Cox)")
    print("    → Feature engineering e seleção de features relevantes")
    
    print("\n📊 DISTRIBUIÇÃO DA VARIÁVEL ALVO (DECIS - 10 FAIXAS):")
    
    # Criar decis
    deciles = pd.qcut(target, q=10, duplicates='drop')
    decile_counts = deciles.value_counts().sort_index()
    
    print(f"\n  Decil    | Count | Percentual | Visualização")
    print(f"  {'-'*50}")
    
    for idx, (interval, count) in enumerate(decile_counts.items(), 1):
        pct = (count / len(target) * 100)
        bar = "█" * int(pct / 2)
        print(f"  D{idx:>2d}     | {count:>5d} | {pct:>8.2f}% | {bar}")
    
    print("\n  Conclusão: Distribuição verificada apenas para exploração.")
    print("  Nenhuma ação de rebalanceamento é necessária em regressão.")

else:
    print("⚠️  Coluna 'baseRent' não encontrada no dataset")

# ============================================================================
# BLOCO 7: RESUMO EXECUTIVO
# ============================================================================
print("\n" + "▓"*100)
print("BLOCO 7 — RESUMO EXECUTIVO E PIPELINE DE PRÉ-PROCESSAMENTO")
print("▓"*100 + "\n")

print("📋 RECOMENDAÇÕES CONSOLIDADAS:\n")

# 1. Colunas a remover
print("1️⃣  COLUNAS A REMOVER (>70% missing ou alta cardinalidade):")
cols_to_drop = missing_diagnosis[missing_diagnosis['Recommendation'] == 'DROP']['Column'].tolist()
if len(cols_to_drop) > 0:
    for col in cols_to_drop:
        print(f"    ✗ {col}")
else:
    print("    ✓ Nenhuma coluna recomendada para remoção por missing values")

# 2. Colunas a imputar
print("\n2️⃣  COLUNAS A IMPUTAR (10-30% missing):")
cols_to_impute = missing_diagnosis[missing_diagnosis['Recommendation'] == 'IMPUTE']['Column'].tolist()
if len(cols_to_impute) > 0:
    for col in cols_to_impute:
        missing_pct_val = missing_diagnosis[missing_diagnosis['Column'] == col]['Missing_%'].values[0]
        if col in numeric_cols:
            print(f"    → {col:25s} ({missing_pct_val:.2f}% missing) - Usar: MÉDIA ou MEDIANA")
        else:
            print(f"    → {col:25s} ({missing_pct_val:.2f}% missing) - Usar: MODA ou criar categoria 'Unknown'")
else:
    print("    ✓ Nenhuma coluna recomendada para imputação")

# 3. Variáveis com transformação logarítmica
print("\n3️⃣  VARIÁVEIS COM TRANSFORMAÇÃO RECOMENDADA:")
log_transforms = skewness_df[skewness_df['Recommendation'].isin(['LOG1P', 'BOX-COX'])]
if len(log_transforms) > 0:
    for idx, row in log_transforms.iterrows():
        print(f"    → {row['Column']:25s} (skewness: {row['Skewness']:7.4f}) - Transformação: {row['Recommendation']}")
else:
    print("    ✓ Poucas variáveis necessitam transformação")

# 4. Variáveis com muitos outliers
print("\n4️⃣  VARIÁVEIS COM MUITOS OUTLIERS (>5%):")
outlier_heavy = outlier_df[outlier_df['Outlier_%'] > 5].sort_values('Outlier_%', ascending=False)
if len(outlier_heavy) > 0:
    for idx, row in outlier_heavy.iterrows():
        print(f"    → {row['Column']:25s} ({row['Outlier_Count']:6d} outliers, {row['Outlier_%']:6.2f}%)")
        print(f"       Ação: Investigar ou usar RobustScaler")
else:
    print("    ✓ Dados relativamente limpos de outliers")

# 5. Encoding para variáveis categóricas
print("\n5️⃣  ENCODING RECOMENDADO PARA VARIÁVEIS CATEGÓRICAS:")
for idx, row in categorical_df.iterrows():
    print(f"    → {row['Column']:25s} ({row['Unique_Values']:3d} categorias) - {row['Encoding_Recommendation']}")

# 6. Scaler recomendado
print("\n6️⃣  SCALER RECOMENDADO PARA VARIÁVEIS NUMÉRICAS:")
robust_scalers = scaling_df[scaling_df['Scaler_Recommendation'].str.contains('RobustScaler')]
standard_scalers = scaling_df[scaling_df['Scaler_Recommendation'].str.contains('StandardScaler')]

if len(robust_scalers) > 0:
    print(f"    • RobustScaler: {len(robust_scalers)} variável(is) com outliers significativos")
if len(standard_scalers) > 0:
    print(f"    • StandardScaler: {len(standard_scalers)} variável(is) com distribuição mais normal")

# 7. Pipeline sugerido
print("\n7️⃣  PIPELINE DE PRÉ-PROCESSAMENTO SUGERIDO:")
print("""
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 1: REMOÇÃO                                     │
    │  • Remover colunas com >70% missing                 │
    │  • Remover colunas com alta cardinalidade (>30)     │
    │  • Remover duplicatas                               │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 2: IMPUTAÇÃO                                   │
    │  • Imputar numéricas com MEDIANA (robusta)          │
    │  • Imputar categóricas com MODA ou 'Unknown'        │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 3: TRATAMENTO DE OUTLIERS                      │
    │  • Manter outliers legítimos                         │
    │  • Investigar valores suspeitos (999, 9999999)      │
    │  • Usar RobustScaler para variáveis com outliers    │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 4: TRANSFORMAÇÕES                              │
    │  • Aplicar LOG1P, Box-Cox ou Yeo-Johnson            │
    │  • Validar distribuição após transformação          │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 5: ENCODING CATEGÓRICAS                        │
    │  • One-Hot Encoding para baixa cardinalidade        │
    │  • Target Encoding para alta cardinalidade          │
    │  • Ordinal Encoding onde apropriado                 │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 6: ESCALONAMENTO                               │
    │  • StandardScaler (para variáveis normais)          │
    │  • RobustScaler (para variáveis com outliers)       │
    │  • Fit em TREINO, apply em TESTE                    │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │ ETAPA 7: VALIDAÇÃO                                   │
    │  • Verificar distribuições                          │
    │  • Testar correlações com variável alvo             │
    │  • Pronto para Machine Learning!                    │
    └─────────────────────────────────────────────────────┘
""")

print("\n" + "="*100)
print("FIM DO RELATÓRIO DE DATA ENGINEERING AND DATA CLEANSING")
print("="*100 + "\n")

print("✅ Análise concluída. Próximas etapas:")
print("  1. Revisar as recomendações acima")
print("  2. Implementar limpeza conforme recomendações")
print("  3. Criar pipeline de preprocessamento com sklearn.pipeline")
print("  4. Validar performance em dados de treino vs teste")
print("  5. Prosseguir com Feature Engineering e Model Selection\n")
