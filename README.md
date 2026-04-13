# Imóveis Alemanha - Data Science

Projeto de análise e previsão de preços de aluguel em apartamentos alemães usando o dataset ImmoScout24 (Kaggle: `corrieaar/apartment-rental-offers-in-germany`).

## Objetivo

Construir um pipeline de Data Science ponta a ponta para prever `baseRent` (aluguel base mensal), incluindo:
- análise exploratória completa (EDA),
- testes estatísticos de hipóteses,
- visualizações para suporte à decisão,
- preparação para modelagem de regressão.

## Status Atual

EDA completa implementada em `main.py` com:
- análise descritiva do dataset (instâncias, features, tipos, missing values, class ratio),
- 20 análises univariadas no formato **Pergunta -> Hipótese -> Análise -> Discussão**,
- 10 análises multivariadas com testes estatísticos (ANOVA, t-test, Pearson, chi-square, regressão linear),
- 5 visualizações efetivas salvas como PNG.

## Dataset

- Fonte: ImmoScout24 via Kaggle
- Registros brutos: **268.850**
- Features: **49**
- Filtro base usado na EDA: `baseRent > 0` e `livingSpace > 0`

## Estrutura do Projeto

```text
data_science/
├── Dataset/
│   ├── baixarDataset.py
│   └── apresentacao_Imobiliario.md
├── Exploração Básica/
│   ├── exploracaoBasica.py
│   ├── dashboard_exploracao_basica.png
│   ├── heatmap_correlacoes.png
│   ├── boxplot_basrent_por_regiao.png
│   ├── scatter_area_aluguel.png
│   ├── boxplot_basrent_eficiencia.png
│   └── violinplot_basrent_qualidade.png
├── Task Specifications and Planning/
│   ├── planejamento.md
│   └── project_description.pdf
└── main.py
```

## Como Executar

### 1) Criar e ativar ambiente virtual (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependências

```powershell
pip install pandas numpy matplotlib seaborn scipy kagglehub
```

### 3) Rodar a EDA completa

```powershell
python main.py
```

## Saídas Geradas

Ao executar `main.py`, o projeto gera:
- relatório completo no terminal com todas as análises em formato Q/H/A/D,
- figuras em `Exploração Básica/`:
  - `heatmap_correlacoes.png`
  - `boxplot_basrent_por_regiao.png`
  - `scatter_area_aluguel.png`
  - `boxplot_basrent_eficiencia.png`
  - `violinplot_basrent_qualidade.png`

## Próximos Passos

- limpeza avançada e tratamento de outliers para modelagem,
- encoding de variáveis categóricas,
- normalização/escalonamento de features,
- treino e avaliação de modelos de regressão para `baseRent`.
