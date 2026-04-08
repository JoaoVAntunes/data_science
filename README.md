# Imóveis Alemanha - Data Science

Projeto de análise e previsão de preços de aluguel em apartamentos alemães usando dataset ImmoScout24 (268.850 registros, 49 features). **Objetivo:** construir modelo de regressão para `baseRent`, passando por exploração, limpeza, análise estatística com 5 hipóteses e modelagem ML.

**Fase 1 ✅ (Exploração Básica concluída):** Carregado dataset (268.850 linhas, 49 colunas), analisadas estatísticas descritivas (baseRent: média €694, mediana €490), identificados 19.28% dados faltantes, 5.4% outliers em preço. Documentadas características do dataset e plano de 8 passos. **Próximos:** Limpeza de dados → Análise de hipóteses → Testes estatísticos → Modelagem ML em `main.py`.

```
TDE/ → Dataset/ (immo_data.csv) | Exploração Básica/ (exploracaoBasica.py) | Task Specifications and Planning/ (planejamento.md) | main.py
```

**Rodar:** `python Exploração\ Básica/exploracaoBasica.py` | **Deps:** pandas, numpy, matplotlib, seaborn, scipy
