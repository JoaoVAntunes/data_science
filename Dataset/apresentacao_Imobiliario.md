# Apresentação do Dataset: Apartment Rental Offers in Germany

## 1. Identificação do Dataset

- **Nome:** Apartment Rental Offers in Germany
- **Fonte:** ImmoScout24 (via Kaggle)
- **URL:** https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany
- **Tema:** Informações sobre ofertas de aluguel de apartamentos na Alemanha — preço base, espaço de vida, localização, características do imóvel, eficiência energética, custos adicionais, entre outros.
- **Relevância:** O mercado imobiliário alemão é um dos maiores e mais dinâmicos da Europa. Analisar dados de aluguel permite identificar tendências de preços, preferências regionais, impacto de características do imóvel (localização, tamanho, eficiência energética) e auxiliar na tomada de decisão de proprietários, corretoras imobiliárias, plataformas de busca e players de investimento imobiliário.

---

## 2. Características Técnicas

### Estrutura

| Métrica               | Valor                  |
|----------------------|------------------------|
| Tamanho do arquivo   | ~50 MB                 |
| Nº de registros      | 268.850                |
| Nº de colunas        | 49                     |

### Colunas e Tipos de Dados (Principais)

| Coluna                    | Tipo         | Exemplo                           | Descrição                                          |
|---------------------------|--------------|-----------------------------------|----------------------------------------------------|
| scoutId                   | Inteiro      | 1234567                           | ID único do imóvel                                 |
| baseRent                  | Numérico     | 750.00                            | Aluguel base mensal (€)                            |
| totalRent                 | Numérico     | 900.00                            | Aluguel total mensal (€) = baseRent + serviceCharge |
| livingSpace               | Numérico     | 75.50                             | Área útil (m²)                                     |
| noRooms                   | Numérico     | 3.0                               | Número de quartos                                  |
| yearConstructed           | Numérico     | 1985.0                            | Ano de construção do imóvel                        |
| regio1                    | Categórico   | "Nordrhein_Westfalen"             | Estado alemão (região de nível 1)                  |
| regio2                    | Categórico   | "Düsseldorf"                      | Cidade/Distrito (região de nível 2)                |
| geo_plz                   | Inteiro      | 40219                             | Código postal (plz)                                |
| hasKitchen                | Booleano     | True/False                        | Possui cozinha montada                             |
| balcony                   | Booleano     | True/False                        | Possui varanda                                     |
| cellar                    | Booleano     | True/False                        | Possui porão                                       |
| garden                    | Booleano     | True/False                        | Possui jardim                                      |
| lift                      | Booleano     | True/False                        | Possui elevador                                    |
| newlyConst                | Booleano     | True/False                        | Construção recente                                 |
| typeOfFlat                | Categórico   | "apartment", "penthouse"          | Tipo de imóvel                                     |
| condition                 | Categórico   | "well_kept", "refurbished"        | Condição do imóvel                                 |
| interiorQual              | Categórico   | "normal", "luxury"                | Qualidade do interior                              |
| energyEfficiencyClass     | Categórico   | "C", "B", "D"                     | Classe de eficiência energética (EU)               |
| heatingCosts              | Numérico     | 125.50                            | Custo mensal de aquecimento (€)                    |
| serviceCharge             | Numérico     | 150.00                            | Custo mensal de serviços (€)                       |
| petsAllowed               | Categórico   | "yes", "no"                       | Permite animais de estimação                       |
| date                      | Texto        | "May19", "Oct19"                  | Período de coleta dos dados                        |

### Desafios de Qualidade de Dados

- **Dados faltantes (nulos):** Diversas colunas possuem valores ausentes significativos:
  - `telekomHybridUploadSpeed`: 83.25%
  - `electricityBasePrice`, `electricityKwhPrice`: 82.58%
  - `energyEfficiencyClass`: 71.07%
  - `lastRefurbish`: 69.98%
  - `heatingCosts`: 68.19%
  - `petsAllowed`, `interiorQual`, `thermalChar`: 42-40%
  - Total de 19.28% de dados faltantes no dataset.

- **Valores extremos e outliers:** 
  - `baseRent`: Máximo de €9.999.999 (likely dados inválidos); 5.4% de outliers detectados pelo método IQR.
  - `livingSpace`: Máximo de 111.111 m² (deve ser filtrado); 4.2% de outliers.
  - 89 registros com `baseRent <= 0` e 236 com `totalRent <= 0` (devem ser removidos).

- **Assimetrias e distribuições espúrias:**
  - `baseRent`: Assimetria extrema (skewness = 500.11), sugerindo presença de valores não realistas.
  - `livingSpace`: Assimetria elevada (skewness = 373.80), indicando presença de registros com áreas extraordinariamente grandes.

- **Inconsistências de formato:** 
  - Datas em formato abreviado ("May19", "Oct19").
  - Variáveis booleanas com falsos valores nulos (bool type, sem NaN).
  - Caracteres especiais em nomes de regiões (underscores em vez de espaços).

- **Esparsidade de dados:** Colunas como `telekomHybridUploadSpeed`, `electricityBasePrice` têm mais de 80% de ausência, limitando sua utilidade para modelagem.

---

## 3. Aspectos Éticos e Legais

### Dados Sensíveis ou Pessoais
- O dataset contém **endereços parciais** (código postal, rua, número da casa) de imóveis anunciados. Embora estes dados estejam publicamente disponíveis na plataforma ImmoScout24, representam informações de identificação geográfica.
- **Não contém** dados pessoais de proprietários ou inquilinos (nomes, telefones, emails), apenas metadados dos imóveis.

### Conformidade com a LGPD e GDPR
- Os dados são de **domínio público**, provenientes de um portal imobiliário aberto (ImmoScout24), portanto seu uso para fins acadêmicos e de pesquisa é permitido.
- O tratamento se enquadra na base legal de **interesse legítimo** e **dados manifestamente tornados públicos pelo titular** (Art. 7º, §4º da LGPD; Art. 6(1)(f) da GDPR).
- Não há tratamento de dados sensíveis (raça, saúde, religião, orientação sexual, dados biométricos).

### Questões Éticas
- **Viés geográfico:** O dataset é concentrado em grandes centros urbanos alemães (Nordrhein-Westfalen 23.4%, Sachsen 21.6%). Análises sobre preços podem não refletir o mercado de regiões rurais subrrepresentadas.
- **Viés temporal:** Dados coletados em períodos específicos (Feb20, May19, Oct19), não capturando variações sazonais ou tendências de longo prazo.
- **Viés de plataforma:** ImmoScout24 é apenas uma plataforma; dados não representam o mercado imobiliário total alemão.
- **Viés de seleção:** Imóveis com características especiais ou preços extremos podem ter maior ou menor representatividade.
- **Representatividade:** Conclusões tiradas devem considerar estas limitações e não generalizar para todo o mercado imobiliário europeu.

---

## 4. Big Data — Os "Vs" Observados

### Volume
- **~50 MB** em format CSV.
- **268.850 registros** (linhas do dataset).
- **49 colunas** com dados estruturados, semiestruturados (textos descritivos) e booleanos.
- Volume significativo que requer ferramentas como Pandas para processamento eficiente; não é trivial abrir em editores de texto ou planilhas convencionais sem perca de desempenho.

### Variedade
O dataset apresenta grande variedade de tipos de dados:
- **Numérico (contínuo):** baseRent, totalRent, livingSpace, serviceCharge, heatingCosts, yearConstructed, floor, numberOfFloors, pricetrend.
- **Numérico (inteiro):** scoutId, noRooms, picturecount, geo_plz, noParkSpaces.
- **Categórico:** regio1, regio2, typeOfFlat, condition, interiorQual, energyEfficiencyClass, heatingType, petsAllowed.
- **Booleano:** balcony, cellar, garden, lift, hasKitchen, newlyConst.
- **Texto/Descrição:** description, facilities, street.
- **Temporal:** date, lastRefurbish.
- **Identificadores:** scoutId, geo_plz.

### Variabilidade
- **Preços (baseRent):** Variam de €1 a €9.999.999, com assimetria extrema. Mediana de €490, média de €694 (distorção por outliers).
- **Tamanho (livingSpace):** Variam de 1 m² a 111.111 m² (outliers), com mediana de 67.33 m² e média de 74.38 m².
- **Localização:** Concentração em 10 estados (Nordrhein-Westfalen, Sachsen, Bayern, etc.), com distribuição espacial variável.
- **Características:** Tipos de imóveis diversos (apartment 48.9%, roof_storey 12.9%, ground_floor 11.7%, etc.).
- **Temporal:** Dados coletados em múltiplos períodos (Feb20, May19, Oct19), com variações de ofertas ao longo do tempo.

### Velocidade
- O dataset é **estático** (snapshot de um período específico — Feb20, May19, Oct19).
- Porém, a fonte original (ImmoScout24) é uma plataforma **dinâmica**, com atualizações diárias de novas ofertas e remoção de listagens expiradas.
- Em um cenário de Big Data real, novos dados imobiliários poderiam ser ingeridos continuamente via APIs da ImmoScout24 ou web scraping para alimentar pipelines de análise em tempo real.

### Veracidade
- **Moderada confiabilidade:** Dados provenientes de um portal confiável (ImmoScout24), mas submetidos por agentes imobiliários/proprietários com potencial conflito de interesse (incentivo a superestimar características).
- **Presença de erros e inconsistências:** 
  - 89 registros com baseRent <= 0 (provavelmente anúncios inativos ou testes).
  - 236 registros com totalRent <= 0.
  - Valores extremos (baseRent de €9M, livingSpace de 111k m²) que destoam da realidade.
  - 5.4% de outliers em baseRent e 4.2% em livingSpace (devem ser tratados).
- **Dados incompletos:** 19.28% de valores faltantes no conjunto, com alguns campos quase totalmente vazios (telekomHybridUploadSpeed 83%).

### Valor
- **Análise de mercado:** Identificar padrões de preços por região, tipo de imóvel, tamanho e características; prever tendências de valorização/desvalorização.
- **Estratégia de precificação:** Proprietários e agências podem usar insights para definir preços competitivos baseado em comparáveis (benchmarking).
- **Investigação de desigualdades:** Analisar disparidades de preços por região, impacto de características sociodemográficas (viés de preços).
- **Pesquisa econômica:** Estudar elasticidade de preços, efeitos de eficiência energética, impacto de proximidade a transporte/serviços.
- **Sistemas de recomendação:** Base para construir motores de busca imobiliários (Immonet, Facebook Marketplace, etc.).
- **Decisões de investimento:** Identificar mercados quentes para compra/aluguel de imóveis com potencial de rentabilidade.

### Visualização
Os dados foram explorados através de:
- **Histogramas:** Distribuição de baseRent (altamente assimétrica) e livingSpace.
- **Boxplots:** Identificação de outliers e quartis em baseRent e livingSpace.
- **Gráficos de barras:** Top 10 regiões (Nordrhein-Westfalen lidera com 62.863 imóveis).
- **Scatter plot:** Relação entre livingSpace e baseRent (correlação visual).
- **Tabelas de frequência:** Distribuição de hasKitchen, typeOfFlat, energyEfficiencyClass.

Para datasets ainda maiores, técnicas como **heatmaps de correlação**, **treemaps de distribuição geográfica** e **dashboards interativos** (Plotly, Tableau) seriam recomendadas.

---

## 5. Extração de Informações

*(Já implementada no código `exploracaoBasica.py` e planejado para `main.py`)*

### Análises Básicas (Implementadas)
1. **Média de baseRent:** €694.36 (com valores > 0)
2. **Mediana de baseRent:** €490.00
3. **Desvio Padrão de baseRent:** €19.539.25 (elevado devido a outliers)
4. **Contagem de imóveis por regio1:** Nordrhein-Westfalen lidera com 62.863 (23.4%)
5. **Distribuição de hasKitchen:** 34.2% com cozinha montada, 65.8% sem

### Análises Intermediárias (Planejadas)
6. **Média de baseRent por regio1:** Comparar preços entre estados
7. **Média de baseRent por typeOfFlat:** Comparar preços por tipo de imóvel
8. **Distribuição regio1 + hasKitchen:** Impacto de localização e cozinha no preço
9. **Média de livingSpace por condition:** Relação entre condição e tamanho
10. **Top 10 energyEfficiencyClass mais frequentes:** C (5.4%), NO_INFORMATION (5.3%), D (5.2%)

### Análises Avançadas (Planejadas para Hipóteses)
11. **H1 - Impacto de localização (regio1) vs. eficiência energética (energyEfficiencyClass) no preço:** ANOVA ou regressão múltipla
12. **H2 - Efeito de hasKitchen no baseRent:** Teste t com imóveis com/sem cozinha montada
13. **H3 - Correlação linear entre livingSpace e baseRent:** Pearson correlation + scatter plot com regressão
14. **H4 - Influência de yearConstructed nos heatingCosts:** Regressão linear, análise de resíduos
15. **H5 - Variação de heatingType entre regiões:** Chi-square test, heatmap de crosstab

---

## 6. Estrutura de Dados para Modelagem

### Variáveis Alvo (Target)
- **Primária:** `baseRent` (Regressão) — Predizer preço base de aluguel
- **Secundária:** `totalRent` — Preço total incluindo custos de serviço
- **Possível:**  `condition`, `energyEfficiencyClass` (Classificação multiclasse)

### Variáveis Preditoras Principais
| Categoria          | Variáveis                                                      |
|-------------------|----------------------------------------------------------------|
| Localização        | regio1, regio2, geo_plz, geo_bln, geo_krs                      |
| Tamanho            | livingSpace, noRooms, numberOfFloors, floor                    |
| Características    | hasKitchen, balcony, cellar, garden, lift, newlyConst          |
| Qualidade/Condição | condition, interiorQual, energyEfficiencyClass                 |
| Idade              | yearConstructed, lastRefurbish                                 |
| Custos Adicionais  | serviceCharge, heatingCosts, electricityBasePrice, electricityKwhPrice |
| Outros             | picturecount, petsAllowed, heatingType, firingTypes            |

### Preparação Necessária (Passo 2 - Limpeza)
1. **Remoção de outliers:** baseRent > €9999, livingSpace > 300 m²
2. **Tratamento de NaN:** Impute heatingCosts com mediana, remova linhas com energyEfficiencyClass faltante (71%)
3. **Encoding:** One-hot para regio1, typeOfFlat, energyEfficiencyClass, heatingType
4. **Normalização:** StandardScaler para variáveis numéricas com diferentes escalas
5. **Feature Engineering:** Razão baseRent/livingSpace, idade = 2024 - yearConstructed, etc.

---

## 7. Conclusão

### Conclusão do Trabalho
A análise do dataset Apartment Rental Offers in Germany permite explorar de forma prática os conceitos de Big Data aplicados a um mercado real e relevante. Com 268.850 registros e 49 atributos diversos, foi possível identificar desde características estatísticas básicas (médias, distribuições, outliers) até padrões complexos de precificação imobiliária por região, tipo de imóvel e características.

As análises exploratórias evidenciam:
- Concentração geográfica em Nordrhein-Westfalen e Sachsen
- Forte assimetria em preços e áreas (presença de outliers e dados inválidos)
- Apenas 34.2% dos imóveis com cozinha montada
- Correlação moderada entre livingSpace e baseRent (indicando outros fatores impactam preço)
- Alto percentual de dados faltantes em alguns atributos (>80% em características de conectividade)

### Justificativa da Escolha do Dataset
O dataset foi escolhido por:
1. **Volume adequado** (~50 MB, 268k registros) — experiência real com Big Data, sem ser impraticável para análise acadêmica.
2. **Riqueza de atributos** (49 colunas) — permite análises multidimensionais: numéricas, categóricas, espaciais, temporais.
3. **Tema relevante e acessível** — mercado imobiliário é universal; dados são intuitivos e fáceis de interpretar.
4. **Desafios reais de qualidade** — outliers, dados faltantes significativos, inconcistências e ruído proporcionam experiência prática essencial.
5. **Fonte confiável e pública** — ImmoScout24 é líder no mercado germânico; dados disponíveis em Kaggle com licença para uso acadêmico.
6. **Aplicação prática imediata** — os mesmos dados/metodologias são usados por corretoras, plataformas de busca, fundos imobiliários e pesquisadores econômicos para análise de mercado, precificação competitiva e investigação de desigualdades.
7. **Alinhamento com projeto de ML:** Tarefa de **regressão** (prever baseRent) e potencial para **classificação** (prever condition, energyEfficiencyClass) — modelos práticos e de valor comercial.
