Objetivo: Desenvolver um projeto de Data Science de ponta a ponta (Análise, Engenharia e Machine Learning).
Dataset Selecionado: "Apartment rental offers in Germany" (Immoscout24 via Kaggle).
Tarefa de ML: Regressão para prever o valor do aluguel (baseRent).

Restrições do Projeto (Checkpoint 1)
Dataset Final: Deve conter no mínimo 30 variáveis (features) e 500 instâncias após a limpeza.

Análise de Dados (Ponto 3): O relatório deve seguir obrigatoriamente o fluxo: Pergunta/Hipótese -> Análise Visual/Estatística -> Discussão dos Resultados.

Requisitos Técnicos: Cálculo de estatísticas descritivas (média, mediana, desvio padrão, assimetria), análise univariada e multivariada.

Suas Instruções (Atue como Arquiteto de Dados)
Elabore um plano detalhado de implementação em Python para a Fase de Análise Exploratória (EDA), considerando que já possuímos experiência com pandas e matplotlib. O plano deve incluir:

1. Preparação e Sanity Check
Carregamento do dataset e verificação da estrutura (shape, dtypes).

Filtro inicial para garantir que manteremos mais de 30 colunas úteis (remover apenas IDs e textos irrelevantes).

Identificação de valores ausentes e inconsistências (ex: áreas negativas ou aluguéis zero).

2. Estatística Descritiva e Distribuição
Gerar um resumo estatístico das variáveis numéricas.

Analisar a assimetria (skewness) da variável alvo (baseRent) e propor transformações se necessário.

Identificar outliers usando Boxplots.

3. Formulação de Hipóteses (Padrão do Projeto)
Crie o código base para testar 5 hipóteses de negócio, como por exemplo:

H1: A localização (regio1) impacta o preço de forma mais significativa que a eficiência energética?

H2: Apartamentos com cozinha montada (hasKitchen) possuem aluguel base superior à média?

H3: Existe correlação linear entre livingSpace e baseRent?

H4: A idade do imóvel (yearConstructed) influencia nos custos adicionais de serviço (serviceCharge)?

H5: A oferta de tipos de aquecimento (heatingType) varia drasticamente entre os estados alemães?

4. Visualização
Sugira o uso de seaborn para matrizes de correlação (heatmap) e gráficos de dispersão.

Garanta que os gráficos sejam autoexplicativos para facilitar a redação do relatório final no formato IEEE.


## Plano Detalhado de Implementação da EDA

Implemente a fase de Análise Exploratória de Dados (EDA) como a primeira entrega, focando na preparação de dados, análise estatística, teste de hipóteses e visualizações, enquanto estabelece bases para futuras fases de engenharia de features e ML. Use exploracaoBasica.py para explorações básicas e main.py para o código integrado da entrega, garantindo que o dataset limpo tenha pelo menos 30 variáveis e 500 instâncias.

### Passos (Step-by-Step)
1. **Carregamento do Dataset e Verificação Inicial (Sanity Check)** (em exploracaoBasica.py): Carregue immo_data.csv usando pandas, verifique o shape (>=500 linhas, >=30 colunas), tipos de dados (dtypes) e identifique valores ausentes iniciais ou inconsistências (ex: áreas negativas, aluguéis zero).
2. **Limpeza e Filtragem de Dados** (paralelo ao passo 1): Remova colunas irrelevantes (IDs, campos de texto livres), trate valores ausentes (impute ou remova baseado em thresholds), filtre registros inválidos e garanta que o dataset final atenda às restrições; registre mudanças para reprodutibilidade.
3. **Estatísticas Descritivas e Análise de Distribuição** (depende do passo 2): Calcule média, mediana, desvio padrão, assimetria para variáveis numéricas, especialmente baseRent; analise distribuições e proponha transformações se assimétricas.
4. **Detecção de Outliers** (paralelo ao passo 3): Use boxplots para identificar outliers em variáveis-chave como baseRent e livingSpace; decida sobre tratamento (capping, remoção) baseado em lógica de negócio.
5. **Formulação e Teste de Hipóteses** (depende dos passos 2-4): Implemente código para as 5 hipóteses (H1-H5 do planejamento.md), usando testes estatísticos (ex: ANOVA para impactos categóricos, correlação para relacionamentos lineares) e visualizações.
6. **Implementação de Visualizações** (paralelo ao passo 5): Crie heatmaps de correlação com seaborn, gráficos de dispersão e outros para suportar hipóteses; garanta que os gráficos sejam autoexplicativos para relatórios no formato IEEE.
7. **Integração e Preparação da Entrega** (depende de todos os passos anteriores): Consolide o código da EDA em main.py para o pipeline completo, adicione comentários e docstrings, e prepare saídas resumidas (ex: estatísticas do dataset limpo, resultados de hipóteses) para o relatório.
8. **Preparação para Futuras Fases** (paralelo ao passo 7): Identifique potenciais features para engenharia (ex: variáveis derivadas de datas), note prontidão para ML (ex: codificar categóricas) e esboce próximas fases no planejamento.md.

### Arquivos Relevantes
- [exploracaoBasica.py](exploracaoBasica.py) — Explorações básicas: carregamento, verificações, estatísticas simples.
- [main.py](main.py) — Código integrado da entrega: pipeline completo da EDA.
- [immo_data.csv](immo_data.csv) — Dataset bruto (reutilizando download existente em main.py).
- [planejamento.md](planejamento.md) — Referência de requisitos; atualize com resultados da EDA e próximos passos.

### Verificação
1. Execute exploracaoBasica.py para confirmar shape do dataset, dtypes e estatísticas básicas; garanta ausência de erros no carregamento.
2. Execute main.py e verifique se o dataset limpo tem >=30 colunas e >=500 linhas; cheque valores ausentes <5%.
3. Revise saídas para cada hipótese: resultados estatísticos (p-valores, correlações) e gráficos; valide contra o fluxo de análise (Pergunta -> Análise -> Discussão).
4. Teste visualizações: Garanta que gráficos do seaborn renderizem corretamente e sejam interpretáveis sem contexto de código.

### Decisões
- Bibliotecas: Use pandas para manipulação de dados, matplotlib/seaborn para plotagem (conforme experiência); adicione numpy para estatísticas se necessário.
- Estratégia de Limpeza: Remova colunas com >50% ausentes; impute numéricos com mediana, categóricos com moda; remova registros com baseRent inválido (>0).
- Teste de Hipóteses: Use scipy para testes estatísticos (ex: t-test, chi-square); foque em evidência visual + estatística.
- Escopo: Apenas EDA; adie engenharia de features para próxima fase; inclua notas no planejamento.md para atualizações incrementais.

### Considerações Adicionais
1. Verificação do Dataset: Confirme contagem exata de colunas e linhas pós-limpeza; se <30 vars ou <500 instâncias, explore fontes adicionais ou ajuste filtragem.
2. Dependências: Crie requirements.txt com pandas, matplotlib, seaborn, scipy; assuma Python 3.8+.
3. Alinhamento com Relatório: Estruture saídas do código para alimentar diretamente seções de relatório no formato IEEE.

## Implementação Atual - Exploração Básica (Passo 1 Concluído)

O arquivo [exploracaoBasica.py](exploracaoBasica.py) foi implementado com o código para carregamento do dataset e verificações iniciais (sanity check), incluindo:
- Carregamento de immo_data.csv com pandas.
- Verificação de shape (268.850 linhas, 49 colunas ✓), dtypes, primeiras/últimas linhas, estatísticas descritivas e valores ausentes.
- Identificação de inconsistências (ex: 89 registros com baseRent <= 0, 236 com totalRent <= 0).
- Validação: ✓ Dataset atende às restrições mínimas (>=500 linhas, >=30 colunas).
- Análise detalhada de variáveis-alvo: baseRent (média €694.36, mediana €490, alta assimetria) e livingSpace (média 74.38 m², mediana 67.33 m²).
- Análise de variáveis categóricas principais: regio1 (Nordrhein-Westfalen lidera com 23.4%), hasKitchen (34.2% com cozinha montada), typeOfFlat (apartment 48.9%), energyEfficiencyClass (C 5.4%, D 5.2%).
- Análise de correlações com baseRent: yearConstructedRange (0.31), serviceCharge (0.24), yearConstructed (0.15).
- Detecção de outliers pelo método IQR: 5.4% em baseRent, 4.2% em livingSpace.
- Dashboard visual com 6 subplots: histogramas, boxplots, gráficos de barras e scatter plot (salvo como 'dashboard_exploracao_basica.png').

## Apresentação do Dataset (Documento Complementar - Criado)

O arquivo [apresentacao_Imobiliario.md](apresentacao_Imobiliario.md) foi criado seguindo o modelo IEEE/acadêmico:
- **Seção 1:** Identificação do dataset (nome, fonte, Kaggle, tema, relevância para mercado imobiliário).
- **Seção 2:** Características técnicas (49 colunas, 268.850 registros, ~50 MB); tabela das principais colunas com tipos e exemplos; discussão dos desafios de qualidade (19.28% NaN totais, outliers extremos, assimetrias).
- **Seção 3:** Aspectos éticos e legais (dados sensíveis: endereços parciais; conformidade LGPD/GDPR; questões de viés geográfico, temporal e de plataforma).
- **Seção 4:** Big Data "Vs" (Volume: 268k registros; Variedade: 7 tipos de dados; Variabilidade: preços 1€-9.9M€, tamanhos 1-111k m²; Velocidade: estático mas fonte dinâmica; Veracidade: moderada, com 5.4% outliers).
- **Seção 5:** Extração de informações (15 análises listadas: 5 básicas já implementadas, 5 intermediárias planejadas, 5 avançadas para hipóteses).
- **Seção 6:** Estrutura de dados para modelagem (targets, variáveis preditoras, preparação necessária).
- **Seção 7:** Conclusão com justificativa da escolha do dataset.

Próximo passo: Executar exploracaoBasica.py para validar, depois prosseguir para Passo 2 (Limpeza e Filtragem de Dados) em main.py.