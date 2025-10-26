# 🍷 Desafio de Análise e Predição de Qualidade de Vinhos

## 📋 Descrição do Desafio

Bem-vindo ao **Desafio de Análise e Predição de Qualidade de Vinhos**! Este é um desafio completo de Machine Learning e Data Science que aborda múltiplos aspectos da análise de dados, desde a exploração até a modelagem preditiva e imputação de valores faltantes.

---

## 🎯 Objetivos do Desafio

Este desafio é dividido em **5 etapas principais**, cada uma com seus próprios objetivos e complexidades:

### 1. **Análise Exploratória de Dados (EDA)**
- Carregar e explorar os datasets de treino e teste
- Entender a distribuição das variáveis
- Identificar correlações entre features
- Visualizar padrões nos dados

### 2. **Modelagem Preditiva de Regressão**
- Criar modelos para prever a **qualidade do vinho**
- Comparar múltiplos algoritmos de regressão
- Avaliar performance usando métricas adequadas

### 3. **Análise de Clustering**
- Identificar grupos naturais nos dados de vinhos
- Aplicar algoritmos de clustering não-supervisionado
- Determinar o número ótimo de clusters
- Visualizar os clusters em espaço reduzido

### 4. **Simulação de Dados Faltantes**
- Criar uma versão do dataset de teste com valores faltantes
- Simular um cenário realista de dados incompletos

### 5. **Imputação de Valores Faltantes**
- Desenvolver um sistema de imputação baseado em Machine Learning
- Prever valores faltantes usando modelos treinados
- Validar a qualidade da imputação comparando com dados originais

---

## 📊 Dados Disponíveis

### Dataset de Treino (`train.csv`)
- **Amostras**: ~15.000 vinhos
- **Features**: 11 características físico-químicas
- **Target**: Qualidade do vinho (quality)

### Dataset de Teste (`test.csv`)
- **Amostras**: Conjunto para avaliação
- **Features**: Mesmas 11 características do treino
- **Target**: Não disponível (para ser previsto)

### Features Disponíveis:
1. `id` - Identificador único
2. `fixed acidity` - Acidez fixa
3. `volatile acidity` - Acidez volátil
4. `citric acid` - Ácido cítrico
5. `residual sugar` - Açúcar residual
6. `chlorides` - Cloretos
7. `free sulfur dioxide` - Dióxido de enxofre livre
8. `total sulfur dioxide` - Dióxido de enxofre total
9. `density` - Densidade
10. `pH` - pH
11. `sulphates` - Sulfatos
12. `alcohol` - Teor alcoólico
13. `quality` - **Qualidade do vinho** (variável alvo, apenas no treino)

---

## 🎓 Etapa 1: Análise Exploratória de Dados

### Objetivos:
- [ ] Carregar os datasets de treino e teste
- [ ] Exibir informações básicas (shape, tipos de dados, valores nulos)
- [ ] Calcular estatísticas descritivas
- [ ] Visualizar a distribuição da variável alvo (`quality`)
- [ ] Criar matriz de correlação
- [ ] Visualizar distribuição de todas as features numéricas

### Pontos de Atenção:
- Existem valores nulos nos dados originais?
- Como é a distribuição da qualidade dos vinhos?
- Quais features têm maior correlação com a qualidade?
- Há outliers significativos nos dados?

### Visualizações Esperadas:
1. Gráfico de barras da distribuição de qualidade
2. Histograma da qualidade
3. Matriz de correlação (heatmap)
4. Histogramas de todas as features numéricas

---

## 🤖 Etapa 2: Modelagem Preditiva de Regressão

### Objetivos:
- [ ] Preparar dados para modelagem (separar X e y)
- [ ] Dividir dados em treino e validação (80/20)
- [ ] Normalizar os dados usando StandardScaler
- [ ] Treinar pelo menos 3 modelos diferentes de regressão
- [ ] Avaliar modelos usando múltiplas métricas

### Modelos Sugeridos:
1. **Regressão Linear** - Baseline simples
2. **Random Forest Regressor** - Modelo ensemble baseado em árvores
3. **Gradient Boosting Regressor** - Modelo de boosting

### Métricas de Avaliação:
- **RMSE** (Root Mean Squared Error) - Para treino e validação
- **R²** (R-squared) - Coeficiente de determinação
- **MAE** (Mean Absolute Error) - Erro absoluto médio

### Desafio Extra:
- Crie visualizações comparando a performance dos modelos
- Identifique qual modelo tem melhor generalização (menor overfitting)
- Gráficos de comparação side-by-side de RMSE e R²

### Critérios de Sucesso:
- RMSE de validação < 0.65
- R² de validação > 0.35
- Diferença entre RMSE treino/validação < 0.15 (evitar overfitting)

---

## 🎨 Etapa 3: Análise de Clustering

### Objetivos:
- [ ] Aplicar método do cotovelo para determinar k ótimo
- [ ] Treinar modelo K-Means com k escolhido
- [ ] Analisar distribuição de vinhos por cluster
- [ ] Calcular qualidade média por cluster
- [ ] Visualizar clusters usando PCA (2 componentes)

### Método do Cotovelo:
- Testar valores de k de 2 a 10
- Calcular inércia para cada k
- Plotar curva e identificar "cotovelo"

### Análise Esperada:
- Qual é o número ótimo de clusters?
- Os clusters representam diferentes níveis de qualidade?
- Quantos vinhos há em cada cluster?
- Qual cluster contém vinhos de maior qualidade média?

### Visualização PCA:
- Reduzir dimensionalidade para 2 componentes principais
- Plotar scatter plot colorido por cluster
- Mostrar variância explicada por cada componente

### Critérios de Sucesso:
- Identificar k ótimo (provavelmente entre 3-5)
- Clusters bem separados na visualização PCA
- Variância explicada pelos 2 primeiros componentes > 40%

---

## 🎲 Etapa 4: Simulação de Dados Faltantes

### Objetivos:
- [ ] Criar cópia do dataset de teste original
- [ ] Remover valores aleatoriamente de forma controlada
- [ ] Aplicar percentual uniforme de missing values (~20%)
- [ ] Salvar dataset modificado
- [ ] Documentar estatísticas dos valores faltantes

### Especificações:
- **Percentual de valores faltantes**: 20% por coluna
- **Seed para reprodutibilidade**: 42
- **Colunas afetadas**: Todas exceto `id`
- **Distribuição**: Aleatória e uniforme

### Output Esperado:
- Arquivo `test_with_missing_features.csv`
- Relatório mostrando:
  - Número total de valores faltantes
  - Valores faltantes por coluna
  - Percentual de missing values por coluna

### Critérios de Sucesso:
- Aproximadamente 20% de valores faltantes por feature
- Distribuição aleatória (não concentrada em poucas linhas)
- ID preservado em todas as linhas

---

## 🔮 Etapa 5: Imputação de Valores Faltantes

Esta é a **etapa mais desafiadora** do projeto!

### Objetivos:
- [ ] Criar função para treinar modelo de predição por feature
- [ ] Treinar um modelo específico para cada feature com valores faltantes
- [ ] Implementar imputação iterativa
- [ ] Aplicar imputação ao dataset de teste
- [ ] Validar qualidade da imputação

### Abordagem Recomendada:

#### 1. Treinamento de Modelos por Feature
Para cada feature com valores faltantes:
- Usar feature como **target**
- Usar todas as outras features como **preditoras**
- Treinar modelo Random Forest
- Armazenar modelo + scaler + features usadas

#### 2. Imputação Iterativa
- **Iteração máxima**: 5 iterações
- Para cada iteração:
  - Identificar valores faltantes
  - Prever usando modelos treinados
  - Preencher valores temporariamente com média se necessário
  - Imputar valores previstos
- Parar quando não houver mais valores faltantes

#### 3. Função `treinar_modelo_feature()`
```python
def treinar_modelo_feature(feature_name, train_data):
    """
    Treina um modelo para prever uma feature específica
    """
    # Separar features preditoras (todas exceto id, quality, e a feature alvo)
    # Normalizar dados com StandardScaler
    # Treinar RandomForestRegressor
    # Retornar modelo, scaler e lista de features
    pass
```

#### 4. Função `imputar_valores_faltantes()`
```python
def imputar_valores_faltantes(df_com_missing, modelos_dict, max_iterations=5):
    """
    Imputa valores faltantes iterativamente
    """
    # Para cada iteração até max_iterations:
    #   Para cada feature com missing values:
    #     Identificar linhas com valores faltantes
    #     Preparar dados para predição
    #     Preencher temporariamente valores faltantes nas features preditoras
    #     Normalizar e fazer predição
    #     Imputar valores previstos
    # Retornar dataframe completo
    pass
```

### Validação:
- [ ] Comparar distribuições: original vs imputado
- [ ] Calcular diferença de média e desvio padrão
- [ ] Criar visualizações comparativas (histogramas + boxplots)
- [ ] Salvar dataset imputado

### Visualizações Esperadas:
1. **Histograma Original** - Distribuição dos valores originais
2. **Histograma Imputado** - Distribuição após imputação
3. **Boxplot Comparativo** - Comparação lado a lado

### Critérios de Sucesso:
- **Zero valores faltantes** após imputação
- Diferença de **média < 5%** entre original e imputado
- Diferença de **std < 10%** entre original e imputado
- Distribuições visualmente similares

---

## 📈 Métricas de Avaliação Final

### Pontuação por Etapa:

| Etapa | Peso | Critérios |
|-------|------|-----------|
| **EDA** | 15% | Completude das análises e visualizações |
| **Modelagem** | 25% | Performance dos modelos (RMSE, R²) |
| **Clustering** | 20% | Identificação de k ótimo e análise dos clusters |
| **Simulação** | 10% | Correta geração de valores faltantes |
| **Imputação** | 30% | Qualidade da imputação e preservação de distribuições |

### Pontuação Extra:
- **+5%**: Código bem documentado com comentários claros
- **+5%**: Visualizações criativas e informativas além das solicitadas
- **+5%**: Análises adicionais ou insights interessantes
- **+5%**: Implementação de técnicas avançadas (ex: hyperparameter tuning)

---

## 🛠️ Ferramentas e Bibliotecas

### Obrigatórias:
```python
# Manipulação de dados
import pandas as pd
import numpy as np

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

### Opcionais (para exploração avançada):
- XGBoost, LightGBM, CatBoost (modelos de boosting avançados)
- Optuna ou GridSearchCV (hyperparameter tuning)
- UMAP (alternativa ao PCA para visualização)
- Plotly (visualizações interativas)

---

## 📝 Estrutura Sugerida do Notebook

```
1. Título e Introdução
2. Importação de Bibliotecas
3. Seção 1: Carregamento e Preparação de Dados
   - Carregar dados originais
   - Criar dataset com valores faltantes
4. Seção 2: Análise Exploratória
   - Estatísticas descritivas
   - Visualizações de distribuição
   - Matriz de correlação
5. Seção 3: Modelagem Preditiva
   - Preparação dos dados
   - Treinamento de modelos
   - Comparação de performance
6. Seção 4: Análise de Clustering
   - Método do cotovelo
   - K-Means
   - Visualização PCA
7. Seção 5: Imputação de Valores Faltantes
   - Funções de treinamento
   - Imputação iterativa
   - Validação e comparação
8. Conclusões e Insights
```

---

## 🏆 Desafios Bônus

### Desafio Bônus 1: Ensemble de Imputação
- Treinar múltiplos modelos (RF, GB, XGB) para cada feature
- Usar média/mediana das predições para imputação final
- Comparar com abordagem de modelo único

### Desafio Bônus 2: Análise de Feature Importance
- Para o melhor modelo de regressão, extrair feature importance
- Visualizar top 10 features mais importantes
- Retreinar modelo apenas com top features

### Desafio Bônus 3: Predição Final
- Usar melhor modelo treinado na Etapa 2
- Prever qualidade do vinho no dataset de teste (após imputação)
- Criar arquivo de submission (`predictions.csv`)

### Desafio Bônus 4: Análise de Outliers
- Identificar outliers usando IQR ou Z-score
- Visualizar outliers em boxplots
- Comparar performance dos modelos com/sem outliers

### Desafio Bônus 5: Validação Cruzada
- Implementar K-Fold Cross-Validation (k=5)
- Calcular média e desvio padrão das métricas
- Comparar estabilidade entre modelos

---

## 📚 Recursos de Apoio

### Documentação Oficial:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Conceitos Importantes:
- **Regressão**: Predição de valores contínuos
- **Clustering**: Agrupamento não-supervisionado
- **Imputação**: Preenchimento de valores faltantes
- **Normalização**: Padronização de escalas
- **Overfitting**: Quando modelo memoriza ao invés de generalizar
- **PCA**: Redução de dimensionalidade preservando variância

---

## 🎯 Critérios de Entrega

### Arquivo Principal:
- **Nome**: `solution.ipynb`
- **Formato**: Jupyter Notebook
- **Tamanho**: Não há limite, mas seja eficiente

### Arquivos Gerados:
1. `test_with_missing_features.csv` - Dataset com valores faltantes
2. `test_imputed.csv` - Dataset após imputação

### Arquivos Opcionais:
3. `predictions.csv` - Predições de qualidade (Desafio Bônus 3)
4. `README.md` - Documentação adicional do seu projeto

### Organização do Código:
- ✅ Células bem organizadas e comentadas
- ✅ Títulos e descrições em Markdown
- ✅ Outputs de todas as células visíveis
- ✅ Código executável do início ao fim
- ✅ Seed fixado para reprodutibilidade

---

## 💡 Dicas para Sucesso

### Gerais:
1. **Comece pela EDA**: Entenda seus dados antes de modelar
2. **Documente tudo**: Explique suas escolhas e descobertas
3. **Visualize bastante**: Um gráfico vale mais que mil números
4. **Valide sempre**: Compare resultados com expectativas

### Específicas de Imputação:
1. Use `random_state=42` para reprodutibilidade
2. Normalize dados antes de treinar modelos
3. Imputação iterativa é melhor que single-pass
4. Valide que distribuições se preservam após imputação

### Otimização:
1. Use `n_jobs=-1` em modelos para paralelizar
2. Comece com poucos estimadores, aumente depois
3. Cache intermediários para evitar reprocessamento
4. Monitore tempo de execução das células

---

## ❓ FAQ - Perguntas Frequentes

**Q: Posso usar modelos diferentes dos sugeridos?**  
R: Sim! Sinta-se livre para experimentar. Os modelos sugeridos são apenas um ponto de partida.

**Q: Preciso fazer todos os desafios bônus?**  
R: Não, são opcionais. Faça se quiser explorar mais ou melhorar sua pontuação.

**Q: Como sei se minha imputação está boa?**  
R: Compare as distribuições visualmente e numericamente. Diferenças < 5% na média são excelentes.

**Q: Posso usar técnicas de deep learning?**  
R: Sim, mas não é necessário. Métodos tradicionais já resolvem bem este problema.

**Q: Quanto tempo devo levar para completar?**  
R: Estimativa: 4-6 horas para todas as etapas obrigatórias. Bônus podem adicionar 2-4 horas.

**Q: Posso trabalhar em equipe?**  
R: Depende das regras do seu instrutor. Mas aprender colaborando é sempre bom!

---

## 🚀 Começando

1. Clone/baixe os arquivos do desafio
2. Configure seu ambiente Python com as bibliotecas necessárias
3. Abra o Jupyter Notebook
4. Siga as etapas na ordem sugerida
5. Documente suas descobertas e insights
6. Divirta-se explorando os dados! 🎉

---

## 📊 Checklist Final

Antes de submeter, verifique:

- [ ] Todas as 5 etapas foram completadas
- [ ] Todas as células executam sem erros
- [ ] Visualizações estão renderizadas
- [ ] Código está comentado e organizado
- [ ] Arquivos CSV foram gerados corretamente
- [ ] Métricas de validação atendem aos critérios mínimos
- [ ] Notebook conta uma "história" clara dos dados

---

## 🎓 Aprendizados Esperados

Ao completar este desafio, você terá praticado:

✅ Análise exploratória de dados completa  
✅ Visualização de dados com matplotlib e seaborn  
✅ Modelagem preditiva com múltiplos algoritmos  
✅ Avaliação e comparação de modelos  
✅ Clustering não-supervisionado  
✅ Redução de dimensionalidade com PCA  
✅ Imputação inteligente de dados faltantes  
✅ Validação de resultados e qualidade de dados  
✅ Boas práticas de organização de código  
✅ Pensamento crítico sobre dados e resultados  

---

## 🏅 Boa Sorte!

Este é um desafio completo e realista de Data Science. Não se preocupe se não conseguir tudo de primeira - o importante é aprender no processo!

**Lembre-se**: O objetivo não é apenas completar, mas **entender** cada etapa e ser capaz de explicar suas escolhas.

---

**Versão**: 1.0  
**Autor do Desafio**: Competição Neuron UFLA  
**Data**: 2025  
**Licença**: Educacional

---

*"In God we trust, all others must bring data." - W. Edwards Deming*
