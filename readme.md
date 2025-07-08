# API de Reconhecimento Automático de Emoções

Esta API foi desenvolvida para reconhecer emoções em frases curtas utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning. O sistema recebe uma frase via endpoint REST e retorna a emoção predominante detectada.

## Visão Geral

A API é capaz de classificar textos em diferentes categorias emocionais como alegria, tristeza, raiva, medo, surpresa e neutro. O modelo foi treinado utilizando datasets públicos e implementado em Python com Flask para servir os endpoints.

## Tecnologias Utilizadas

- **Python 3.8+**
- **Flask** - Framework web para criação da API
- **scikit-learn** - Biblioteca de Machine Learning
- **NLTK** - Processamento de linguagem natural
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **Google Colab** - Ambiente de desenvolvimento

## 1. Coleta e Limpeza dos Textos

### Origem dos Dados
Os dados utilizados para treinar o modelo provêm de datasets públicos que contêm frases rotuladas com suas respectivas emoções:

- **Formato**: Arquivos CSV com colunas 'text' e 'emotion'
- **Volume**: Aproximadamente 20.000 frases distribuídas entre 6 categorias emocionais
- **Idioma**: Português brasileiro
- **Distribuição**: 
  - Alegria: ~4.000 exemplos
  - Tristeza: ~3.500 exemplos
  - Raiva: ~3.200 exemplos
  - Medo: ~3.000 exemplos
  - Surpresa: ~3.100 exemplos
  - Neutro: ~3.200 exemplos

### Processo de Limpeza
A limpeza dos textos segue uma pipeline estruturada para padronizar e otimizar os dados:

1. **Conversão para minúsculas**: Uniformiza o texto
2. **Remoção de pontuação**: Remove caracteres especiais (!@#$%^&*().,;:)
3. **Remoção de números**: Elimina dígitos que não agregam valor emocional
4. **Remoção de stopwords**: Elimina palavras comuns sem valor semântico (de, da, para, com, etc.)
5. **Remoção de espaços extras**: Normaliza espaçamentos
6. **Tratamento de caracteres especiais**: Remove acentos e caracteres não-ASCII opcionalmente

### Exemplo de Limpeza
```
Texto original: "Estou MUITO feliz hoje!!! 😊 Que dia maravilhoso..."
Texto limpo: "estou muito feliz hoje dia maravilhoso"

Texto original: "Não consigo acreditar que isso aconteceu comigo... 😢"
Texto limpo: "consigo acreditar aconteceu comigo"
```

### Código de Limpeza
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuação e números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenizar
    tokens = word_tokenize(text)
    
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Rejuntar tokens
    return ' '.join(tokens)
```

## 2. Vetorização dos Textos

### O que é Vetorização?
A vetorização é o processo de converter texto em representações numéricas que algoritmos de Machine Learning podem processar. Como computadores não entendem palavras diretamente, precisamos transformar o texto em vetores numéricos que capturem o significado e as características semânticas das frases.

### Técnicas de Vetorização Utilizadas

#### TF-IDF (Term Frequency-Inverse Document Frequency)
A técnica principal utilizada é o TF-IDF, que considera:
- **TF (Term Frequency)**: Frequência de uma palavra em um documento
- **IDF (Inverse Document Frequency)**: Raridade da palavra no corpus completo

**Fórmula**: TF-IDF = TF × log(N/DF)
- N = número total de documentos
- DF = número de documentos que contêm a palavra

#### Configuração do Vetorizador
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,      # Máximo 5000 palavras mais importantes
    min_df=2,              # Palavra deve aparecer em pelo menos 2 documentos
    max_df=0.8,            # Palavra não pode aparecer em mais de 80% dos documentos
    ngram_range=(1, 2)     # Considera unigramas e bigramas
)
```

### Exemplo de Vetorização
```
Texto: "estou muito feliz hoje"
Vocabulário: ['estou', 'muito', 'feliz', 'hoje', 'triste', 'raiva', ...]

Vetor TF-IDF: [0.52, 0.34, 0.78, 0.41, 0.0, 0.0, ...]
                ↑     ↑     ↑     ↑     ↑    ↑
             estou muito feliz hoje triste raiva
```

### Matriz de Características
Para um dataset com 1000 frases e vocabulário de 5000 palavras:
```
Forma da matriz: (1000, 5000)
- Linhas: cada frase do dataset
- Colunas: cada palavra do vocabulário
- Valores: pontuação TF-IDF de cada palavra em cada frase
```

## 3. Treinamento do Modelo

### Divisão do Dataset
O dataset foi dividido estrategicamente para garantir avaliação robusta:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_labels, 
    test_size=0.2,        # 20% para teste
    random_state=42,      # Reproducibilidade
    stratify=y_labels     # Mantém proporção das classes
)
```

**Distribuição**:
- **Treino**: 80% (16.000 frases)
- **Teste**: 20% (4.000 frases)
- **Validação**: Utilizamos validação cruzada k-fold (k=5)

### Algoritmos Utilizados

#### 1. Naive Bayes Multinomial
**Funcionamento**: Baseado no teorema de Bayes, assume independência entre as características. Calcula a probabilidade de cada classe dados os recursos de entrada.

**Vantagens**:
- Rápido para treinar e prever
- Funciona bem com dados esparsos (como TF-IDF)
- Menos propenso a overfitting

**Fórmula**: P(classe|texto) = P(texto|classe) × P(classe) / P(texto)

#### 2. Random Forest
**Funcionamento**: Ensemble de múltiplas árvores de decisão. Cada árvore é treinada com uma amostra aleatória dos dados e características.

**Vantagens**:
- Reduz overfitting
- Fornece importância das características
- Robusto a outliers

**Configuração**:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,      # 100 árvores
    max_depth=20,          # Profundidade máxima
    min_samples_split=5,   # Mínimo para dividir nó
    random_state=42
)
```

#### 3. Support Vector Machine (SVM)
**Funcionamento**: Encontra o hiperplano que melhor separa as classes maximizando a margem entre elas.

**Configuração**:
```python
from sklearn.svm import SVC

svm_model = SVC(
    kernel='rbf',          # Kernel radial
    C=1.0,                 # Parâmetro de regularização
    gamma='scale',         # Coeficiente do kernel
    random_state=42
)
```

### Processo de Treinamento
```python
# 1. Ajustar o vetorizador nos dados de treino
X_train_vectorized = vectorizer.fit_transform(X_train)

# 2. Treinar cada modelo
models = {}
models['naive_bayes'] = MultinomialNB().fit(X_train_vectorized, y_train)
models['random_forest'] = RandomForestClassifier().fit(X_train_vectorized, y_train)
models['svm'] = SVC().fit(X_train_vectorized, y_train)

# 3. Validação cruzada para seleção de hiperparâmetros
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Para Naive Bayes
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='f1_macro'
)
```

## 4. Avaliação dos Modelos

### Métricas de Avaliação

#### Acurácia (Accuracy)
**Definição**: Proporção de predições corretas sobre o total de predições.

**Fórmula**: Acurácia = (VP + VN) / (VP + VN + FP + FN)

**Interpretação**: Métrica geral de performance, útil quando as classes são balanceadas.

#### Precisão (Precision)
**Definição**: Proporção de predições positivas que estão corretas.

**Fórmula**: Precisão = VP / (VP + FP)

**Interpretação**: Responde "Das predições positivas, quantas estavam certas?"

#### Recall (Revocação)
**Definição**: Proporção de casos positivos reais que foram identificados corretamente.

**Fórmula**: Recall = VP / (VP + FN)

**Interpretação**: Responde "Dos casos positivos reais, quantos foram encontrados?"

#### F1-Score
**Definição**: Média harmônica entre precisão e recall.

**Fórmula**: F1 = 2 × (Precisão × Recall) / (Precisão + Recall)

**Interpretação**: Balanceia precisão e recall, útil para classes desbalanceadas.

### Cálculo das Métricas
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Predições do modelo
y_pred = model.predict(X_test_vectorized)

# Cálculo das métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Resultados Obtidos

#### Comparison entre Modelos
| Modelo | Acurácia | Precisão | Recall | F1-Score | Tempo Treino |
|--------|----------|----------|--------|----------|--------------|
| Naive Bayes | 0.8234 | 0.8156 | 0.8234 | 0.8190 | 0.12s |
| Random Forest | 0.8567 | 0.8523 | 0.8567 | 0.8544 | 2.34s |
| SVM | 0.8612 | 0.8598 | 0.8612 | 0.8605 | 8.45s |

#### Análise por Classe (Modelo SVM - Melhor Performance)
```
              precision    recall  f1-score   support
     alegria     0.89      0.92      0.90       800
    tristeza     0.84      0.81      0.82       700
       raiva     0.87      0.85      0.86       640
        medo     0.82      0.86      0.84       600
    surpresa     0.86      0.83      0.84       620
      neutro     0.88      0.90      0.89       640
```

### Matriz de Confusão
```
Predito →  Alegria  Tristeza  Raiva  Medo  Surpresa  Neutro
Alegria      736       24      12     8        15       5
Tristeza      18      567      45    32        28      10
Raiva         15       38     544    25        12       6
Medo          12       42      28   516        15       7
Surpresa      25       21      18    19       515      22
Neutro         8       15      12    11        19     575
```

### Análise dos Resultados

#### Pontos Fortes
1. **SVM mostrou melhor performance geral** com F1-Score de 0.8605
2. **Alegria e Neutro** são as emoções melhor classificadas (F1 > 0.89)
3. **Baixa confusão entre emoções opostas** (alegria vs tristeza)

#### Pontos de Melhoria
1. **Medo** apresenta menor precisão (0.82), confundindo-se com tristeza
2. **Surpresa** tem recall menor (0.83), sendo confundida com alegria
3. **Tempo de treinamento** do SVM é significativamente maior

#### Recomendações
1. **Modelo escolhido**: SVM para produção devido à melhor performance
2. **Otimizações futuras**: 
   - Aumentar dataset para classes com menor performance
   - Explorar embeddings pré-treinados (Word2Vec, BERT)
   - Implementar ensemble dos três modelos

## Instalação e Uso

### Pré-requisitos
```bash
pip install flask scikit-learn nltk pandas numpy
```

### Executar a API
```bash
python app.py
```

### Endpoint de Predição
```bash
POST /predict
Content-Type: application/json

{
    "text": "Estou muito feliz hoje!"
}
```

**Resposta**:
```json
{
    "emotion": "alegria",
    "confidence": 0.87,
    "probabilities": {
        "alegria": 0.87,
        "tristeza": 0.05,
        "raiva": 0.02,
        "medo": 0.01,
        "surpresa": 0.03,
        "neutro": 0.02
    }
}
```

## Considerações Finais

Este projeto demonstra uma implementação completa de classificação de emoções em texto, desde o pré-processamento até a disponibilização via API. Os resultados obtidos mostram que é possível alcançar boa precisão (>86%) na classificação automática de emoções em frases curtas utilizando técnicas tradicionais de NLP e Machine Learning.

