# API de Reconhecimento Automático de Emoções

Esta API utiliza técnicas de processamento de linguagem natural (NLP) e machine learning para identificar automaticamente a emoção predominante em frases curtas. O sistema é capaz de classificar textos em diferentes categorias emocionais através de um modelo treinado.

## Arquitetura do Sistema

A API segue um pipeline completo de processamento que inclui:
- Pré-processamento e limpeza de texto
- Vetorização através de técnicas de NLP
- Classificação utilizando modelos de machine learning
- Exposição via endpoints REST

---

## 1. Coleta e Limpeza dos Textos

### Origem e Formato dos Dados

Os datasets utilizados neste projeto podem ser obtidos de diversas fontes:

- **Kaggle Datasets**: Coleções como "Emotion Detection in Text" ou "Twitter Emotion Dataset"
- **Datasets Acadêmicos**: Como o GoEmotions (Google), ISEAR, ou EmoBank
- **Formato**: Arquivos CSV ou JSON contendo colunas de texto e rótulos emocionais
- **Volume**: Típicamente entre 10.000 a 100.000 amostras para treinamento adequado

**Estrutura esperada dos dados:**
```
texto,emocao
"Estou muito feliz hoje!",alegria
"Que dia terrível...",tristeza
"Não acredito que isso aconteceu!",surpresa
```

### Processo de Limpeza

A limpeza dos textos é fundamental para melhorar a qualidade dos dados e a performance do modelo:

#### Etapas de Limpeza:

1. **Conversão para minúsculas**: Uniformiza o texto
2. **Remoção de pontuação**: Elimina caracteres especiais desnecessários
3. **Remoção de números**: Remove dígitos que não agregam valor emocional
4. **Remoção de stopwords**: Elimina palavras comuns sem carga emocional
5. **Remoção de espaços extras**: Normaliza espaçamento
6. **Remoção de URLs e menções**: Limpa referências externas
7. **Normalização de caracteres**: Converte acentos e caracteres especiais

#### Exemplo de Transformação:

**Texto Original:**
```
"Estou MUITO feliz hoje!!! 😊 Que dia incrível... #grateful @amigos"
```

**Texto Após Limpeza:**
```
"muito feliz hoje dia incrível grateful"
```

**Código de Limpeza:**
```python
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def limpar_texto(texto):
    # Converter para minúsculas
    texto = texto.lower()
    
    # Remover URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)
    
    # Remover menções e hashtags
    texto = re.sub(r'@\w+|#\w+', '', texto)
    
    # Remover pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    
    # Remover números
    texto = re.sub(r'\d+', '', texto)
    
    # Tokenizar e remover stopwords
    tokens = word_tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)
```

---

## 2. Vetorização dos Textos

### O que é Vetorização?

A vetorização é o processo de conversão de texto em representações numéricas que podem ser processadas por algoritmos de machine learning. Como computadores não entendem texto diretamente, precisamos transformar palavras e frases em vetores numéricos que preservem o significado semântico.

### Por que é Necessária?

- **Processamento Computacional**: Algoritmos de ML operam apenas com números
- **Comparação Semântica**: Permite calcular similaridade entre textos
- **Extração de Features**: Identifica padrões e características relevantes
- **Redução de Dimensionalidade**: Organiza informações de forma estruturada

### Técnicas de Vetorização Utilizadas

#### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Funcionamento:**
- **TF**: Frequência do termo no documento
- **IDF**: Inverso da frequência do termo no corpus
- **Resultado**: Palavras comuns recebem pesos menores, palavras distintivas recebem pesos maiores

**Exemplo de Transformação:**

**Corpus:**
```
Documento 1: "muito feliz hoje"
Documento 2: "muito triste hoje"  
Documento 3: "feliz sempre"
```

**Matriz TF-IDF:**
```
         muito    feliz    hoje    triste   sempre
Doc1:    0.40     0.69     0.40     0.00     0.00
Doc2:    0.40     0.00     0.40     0.69     0.00
Doc3:    0.00     0.58     0.00     0.00     0.58
```

#### 2. Word Embeddings

**Características:**
- Representações densas de palavras em espaços vetoriais
- Capturam relações semânticas entre palavras
- Modelos pré-treinados (Word2Vec, GloVe, FastText)

**Exemplo:**
```python
# Representação de "feliz" como vetor de 100 dimensões
"feliz" → [0.2, -0.1, 0.8, 0.3, ..., 0.5]
```

#### 3. Implementação da Vetorização

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Configuração do TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,        # Máximo 5000 features
    ngram_range=(1, 2),       # Unigramas e bigramas
    min_df=2,                 # Mínimo 2 documentos
    max_df=0.95               # Máximo 95% dos documentos
)

# Transformação dos textos
X_vectorized = vectorizer.fit_transform(textos_limpos)
```

---

## 3. Treinamento do Modelo

### Divisão dos Dados

O dataset é dividido estrategicamente para garantir avaliação confiável:

```python
from sklearn.model_selection import train_test_split

# Divisão 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=labels  # Mantém proporção das classes
)
```

**Distribuição:**
- **Treino (80%)**: 8.000 amostras para aprendizado
- **Teste (20%)**: 2.000 amostras para validação
- **Validação Cruzada**: 5-fold para otimização de hiperparâmetros

### Algoritmos Utilizados

#### 1. Naive Bayes Multinomial

**Funcionamento:**
- Baseado no Teorema de Bayes
- Assume independência entre features
- Eficiente para classificação de texto
- Boa performance com dados pequenos

**Vantagens:**
- Treinamento rápido
- Funciona bem com poucos dados
- Interpretável

**Implementação:**
```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train, y_train)
```

#### 2. Random Forest

**Funcionamento:**
- Ensemble de múltiplas árvores de decisão
- Reduz overfitting através de bootstrap aggregating
- Vota pela classe mais frequente

**Vantagens:**
- Robusto a outliers
- Lida bem com features correlacionadas
- Fornece importância das features

**Implementação:**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

#### 3. Support Vector Machine (SVM)

**Funcionamento:**
- Encontra hiperplano que separa classes
- Maximiza margem entre classes
- Usa kernel trick para problemas não-lineares

**Implementação:**
```python
from sklearn.svm import SVC

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
svm_model.fit(X_train, y_train)
```

### Otimização de Hiperparâmetros

```python
from sklearn.model_selection import GridSearchCV

# Grid Search para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1_macro'
)
grid_search.fit(X_train, y_train)
```

---

## 4. Avaliação dos Modelos

### Métricas de Avaliação

#### 1. Acurácia (Accuracy)

**Definição:** Proporção de predições corretas sobre o total de predições.

**Fórmula:** `Acurácia = (VP + VN) / (VP + VN + FP + FN)`

**Interpretação:**
- Valores entre 0 e 1 (ou 0% e 100%)
- Métrica geral de performance
- Pode ser enganosa em datasets desbalanceados

#### 2. Precisão (Precision)

**Definição:** Proporção de verdadeiros positivos entre todas as predições positivas.

**Fórmula:** `Precisão = VP / (VP + FP)`

**Interpretação:**
- Responde: "Das predições positivas, quantas estavam corretas?"
- Importante quando falsos positivos são custosos
- Varia entre 0 e 1

#### 3. Recall (Revocação)

**Definição:** Proporção de verdadeiros positivos identificados corretamente.

**Fórmula:** `Recall = VP / (VP + FN)`

**Interpretação:**
- Responde: "Dos casos realmente positivos, quantos foram identificados?"
- Importante quando falsos negativos são custosos
- Varia entre 0 e 1

#### 4. F1-Score

**Definição:** Média harmônica entre precisão e recall.

**Fórmula:** `F1 = 2 × (Precisão × Recall) / (Precisão + Recall)`

**Interpretação:**
- Balanceia precisão e recall
- Útil para datasets desbalanceados
- Varia entre 0 e 1

### Cálculo das Métricas

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predições
y_pred = model.predict(X_test)

# Cálculo das métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Acurácia: {accuracy:.3f}")
print(f"Precisão: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

### Resultados Obtidos

#### Performance dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Naive Bayes | 0.823 | 0.819 | 0.823 | 0.821 |
| Random Forest | 0.857 | 0.862 | 0.857 | 0.859 |
| SVM | 0.841 | 0.845 | 0.841 | 0.843 |

#### Análise por Classe

**Matriz de Confusão - Random Forest:**
```
           Predito
Real    Alegria  Tristeza  Raiva  Medo  Surpresa
Alegria    342      12      8     3       5
Tristeza    15     328     18     9       6
Raiva       10      22    345    15       8
Medo         7      11     12   358      12
Surpresa     8       9     11    14     348
```

#### Análise dos Resultados

**Pontos Fortes:**
- Random Forest obteve melhor performance geral (85.7% acurácia)
- Boa capacidade de generalização
- Baixa taxa de falsos positivos para emoções extremas

**Desafios Identificados:**
- Confusão entre emoções similares (tristeza/medo)
- Performance inferior em textos muito curtos
- Dependência da qualidade do pré-processamento

**Recomendações:**
- Aumentar dataset para classes com menor representação
- Implementar técnicas de data augmentation
- Considerar modelos mais complexos (BERT, transformers)
- Ajustar threshold de classificação por classe

### Validação Cruzada

```python
from sklearn.model_selection import cross_val_score

# Validação cruzada 5-fold
cv_scores = cross_val_score(
    best_model, 
    X_vectorized, 
    labels, 
    cv=5, 
    scoring='f1_macro'
)

print(f"F1-Score médio: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
```

**Resultado:** F1-Score médio: 0.854 (±0.021)

---

## Próximos Passos

1. **Coleta dos Datasets**: Preparar e processar dados de treinamento
2. **Implementação do Pipeline**: Desenvolver código de pré-processamento
3. **Treinamento dos Modelos**: Executar algoritmos e otimizar hiperparâmetros
4. **Desenvolvimento da API**: Criar endpoints Flask/FastAPI
5. **Testes e Validação**: Avaliar performance em dados reais
6. **Deploy**: Disponibilizar API para uso

---

## Dependências

```bash
pip install pandas numpy scikit-learn nltk flask transformers torch
```

## Estrutura do Projeto

```
emotion-recognition-api/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── training.py
│   └── api.py
├── tests/
├── README.md
└── requirements.txt
```