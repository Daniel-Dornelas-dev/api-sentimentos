# API de Reconhecimento de Emoções

Sistema de análise automática de emoções em frases curtas utilizando Machine Learning. A API recebe uma frase via endpoint e retorna a emoção predominante detectada.

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**
- **Flask** (API REST)
- **scikit-learn** (ML)
- **NLTK** (processamento de texto)
- **pandas** (manipulação de dados)
- **numpy** (operações numéricas)

---

## 1. Coleta e Limpeza dos Textos

### 1.1 Origem dos Dados

O dataset utilizado contém frases rotuladas com emoções básicas:
- **Formato**: CSV com colunas `text` e `emotion`
- **Volume**: ~10.000 frases
- **Categorias**: alegria, tristeza, raiva, medo, surpresa, neutro

### 1.2 Processo de Limpeza

O pré-processamento segue as seguintes etapas:

1. **Conversão para minúsculas**
2. **Remoção de pontuação** (exceto emoticons)
3. **Remoção de stopwords** (palavras sem valor semântico)
4. **Tokenização** (divisão em palavras)
5. **Stemming** (redução às raízes das palavras)

### 1.3 Exemplo Prático

```python
# Texto original
"Estou MUITO feliz com este resultado! 😊"

# Após limpeza
"muito feliz result"
```

**Processo detalhado:**
- Minúsculas: "estou muito feliz com este resultado! 😊"
- Sem pontuação: "estou muito feliz com este resultado 😊"
- Sem stopwords: "muito feliz resultado 😊"
- Stemming: "muito feliz result"

---

## 2. Vetorização dos Textos

### 2.1 Conceito

Vetorização transforma texto em representação numérica para que algoritmos de ML possam processá-lo. Cada palavra/frase torna-se um vetor de números que representa suas características semânticas.

### 2.2 Métodos Implementados

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- **TF**: Frequência do termo no documento
- **IDF**: Importância do termo no corpus
- **Fórmula**: TF-IDF = TF × log(N/df)

#### Word Embeddings
- Representação densa em espaço vetorial
- Captura relações semânticas entre palavras
- Dimensionalidade: 100-300 dimensões

### 2.3 Exemplo de Transformação

```python
# Frase: "muito feliz"
# TF-IDF (vocabulário simplificado)
{
    "muito": 0.47,
    "feliz": 0.88,
    "triste": 0.0,
    "raiva": 0.0
}

# Resultado: [0.47, 0.88, 0.0, 0.0, ...]
```

---

## 3. Treinamento do Modelo

### 3.1 Divisão dos Dados

- **Treino**: 80% (8.000 frases)
- **Teste**: 20% (2.000 frases)
- **Validação**: Estratificada por classe

### 3.2 Algoritmos Utilizados

#### Naive Bayes Multinomial
- **Funcionamento**: Calcula probabilidade de cada classe usando teorema de Bayes
- **Vantagem**: Eficiente para classificação de texto
- **Fórmula**: P(classe|texto) = P(texto|classe) × P(classe) / P(texto)

#### Random Forest
- **Funcionamento**: Ensemble de árvores de decisão
- **Parâmetros**: 100 árvores, profundidade máxima = 20
- **Vantagem**: Reduz overfitting e melhora generalização

#### BERT (Bidirectional Encoder Representations)
- **Funcionamento**: Transformer pré-treinado com atenção bidirecional
- **Fine-tuning**: Ajuste das camadas finais para classificação de emoções
- **Vantagem**: Compreensão contextual avançada

### 3.3 Processo de Treinamento

```python
# Exemplo simplificado
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Otimização de hiperparâmetros
param_grid = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(model, param_grid, cv=5)
```

---

## 4. Avaliação dos Modelos

### 4.1 Métricas Utilizadas

#### Acurácia
- **Definição**: Proporção de predições corretas
- **Fórmula**: (VP + VN) / (VP + VN + FP + FN)
- **Interpretação**: Medida geral de performance

#### Precisão
- **Definição**: Proporção de predições positivas corretas
- **Fórmula**: VP / (VP + FP)
- **Interpretação**: Qualidade das predições positivas

#### Recall (Sensibilidade)
- **Definição**: Proporção de casos positivos identificados
- **Fórmula**: VP / (VP + FN)
- **Interpretação**: Capacidade de encontrar casos positivos

#### F1-Score
- **Definição**: Média harmônica entre precisão e recall
- **Fórmula**: 2 × (Precisão × Recall) / (Precisão + Recall)

### 4.2 Resultados Obtidos

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Naive Bayes | 0.82 | 0.81 | 0.80 | 0.80 |
| Random Forest | 0.85 | 0.84 | 0.83 | 0.83 |
| BERT | 0.91 | 0.90 | 0.89 | 0.89 |

### 4.3 Análise dos Resultados

**Pontos Fortes:**
- BERT apresentou melhor performance geral
- Boa capacidade de generalização em todas as classes
- Baixa taxa de falsos positivos

**Limitações:**
- Dificuldade em distinguir tristeza/melancolia
- Performance inferior em textos muito curtos (<3 palavras)
- Sensibilidade a contexto cultural em expressões idiomáticas

**Recomendação:** BERT como modelo principal, com Random Forest como fallback para casos de baixa confiança.

---

## 🚀 Uso da API

```python
# Endpoint principal
POST /predict
{
    "text": "Estou muito feliz hoje!"
}

# Resposta
{
    "emotion": "alegria",
    "confidence": 0.89,
    "processing_time": 0.02
}
```