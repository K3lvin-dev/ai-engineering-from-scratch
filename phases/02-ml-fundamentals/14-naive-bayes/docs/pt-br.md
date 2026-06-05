# Naive Bayes

> A suposição "ingênua" está errada, e funciona mesmo assim. Essa é a beleza disso.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 2, Aulas 01-07 (classificação, teorema de Bayes)
**Tempo:** ~75 minutos

## Objetivos de Aprendizado

- Implementar Multinomial Naive Bayes do zero com suavização Laplace para classificação de texto
- Explicar por que a suposição de independência ingênua é matematicamente errada mas produz classificações corretas na prática
- Comparar as variantes Multinomial, Bernoulli e Gaussian Naive Bayes e selecionar a correta para um dado tipo de feature
- Avaliar Naive Bayes contra regressão logística em dados esparsos de alta dimensão e explicar o trade-off viés-variância em ação

## O Problema

Você precisa classificar texto. Emails em spam ou não-spam. Reviews de clientes em positivos ou negativos. Tickets de suporte em categorias. Você tem milhares de features (uma por palavra) e dados de treino limitados.

A maioria dos classificadores trava aqui. Regressão logística precisa de amostras suficientes para estimar milhares de pesos de forma confiável. Árvores de decisão dividem em uma palavra de cada vez e overfittam violentamente. KNN em 10.000 dimensões não faz sentido porque cada ponto está igualmente distante de todos os outros.

Naive Bayes lida com isso. Ele faz uma suposição matematicamente errada (que toda feature é independente de toda outra feature dada a classe), e ainda assim supera modelos "mais inteligentes" em classificação de texto, especialmente com conjuntos de treino pequenos. Ele treina em uma única passada pelos dados. Ele escala para milhões de features. Ele produz estimativas de probabilidade (embora frequentemente mal calibradas devido à suposição de independência).

Entender por que uma suposição errada leva a boas previsões te ensina algo fundamental sobre machine learning: o melhor modelo não é o mais correto, é o com o melhor trade-off viés-variância para seus dados.

## O Conceito

### Teorema de Bayes (Revisão Rápida)

O teorema de Bayes inverte probabilidades condicionais:

```
P(classe | features) = P(features | classe) * P(classe) / P(features)
```

Queremos `P(classe | features)` — a probabilidade de um documento pertencer a uma classe dadas as palavras nele. Podemos computar isso a partir de:
- `P(features | classe)` — a verossimilhança de ver estas palavras em documentos desta classe
- `P(classe)` — a probabilidade a priori da classe (quão comum é spam em geral?)
- `P(features)` — a evidência, igual para todas as classes, então podemos ignorá-la ao comparar

A classe com maior `P(classe | features)` vence.

### A Suposição de Independência Ingênua

Computar `P(features | classe)` exatamente requer estimar a probabilidade conjunta de todas as features juntas. Com um vocabulário de 10.000 palavras, você precisaria estimar uma distribuição sobre 2^10.000 combinações possíveis. Impossível.

A suposição ingênua: toda feature é condicionalmente independente dada a classe.

```
P(w1, w2, ..., wn | classe) = P(w1 | classe) * P(w2 | classe) * ... * P(wn | classe)
```

Em vez de uma distribuição conjunta impossível, você estima n distribuições simples por feature. Cada uma precisa apenas de uma contagem.

Esta suposição é obviamente errada. As palavras "machine" e "learning" não são independentes em nenhum documento. Mas o classificador não precisa de estimativas de probabilidade corretas. Ele precisa de rankings corretos — qual classe tem a maior probabilidade. A suposição de independência introduz erros sistemáticos, mas esses erros afetam todas as classes similarmente, então o ranking permanece correto.

### Por Que Ainda Funciona

Três razões:

1. **Ranking acima de calibração.** Classificação só precisa que a classe do topo do ranking esteja correta. Mesmo que P(spam) = 0.99999 quando a probabilidade verdadeira é 0.7, o classificador ainda escolhe spam corretamente. Não precisamos de probabilidades corretas. Precisamos do vencedor correto.

2. **Viés alto, variância baixa.** A suposição de independência é um prior forte. Ela restringe o modelo pesadamente, o que previne overfitting. Com dados de treino limitados, um modelo ligeiramente errado mas estável vence um modelo teoricamente correto mas violentamente instável. Este é o trade-off viés-variância em ação.

3. **Redundância de features se cancela.** Features correlacionadas fornecem evidência redundante. O classificador conta essa evidência duas vezes, mas a conta duas vezes para a classe correta também. Se "machine" e "learning" sempre aparecem juntas, ambas fornecem evidência para a classe "tech". NB as conta duas vezes, mas as conta duas vezes para a classe certa.

Uma quarta razão prática: Naive Bayes é extremamente rápido. Treinar é uma única passada pelos dados contando frequências. Predição é uma multiplicação de matrizes. Você pode treinar em um milhão de documentos em segundos. Essa velocidade significa que você pode iterar mais rápido, testar mais conjuntos de features e executar mais experimentos do que com modelos mais lentos.

### A Matemática Passo a Passo

Vamos percorrer um exemplo concreto. Suponha que temos duas classes: spam e não-spam. Nosso vocabulário tem três palavras: "grátis", "dinheiro", "reunião".

Dados de treino:
- Spam menciona "grátis" 80 vezes, "dinheiro" 60 vezes, "reunião" 10 vezes (150 palavras no total)
- Não-spam menciona "grátis" 5 vezes, "dinheiro" 10 vezes, "reunião" 100 vezes (115 palavras no total)
- 40% dos emails são spam, 60% são não-spam

Com suavização Laplace (alpha=1):

```
P(grátis | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(dinheiro | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(reunião | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(grátis | não-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(dinheiro | não-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(reunião | não-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Novo email contém: "grátis" (2 vezes), "dinheiro" (1 vez), "reunião" (0 vezes).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(não-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam vence por uma larga margem. A palavra "grátis" aparecendo duas vezes é forte evidência para spam. Note que "reunião" não aparecer contribui zero para ambas as somas log (0 * log(P)) — no Multinomial NB, palavras ausentes não têm efeito. É o Bernoulli NB que modela explicitamente a ausência de palavras.

### Três Variantes

O Naive Bayes vem em três sabores. Cada um modela `P(feature | classe)` diferentemente.

#### Multinomial Naive Bayes

Modela cada feature como uma contagem. Melhor para dados de texto onde features são frequências de palavras ou valores TF-IDF.

```
P(palavra_i | classe) = (contagem de palavra_i na classe + alpha) / (total de palavras na classe + alpha * tam_vocabulario)
```

O `alpha` é a suavização Laplace (explicada abaixo). Esta variante é o cavalo de batalha para classificação de texto.

#### Gaussian Naive Bayes

Modela cada feature como uma distribuição normal. Melhor para features contínuas.

```
P(x_i | classe) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - media)^2 / (2 * var))
```

Cada classe obtém sua própria média e variância por feature. Funciona bem quando as features realmente seguem uma curva sino dentro de cada classe.

#### Bernoulli Naive Bayes

Modela cada feature como binária (presente ou ausente). Melhor para texto curto ou vetores de features binários.

```
P(palavra_i | classe) = (docs na classe contendo palavra_i + alpha) / (total de docs na classe + 2 * alpha)
```

Diferente do Multinomial, Bernoulli penaliza explicitamente a ausência de uma palavra. Se "grátis" tipicamente aparece em spam mas está ausente deste email, Bernoulli conta isso como evidência contra spam.

### Quando Usar Cada Variante

| Variante | Tipo de Feature | Melhor Para | Exemplo |
|---------|----------------|-------------|---------|
| Multinomial | Contagens ou frequências | Classificação de texto, bag-of-words | Spam email, classificação de tópicos |
| Gaussian | Valores contínuos | Dados tabulares com features aproximadamente normais | Classificação Iris, dados de sensores |
| Bernoulli | Binário (0/1) | Texto curto, vetores binários | Spam SMS, features de presença/ausência |

### Suavização Laplace

O que acontece quando uma palavra aparece nos dados de teste mas nunca apareceu nos dados de treino para uma classe particular?

Sem suavização: `P(palavra | classe) = 0/N = 0`. Um zero multiplicado por todo o produto faz `P(classe | features) = 0`, independentemente de toda outra evidência. Uma única palavra não vista destrói toda a predição, não importa quanta outra evidência a suporte.

A suavização Laplace adiciona uma pequena contagem `alpha` (geralmente 1) a cada contagem de feature:

```
P(palavra_i | classe) = (contagem(palavra_i, classe) + alpha) / (total_palavras_na_classe + alpha * tam_vocabulario)
```

Com alpha=1, cada palavra recebe pelo menos uma probabilidade minúscula. A palavra "descombobular" aparecendo em um email de teste não mata mais a probabilidade de spam. A suavização tem uma interpretação Bayesiana: é equivalente a colocar um prior Dirichlet uniforme nas distribuições de palavras.

Alpha mais alto significa suavização mais forte (distribuições mais uniformes). Alpha mais baixo significa que o modelo confia mais nos dados. Alpha é um hiperparâmetro que você ajusta.

O efeito de alpha:

| Alpha | Efeito | Quando usar |
|-------|--------|-------------|
| 0.001 | Quase sem suavização, confia nos dados | Conjunto de treino muito grande, nenhuma feature não vista esperada |
| 0.1 | Suavização leve | Conjunto de treino grande |
| 1.0 | Suavização Laplace padrão | Ponto de partida padrão |
| 10.0 | Suavização pesada, achata distribuições | Conjunto de treino muito pequeno, muitas features não vistas esperadas |

### Computação em Espaço Log

Multiplicar centenas de probabilidades (cada uma menor que 1) causa underflow de ponto flutuante. O produto se torna zero em ponto flutuante mesmo que o valor verdadeiro seja um número positivo muito pequeno.

A solução: trabalhe em espaço log. Em vez de multiplicar probabilidades, some seus logaritmos:

```
log P(classe | x1, x2, ..., xn) = log P(classe) + sum_i log P(xi | classe)
```

Isso transforma a predição em um produto escalar:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Multiplicação de matrizes. É por isso que a predição do Naive Bayes é tão rápida — é a mesma operação que um modelo linear de camada única.

### Naive Bayes vs Regressão Logística

Ambos são classificadores lineares para texto. A diferença está no que eles modelam.

| Aspecto | Naive Bayes | Regressão Logística |
|--------|------------|-------------------|
| Tipo | Generativo (modela P(X\|Y)) | Discriminativo (modela P(Y\|X)) |
| Treino | Contar frequências | Otimizar função de perda |
| Dados pequenos | Melhor (prior forte ajuda) | Pior (não o suficiente para estimar pesos) |
| Dados grandes | Pior (suposição errada prejudica) | Melhor (fronteira flexível) |
| Features | Assume independência | Lida com correlações |
| Velocidade | Única passada, muito rápido | Otimização iterativa |
| Calibração | Probabilidades ruins | Probabilidades melhores |

Regra prática: comece com Naive Bayes. Se você tem dados suficientes e NB estagna, mude para regressão logística.

### Pipeline de Classificação

```mermaid
flowchart LR
    A[Texto Bruto] --> B[Tokenizar]
    B --> C[Construir Vocabulário]
    C --> D[Contar Frequências de Palavras]
    D --> E[Aplicar Suavização]
    E --> F[Computar Log Probabilidades]
    F --> G[Predizer: argmax P classe dadas palavras]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

Na prática, trabalhamos em espaço log para evitar underflow de ponto flutuante. Em vez de multiplicar muitas probabilidades pequenas, somamos seus logaritmos:

```
log P(classe | features) = log P(classe) + sum_i log P(feature_i | classe)
```

## Construa

O código em `code/naive_bayes.py` implementa tanto MultinomialNB quanto GaussianNB do zero.

### MultinomialNB

A implementação feita do zero:

1. **fit(X, y)**: Para cada classe, conte a frequência de cada feature. Adicione suavização Laplace. Compute log probabilidades. Armazene priors das classes (log das frequências das classes).

2. **predict_log_proba(X)**: Para cada amostra, compute log P(classe) + soma de log P(feature_i | classe) para todas as classes. Isto é uma multiplicação de matrizes: X @ log_probs.T + log_priors.

3. **predict(X)**: Retorne a classe com maior log probabilidade.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

A percepção chave: após o ajuste, a predição é apenas multiplicação de matrizes mais um viés. É por isso que Naive Bayes é tão rápido.

### GaussianNB

Para features contínuas, estimamos média e variância por classe por feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

A predição usa a PDF Gaussiana por feature, multiplicada entre features (somada em espaço log).

### Demo: Classificação de Texto

O código gera dados bag-of-words sintéticos simulando duas classes (artigos de tech vs artigos de esportes). Cada classe tem uma distribuição de frequência de palavras diferente. MultinomialNB os classifica usando contagens de palavras.

Os dados sintéticos funcionam assim: criamos 200 "palavras" (colunas de features). Palavras 0-39 têm alta frequência em artigos de tech e baixa em esportes. Palavras 80-119 têm alta frequência em esportes e baixa em tech. Palavras 40-79 são frequência média em ambos. Isso cria um cenário realista onde algumas palavras são fortes indicadores de classe e outras são ruído.

### Demo: Features Contínuas

O código gera dados similares a Iris (3 classes, 4 features, clusters Gaussianos). GaussianNB classifica usando média e variância por classe. Cada classe tem um centro diferente (vetor de médias) e uma dispersão diferente (variância), imitando dados do mundo real onde medições diferem sistematicamente entre categorias.

O código também demonstra:
- **Comparação de suavização:** Treinar MultinomialNB com diferentes valores de alpha para mostrar o efeito da força da suavização na acurácia.
- **Experimento de tamanho de treino:** Como a acurácia do NB melhora conforme os dados de treino crescem de 20 para 1600 amostras. NB atinge acurácia decente mesmo com muito poucas amostras — esta é sua principal vantagem.
- **Matriz de confusão:** Precisão, recall e F1 por classe para mostrar onde NB comete erros.

### Velocidade de Predição

A predição do Naive Bayes é uma multiplicação de matrizes. Para n amostras com d features e k classes:
- MultinomialNB: uma multiplicação de matrizes (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k avaliações de PDF Gaussiana, cada uma sobre d features = O(n * d * k)

Ambos são lineares em todas as dimensões. Compare com KNN (que requer computação de distância para todos os pontos de treino) ou SVM com kernel RBF (que requer avaliação de kernel contra todos os vetores de suporte). NB é ordens de magnitude mais rápido no momento da predição.

## Use

Com sklearn, ambas as variantes são one-liners:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Acurácia GaussianNB: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"Acurácia MultinomialNB: {mnb.score(X_test_counts, y_test):.3f}")
```

Para classificação de texto com sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

O código em `naive_bayes.py` compara implementações feitas do zero contra sklearn nos mesmos dados para verificar a correção.

### TF-IDF com Naive Bayes

Contagens brutas de palavras dão a cada palavra peso igual por ocorrência. Mas palavras comuns como "o", "a", "de" aparecem frequentemente em toda classe — elas não carregam informação. TF-IDF (Term Frequency - Inverse Document Frequency) diminui o peso de palavras comuns e aumenta o peso de palavras raras e discriminativas.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Valores TF-IDF são não-negativos, então funcionam com MultinomialNB. A combinação de TF-IDF + MultinomialNB é um dos baselines mais fortes para classificação de texto. Frequentemente supera modelos mais complexos em datasets com menos de 10.000 amostras de treino.

### BernoulliNB para Texto Curto

Para texto curto (tweets, SMS, mensagens de chat), BernoulliNB pode superar MultinomialNB. Textos curtos têm baixas contagens de palavras, então a informação de frequência em que MultinomialNB confia é ruidosa. BernoulliNB só se importa com presença ou ausência, que é mais confiável com texto curto.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

A flag `binary=True` no CountVectorizer converte todas as contagens para 0/1. Sem ela, BernoulliNB ainda funciona mas está vendo contagens para as quais não foi projetado.

### Calibrando Probabilidades NB

As probabilidades do NB são mal calibradas. Quando NB diz P(spam) = 0.95, a probabilidade verdadeira pode ser 0.7. Se você precisa de estimativas de probabilidade confiáveis (por exemplo, para definir um limiar ou para combinar com outros modelos), use o CalibratedClassifierCV do sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Isso ajusta uma regressão logística em cima dos scores brutos do NB usando validação cruzada. As probabilidades resultantes são muito mais próximas das frequências verdadeiras das classes.

### Armadilhas Comuns

1. **Valores negativos de features.** MultinomialNB requer features não-negativas. Se você tem valores negativos (como TF-IDF com certas configurações ou features padronizadas), use GaussianNB, ou desloque as features para serem positivas.

2. **Features com variância zero.** GaussianNB divide pela variância. Se uma feature tem variância zero para uma classe (todos os valores idênticos), a computação da probabilidade quebra. O código adiciona um pequeno termo de suavização (1e-9) a todas as variâncias para prevenir isso.

3. **Desbalanceamento de classes.** Se 99% dos emails não são spam, o prior P(não-spam) = 0.99 é tão forte que sobrepuja a evidência da verossimilhança. Você pode definir priors de classe manualmente ou usar o parâmetro class_prior no sklearn.

4. **Escalonamento de features.** MultinomialNB não precisa de escalonamento (funciona em contagens). GaussianNB também não precisa (estima estatísticas por feature). Isto é uma vantagem sobre regressão logística e SVM, que são sensíveis a escalas de features.

## Entregue

Esta lição produz:
- `outputs/skill-naive-bayes-chooser.md` — uma skill de decisão para escolher a variante NB correta
- `code/naive_bayes.py` — MultinomialNB e GaussianNB do zero, com comparação sklearn

### Quando Naive Bayes Falha

NB falha quando a suposição de independência causa rankings incorretos (não apenas probabilidades incorretas). Isso acontece quando:

1. **Interações fortes de features.** Se a classe depende da combinação de duas features mas não de nenhuma isoladamente (padrões tipo XOR), NB vai perder completamente. Cada feature sozinha não fornece evidência, e NB não pode combiná-las não-linearmente.

2. **Features altamente correlacionadas com evidência oposta.** Se a feature A diz "spam" e a feature B diz "não-spam", mas A e B são perfeitamente correlacionadas (elas sempre concordam na realidade), NB verá evidência conflitante onde não há nenhuma.

3. **Conjuntos de treino muito grandes.** Com dados suficientes, modelos discriminativos como regressão logística aprendem a verdadeira fronteira de decisão e superam NB. A suposição de independência que ajudava com dados pequenos agora segura o modelo.

Na prática, esses modos de falha são raros para classificação de texto. Features de texto são numerosas, individualmente fracas, e os erros da suposição de independência tendem a se cancelar. Para dados tabulares com poucas features fortemente correlacionadas, considere regressão logística ou modelos baseados em árvore primeiro.

## Exercícios

1. **Experimento de suavização.** Treine MultinomialNB em dados de texto com valores de alpha de 0.01, 0.1, 1.0, 10.0 e 100.0. Plote acurácia vs alpha. Onde a performance atinge o pico? Por que alpha muito alto prejudica?

2. **Teste de independência de features.** Pegue um dataset de texto real. Escolha duas palavras que são obviamente correlacionadas ("machine" e "learning"). Compute P(word1 | classe) * P(word2 | classe) e compare com P(word1 AND word2 | classe). Quão errada é a suposição de independência? Isso afeta a acurácia da classificação?

3. **Implementação Bernoulli.** Estenda o código com uma classe BernoulliNB. Converta bag-of-words para binário (presente/ausente) e compare a acurácia contra MultinomialNB em dados de texto. Quando Bernoulli vence?

4. **NB vs Regressão Logística.** Treine ambos em dados de texto. Comece com 100 amostras de treino e aumente para 10.000. Plote acurácia vs tamanho do treino para ambos. Em que ponto a Regressão Logística ultrapassa Naive Bayes?

5. **Filtro de spam.** Construa um classificador de spam completo: tokenize texto de email bruto, construa vocabulário, crie features bag-of-words, treine MultinomialNB, avalie com precisão e recall (não apenas acurácia — por quê?).

## Termos-Chave

| Termo | O que o pessoal diz | O que realmente significa |
|-------|--------------------|-----------------------|
| Naive Bayes | "Classificador probabilístico simples" | Um classificador que aplica o teorema de Bayes com a suposição de que features são condicionalmente independentes dada a classe |
| Independência condicional | "Features não se afetam" | P(A, B \| C) = P(A \| C) * P(B \| C) — saber B não te diz nada novo sobre A uma vez que você sabe C |
| Suavização Laplace | "Suavização aditiva" | Adicionar uma pequena contagem a cada feature para evitar que probabilidades zero dominem a predição |
| Prior | "O que você acreditava antes de ver os dados" | P(classe) — a probabilidade de cada classe antes de observar qualquer feature |
| Verossimilhança | "Quão bem os dados se encaixam" | P(features \| classe) — a probabilidade de observar estas features se a classe é conhecida |
| Posteriori | "O que você acredita após ver os dados" | P(classe \| features) — a probabilidade atualizada da classe após observar as features |
| Modelo generativo | "Modela como os dados são gerados" | Um modelo que aprende P(X \| Y) e P(Y), depois usa o teorema de Bayes para obter P(Y \| X) |
| Modelo discriminativo | "Modela a fronteira de decisão" | Um modelo que aprende P(Y \| X) diretamente sem modelar como X é gerado |
| Log probabilidade | "Evitar underflow" | Trabalhar com log P em vez de P para prevenir que o produto de muitos números pequenos se torne zero em ponto flutuante |

## Leitura Adicional

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) — todas as três variantes com detalhes matemáticos
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) — a comparação clássica de Multinomial vs Bernoulli para texto
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) — melhorias para NB em texto
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) — prova que NB converge mais rápido que LR com menos dados
