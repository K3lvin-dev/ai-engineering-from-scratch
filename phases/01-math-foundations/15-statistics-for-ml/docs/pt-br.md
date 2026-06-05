# Estatística para Machine Learning

> Estatística é como você sabe se seu modelo realmente funciona ou só teve sorte.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 06 (Probabilidade e Distribuições), 07 (Teorema de Bayes)
**Tempo:** ~120 minutos

## Objetivos de Aprendizado

- Computar estatísticas descritivas, correlação de Pearson/Spearman e matrizes de covariância do zero
- Realizar testes de hipótese (t-test, chi-quadrado) e interpretar p-valores e intervalos de confiança corretamente
- Usar bootstrap resampling para construir intervalos de confiança para qualquer métrica sem suposições de distribuição
- Distinguir significância estatística de significância prática usando medidas de tamanho de efeito

## O Problema

Você treinou dois modelos. Modelo A pontua 0.87 no seu teste. Modelo B pontua 0.89. Você deploya o Modelo B. Três semanas depois, as métricas de produção estão piores. O que aconteceu?

O Modelo B não superou realmente o Modelo A. A diferença de 0.02 era ruído. Seu conjunto de teste era pequeno demais, ou a variância alta demais, ou ambos. Você entregou aleatoriedade disfarçada de melhoria.

Isso acontece o tempo todo. Mudanças no ranking do Kaggle. Papers que falham em reproduzir. Testes A/B que declaram vencedores com poucas centenas de amostras. A causa raiz é sempre a mesma: alguém pulou a estatística.

Estatística te dá as ferramentas para distinguir sinal de ruído. Ela diz quando uma diferença é real, quão confiante você deve ser, e quantos dados você precisa antes de confiar em um resultado. Todo pipeline de ML, toda comparação de modelos, todo experimento precisa de estatística. Sem ela, você está chutando.

## O Conceito

### Estatísticas Descritivas: Resumindo Seus Dados

Antes de modelar qualquer coisa, você precisa saber como seus dados são. Estatísticas descritivas comprimem um dataset em alguns números que capturam sua forma.

**Medidas de tendência central** respondem "onde está o meio?"

```
Média:     soma de todos os valores / contagem
          mu = (1/n) * sum(x_i)

Mediana:   valor do meio quando ordenado
          Robusta a outliers. Se você tem [1, 2, 3, 4, 1000], a média é 202
          mas a mediana é 3.

Moda:      valor mais frequente
          Útil para dados categóricos. Para dados contínuos, raramente informativa.
```

A média é o ponto de equilíbrio. A mediana é a marca da metade. Quando divergem, sua distribuição é assimétrica. Distribuições de renda têm média >> mediana (assimetria direita dos bilionários). Distribuições de perda durante treino geralmente têm média << mediana (assimetria esquerda de amostras fáceis).

**Medidas de dispersão** respondem "quão dispersos são os dados?"

```
Variância:          média dos desvios quadrados da média
                    sigma^2 = (1/n) * sum((x_i - mu)^2)

Desvio padrão:      raiz quadrada da variância
                    sigma = sqrt(sigma^2)
                    Mesmas unidades dos dados, mais interpretável.

Amplitude:          max - min
                    Sensível a outliers. Quase nunca útil sozinha.

IQR:                Q3 - Q1 (intervalo interquartil)
                    A amplitude dos 50% centrais dos dados.
                    Robusta a outliers. Usada para box plots e detecção de outliers.
```

**Percentis** dividem dados ordenados em 100 partes iguais. O 25º percentil (Q1) significa que 25% dos valores caem abaixo deste ponto. O 50º percentil é a mediana. O 75º percentil é Q3.

```
Para monitoramento de latência:
  P50 = latência mediana        (experiência típica do usuário)
  P95 = 95º percentil           (ruim mas não pior caso)
  P99 = 99º percentil           (latência de cauda, geralmente 10x a mediana)
```

Em ML, você se importa com percentis para latência de inferência, distribuições de confiança de previsão e entendimento de distribuições de erro. Um modelo com baixo erro médio mas erro P99 terrível pode ser inútil para aplicações críticas de segurança.

**Estatística amostral vs populacional.** Ao computar variância de uma amostra, divida por (n-1) em vez de n. Esta é a correção de Bessel. Compensa o fato de que sua média amostral não é a verdadeira média populacional. Com n no denominador, você subestima sistematicamente a verdadeira variância. Com (n-1), a estimativa é não-viesada.

```
Variância populacional: sigma^2 = (1/N) * sum((x_i - mu)^2)
Variância amostral:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Na prática: se n é grande (milhares de amostras), a diferença é desprezível. Se n é pequeno (dezenas de amostras), importa.

### Correlação: Como Variáveis se Movem Juntas

Correlação mede a força e direção de uma relação linear entre duas variáveis.

**Coeficiente de correlação de Pearson** mede associação linear:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  relação linear positiva perfeita
r = -1:  relação linear negativa perfeita
r =  0:  sem relação linear (mas pode haver uma não-linear!)

Intervalo: [-1, 1]
```

Pearson assume que a relação é linear e ambas as variáveis são aproximadamente normalmente distribuídas. É sensível a outliers. Um único ponto extremo pode arrastar r de 0.1 para 0.9.

**Correlação de Spearman** mede associação monotônica:

```
1. Substitua cada valor por seu ranking (1, 2, 3, ...)
2. Compute correlação de Pearson nos rankings

Spearman captura qualquer relação monotônica, não apenas linear.
Se y = x^3, Pearson dá r < 1 mas Spearman dá rho = 1.
```

**Quando usar cada um:**

```
Pearson:    Ambas variáveis são contínuas e aproximadamente normais.
            Você se importa especificamente com a relação linear.
            Sem outliers extremos.

Spearman:   Dados ordinais (rankings, avaliações).
            Dados não são normalmente distribuídos.
            Suspeita-se de relação monotônica mas não linear.
            Outliers estão presentes.
```

**A regra de ouro:** correlação não implica causalidade. Vendas de sorvete e mortes por afogamento são correlacionadas porque ambas aumentam no verão. A acurácia do seu modelo e o número de parâmetros são correlacionados, mas adicionar parâmetros não melhora automaticamente a acurácia (veja: overfitting).

### Matriz de Covariância

A covariância entre duas variáveis mede como elas variam juntas:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X e Y tendem a aumentar juntos
Cov(X, Y) < 0:  quando X aumenta, Y tende a diminuir
Cov(X, Y) = 0:  sem co-movimento linear
```

Para d features, a matriz de covariância C é uma matriz d x d onde C[i][j] = Cov(feature_i, feature_j). As entradas diagonais C[i][i] são as variâncias de cada feature.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Propriedades:
  - Simétrica: C[i][j] = C[j][i]
  - Semidefinida positiva: todos autovalores >= 0
  - Diagonal = variâncias
  - Fora da diagonal = covariâncias
```

**Conexão com PCA.** PCA decompõe a matriz de covariância em autovalores e autovetores. Os autovetores são os componentes principais (direções de máxima variância). Os autovalores dizem quanta variância cada componente captura. Isso é exatamente o que a Lição 10 cobriu, mas agora você vê por que a matriz de covariância é a coisa certa a decompor: ela codifica todas as relações lineares pareadas em seus dados.

**Conexão com correlação.** A matriz de correlação é a matriz de covariância de variáveis padronizadas (cada uma dividida por seu desvio padrão). Correlação normaliza a covariância para que todos os valores fiquem em [-1, 1].

### Testes de Hipótese

Testes de hipótese são um framework para tomar decisões sob incerteza. Você começa com uma afirmação, coleta dados e determina se os dados são consistentes com a afirmação.

**A configuração:**

```
Hipótese nula (H0):        a suposição padrão, geralmente "nenhum efeito"
Hipótese alternativa (H1): o que você está tentando mostrar

Exemplo:
  H0: Modelo A e Modelo B têm a mesma acurácia
  H1: Modelo B tem acurácia maior que Modelo A
```

**O p-valor** é a probabilidade de ver dados tão extremos quanto os observados, assumindo que H0 é verdadeira. NÃO é a probabilidade de que H0 seja verdadeira. Este é o equívoco mais comum em estatística.

```
p-valor = P(dados tão extremos | H0 é verdadeira)

Se p-valor < alpha (tipicamente 0.05):
    Rejeite H0. O resultado é "estatisticamente significativo."
Se p-valor >= alpha:
    Falha em rejeitar H0. Você não tem evidência suficiente.
    Isso NÃO significa que H0 é verdadeira.
```

**Intervalos de confiança** dão uma faixa de valores plausíveis para um parâmetro:

```
Intervalo de confiança de 95% para a média:
    x_bar +/- z * (s / sqrt(n))

onde z = 1.96 para 95% de confiança

Interpretação: se você repetisse este experimento muitas vezes, 95% dos
intervalos calculados conteriam a verdadeira média. NÃO significa que há
95% de probabilidade da verdadeira média estar neste intervalo específico.
```

A largura do intervalo de confiança te diz sobre precisão. Intervalos largos significam alta incerteza. Intervalos estreitos significam que sua estimativa é precisa (mas não necessariamente acurada, se seus dados forem viesados).

### O Teste t

O teste t compara médias. Existem várias variações.

**Teste t de uma amostra:** a média populacional é diferente de um valor hipotético?

```
t = (x_bar - mu_0) / (s / sqrt(n))

graus de liberdade = n - 1
```

**Teste t de duas amostras (independentes):** duas médias de grupo são diferentes?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

Este é o teste t de Welch, que não assume variâncias iguais.
Sempre use Welch a menos que você tenha uma razão específica para variâncias iguais.
```

**Teste t pareado:** quando as medições vêm em pares (mesmo modelo avaliado nas mesmas divisões de dados):

```
Compute d_i = x_i - y_i para cada par
Depois execute um teste t de uma amostra nos valores d_i contra mu_0 = 0
```

Em ML, o teste t pareado é comum: você executa ambos os modelos nas mesmas 10 dobras de validação cruzada e compara seus escores pareados.

### Teste Chi-Quadrado

O teste chi-quadrado verifica se frequências observadas combinam com frequências esperadas. Útil para dados categóricos.

```
chi^2 = sum((observado - esperado)^2 / esperado)

Exemplo: a distribuição de saída de um modelo de linguagem combina com a
distribuição de treino entre categorias?

Categoria    Observado   Esperado
Positivo       120        100
Negativo        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

Com 1 grau de liberdade, chi^2 = 8 dá p < 0.005.
A diferença é significativa.
```

### Teste A/B para Modelos de ML

Teste A/B em ML não é o mesmo que teste A/B web. Comparação de modelos tem desafios específicos:

```
1. Mesmo conjunto de teste: Ambos modelos devem ser avaliados nos mesmos dados.
                            Conjuntos de teste diferentes tornam a comparação sem sentido.

2. Múltiplas métricas: Acurácia sozinha não basta. Você precisa de precisão,
                       recall, F1, latência e métricas de justiça.

3. Variância:          Use validação cruzada ou bootstrap para estimar
                       a variância de cada métrica, não apenas estimativas pontuais.

4. Vazamento de dados: Se o conjunto de teste foi usado durante seleção de modelo,
                       sua comparação é viesada. Segure um conjunto de teste final.
```

**O procedimento:**

```
1. Defina sua métrica e nível de significância (alpha = 0.05)
2. Execute ambos os modelos nas mesmas divisões de validação cruzada k-fold
3. Colete escores pareados: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute diferenças: d_i = b_i - a_i
5. Execute um teste t pareado nas diferenças
6. Verifique: a média das diferenças é significativamente diferente de 0?
7. Compute um intervalo de confiança para a diferença média
8. Compute tamanho do efeito (Cohen's d) para julgar significância prática
```

### Significância Estatística vs Significância Prática

Um resultado pode ser estatisticamente significativo mas praticamente irrelevante. Com dados suficientes, até uma diferença trivial se torna estatisticamente significativa.

```
Exemplo:
  Acurácia Modelo A: 0.9234
  Acurácia Modelo B: 0.9237
  n = 1.000.000 amostras de teste
  p-valor = 0.001

Estatisticamente significativo? Sim.
Praticamente significativo? Uma melhora de 0.03% não vale o
custo de engenharia de implantar um novo modelo.
```

**Tamanho do efeito** quantifica o quão grande é a diferença, independente do tamanho amostral:

```
Cohen's d = (média_1 - média_2) / desvio_padrão_combinado

d = 0.2:  efeito pequeno
d = 0.5:  efeito médio
d = 0.8:  efeito grande
```

Sempre relate tanto o p-valor quanto o tamanho do efeito. O p-valor te diz se a diferença é real. O tamanho do efeito te diz se ela importa.

### Problema de Comparações Múltiplas

Quando você testa muitas hipóteses, algumas serão "significativas" por acaso. Se você testar 20 coisas com alpha = 0.05, espera 1 falso positivo mesmo quando nada é real.

```
P(pelo menos um falso positivo) = 1 - (1 - alpha)^m

m = 20 testes, alpha = 0.05:
P(falso positivo) = 1 - 0.95^20 = 0.64

Você tem 64% de chance de pelo menos um falso positivo.
```

**Correção de Bonferroni:** divida alpha pelo número de testes.

```
Alpha ajustado = alpha / m = 0.05 / 20 = 0.0025

Só rejeite H0 se p-valor < 0.0025.
Conservador mas simples. Funciona quando os testes são independentes.
```

Em ML, isso importa quando você compara um modelo através de múltiplas métricas, testa muitas configurações de hiperparâmetros, ou avalia em múltiplos datasets.

### Métodos Bootstrap

Bootstrapping estima a distribuição de amostragem de uma estatística reamostrando seus dados com reposição. Sem suposições sobre a distribuição subjacente.

**O algoritmo:**

```
1. Você tem n pontos de dados
2. Tire n amostras COM reposição (alguns pontos aparecem múltiplas vezes,
   alguns nem aparecem)
3. Compute sua estatística nesta amostra bootstrap
4. Repita B vezes (tipicamente B = 1000 a 10000)
5. A distribuição das estatísticas bootstrap aproxima a
   distribuição de amostragem
```

**Intervalo de confiança bootstrap (método do percentil):**

```
Ordene as B estatísticas bootstrap
IC 95% = [2.5º percentil, 97.5º percentil]
```

**Por que bootstrap importa para ML:**

```
- Acurácia do conjunto de teste é uma estimativa pontual. Bootstrap te dá
  intervalos de confiança.
- Você não pode assumir que distribuições de métricas são normais (especialmente
  para AUC, F1, precisão em k).
- Bootstrap funciona para QUALQUER estatística: mediana, razão de duas médias,
  diferença em AUC entre dois modelos.
- Nenhuma fórmula fechada necessária.
```

**Bootstrap para comparação de modelos:**

```
1. Você tem previsões do Modelo A e Modelo B no mesmo conjunto de teste
2. Para cada iteração bootstrap:
   a. Reamostre índices de teste com reposição
   b. Compute métrica_A e métrica_B no conjunto reamostrado
   c. Armazene diff = métrica_B - métrica_A
3. IC 95% para a diferença:
   [2.5º percentil das diffs, 97.5º percentil das diffs]
4. Se o IC não contém 0, a diferença é significativa
```

Isso é mais robusto que o teste t pareado porque não faz suposições distribucionais.

### Testes Paramétricos vs Não-Paramétricos

**Testes paramétricos** assumem uma distribuição específica (geralmente normal):

```
Teste t:          assume dados normalmente distribuídos (ou n grande pelo TLC)
ANOVA:            assume normalidade e variâncias iguais
Pearson r:        assume normalidade bivariada
```

**Testes não-paramétricos** não fazem suposições distribucionais:

```
Mann-Whitney U:     compara dois grupos (substitui teste t independente)
Wilcoxon pareado:   compara dados pareados (substitui teste t pareado)
Spearman rho:       correlação em rankings (substitui Pearson)
Kruskal-Wallis:     compara múltiplos grupos (substitui ANOVA)
```

**Quando usar não-paramétrico:**

```
- Tamanho amostral pequeno (n < 30) e dados claramente não-normais
- Dados ordinais (avaliações, rankings)
- Outliers pesados que você não pode remover
- Distribuições assimétricas
```

**Quando usar paramétrico:**

```
- Tamanho amostral grande (TLC torna a estatística de teste aproximadamente normal)
- Dados aproximadamente simétricos sem outliers extremos
- Mais poder estatístico (melhor em detectar diferenças reais)
```

Em experimentos de ML, você tipicamente tem n pequeno (5 ou 10 dobras de validação cruzada), então testes não-paramétricos como Wilcoxon pareado são frequentemente mais apropriados que testes t.

### Teorema Central do Limite: Implicações Práticas

O TLC diz que a distribuição das médias amostrais se aproxima de uma distribuição normal conforme n cresce, independentemente da distribuição populacional subjacente.

```
Se X_1, X_2, ..., X_n são iid com média mu e variância sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    quando n -> infinito

Funciona para n >= 30 na maioria dos casos.
Para distribuições altamente assimétricas, você pode precisar n >= 100.
```

**Por que isso importa para ML:**

```
1. Justifica intervalos de confiança e testes t em métricas agregadas
2. Explica por que a média sobre dobras de validação cruzada dá estimativas
   estáveis mesmo quando dobras individuais variam muito
3. Descida do gradiente em mini-batches funciona porque o gradiente médio
   sobre um lote aproxima o gradiente verdadeiro (TLC em ação)
4. Métodos ensemble: a média das previsões de muitos modelos dá
   saída mais estável que qualquer modelo individual
```

**O que TLC NÃO faz:**

```
- NÃO torna seus dados normais. Torna a MÉDIA das amostras normal.
- NÃO funciona para distribuições de cauda pesada com variância infinita
  (distribuição de Cauchy).
- NÃO se aplica a dados dependentes (séries temporais sem correção).
```

### Erros Estatísticos Comuns em Papers de ML

1. **Testar no conjunto de treino.** Garante overfitting. Sempre segure dados que o modelo nunca vê durante o treino.

2. **Sem intervalos de confiança.** Relatar um único número de acurácia sem incerteza torna resultados irreproduzíveis e não verificáveis.

3. **Ignorar comparações múltiplas.** Testar 50 configurações e relatar a melhor sem correção infla as taxas de falso positivo.

4. **Confundir significância estatística e prática.** Um p-valor de 0.001 em uma melhora de 0.01% de acurácia não é significativo.

5. **Usar acurácia em dados desbalanceados.** 99% de acurácia em um dataset com 99% de classe negativa significa que o modelo não aprendeu nada. Use precisão, recall, F1 ou AUC.

6. **Escolher métricas a dedo.** Relatar apenas a métrica onde seu modelo ganha. Avaliação honesta relata todas as métricas relevantes.

7. **Vazar informação através das divisões treino/teste.** Normalizar antes de dividir, ou usar dados futuros para prever o passado.

8. **Conjuntos de teste pequenos sem estimativas de variância.** Avaliar em 100 amostras e reivindicar melhora de 2% é ruído, não sinal.

9. **Assumir independência quando dados não são independentes.** Imagens médicas do mesmo paciente, múltiplas sentenças do mesmo documento. Observações dentro de um grupo são correlacionadas.

10. **P-hacking.** Tentar diferentes testes, subconjuntos ou critérios de exclusão até obter p < 0.05. O resultado é um artefato da busca.

## Construa

Você implementará:

1. **Estatísticas descritivas do zero** (média, mediana, moda, desvio padrão, percentis, IQR)
2. **Funções de correlação** (Pearson e Spearman, com a matriz de covariância)
3. **Testes de hipótese** (teste t de uma amostra, duas amostras, teste chi-quadrado)
4. **Intervalos de confiança bootstrap** (para qualquer estatística, sem suposições)
5. **Simulador de teste A/B** (gerar dados, testar, verificar erros Tipo I e Tipo II)
6. **Demonstração de significância estatística vs prática** (mostrando que n grande torna tudo "significante")

Tudo do zero, usando apenas `math` e `random`. Sem numpy, sem scipy.

## Termos-Chave

| Termo | Definição |
|-------|-----------|
| Média | Soma dos valores dividida pela contagem. Sensível a outliers. |
| Mediana | Valor do meio dos dados ordenados. Robusta a outliers. |
| Desvio padrão | Raiz quadrada da variância. Mede dispersão nas unidades originais. |
| Percentil | Valor abaixo do qual uma dada porcentagem dos dados cai. |
| IQR | Intervalo interquartil. Q3 menos Q1. A dispersão dos 50% centrais. |
| Correlação Pearson | Mede associação linear entre duas variáveis. Intervalo [-1, 1]. |
| Correlação Spearman | Mede associação monotônica usando rankings. |
| Matriz de covariância | Matriz de covariâncias pareadas entre todas as features. |
| Hipótese nula | Suposição padrão de nenhum efeito ou nenhuma diferença. |
| p-valor | Probabilidade dos dados serem tão extremos dado que a hipótese nula é verdadeira. |
| Intervalo de confiança | Faixa de valores plausíveis para um parâmetro em um dado nível de confiança. |
| Teste t | Testa se médias diferem significativamente. Usa a distribuição t. |
| Teste chi-quadrado | Testa se frequências observadas diferem das esperadas. |
| Tamanho do efeito | Magnitude de uma diferença, independente do tamanho amostral. Cohen's d é comum. |
| Correção de Bonferroni | Divide o limiar de significância pelo número de testes para controlar falsos positivos. |
| Bootstrap | Reamostragem com reposição para estimar distribuições de amostragem. |
| Erro Tipo I | Falso positivo. Rejeitar H0 quando ela é verdadeira. |
| Erro Tipo II | Falso negativo. Falhar em rejeitar H0 quando ela é falsa. |
| Poder estatístico | Probabilidade de rejeitar corretamente uma H0 falsa. Poder = 1 - taxa de erro Tipo II. |
| Teorema central do limite | Médias amostrais convergem para distribuição normal conforme o tamanho amostral cresce. |
| Teste paramétrico | Assume uma distribuição específica para os dados (geralmente normal). |
| Teste não-paramétrico | Não faz suposições distribucionais. Funciona em rankings ou sinais. |

## Leitura Adicional

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- referência acessível para estatística em ML
- [The Art of Statistics (Spiegelhalter)](https://www.amazon.com/Art-Statistics-Learning-Data-Using/dp/1541618518) -- estatística com intuição e exemplos reais
- [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) -- notas de curso sobre estatística para ML
- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) -- referência prática para testes estatísticos em Python
