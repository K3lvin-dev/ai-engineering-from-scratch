# Métodos de Amostragem

> Amostragem é como a IA explora o espaço de possibilidades.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 06-07 (Probabilidade, Teorema de Bayes)
**Tempo:** ~120 minutos

## Objetivos de Aprendizado

- Implementar amostragem por CDF inversa, rejeição e importância usando apenas números aleatórios uniformes
- Construir amostragem por temperatura, top-k e top-p (núcleo) para geração de tokens de modelo de linguagem
- Explicar o truque de reparametrização e por que ele permite backpropagation através de amostragem em VAEs
- Executar Metropolis-Hastings MCMC para amostrar de uma distribuição alvo não-normalizada

## O Problema

Um modelo de linguagem termina de processar seu prompt e produz um vetor de 50.000 logits. Um para cada token no vocabulário. Agora ele tem que escolher um. Como?

Se ele sempre escolhe o token de maior probabilidade, toda resposta é idêntica. Determinística. Chata. Se escolhe uniformemente ao acaso, a saída é bobagem. A resposta vive em algum lugar entre esses extremos, e esse lugar é controlado pela amostragem.

Amostragem não se limita a geração de texto. Aprendizado por reforço estima gradientes de política amostrando trajetórias. VAEs aprendem representações latentes amostrando de distribuições aprendidas e retropropagando através da aleatoriedade. Modelos de difusão geram imagens amostrando ruído e denoising iterativamente. Métodos Monte Carlo estimam integrais que não têm solução fechada. Algoritmos MCMC exploram distribuições posteriores de alta dimensão que são impossíveis de enumerar.

Cada sistema de IA generativa é um sistema de amostragem. A estratégia de amostragem determina a qualidade, diversidade e controlabilidade da saída. Esta lição constrói cada método de amostragem principal do zero, começando de números aleatórios uniformes e terminando com as técnicas que alimentam LLMs modernos e modelos generativos.

## O Conceito

### Por que Amostragem Importa

Amostragem aparece em quatro papéis fundamentais na IA e machine learning:

**Geração.** Modelos de linguagem, modelos de difusão e GANs produzem saída por amostragem. O algoritmo de amostragem controla diretamente criatividade, coerência e diversidade. Temperatura, top-k e núcleo são os botões que engenheiros ajustam diariamente.

**Treino.** Descida do gradiente estocástico amostra mini-batches. Dropout amostra neurônios para desativar. Aumento de dados amostra transformações aleatórias. Amostragem por importância repondera amostras para reduzir variância do gradiente em aprendizado por reforço (PPO, TRPO).

**Estimação.** Muitas quantidades em ML não têm solução fechada. A perda esperada sobre uma distribuição de dados, a função de partição de um modelo baseado em energia, a evidência em inferência Bayesiana. Estimação Monte Carlo aproxima todas estas pela média sobre amostras.

**Exploração.** Algoritmos MCMC exploram distribuições posteriores em inferência Bayesiana. Estratégias evolucionárias amostram perturbações de parâmetros. Amostragem de Thompson equilibra exploração e explotação em bandits.

O desafio central: você só pode amostrar diretamente de distribuições simples (uniforme, normal). Para todo o resto, você precisa de um método para converter amostras simples em amostras de sua distribuição alvo.

### Amostragem Uniforme Aleatória

Todo método de amostragem começa aqui. Um gerador de números aleatórios uniformes produz valores em [0, 1) onde todo sub-intervalo de igual comprimento tem igual probabilidade.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    para 0 <= a <= b <= 1

Propriedades:
  E[U] = 0.5
  Var(U) = 1/12
```

Para amostrar uniformemente de um conjunto discreto de n itens, gere U e retorne floor(n * U). Para amostrar de uma faixa contínua [a, b], compute a + (b - a) * U.

A ideia chave: um único número aleatório uniforme contém exatamente a quantidade certa de aleatoriedade para produzir uma amostra de qualquer distribuição. O truque é encontrar a transformação certa.

### Método CDF Inversa (Amostragem por Transformada Inversa)

A função de distribuição acumulada (CDF) mapeia valores para probabilidades:

```
F(x) = P(X <= x)

Propriedades:
  F é não-decrescente
  F(-inf) = 0
  F(+inf) = 1
  F mapeia a reta real para [0, 1]
```

A CDF inversa mapeia probabilidades de volta para valores. Se U ~ Uniform(0, 1), então X = F_inverso(U) segue a distribuição alvo.

```
Algoritmo:
  1. Gere u ~ Uniform(0, 1)
  2. Retorne F_inverso(u)

Por que funciona:
  P(X <= x) = P(F_inverso(U) <= x) = P(U <= F(x)) = F(x)
```

**Exemplo de distribuição exponencial:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Resolva F(x) = u para x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Como (1 - U) e U têm a mesma distribuição:
  x = -ln(u) / lambda
```

Isso funciona perfeitamente quando você pode escrever F_inverso em forma fechada. Para a distribuição normal, não há CDF inversa de forma fechada, então usamos outros métodos (Box-Muller, ou aproximação numérica).

**Versão discreta:** Para distribuições discretas, construa a CDF como uma soma acumulada, gere U e encontre o primeiro índice onde a soma acumulada excede U. É assim que `sample_categorical` funciona na Lição 06.

### Amostragem por Rejeição

Quando você não pode inverter a CDF mas pode avaliar a PDF alvo até uma constante, a amostragem por rejeição funciona.

```
Distribuição alvo: p(x)  (pode avaliar, possivelmente não normalizada)
Distribuição proposta: q(x)  (pode amostrar)
Limite M: tal que p(x) <= M * q(x) para todo x

Algoritmo:
  1. Amostre x ~ q(x)
  2. Amostre u ~ Uniform(0, 1)
  3. Se u < p(x) / (M * q(x)), aceite x
  4. Caso contrário, rejeite e vá para o passo 1

Taxa de aceitação = 1/M
```

Quanto mais apertado o limite M, maior a taxa de aceitação. Em baixas dimensões (1-3), a amostragem por rejeição funciona bem. Em altas dimensões, a taxa de aceitação cai exponencialmente porque a maior parte do volume proposto é rejeitada. Esta é a maldição da dimensionalidade para amostragem por rejeição.

**Exemplo: amostragem de uma normal truncada.** Use uma proposta uniforme sobre o intervalo truncado. O envelope M é o máximo da PDF normal naquele intervalo.

**Exemplo: amostragem de um semicírculo.** Proponha uniformemente no retângulo delimitador. Aceite se o ponto cair dentro do semicírculo. É assim que Monte Carlo computa pi: a taxa de aceitação é igual à razão de área pi/4.

### Amostragem por Importância

Às vezes você não precisa de amostras da distribuição alvo p(x). Você precisa estimar uma expectativa sob p(x), e você tem amostras de uma distribuição diferente q(x).

```
Objetivo: estimar E_p[f(x)] = integral de f(x) * p(x) dx

Reescreva:
  E_p[f(x)] = integral de f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

onde w(x) = p(x) / q(x)  são os pesos de importância.

Estimador:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    onde x_i ~ q(x)
```

Isto é crítico em aprendizado por reforço. Em PPO (Proximal Policy Optimization), você coleta trajetórias sob uma política antiga pi_old mas quer otimizar uma nova política pi_new. O peso de importância é pi_new(a|s) / pi_old(a|s). PPO limita estes pesos para evitar que a nova política divirja muito da antiga.

A variância do estimador de amostragem por importância depende de quão similar q é a p. Se q é muito diferente de p, algumas amostras recebem pesos enormes e dominam a estimativa. A amostragem por importância auto-normalizada divide pela soma dos pesos para reduzir este problema:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Estimação Monte Carlo

A estimação Monte Carlo aproxima integrais pela média de amostras aleatórias. A lei dos grandes números garante convergência.

```
Objetivo: estimar I = integral de g(x) dx sobre domínio D

Método:
  1. Amostre x_1, ..., x_N uniformemente de D
  2. I ~ (Volume de D / N) * sum(g(x_i))

Erro: O(1 / sqrt(N))   independentemente da dimensão
```

A taxa de erro é independente da dimensão. É por isso que métodos Monte Carlo dominam em altas dimensões onde integração baseada em grade é impossível.

**Estimando pi:**

```
Amostre (x, y) uniformemente de [-1, 1] x [-1, 1]
Conte quantos caem dentro do círculo unitário: x^2 + y^2 <= 1
pi ~ 4 * (dentro) / (total)
```

**Estimando expectativas:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    onde x_i ~ p(x)

A média amostral converge para a expectativa verdadeira.
Variância do estimador = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC constrói uma cadeia de Markov cuja distribuição estacionária é a distribuição alvo p(x). Depois de passos suficientes, amostras da cadeia são (aproximadamente) amostras de p(x).

```
Alvo: p(x)  (conhecido até uma constante de normalização)
Proposta: q(x'|x)  (como propor o próximo estado dado o estado atual)

Algoritmo Metropolis-Hastings:
  1. Comece em algum x_0
  2. Para t = 1, 2, ..., T:
     a. Proponha x' ~ q(x'|x_t)
     b. Compute razão de aceitação:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Aceite com probabilidade min(1, alpha):
        - Se u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Caso contrário: x_{t+1} = x_t
  3. Descarte as primeiras B amostras (burn-in)
  4. Retorne as amostras restantes
```

Para propostas simétricas (q(x'|x) = q(x|x')), a razão simplifica para p(x')/p(x). Este é o algoritmo Metropolis original.

**Por que funciona.** A regra de aceitação garante balanço detalhado: a probabilidade de estar em x e mover para x' é igual à probabilidade de estar em x' e mover para x. Balanço detalhado implica que p(x) é a distribuição estacionária da cadeia.

**Considerações práticas:**
- Burn-in: descarte amostras iniciais antes que a cadeia atinja equilíbrio
- Thinning: mantenha cada k-ésima amostra para reduzir autocorrelação
- Escala da proposta: muito pequena e a cadeia se move lentamente (alta aceitação, exploração lenta); muito grande e a maioria das propostas é rejeitada (baixa aceitação, preso no lugar)
- A taxa de aceitação ótima para uma proposta Gaussiana em altas dimensões é aproximadamente 0.234

### Amostragem Gibbs

A amostragem Gibbs é um caso especial de MCMC para distribuições multivariadas. Em vez de propor um movimento em todas as dimensões de uma vez, atualiza uma variável por vez a partir de sua distribuição condicional.

```
Alvo: p(x_1, x_2, ..., x_d)

Algoritmo:
  Para cada iteração t:
    Amostre x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Amostre x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Amostre x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

A amostragem Gibbs requer que você possa amostrar de cada distribuição condicional p(x_i | x_{-i}). Isto é direto para muitos modelos:
- Redes Bayesianas: condicionais seguem da estrutura do grafo
- Misturas Gaussianas: condicionais são Gaussianas
- Modelos Ising: o condicional de cada spin depende apenas de seus vizinhos

A taxa de aceitação é sempre 1 (toda proposta é aceita) porque amostrar da condicional exata satisfaz automaticamente o balanço detalhado.

**Limitação.** Quando variáveis são altamente correlacionadas, a amostragem Gibbs mistura lentamente porque atualizar uma variável por vez não pode fazer grandes movimentos diagonais através da distribuição.

### Amostragem por Temperatura (Usada em LLMs)

Modelos de linguagem produzem logits z_1, ..., z_V para cada token no vocabulário. Softmax converte em probabilidades. A temperatura reescala os logits antes do softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: softmax padrão (distribuição original)
T -> 0:  argmax (determinístico, sempre escolhe o maior logit)
T -> inf: uniforme (todos tokens igualmente prováveis)
T < 1.0: agudiza a distribuição (mais confiante, menos diverso)
T > 1.0: achata a distribuição (menos confiante, mais diverso)
```

**Por que funciona.** Dividir logits por T < 1 amplifica diferenças entre logits. Se z_1 = 2 e z_2 = 1, dividir por T = 0.5 dá z_1/T = 4 e z_2/T = 2, tornando a lacuna maior. Após softmax, o token de maior logit recebe uma parcela muito maior.

**Na prática:**
- T = 0.0: decodificação gulosa, melhor para Q&A factual
- T = 0.3-0.7: levemente criativo, bom para geração de código
- T = 0.7-1.0: equilibrado, bom para conversação geral
- T = 1.0-1.5: escrita criativa, brainstorming
- T > 1.5: cada vez mais aleatório, raramente útil

A temperatura não muda quais tokens são possíveis. Ela muda a massa de probabilidade alocada para cada token.

### Amostragem Top-k

A amostragem top-k restringe o conjunto candidato aos k tokens com maiores probabilidades, então renormaliza e amostra desse conjunto restrito.

```
Algoritmo:
  1. Compute probabilidades softmax para todos V tokens
  2. Ordene tokens por probabilidade (decrescente)
  3. Mantenha apenas os k primeiros tokens
  4. Renormalize: p_i' = p_i / sum(p_j para j em top-k)
  5. Amostre da distribuição renormalizada

k = 1:  decodificação gulosa
k = V:  sem filtragem (amostragem padrão)
k = 40: configuração típica, remove a cauda longa de tokens improváveis
```

Top-k impede que o modelo selecione tokens extremamente improváveis (erros de digitação, nonsense) que existem na cauda longa da distribuição do vocabulário. O problema: k é fixo independentemente do contexto. Quando o modelo está confiante (um token tem 95% de probabilidade), k = 40 ainda permite 39 alternativas. Quando o modelo está incerto (probabilidade espalhada por 1000 tokens), k = 40 corta opções plausíveis.

### Amostragem Top-p (Núcleo)

A amostragem top-p ajusta dinamicamente o tamanho do conjunto candidato. Em vez de manter um número fixo de tokens, mantém o menor conjunto de tokens cuja probabilidade acumulada excede p.

```
Algoritmo:
  1. Compute probabilidades softmax para todos V tokens
  2. Ordene tokens por probabilidade (decrescente)
  3. Encontre o menor k tal que soma das top-k probabilidades >= p
  4. Mantenha apenas esses k tokens
  5. Renormalize e amostre

p = 0.9:  mantém tokens cobrindo 90% da massa de probabilidade
p = 1.0:  sem filtragem
p = 0.1:  muito restritivo, quase guloso
```

Quando o modelo está confiante, a amostragem por núcleo mantém poucos tokens (talvez 2-3). Quando o modelo está incerto, mantém muitos (talvez 200). Este comportamento adaptativo é por que a amostragem por núcleo geralmente produz melhor texto que top-k.

**Combinações comuns:**
- Temperatura 0.7 + top-p 0.9: boa configuração de uso geral
- Temperatura 0.0 (guloso): melhor para tarefas determinísticas
- Temperatura 1.0 + top-k 50: configuração do artigo original de Fan et al. (2018)

Top-k e top-p podem ser combinados. Aplique top-k primeiro, depois top-p no conjunto restante.

### Truque de Reparametrização (Usado em VAEs)

Autoencoders variacionais (VAEs) aprendem codificando entradas em uma distribuição no espaço latente, amostrando dessa distribuição, e decodificando a amostra de volta. O problema: você não pode retropropagar através de uma operação de amostragem.

```
Amostragem padrão (não diferenciável):
  z ~ N(mu, sigma^2)

Truque de reparametrização (diferenciável):
  z = mu + sigma * epsilon    onde epsilon ~ N(0, 1)
```

A chave: em vez de amostrar z diretamente de uma distribuição parametrizada por mu e sigma, você amostra ruído de uma distribuição fixa (N(0, 1)) e o transforma deterministicamente. A amostra é uma função diferenciável dos parâmetros.

```
Sem reparametrização:
  z ~ N(mu, sigma^2)
  dz/dmu = ?    (a amostragem não é uma função)
  dz/dsigma = ? (não há gradiente)

Com reparametrização:
  z = mu + sigma * epsilon
  dz/dmu = 1
  dz/dsigma = epsilon
```

**No loop de treino do VAE:**
1. O encoder produz mu e log(sigma^2) para cada entrada
2. Amostre epsilon ~ N(0, 1)
3. Compute z = mu + sigma * epsilon
4. Decodifique z para reconstruir a entrada
5. Retropropague através dos passos 4, 3, 2, 1 (possível porque o passo 3 é diferenciável)

Sem o truque de reparametrização, VAEs não podem ser treinados com retropropagação padrão. Esta única percepção tornou VAEs práticos.

### Gumbel-Softmax (Amostragem Categórica Diferenciável)

O truque de reparametrização funciona para distribuições contínuas (Gaussiana). Para distribuições categóricas discretas, precisamos de uma abordagem diferente. Gumbel-Softmax fornece uma aproximação diferenciável para amostragem categórica.

**O truque Gumbel-Max (não diferenciável):**

```
Para amostrar de uma distribuição categórica com log-probabilidades log(p_1), ..., log(p_k):
  1. Amostre g_i ~ Gumbel(0, 1) para cada categoria
     (g = -log(-log(u)), onde u ~ Uniform(0, 1))
  2. Retorne argmax(log(p_i) + g_i)

Isso produz amostras categóricas exatas.
```

**Gumbel-Softmax (aproximação diferenciável):**

```
Substitua o argmax rígido por um softmax suave:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperatura) controla a aproximação:
  tau -> 0:  aproxima um vetor one-hot (categórico rígido)
  tau -> inf: aproxima uniforme (1/k, 1/k, ..., 1/k)
  tau = 1.0: aproximação suave
```

Gumbel-Softmax produz um relaxamento contínuo de uma amostra discreta. A saída é um vetor de probabilidade (one-hot suave) em vez de um one-hot rígido. Gradientes fluem através do softmax. Durante o passeio direto no treino, você pode usar o estimador "straight-through": use o argmax rígido para o passeio direto mas os gradientes suaves do Gumbel-Softmax para o passeio reverso.

**Aplicações:**
- Variáveis latentes discretas em VAEs
- Busca de arquitetura neural (escolhendo operações discretas)
- Mecanismos de atenção rígida
- Aprendizado por reforço com ações discretas

### Amostragem Estratificada

A amostragem Monte Carlo padrão pode deixar lacunas no espaço amostral por acaso. A amostragem estratificada força cobertura uniforme dividindo o espaço em estratos e amostrando de cada um.

```
Monte Carlo padrão:
  Amostre N pontos uniformemente de [0, 1]
  Algumas regiões podem ter aglomerados, outras lacunas

Amostragem estratificada:
  Divida [0, 1] em N estratos iguais: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Amostre um ponto uniformemente dentro de cada estrato
  x_i = (i + u_i) / N   onde u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

A amostragem estratificada sempre tem variância menor ou igual comparada ao Monte Carlo padrão:

```
Var(estratificada) <= Var(Monte Carlo padrão)

A melhoria é maior quando f(x) varia suavemente.
Para funções constantes por partes, a amostragem estratificada é exata.
```

**Aplicações:**
- Integração numérica (quasi-Monte Carlo)
- Divisões de dados de treino (garantindo equilíbrio de classe em cada dobra)
- Amostragem por importância com estratificação (combinando ambas técnicas)
- NeRF (Neural Radiance Fields) usa amostragem estratificada ao longo dos raios da câmera

### Conexão com Modelos de Difusão

Modelos de difusão geram imagens através de um processo de amostragem. O processo direto adiciona ruído Gaussiano a uma imagem ao longo de T passos até que se torne ruído puro. O processo reverso aprende a denoisear, recuperando a imagem original passo a passo.

```
Processo direto (conhecido):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  onde epsilon ~ N(0, I)

  Após T passos: x_T ~ N(0, I)  (ruído puro)

Processo reverso (aprendido):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  onde z ~ N(0, I)

  Cada passo de denoising é um passo de amostragem.
```

A conexão com os métodos desta lição:
- Cada passo de denoising usa o truque de reparametrização (amostre ruído, aplique transformação determinística)
- O cronograma de ruído {alpha_t} controla uma forma de recozimento de temperatura
- O treino usa estimação Monte Carlo para aproximar o ELBO (evidence lower bound)
- Amostragem ancestral em modelos de difusão é uma cadeia de Markov (cada passo depende apenas do estado atual)

Todo o processo de geração de imagens é amostragem iterativa: comece do ruído, e a cada passo, amostre uma versão ligeiramente menos ruidosa condicionada no modelo de denoising aprendido.

## Construa

### Passo 1: Amostragem uniforme e CDF inversa

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Gere 10.000 amostras exponenciais e verifique que a média é 1/lambda.

### Passo 2: Amostragem por rejeição

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Use amostragem por rejeição para amostrar de uma distribuição normal truncada. Verifique a forma através do histograma das amostras.

### Passo 3: Amostragem por importância

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Estime E[X^2] sob uma distribuição normal usando uma proposta uniforme. Compare com a resposta conhecida (mu^2 + sigma^2).

### Passo 4: Estimação Monte Carlo de pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Passo 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Amostre de uma distribuição bimodal (mistura de duas Gaussianas). Visualize a trajetória da cadeia.

### Passo 6: Amostragem Gibbs

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Passo 7: Amostragem por temperatura

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Mostre como a temperatura muda a distribuição de saída para um conjunto de logits de tokens.

### Passo 8: Amostragem top-k e top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Passo 9: Truque de reparametrização

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Demonstre que gradientes fluem através da amostra reparametrizada mas não através da amostragem direta.

### Passo 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Mostre como a diminuição da temperatura faz a saída se aproximar de um vetor one-hot.

Implementações completas com todas as visualizações estão em `code/sampling.py`.

## Use

Com NumPy e SciPy, as versões de produção:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Média exponencial: {exponential_samples.mean():.4f} (esperado 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF em 1.96: {normal.cdf(1.96):.4f}")
print(f"CDF inversa em 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Índice do token amostrado: {token}")
```

Para MCMC em escala, use bibliotecas dedicadas:
- PyMC: modelagem Bayesiana completa com NUTS (HMC adaptativo)
- emcee: amostrador MCMC ensemble
- NumPyro/JAX: MCMC acelerado por GPU

Você construiu estes do zero. Agora você sabe o que as chamadas de biblioteca estão fazendo.

## Exercícios

1. Implemente amostragem por CDF inversa para a distribuição de Cauchy. A CDF é F(x) = 0.5 + arctan(x)/pi. Gere 10.000 amostras e plote o histograma contra a PDF verdadeira. Observe as caudas pesadas (valores extremos longe do centro).

2. Use amostragem por rejeição para gerar amostras de uma distribuição Beta(2, 5) usando uma proposta Uniform(0, 1). Plote as amostras aceitas contra a PDF Beta verdadeira. Qual é a taxa de aceitação teórica?

3. Estime a integral de sin(x) de 0 a pi usando Monte Carlo com 1.000, 10.000 e 100.000 amostras. Compare o erro em cada nível. Verifique que o erro escala como O(1/sqrt(N)).

4. Implemente Metropolis-Hastings para amostrar de uma distribuição 2D p(x, y) proporcional a exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plote as amostras e a trajetória da cadeia. Experimente com diferentes desvios padrão da proposta.

5. Construa uma demonstração completa de geração de texto: dado um vocabulário de 10 palavras com logits, gere sequências de 20 tokens usando (a) guloso, (b) temperatura=0.7, (c) top-k=3, (d) top-p=0.9. Compare a diversidade das saídas ao longo de 5 execuções.

## Termos-Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|----------------------|--------------------------|
| Amostragem | "Tirar valores aleatórios" | Gerar valores de acordo com uma distribuição de probabilidade. O mecanismo por trás de toda IA generativa |
| Distribuição uniforme | "Todos igualmente prováveis" | Cada valor em [a, b] tem densidade de probabilidade igual 1/(b-a). O ponto de partida para todos os métodos de amostragem |
| CDF inversa | "Transformação de probabilidade" | F_inverso(U) converte amostra uniforme em amostra de qualquer distribuição com CDF conhecida. Exato e eficiente |
| Amostragem por rejeição | "Propor e aceitar/rejeitar" | Gerar de uma proposta simples, aceitar com probabilidade proporcional à razão alvo/proposta. Exato mas desperdiça amostras |
| Amostragem por importância | "Reponderar amostras" | Estimar expectativas sob p(x) usando amostras de q(x) ponderando cada amostra por p(x)/q(x). Essencial para PPO em RL |
| Monte Carlo | "Média de amostras aleatórias" | Aproximar integrais como médias de amostras. Erro O(1/sqrt(N)) independente da dimensão |
| MCMC | "Caminhada aleatória que converge" | Construir uma cadeia de Markov cuja distribuição estacionária é o alvo. Metropolis-Hastings é o algoritmo fundamental |
| Metropolis-Hastings | "Aceitar ladeira acima, às vezes ladeira abaixo" | Propor movimentos, aceitar baseado na razão de densidades. Balanço detalhado garante convergência à distribuição alvo |
| Amostragem Gibbs | "Uma variável por vez" | Atualizar cada variável a partir de sua distribuição condicional mantendo as outras fixas. Taxa de aceitação de 100% |
| Temperatura | "Botão de confiança" | Divide logits por T antes do softmax. T<1 agudiza (mais confiante), T>1 achata (mais diverso) |
| Amostragem top-k | "Manter os k melhores" | Zerar todos exceto os k tokens de maior probabilidade, renormalizar, amostrar. Tamanho de conjunto candidato fixo |
| Amostragem núcleo (top-p) | "Manter os prováveis" | Manter o menor conjunto de tokens cuja probabilidade acumulada excede p. Tamanho de conjunto candidato adaptativo |
| Truque de reparametrização | "Mover aleatoriedade para fora" | Escrever z = mu + sigma * epsilon onde epsilon ~ N(0,1). Torna a amostragem diferenciável. Essencial para treino de VAE |
| Gumbel-Softmax | "Amostragem categórica suave" | Aproximação diferenciável para amostragem categórica usando ruído Gumbel + softmax com temperatura |
| Amostragem estratificada | "Cobertura forçada" | Dividir o espaço amostral em estratos, amostrar de cada. Sempre variância menor que Monte Carlo ingênuo |
| Burn-in | "Período de aquecimento" | Amostras MCMC iniciais descartadas antes que a cadeia atinja sua distribuição estacionária |
| Balanço detalhado | "Condição de reversibilidade" | p(x) * T(x->y) = p(y) * T(y->x). Condição suficiente para p ser a distribuição estacionária de uma cadeia de Markov |
| Amostragem por difusão | "Denoising iterativo" | Gerar dados começando de ruído e aplicando passos de denoising aprendidos. Cada passo é uma operação de amostragem condicional |

## Leitura Adicional

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - tutorial detalhado sobre fundamentos de MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - artigo original do Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - artigo sobre amostragem núcleo (top-p)
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - artigo VAE que introduziu o truque de reparametrização
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM conecta amostragem à geração de imagens
