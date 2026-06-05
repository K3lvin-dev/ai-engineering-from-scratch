# Estabilidade Numérica

> Ponto flutuante é uma abstração com vazamentos. Ele vai te morder durante o treino, e você não vai ver chegar.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 01-04
**Tempo:** ~120 minutos

## Objetivos de Aprendizado

- Implementar softmax e log-sum-exp numericamente estáveis usando o truque de subtração do máximo
- Identificar overflow, underflow e cancelamento catastrófico em computações de ponto flutuante
- Verificar gradientes analíticos contra gradientes numéricos usando diferenças finitas centralizadas
- Explicar por que bfloat16 é preferido sobre float16 para treino e como loss scaling previne underflow de gradiente

## O Problema

Seu modelo treina por três horas, depois a perda vira NaN. Você adiciona um print. Os logits estão ok no passo 9.000. No passo 9.001 são `inf`. No passo 9.002 todo gradiente é `nan` e o treino morreu.

Ou: seu modelo treina até o final mas a acurácia é 2% pior que o paper diz. Você verifica tudo. Arquitetura bate. Hiperparâmetros batem. Dados batem. O problema é que o paper usou float32 e você usou float16 sem o escalonamento certo. Trinta e dois bits de erro de arredondamento acumulado comeram sua acurácia silenciosamente.

Ou: você implementa cross-entropy loss do zero. Funciona em logits pequenos. Quando logits passam de 100, retorna `inf`. O softmax fez overflow porque `exp(100)` é maior que float32 consegue representar. Todo framework de ML lida com isso com um truque de duas linhas. Você não sabia que o truque existia.

Estabilidade numérica não é uma preocupação teórica. É a diferença entre um treino que funciona e um que falha silenciosamente. Todo bug sério de ML que você vai debugar eventualmente se resume a ponto flutuante.

## O Conceito

### IEEE 754: Como Computadores Armazenam Números Reais

Computadores armazenam números reais como valores de ponto flutuante seguindo o padrão IEEE 754. Um float tem três partes: um bit de sinal, um expoente e uma mantissa (significando).

```
Layout do Float32 (32 bits no total):
[1 sinal] [8 expoente] [23 mantissa]

Valor = (-1)^sinal * 2^(expoente - 127) * 1.mantissa
```

A mantissa determina precisão (quantos dígitos significativos). O expoente determina o intervalo (quão grande ou pequeno um número pode ser).

```
Formato    Bits  Expoente  Mantissa  Dígitos decimais  Intervalo (aprox)
float64    64    11        52        ~15-16             +/- 1.8e308
float32    32    8         23        ~7-8               +/- 3.4e38
float16    16    5         10        ~3-4               +/- 65.504
bfloat16   16    8         7         ~2-3               +/- 3.4e38
```

float32 dá cerca de 7 dígitos decimais de precisão. Isso significa que ele consegue distinguir 1.0000001 e 1.0000002, mas não 1.00000001 e 1.00000002. Depois de 7 dígitos, tudo é ruído de arredondamento.

float16 dá cerca de 3 dígitos. O maior número que ele representa é 65.504. Isso é alarmantemente pequeno para ML onde logits, gradientes e ativações rotineiramente excedem isso.

bfloat16 é a resposta do Google para o problema de intervalo do float16. Tem o mesmo expoente de 8 bits do float32 (mesmo intervalo, até 3.4e38) mas apenas 7 bits de mantissa (menos precisão que float16). Para treinar redes neurais, intervalo importa mais que precisão, então bfloat16 geralmente vence.

### Por que 0.1 + 0.2 != 0.3

O número 0.1 não pode ser representado exatamente em ponto flutuante binário. Em base 2, é uma fração periódica:

```
0.1 em binário = 0.0001100110011001100110011... (repete para sempre)
```

Float32 trunca isso para 23 bits de mantissa. O valor armazenado é aproximadamente 0.100000001490116. Similarmente, 0.2 é armazenado como aproximadamente 0.200000002980232. A soma é 0.300000004470348, não 0.3.

```
Em Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Isso importa para ML porque:
1. Comparações de perda como `if loss < threshold` podem dar respostas erradas
2. Acumular muitos valores pequenos (atualizações de gradiente ao longo de milhares de passos) se desvia da soma verdadeira
3. Checksums e testes de reprodutibilidade falham se você comparar floats com `==`

A correção: nunca compare floats com `==`. Use `abs(a - b) < epsilon` ou `math.isclose()`.

### Cancelamento Catastrófico

Quando você subtrai dois números de ponto flutuante quase iguais, os dígitos significativos se cancelam e você fica com ruído de arredondamento promovido a dígitos principais.

```
a = 1.0000001    (armazenado como 1.00000011920929 em float32)
b = 1.0000000    (armazenado como 1.00000000000000 em float32)

Diferença real:  0.0000001
Computada:       0.00000011920929

Erro relativo: 19.2%
```

Isso é um erro relativo de 19% em uma única subtração. Em ML, isso acontece quando você:
- Calcula variância de dados com média grande: `E[x^2] - E[x]^2` quando E[x] é grande
- Subtrai log-probabilidades quase iguais
- Calcula gradientes por diferenças finitas com epsilon muito pequeno

A correção: reorganize fórmulas para evitar subtrair números grandes e quase iguais. Para variância, use o algoritmo de Welford ou centralize os dados primeiro. Para log-probabilidades, trabalhe em espaço de log durante todo o processo.

### Overflow e Underflow

Overflow acontece quando um resultado é grande demais para representar. Underflow acontece quando é pequeno demais (mais perto de zero que o menor número positivo representável).

```
Limites do Float32:
  Máximo:      3.4028235e+38
  Mínimo positivo (normal): 1.175e-38
  Mínimo positivo (denorm): 1.401e-45
  Overflow:    qualquer coisa > 3.4e38 vira inf
  Underflow:   qualquer coisa < 1.4e-45 vira 0.0
```

A função `exp()` é a principal fonte de overflow em ML:

```
exp(88.7)  = 3.40e+38   (mal cabe em float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (mal acima do underflow)
exp(-104)  = 0.0         (underflow para zero)
```

A função `log()` atinge o outro lado:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (ok)
log(1e-46) = -inf        (entrada underflow pra 0, então log(0) = -inf)
```

Em ML, `exp()` aparece em softmax, sigmoid e computações de probabilidade. `log()` aparece em cross-entropy, log-verossimilhanças e divergência KL. A combinação `log(exp(x))` é um campo minado sem os truques certos.

### O Truque Log-Sum-Exp

Computar `log(sum(exp(x_i)))` diretamente é numericamente perigoso. Se qualquer `x_i` for grande, `exp(x_i)` faz overflow. Se todos `x_i` forem muito negativos, todo `exp(x_i)` faz underflow para zero e `log(0)` é `-inf`.

O truque: subtraia o valor máximo antes de exponenciar.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Por que funciona: depois de subtrair `max(x)`, o maior expoente é `exp(0) = 1`. Overflow é impossível. Pelo menos um termo na soma é 1, então a soma é pelo menos 1, e `log(1) = 0`. Underflow para `-inf` é impossível.

Prova:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (soma e subtrai c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (fatora exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Defina `c = max(x)` e overflow é eliminado.

Este truque aparece em todo lugar no ML:
- Normalização softmax
- Computação de cross-entropy loss
- Soma de log-probabilidades em modelos de sequência
- Mistura de Gaussianas
- Inferência variacional

### Por que Softmax Precisa do Truque de Subtração do Máximo

Softmax converte logits em probabilidades:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Sem o truque, logits de [100, 101, 102] causam overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
soma     = 2.99e44

Esses valores overflow em float32 (máx ~3.4e38)? Não, 2.69e43 < 3.4e38? Na verdade:
exp(88.7) já está no limite do float32.
exp(100) = inf em float32.
```

Com o truque, subtraia max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
soma = 1.503

softmax = [0.090, 0.245, 0.665]
```

As probabilidades são idênticas. A computação é segura. Isso não é uma otimização. É um requisito de correção.

### NaN e Inf: Detecção e Prevenção

`nan` (Not a Number) e `inf` (infinito) se propagam viralmente pela computação. Um `nan` em uma atualização de gradiente torna o peso `nan`, que torna toda saída subsequente `nan`. O treino morre em um passo.

Como `inf` aparece:
- `exp()` de um número positivo grande
- Divisão por zero: `1.0 / 0.0`
- Overflow float32 em acumulações

Como `nan` aparece:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` de um número negativo
- `log()` de um número negativo
- Qualquer aritmética envolvendo um `nan` existente

Detecção:

```python
import math

math.isnan(x)       # True se x é nan
math.isinf(x)       # True se x é +inf ou -inf
math.isfinite(x)    # True se x não é nan nem inf
```

Estratégias de prevenção:

1. Limite entradas de `exp()`: `exp(clamp(x, -80, 80))`
2. Adicione epsilon ao denominador: `x / (y + 1e-8)`
3. Adicione epsilon dentro de `log()`: `log(x + 1e-8)`
4. Use implementações estáveis (log-sum-exp, softmax estável)
5. Recorte de gradiente para evitar explosão de pesos
6. Verifique `nan`/`inf` após todo forward pass durante debug

### Verificação de Gradiente Numérico

Gradientes analíticos (da backpropagation) podem ter bugs. A verificação numérica de gradiente os valida computando gradientes com diferenças finitas.

A fórmula de diferença centralizada:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Isso tem precisão O(h^2), muito melhor que a diferença progressiva `(f(x+h) - f(x)) / h` que é apenas O(h).

Escolhendo h: grande demais e a aproximação é errada. Pequeno demais e o cancelamento catastrófico destrói a resposta. `h = 1e-5` a `1e-7` é típico.

A verificação: compute a diferença relativa entre gradientes analíticos e numéricos.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Regras gerais:
- relative_error < 1e-7: perfeito, gradiente correto
- relative_error < 1e-5: aceitável, provavelmente correto
- relative_error > 1e-3: algo está errado
- relative_error > 1: gradiente completamente errado

Sempre verifique gradientes ao implementar uma nova camada ou função de loss. PyTorch fornece `torch.autograd.gradcheck()` para isso.

### Treino com Precisão Mista

GPUs modernas têm hardware especializado (Tensor Cores) que computam multiplicações de matrizes em float16 2-8x mais rápido que float32. O treino com precisão mista explora isso:

```
1. Mantenha cópia-mestre float32 dos pesos
2. Forward pass em float16 (rápido)
3. Compute perda em float32 (previne overflow)
4. Backward pass em float16 (rápido)
5. Escale gradientes para float32
6. Atualize pesos mestre float32
```

O problema com treino puro float16: gradientes são frequentemente muito pequenos (1e-8 ou menor). Float16 sofre underflow abaixo de ~6e-8 para zero. Seu modelo para de aprender porque todas as atualizações de gradiente são zero.

A correção é loss scaling:

```
1. Multiplique a perda por um fator de escala grande (ex.: 1024)
2. Backward pass computa gradientes de (loss * 1024)
3. Todos gradientes são 1024x maiores (empurrados acima do underflow float16)
4. Divida gradientes por 1024 antes de atualizar pesos
5. Efeito líquido: mesma atualização, mas sem underflow
```

Loss scaling dinâmico ajusta o fator de escala automaticamente. Comece com um valor grande (65536). Se gradientes overflow para `inf`, reduza pela metade. Se N passos passarem sem overflow, dobre.

### bfloat16 vs float16: Por que bfloat16 Vence para Treino

```
float16:   [1 sinal] [5 expoente]  [10 mantissa]
bfloat16:  [1 sinal] [8 expoente]  [7 mantissa]
```

float16 tem mais precisão (10 bits de mantissa vs 7) mas intervalo limitado (máx ~65.504). bfloat16 tem menos precisão mas o mesmo intervalo que float32 (máx ~3.4e38).

Para treinar redes neurais:

- Ativações e logits excedem regularmente 65.504 durante picos de treino. float16 faz overflow; bfloat16 lida bem.
- Loss scaling é necessário com float16 mas geralmente desnecessário com bfloat16 porque seu intervalo cobre o espectro de magnitude dos gradientes.
- bfloat16 é uma simples truncatura de float32: descarte os 16 bits inferiores da mantissa. A conversão é trivial e sem perdas no expoente.

float16 é preferido para inferência onde valores são limitados e precisão importa mais. bfloat16 é preferido para treino onde intervalo importa mais. É por isso que TPUs e GPUs NVIDIA modernas (A100, H100) têm suporte nativo a bfloat16.

### Recorte de Gradiente

Gradientes explosivos acontecem quando gradientes crescem exponencialmente através de muitas camadas (comum em RNNs, redes profundas e transformers). Um único gradiente grande pode corromper todos os pesos em um passo.

Dois tipos de recorte:

**Recorte por valor:** limite cada elemento do gradiente independentemente.

```
grad = clamp(grad, -max_val, max_val)
```

Simples mas pode mudar a direção do vetor gradiente.

**Recorte por norma:** escale o vetor gradiente inteiro para que sua norma não exceda um limite.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Preserva a direção do gradiente. É o que `torch.nn.utils.clip_grad_norm_()` faz. É a escolha padrão.

Valores típicos: `max_norm=1.0` para transformers, `max_norm=0.5` para RL, `max_norm=5.0` para redes mais simples.

Recorte de gradiente não é um hack. É um mecanismo de segurança. Sem ele, um único batch outlier pode produzir um gradiente grande o suficiente para arruinar semanas de treino.

### Camadas de Normalização como Estabilizadores Numéricos

Batch normalization, layer normalization e RMS normalization geralmente são apresentadas como regularizadores que ajudam o treino a convergir. Elas também são estabilizadores numéricos.

Sem normalização, ativações podem crescer ou encolher exponencialmente através das camadas:

```
Camada 1: valores em [0, 1]
Camada 5: valores em [0, 100]
Camada 10: valores em [0, 10.000]
Camada 50: valores em [0, inf]
```

A normalização recentraliza e reescala ativações em cada camada:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

O `epsilon` (tipicamente 1e-5) previne divisão por zero quando todas as ativações são idênticas. Os parâmetros aprendidos `gamma` e `beta` permitem que a rede restaure qualquer escala que precisar.

Isso mantém valores em uma faixa numericamente segura através da rede, prevenindo tanto overflow no forward pass quanto explosão de gradiente no backward pass.

### Bugs Numéricos Comuns em ML

**Bug: Perda é NaN depois de algumas épocas.**
Causa: logits cresceram demais, softmax fez overflow. Ou taxa de aprendizado alta demais e pesos divergiram.
Correção: use softmax estável (subtração do máximo), reduza a taxa de aprendizado, adicione recorte de gradiente.

**Bug: Perda fica presa em log(num_classes).**
Causa: saídas do modelo são probabilidades quase uniformes. Frequentemente significa que gradientes estão vanishing ou o modelo não está aprendendo.
Correção: verifique se os rótulos dos dados estão corretos, verifique a função de loss, procure por ReLUs mortos.

**Bug: Acurácia de validação é 1-3% menor que o esperado.**
Causa: precisão mista sem loss scaling adequado. Underflow de gradiente anula pequenas atualizações silenciosamente.
Correção: ative loss scaling dinâmico ou mude para bfloat16.

**Bug: Normas de gradiente são 0.0 para algumas camadas.**
Causa: neurônios ReLU mortos (todas entradas negativas) ou underflow de float16.
Correção: use LeakyReLU ou GELU, use escalonamento de gradiente, verifique inicialização de pesos.

**Bug: Modelo funciona em uma GPU mas dá resultados diferentes em outra.**
Causa: ordem de acumulação de ponto flutuante não-determinística. Reduções paralelas de GPU somam em ordens diferentes em hardwares diferentes, e adição de ponto flutuante não é associativa.
Correção: aceite pequenas diferenças (1e-6), ou defina `torch.use_deterministic_algorithms(True)` e aceite a penalidade de velocidade.

**Bug: `exp()` retorna `inf` na computação de loss.**
Causa: logits crus passados para `exp()` sem o truque de subtração do máximo.
Correção: use `torch.nn.functional.log_softmax()` que implementa log-sum-exp internamente.

**Bug: Treino diverge depois de mudar de float32 para float16.**
Causa: float16 não consegue representar magnitudes de gradiente abaixo de 6e-8 ou ativações acima de 65.504.
Correção: use precisão mista com loss scaling (AMP), ou use bfloat16.

## Construa

### Passo 1: Demonstre limites de precisão de ponto flutuante

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Passo 2: Implemente softmax ingênuo vs estável

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) retornaria [nan, nan, nan]
```

### Passo 3: Implemente log-sum-exp estável

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) retorna inf
```

### Passo 4: Implemente cross-entropy estável

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Passo 5: Verificação de gradiente

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use

### Simulação de precisão mista

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Recorte de gradiente

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Detecção de NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Veja `code/numerical.py` para implementações completas com todos os casos extremos demonstrados.

## Entregue

Esta lição produz:
- `code/numerical.py` com softmax estável, log-sum-exp, cross-entropy, verificação de gradiente e simulação de precisão mista
- `outputs/prompt-numerical-debugger.md` para diagnosticar NaN/Inf e problemas numéricos no treino

Estas implementações estáveis reaparecem na Fase 3 ao construir o loop de treino e na Fase 4 ao implementar mecanismos de atenção.

## Exercícios

1. **Cancelamento catastrófico.** Compute a variância de [1000000.0, 1000001.0, 1000002.0] usando a fórmula ingênua `E[x^2] - E[x]^2` em float32. Depois compute usando o algoritmo online de Welford. Compare os erros contra a variância verdadeira (0.6667).

2. **Caça à precisão.** Encontre o menor valor positivo float32 `x` tal que `1.0 + x == 1.0` em Python. Este é o epsilon da máquina. Verifique que corresponde a `numpy.finfo(numpy.float32).eps`.

3. **Casos extremos de log-sum-exp.** Teste sua função `logsumexp_stable` com: (a) todos valores iguais, (b) um valor muito maior que os outros, (c) todos valores muito negativos (-1000). Verifique que dá resultados corretos onde a versão ingênua falha.

4. **Verificação de gradiente de uma camada de rede neural.** Implemente uma camada linear simples `y = Wx + b` e seu backward pass analítico. Use `numerical_gradient` para verificar a correção para uma matriz de pesos 3x2.

5. **Experimento de loss scaling.** Simule treino com float16: crie gradientes aleatórios na faixa [1e-9, 1e-3], converta para float16 e meça qual fração vira zero. Depois aplique loss scaling (multiplique por 1024), converta para float16, escale de volta e meça a fração de zero novamente.

## Termos-chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|------------------------|---------------------------|
| IEEE 754 | "O padrão de float" | Padrão internacional que define formatos de ponto flutuante binário, regras de arredondamento e valores especiais (inf, nan). Todo CPU e GPU moderno implementa. |
| Machine epsilon | "O limite de precisão" | O menor valor e tal que 1.0 + e != 1.0 em um formato float dado. Para float32, é cerca de 1.19e-7. |
| Cancelamento catastrófico | "Perda de precisão por subtração" | Quando subtrair números de ponto flutuante quase iguais, dígitos significativos se cancelam e ruído de arredondamento domina o resultado. |
| Overflow | "Número grande demais" | Um resultado excede o maior valor representável e vira inf. exp(89) faz overflow em float32. |
| Underflow | "Número pequeno demais" | Um resultado é mais perto de zero que o menor positivo representável e vira 0.0. exp(-104) faz underflow em float32. |
| Truque log-sum-exp | "Subtraia o máximo primeiro" | Computar log(sum(exp(x))) fatorando exp(max(x)) para prevenir overflow e underflow. Usado em softmax, cross-entropy e matemática de log-probabilidade. |
| Softmax estável | "Softmax que não explode" | Subtrair max(logits) antes de exponenciar. Resultado numericamente idêntico, sem overflow possível. |
| Verificação de gradiente | "Verifique sua backprop" | Comparar gradientes analíticos da backpropagation contra gradientes numéricos de diferenças finitas para capturar bugs de implementação. |
| Precisão mista | "Float16 forward, float32 backward" | Usar floats de menor precisão para operações críticas de velocidade e floats de maior precisão para operações numericamente sensíveis. Aceleração típica de 2-3x. |
| Loss scaling | "Prevenir underflow de gradiente" | Multiplicar a perda por uma constante grande antes do backprop para que gradientes fiquem na faixa representável do float16, depois dividir pela mesma constante antes da atualização de pesos. |
| bfloat16 | "Brain floating point" | Formato de 16 bits do Google com 8 bits de expoente (mesmo intervalo que float32) e 7 bits de mantissa (menos precisão que float16). Preferido para treino. |
| Recorte de gradiente | "Limite a norma do gradiente" | Escalar o vetor gradiente para que sua norma não exceda um limite. Previne que gradientes explosivos arruínem os pesos. |
| NaN | "Not a Number" | Valor float especial de operações indefinidas (0/0, inf-inf, sqrt(-1)). Propaga por toda aritmética subsequente. |
| Inf | "Infinito" | Valor float especial de overflow ou divisão por zero. Pode combinar para produzir NaN (inf - inf, inf * 0). |
| Gradiente numérico | "Derivada por força bruta" | Aproximar uma derivada avaliando f(x+h) e f(x-h) dividindo por 2h. Lento mas confiável para verificação. |

## Leitura Adicional

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- a referência definitiva, densa mas completa
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- o paper da NVIDIA que introduziu loss scaling para treino com float16
- [AMP: Automatic Mixed Precision (docs do PyTorch)](https://pytorch.org/docs/stable/amp.html) -- guia prático de precisão mista no PyTorch
- [Formato bfloat16 (docs Google Cloud TPU)](https://cloud.google.com/tpu/docs/bfloat16) -- por que o Google escolheu esse formato para TPUs
- [Soma de Kahan (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algoritmo para reduzir erro de arredondamento em somas de ponto flutuante
