# Introdução ao JAX

> PyTorch muta tensores. TensorFlow constrói grafos. JAX compila funções puras. Essa última coisa muda como você pensa sobre deep learning.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Aulas 01-10 da Fase 03, NumPy básico
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Escrever código de rede neural usando a API funcional do JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explicar a diferença-chave de design entre mutação ansiosa do PyTorch e o modelo de compilação funcional do JAX
- Aplicar compilação jit e vectorização vmap pra acelerar loops de treino comparados com Python ingênuo
- Treinar uma rede simples em JAX e contrastar o gerenciamento explícito de estado com a abordagem orientada a objetos do PyTorch

## O Problema

Você sabe como construir redes neurais em PyTorch. Define um `nn.Module`, chama `.backward()`, avança o otimizador. Funciona. Milhões de pessoas usam.

Mas PyTorch tem uma restrição no seu DNA: ele rastreia operações ansiosamente, uma por vez, em Python. Cada `tensor + tensor` é um disparo de kernel separado. Cada passo de treino re-interpreta o mesmo código Python. Isso funciona até você precisar treinar um modelo de 540 bilhões de parâmetros em 2.048 TPUs. Aí a sobrecarga te mata.

Google DeepMind treina Gemini em JAX. Anthropic treinou Claude em JAX. Não são operações pequenas — são os maiores treinos de redes neurais na Terra. Eles escolheram JAX porque ele trata seu loop de treino como um programa compilável, não uma sequência de chamadas Python.

JAX é NumPy com três superpoderes: diferenciação automática, compilação JIT pra XLA e vectorização automática. Você escreve uma função que processa uma amostra. JAX te dá uma função que processa um lote, computa gradientes, compila pra código de máquina e roda em múltiplos dispositivos. Tudo sem mudar a função original.

## O Conceito

### A Filosofia JAX

JAX é um framework funcional. Sem classes, sem estado mutável, sem método `.backward()`. Em vez disso:

| PyTorch | JAX |
|---------|-----|
| Classe `nn.Module` com estado | Função pura: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Execução ansiosa | Compilação JIT via XLA |
| `for x in batch:` loop manual | `jax.vmap(f)` vectorização automática |
| `DataParallel` / `FSDP` | `jax.pmap(f)` paralelismo automático |
| `model.parameters()` mutável | pytree imutável de arrays |

Isso não é preferência de estilo. É uma restrição do compilador. Compilação JIT requer funções puras — mesmas entradas sempre produzem mesmas saídas, sem efeitos colaterais. Essa restrição é o que torna acelerações de 100x possíveis.

### jax.numpy: A Superfície Familiar

JAX reimplementa a API do NumPy em aceleradores:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Mesmos nomes de função. Mesmas regras de broadcasting. Mesma semântica de slicing. Mas os arrays vivem na GPU/TPU e toda operação é rastreável pelo compilador.

Uma diferença crítica: arrays JAX são imutáveis. Sem `a[0] = 5`. Em vez disso: `a = a.at[0].set(5)`. Isso parece estranho por uma semana, depois faz sentido — imutabilidade é o que torna transformações como `grad`, `jit` e `vmap` combináveis.

### jax.grad: Autodiff Funcional

PyTorch anexa gradientes a tensores (`.grad`). JAX anexa gradientes a funções.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` recebe uma função e retorna uma nova função que computa o gradiente. Sem chamada `.backward()`. Sem grafo computacional armazenado em tensores. O gradiente é só outra função que você pode chamar, compor ou compilar com JIT.

Isso compõe arbitrariamente:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Segundas derivadas. Terceiras derivadas. Jacobianos. Hessianos. Tudo compondo `grad`. PyTorch também consegue (`torch.autograd.functional.hessian`), mas é algo adicional. No JAX, é a fundação.

A restrição: `grad` só funciona em funções puras. Sem prints dentro (rodam durante rastreamento, não execução). Sem mutação de estado externo. Sem geração de números aleatórios sem gerenciamento explícito de chaves.

### jit: Compilar pra XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Na primeira chamada, JAX rastreia a função — registra quais operações acontecem, sem executá-las. Depois passa o rastreamento pro XLA (Accelerated Linear Algebra), o compilador do Google pra TPUs e GPUs. XLA funde operações, elimina cópias de memória redundantes e gera código de máquina otimizado.

Chamadas subsequentes pulam Python completamente. O código compilado roda no acelerador à velocidade C++.

Quando JIT ajuda:
- Passos de treino (mesma computação repetida milhares de vezes)
- Inferência (mesmo modelo, entradas diferentes)
- Qualquer função chamada mais de uma vez com entradas de forma similar

Quando JIT atrapalha:
- Funções com fluxo de controle Python que depende de valores (`if x > 0` onde x é um array rastreado)
- Computações únicas (sobrecarga de compilação excede tempo de execução)
- Debug (rastreamento esconde a execução real)

A restrição de fluxo de controle é real. `jax.lax.cond` substitui `if/else`. `jax.lax.scan` substitui loops `for`. Estes não são opcionais — são o preço da compilação.

### vmap: Vectorização Automática

Você escreve uma função que processa uma amostra:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` eleva pra processar um lote:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` significa: não criar lote sobre `params` (compartilhado), criar lote sobre eixo 0 de `x`. Sem loop `for` manual. Sem redimensionamento. Sem threading de dimensão de lote. JAX descobre a dimensão do lote e vetoriza a computação inteira.

Isso não é açúcar sintático. `vmap` gera código vectorizado fundido que roda 10-100x mais rápido que um loop Python. E compõe com `jit` e `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradientes por amostra. Uma linha. Isso é quase impossível em PyTorch sem truques.

### pmap: Paralelismo de Dados entre Dispositivos

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` replica a função através de todos os dispositivos disponíveis (GPUs/TPUs) e divide o lote. Dentro da função, `jax.lax.pmean` e `jax.lax.psum` sincronizam gradientes entre dispositivos.

Google treina Gemini em milhares de chips TPU v5e usando `pmap` (e seu sucessor `shard_map`). O modelo de programação: escreva a versão de dispositivo único, envolva com `pmap`, pronto.

### Pytrees: A Estrutura de Dados Universal

JAX opera em "pytrees" — combinações aninhadas de listas, tuplas, dicts e arrays. Seus parâmetros de modelo são um pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Toda transformação JAX — `grad`, `jit`, `vmap` — sabe percorrer pytrees. `jax.tree.map(f, tree)` aplica `f` a toda folha. É assim que otimizadores atualizam todos parâmetros de uma vez:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Sem método `.parameters()`. Sem registro de parâmetros. A estrutura da árvore é o modelo.

### Funcional vs Orientado a Objetos

PyTorch armazena estado dentro de objetos:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x)
```

JAX usa funções puras com estado explícito:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Os params são passados. Nada é armazenado. Nada é mutado. Isso torna toda função testável, combinável e compilável. Também significa que você gerencia os params você mesmo — ou usa uma biblioteca como Flax ou Equinox.

### Ecossistema JAX

JAX te dá primitivas. Bibliotecas te dão ergonomia:

| Biblioteca | Papel | Estilo |
|-----------|-------|--------|
| **Flax** (Google) | Camadas de rede neural | `nn.Module` com estado explícito |
| **Equinox** (Patrick Kidger) | Camadas de rede neural | Baseado em pytree, Pythonico |
| **Optax** (DeepMind) | Otimizadores + agendamentos LR | Transformações de gradiente compostas |
| **Orbax** (Google) | Checkpointing | Salvar/restaurar pytrees |
| **CLU** (Google) | Métricas + logging | Utilitários de loop de treino |

Optax é a biblioteca de otimizadores padrão. Ela separa a transformação do gradiente (Adam, SGD, clipping) da atualização do parâmetro, tornando trivial compor:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Quando Usar JAX vs PyTorch

| Fator | JAX | PyTorch |
|-------|-----|---------|
| Suporte TPU | Primeira classe (Google construiu ambos) | Mantido pela comunidade (torch_xla) |
| Suporte GPU | Bom (CUDA via XLA) | Melhor da classe (CUDA nativo) |
| Debug | Difícil (rastreamento + compilação) | Fácil (ansioso, linha por linha) |
| Ecossistema | Focado em pesquisa (Flax, Equinox) | Massivo (HuggingFace, torchvision, etc.) |
| Contratação | Nicho (Google/DeepMind/Anthropic) | Principal (todo lugar) |
| Treino em grande escala | Superior (XLA, pmap, mesh) | Bom (FSDP, DeepSpeed) |
| Prototipagem | Mais lento (sobrecarga funcional) | Mais rápido (mutar e seguir) |
| Inferência em produção | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Quem usa | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

A resposta honesta: use PyTorch a menos que tenha uma razão específica pra usar JAX. Essas razões são: acesso a TPU, necessidade de gradientes por amostra, treino multi-dispositivo em escala massiva, ou trabalhar no Google/DeepMind/Anthropic.

### Números Aleatórios no JAX

JAX não tem estado aleatório global. Toda operação aleatória requer uma PRNG key explícita:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Isso é chato no começo. Mas garante reprodutibilidade entre dispositivos e compilações — uma propriedade que o `torch.manual_seed` do PyTorch não consegue garantir em configurações multi-GPU.

## Construa

### Passo 1: Setup e Dados

Vamos treinar um MLP de 3 camadas no MNIST usando JAX e Optax. 784 entradas, duas camadas ocultas de 256 e 128 neurônios, 10 classes de saída.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Passo 2: Inicializar Parâmetros

Sem classe. Só uma função que retorna um pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Inicialização He, feita manualmente. Três chaves PRNG divididas de uma semente. Cada peso é um array imutável num dict aninhado.

### Passo 3: Passo Direto

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Funções puras. Params entram, previsão sai. Sem `self`, sem estado armazenado. `loss_fn` computa entropia cruzada do zero — softmax, log, média negativa.

### Passo 4: Passo de Treino Compilado com JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` retorna tanto o valor da perda quanto os gradientes numa passada. O decorador `@jax.jit` compila ambas as funções pra XLA. Após a primeira chamada, cada passo de treino roda sem tocar Python.

### Passo 5: Loop de Treino

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 épocas. ~97% de acurácia de teste. A primeira época é lenta (compilação JIT). Épocas 2-10 são rápidas.

Repare no que está faltando: sem `.zero_grad()`, sem `.backward()`, sem `.step()`. A atualização inteira é uma chamada de função composta. Gradientes são computados, transformados pelo Adam e aplicados aos parâmetros — tudo dentro de `train_step`.

## Use

### Flax: O Padrão Google

Flax é a biblioteca de rede neural JAX mais comum. Ela adiciona `nn.Module` de volta, mas com gerenciamento de estado explícito:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Mesma estrutura que PyTorch, mas `params` é separado do modelo. `model.init()` cria params. `model.apply(params, x)` roda o passo direto. O objeto modelo não tem estado.

### Equinox: A Alternativa Pythonica

Equinox (por Patrick Kidger) representa modelos como pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

O modelo em si é um pytree. Sem `.apply()` necessário. Os parâmetros são apenas as folhas do modelo. Isso é mais próximo de como JAX pensa.

### Optax: Otimizadores Compostos

Optax desacopla a transformação do gradiente da atualização:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Clipping de gradiente, warmup de taxa de aprendizado, weight decay — tudo composto como uma cadeia de transformações. Cada transformação vê os gradientes, modifica e passa pro próximo. Sem classe de otimizador monolítica.

## Entregue

**Instalação:**

```bash
pip install jax jaxlib optax flax
```

Pra suporte GPU:

```bash
pip install jax[cuda12]
```

Pra TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Armadilhas de performance:**

- Primeira chamada JIT é lenta (compilação). Aqueça antes de benchmarkear.
- Evite loops Python sobre arrays JAX dentro de JIT. Use `jax.lax.scan` ou `jax.lax.fori_loop`.
- `jax.debug.print()` funciona dentro de JIT. `print()` normal não.
- Profile com `jax.profiler` ou TensorBoard. Compilação XLA pode esconder gargalos.
- JAX pré-aloca 75% da memória GPU por padrão. Defina `XLA_PYTHON_CLIENT_PREALLOCATE=false` pra desabilitar.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

Esta aula produz:
- `outputs/prompt-jax-optimizer.md` — um prompt pra escolher a configuração certa de otimizador JAX
- `outputs/skill-jax-patterns.md` — uma habilidade cobrindo padrões funcionais no JAX

## Exercícios

1. Adicione dropout ao MLP. No JAX, dropout requer uma PRNG key — passe uma key pelo passo direto e divida pra cada camada de dropout. Compare acurácia de teste com e sem.

2. Use `jax.vmap` pra computar gradientes por amostra pra um lote de 32 imagens MNIST. Compute a norma do gradiente pra cada amostra. Quais amostras têm os maiores gradientes e por quê?

3. Substitua a função manual de forward por uma `mlp_forward(params, x)` genérica que funcione pra qualquer número de camadas. Use `jax.tree.leaves` pra determinar a profundidade automaticamente.

4. Faça benchmark do passo de treino com e sem `@jax.jit`. Cronometre 100 passos de cada. Qual a aceleração no seu hardware? Qual a sobrecarga de compilação na primeira chamada?

5. Implemente clipping de gradiente compondo `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Treine com e sem clipping. Plote a norma do gradiente durante o treino pra ver o efeito.

## Termos-chave

| Termo | O que o pessoal diz | O que realmente significa |
|-------|---------------------|--------------------------|
| XLA | "A coisa que deixa JAX rápido" | Accelerated Linear Algebra — um compilador que funde operações e gera kernels GPU/TPU otimizados a partir de um grafo computacional |
| JIT | "Compilação just-in-time" | JAX rastreia a função na primeira chamada, compila pra XLA e roda a versão compilada nas chamadas seguintes |
| Função pura | "Sem efeitos colaterais" | Uma função onde a saída depende só das entradas — sem estado global, sem mutação, sem aleatoriedade sem chaves explícitas |
| vmap | "Auto-batching" | Transforma uma função que processa uma amostra em uma que processa um lote, sem reescrever |
| pmap | "Auto-paralelismo" | Replica uma função entre múltiplos dispositivos e divide o lote de entrada |
| Pytree | "Dict aninhado de arrays" | Qualquer estrutura aninhada de listas, tuplas, dicts e arrays que JAX pode percorrer e transformar |
| Rastreamento | "Registrando a computação" | JAX executa a função com valores abstratos pra construir um grafo computacional, sem computar resultados reais |
| Autodiff funcional | "grad de uma função" | Computar derivadas transformando funções, não anexando armazenamento de gradiente a tensores |
| Optax | "Biblioteca de otimizadores do JAX" | Uma biblioteca composta de transformações de gradiente — Adam, SGD, clipping, agendamento — que se encadeiam |
| Flax | "nn.Module do JAX" | A biblioteca de rede neural do Google pra JAX, adicionando abstrações de camada enquanto mantém estado explícito |

## Leituras Complementares

- Documentação JAX: https://jax.readthedocs.io/ — os docs oficiais, com tutoriais excelentes sobre grad, jit e vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) — o paper original explicando a filosofia de design
- Documentação Flax: https://flax.readthedocs.io/ — a biblioteca de rede neural do Google pra JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) — a alternativa Pythonica ao Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" — a biblioteca de otimizadores padrão
- "You Don't Know JAX" (Colin Raffel, 2020) — um guia prático de pegadinhas e padrões JAX, de um dos autores do T5
