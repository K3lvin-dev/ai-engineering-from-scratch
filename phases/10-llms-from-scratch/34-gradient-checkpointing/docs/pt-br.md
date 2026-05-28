# Gradient Checkpointing e Recomputacao de Ativacoes

> Backprop mantem cada ativacao intermediaria. Em 70B parametros e 128K contexto isso sao 3 TB de ativacoes por rank. Checkpointing troca FLOPs por memoria: recomputa ao inves de salvar. A questao e quais segmentos dropar e a resposta nao e "todos."

**Tipo:** Construir
**Linguagens:** Python (com numpy, torch opcional)
**Pre-requisitos:** Fase 10, Aula 04 (Pre-Treinamento Mini-GPT), Fase 10, Aula 05 (Escalabilidade e Distribuido)
**Tempo:** ~70 minutos

## O Problema

Treinar um transformer armazena, pra cada camada, as entradas de toda operacao que e diferenciada no backward: as entradas de attention, as projecoes Q/K/V, a saida do softmax, as entradas de FFN, as saidas de norm e o fluxo residual. Pra uma camada com hidden size `d`, comprimento de sequencia `L`, batch `B`, isso e da ordem de `12 * B * L * d` floats por camada.

Pra `d=8192, L=8192, B=1`, sao 800 MB/camada em BF16. Um modelo de 64 camadas e 51 GB de ativacoes -- e isso e antes de multiplicar por microbatch, antes de somar intermediarios de attention-softmax (`L^2` por head) e antes de levar em conta copias parciais de tensor parallelism.

A conta de dois lados: pesos BF16 mais estado do otimizador podem caber em 80GB, mas ativacoes te empurram pra fora. Gradient checkpointing (aka recomputacao de ativacoes) e a correcao padrao. Dropa a maior parte das ativacoes; refaz o forward durante o backward pra recupera-las. Custo: FLOPs extras. Beneficio: memoria cai pela razao de segmentos de checkpoint por camadas totais.

Feito ingenuamente, checkpointing custa cerca de 33% mais FLOPs de forward por etapa. Feito bem -- checkpointing seletivo pela "selecao inteligente" de Korthikanti et al. -- voce economiza 5x de memoria pra menos de 5% de overhead de FLOPs. E com matmuls FP8, offload FSDP e MoE expert-parallel isso realmente importa: voce nao pode pagar nem a memoria nem o compute desperdicado.

## O Conceito

### O que Backward Realmente Precisa

`output = layer(input)`. Backward quer `grad_input` e `grad_params`. Pra calcula-los precisa:

- `input` (pra calcular `grad_params = input.T @ grad_output` pra camadas lineares)
- alguns intermediarios de derivada de ativacao (a derivada de ReLU/GELU/softmax depende do valor da ativacao)

O forward pass armazena esses automaticamente no grafo autograd. Todo `tensor.retain_grad()` e toda operacao que precisa da sua entrada mantem uma referencia.

### Checkpointing Completo Ingenuo

Divida a rede em `N` segmentos. Durante forward, armazene apenas a *entrada* de cada segmento. Quando backward precisar dos intermediarios, rode o forward do segmento de novo pra materializa-los, depois diferencie.

Exemplo: transformer de 32 camadas dividido em 32 segmentos de 1 camada cada.

- Memoria: 32 entradas-de-camada (pequeno) vs 32 * (volume de ativacao por camada) (enorme).
- Compute extra: 1 forward por segmento, ou seja, ~33% mais FLOPs de forward total (ja que backward e 2x forward, passo completo vira 1 + 1 + 2 = 4 unidades em vez de 1 + 2 = 3).

Essa e a receita original de Chen et al. 2016: um checkpoint a cada `sqrt(L)` camadas pra equilibrar memoria e compute. Pra L=64, sao 8 checkpoints.

### Checkpointing Seletivo (Korthikanti 2022)

Nao todas as ativacoes custam o mesmo. A saida de attention softmax e `B*L*L*heads` e cresce *quadraticamente* com o comprimento da sequencia. A ativacao hidden de FFN e `B*L*4d` e cresce linearmente. Pra sequencias longas o softmax domina.

Checkpointing seletivo mantem as ativacoes baratas de armazenar (projecoes lineares, residuais) e recomputa apenas as caras (attention). Voce paga FLOPs minimos pra recomputar mas economiza a memoria O(L^2).

Megatron-Core implementa isso como recomputacao de ativacoes "seletiva." Usado na maioria dos runs de treinamento de fronteira 2024+.

### Offload

Alternativa pra recomputar: mandar ativacoes pra RAM da CPU entre forward e backward. Requer largura de banda PCIe; e benefico quando largura de banda ociosa excede o custo de rematerializacao. Estrategias misturas sao comuns: checkpoint em algumas camadas, offload em outras.

FSDP2 envia offload como opcao de primeira classe. Offload brilha quando GPU esta no gargalo de memoria mas a transferencia CPU-GPU tem espaco.

### Modelo de Custo de Recomputacao

FLOPs por etapa com checkpointing ingenuo a cada `k` camadas de `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # um forward extra por camada no segmento
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Com checkpointing seletivo voce recomputa apenas o kernel de attention, nao a camada inteira:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Modelo de Economia de Memoria

Volume de ativacao por camada: `A`. Pra `L` camadas, memoria total de ativacao: `L * A`.

Checkpoint completo (tamanho de segmento 1): armazena apenas `L * input_volume` (~`L * 1/10 A` pra um transformer padrao). Economiza ~`9 * L * A * 1/10`.

Checkpoint a cada `k` camadas: armazena `L/k * A` mais o valor dentro do segmento ativo de `k-1` camadas.

Em `k = sqrt(L)`, memoria e custo de recomputacao ambos escalam com `sqrt(L)` -- o tradeoff otimo pra camadas de custo uniforme.

### Quando NAO Checkpointar

- As camadas mais internas de um estagio de pipeline ja em voo. Elas tem que terminar de qualquer jeito.
- A primeira e ultima camada se dominam o compute do estagio (raro em transformers).
- Kernels de attention ja usando FlashAttention -- Flash ja recomputa o softmax rapido, entao checkpointing adicional por camada adiciona pouco.

### Padroes de Implementacao

1. **Wrapper de funcao:** envolver um segmento em `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch armazena apenas `input`, recomputa tudo mais no backward.

2. **Baseado em decorador:** marcar camadas como checkpointable; o treinador decide no tempo de config quais segmentos sao envolvidos.

3. **Recomputacao manual explicita:** escrever o backward voce mesmo, chamando um `recompute_forward` custom que duplica o forward com a entrada armazenada.

Os tres dao o mesmo resultado funcional. Wrappers sao o idiomma padrao.

### Interacao com TP / PP / FP8

- **Tensor parallel:** entradas de checkpoint precisam ser coletadas ou redistribuidas na recomputacao; lidar com o custo de comunicacao.
- **Pipeline parallel:** padrao tipico e checkpointar o forward de cada estagio de pipeline pra microbatches em ordem reversa reutilizarem memoria de ativacao.
- **Recomputacao FP8:** historicos de amax atualizados durante recomputacao precisam combinar com os do forward original, ou a escala FP8 desvia. A maioria dos frameworks tira um snapshot da escala.

## Construir

### Passo 1: Um Modelo Toy Com Segmentos

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Passo 2: Backward Ingenuo Precisando de Todas as Ativacoes

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Passo 3: Checkpoint-a-Cada-k Memoria

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Passo 4: Modelo de Custo

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Passo 5: Estimador de Memoria

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Passo 6: Tamanho Otimo de Segmento

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Passo 7: Decisao de Checkpointing Seletivo

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Usar

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` -- o wrapper canonico no PyTorch. Envolve uma funcao; armazena apenas entradas, recomputa no backward.
- **Recomputacao de ativacoes do Megatron-Core**: suporta modos `selective`, `full` e `block`. Padrao em treinamentos de fronteira 2024+.
- **Offload FSDP2**: `module.to_empty(device="cpu")` com `offload_policy` no FSDP2 fragmenta ativacoes pra CPU ao inves de recomputar.
- **DeepSpeed ZeRO-Offload**: offload de CPU pra estados de otimizador e ativacoes, complementando checkpointing.

## Entregar

Esta aula produz `outputs/prompt-activation-recompute-policy.md` -- um prompt que recebe a config do seu modelo (camadas, hidden, seq, batch) e memoria GPU disponivel e emite uma politica de recomputacao por camada (nenhum / seletivo / completo / offload).

## Exercicios

1. Verificar correcao. Rodar `model_forward` + `model_backward` (ativacoes completas) vs `model_forward_checkpointed` + `model_backward_checkpointed` (segmentos). Gradientes de parametros devem ser identicos ate precisao de maquina.

2. Varer tamanho de segmento `k` de 1 ate `L`. Plotar overhead de FLOPs e memoria. Encontrar o joelho da curva.

3. Implementar checkpointing seletivo: armazenar a entrada do modulo de attention mas nao seus intermediarios. Medir o overhead de FLOPs vs checkpointing de camada inteira pra um modelo de 32 camadas em seq=8192.

4. Adicionar offload. Salvar entradas de segmento em um "buffer de CPU" simulado (uma lista separada). Medir "largura de banda de PCIe" como bytes/tempo e encontrar o ponto de equilibrio entre offload e recomputacao.

5. Fazer benchmark de um transformer PyTorch real com e sem `torch.utils.checkpoint`. Medir memoria (via `torch.cuda.max_memory_allocated`) e tempo por etapa.

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Gradient checkpointing | "Economizar memoria refazendo forward" | Armazenar apenas entradas de segmentos; recomputar intermediarios durante backward pra obter tensores de suporte a gradiente |
| Recomputacao de ativacoes | "Mesmo que checkpointing" | O nome no estilo HPC pra mesma tecnica |
| Tamanho de segmento (k) | "Quantas camadas por checkpoint" | Numero de camadas cujos intermediarios sao dropados e rematerializados juntos |
| Checkpointing seletivo | "O truque do Korthikanti" | Recomputar apenas ativacoes caras de armazenar (attention softmax); manter as baratas |
| Checkpointing completo | "A versao ingenua" | Recomputar os intermediarios de cada camada em cada segmento |
| Checkpointing em bloco | "Granulosidade grossa" | Checkpointar blocos transformer inteiros; maior granulosidade |
| Overhead de FLOPs | "O imposto de compute" | FLOPs extras por etapa = (FLOPs de recomputacao) / (FLOPs fwd + bwd); 33% ingenuo, 5% seletivo |
| Offload de ativacoes | "Mandar pra CPU" | Mover ativacoes pra RAM da CPU entre forward->backward; alternativa a recomputacao |
| Regra sqrt-L | "O otimo classico" | Pra camadas de custo uniforme, espacamento otimo de checkpoint e sqrt(L) camadas |
| Volume de attention-softmax | "O problema O(L^2)" | L^2 * heads * batch floats; domina memoria de ativacao em contextos longos |

## Leitura Complementar

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- o paper original que formalizou gradient checkpointing
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- recomputacao seletiva de ativacoes e analise formal de custo
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- abordagem alternativa de memoria constante via rematerializacao em modo reverso
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- offload de ativacoes em escala
- [Documentacao torch.utils.checkpoint do PyTorch](https://pytorch.org/docs/stable/checkpoint.html) -- a API padrao
- [Documentacao de recomputacao de ativacoes do Megatron-Core](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- modos seletivo, completo e em bloco
