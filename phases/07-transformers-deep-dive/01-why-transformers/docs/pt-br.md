# Por que Transformers — Os Problemas com RNNs

> RNNs processam tokens um de cada vez. Transformers processam todos os tokens de uma vez. Essa aposta arquitetônica mudou todas as curvas de escala do deep learning depois de 2017.

**Tipo:** Aprender
**Linguagens:** Python
**Pré-requisitos:** Fase 3 (Deep Learning Core), Fase 5 · 09 (Sequence-to-Sequence), Fase 5 · 10 (Mecanismo de Attention)
**Tempo:** ~45 minutos

## O Problema

Antes de 2017, todo modelo state-of-the-art de sequência no planeta — linguagem, tradução, fala — era uma rede neural recorrente. LSTMs e GRUs dominaram benchmarks de tradução equivalentes ao ImageNet durante meia década. Eram a única ferramenta que existia.

Tinham três fraquezas fatais. Computação sequencial significava que você não conseguia paralelizar ao longo do eixo temporal: o token `t+1` precisa do estado oculto do token `t`. Uma sequência de 1.024 tokens significava 1.024 passos seriais em uma GPU capaz de fazer 1.000.000 operações de ponto flutuante por ciclo. Tempo de treinamento escala linearmente com o comprimento da sequência em hardware feito para paralelismo.

Gradientes que desapareciam significavam que informações de 50 tokens atrás já tinham sido comprimidas através de 50 não-linearidades. Unidades recorrentes com portões (LSTM, GRU) suavizaram o impacto, mas nunca eliminaram. Dependências de longo alcance — "o livro que eu li no verão passado num avião para Quioto era..." — falhavam rotineiramente.

Estados ocultos de largura fixa significavam que o encoder comprimia toda a sequência de entrada em um único vetor antes que o decoder visse qualquer coisa. Não importa se a fonte tem 5 tokens ou 500; o gargalo tem a mesma forma.

O paper de 2017 "Attention Is All You Need" propôs algo radical: abandonar a recorrência completamente. Deixar cada posição attend em paralelo a todas as outras posições. Treinar em uma única multiplicação de matrizes gigante em vez de 1.024 multiplicações sequenciais.

O resultado domina todas as modalidades até 2026. Linguagem (GPT-5, Claude 4, Llama 4), visão (ViT, DINOv2, SAM 3), áudio (Whisper), biologia (AlphaFold 3), robótica (RT-2). Mesmo bloco, entradas diferentes.

## O Conceito

![RNN compute sequencial vs Transformer attention paralela](../assets/rnn-vs-transformer.svg)

**Recorrência como gargalo.** Uma RNN calcula `h_t = f(h_{t-1}, x_t)`. Cada passo depende do anterior. Você não pode calcular `h_5` antes de `h_4`. Em GPUs modernas com mais de 10.000 cores paralelos, isso desperdiça 99% do silício numa sequência longa.

**Attention como broadcast.** Self-attention calcula `output_i = sum_j(a_ij * v_j)` para cada par `(i, j)` simultaneamente. A matriz inteira de attention N×N é preenchida em uma única matmul em batch. Nenhum passo depende de outro. GPUs adoram.

**A aceleração não é constante.** É a diferença entre profundidade serial de `O(N)` e profundidade serial de `O(1)`. Na prática, transformers treinam 5–10× mais rápido por epoch em hardware equivalente com N=512, e a diferença cresce com o comprimento da sequência até você bater no muro de memória `O(N²)` da attention (que o Flash Attention corrigiu depois — ver Aula 12).

**O custo dos transformers.** Memória da attention escala como `O(N²)`. Para contexto de 2K, tranquilo. Para contexto de 128K, você precisa de janelas deslizantes, extrapolação RoPE, Flash Attention em tiles ou variantes de attention linear. Recorrência era `O(N)` em tempo e memória; transformers trocam tempo por memória e recuperam o tempo de volta via paralelismo.

**A mudança de viés indutivo.** RNNs assumem localidade e recência. Transformers não assumem nada — cada par é candidato a attention. É por isso que transformers precisam de mais dados pra treinar bem, mas escalam mais quando têm. Chinchilla (2022) formalizou isso: dados tokens suficientes, um transformer sempre vence uma RNN com contagem de parâmetros igual.

## Construindo

Sem rede neural aqui — simulamos o gargalo central numericamente pra você sentir a diferença no seu laptop.

### Passo 1: medir profundidade serial

Veja `code/main.py`. Construímos duas funções. Uma codifica uma sequência como uma corrente de adições (serial, como uma RNN). Outra codifica como uma redução paralela (broadcast, como attention). Mesma matemática, grafo de dependências diferente.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Mudamos o tempo de execução em sequências de até 100.000 elementos. A versão RNN é O(N) e usa um pipeline serial de CPU. Mesmo em Python puro, a redução estilo attention vence a partir de comprimento ≥ 1.000 porque o `sum()` do Python é implementado em C e itera sem overhead do interpretador a cada passo.

### Passo 2: contar operações teóricas

Ambos os algoritmos fazem N adições. A diferença é a *profundidade de dependências*: quantas operações precisam acontecer serialmente antes que a próxima possa começar. Profundidade da RNN = N. Profundidade da attention = log(N) com uma redução em árvore, ou 1 com uma varredura paralela. Profundidade, não contagem de operações,决定 o tempo de GPU.

### Passo 3: escala empírica em sequências longas

Imprimimos uma tabela de tempos que torna a diferença de O(N) visível. Num Mac de 2026, sequências abaixo de 1.000 elementos são rápidas demais pra medir. Sequências de 100.000 mostram uma varredura linear limpa. Escale isso para um transformer de 16.384 tokens com um LSTM equivalente de 12 camadas e você entende por que tempo de treinamento era um bloqueador em 2016.

## Usando

Quando ainda escolher uma RNN em 2026:

| Situação | Escolha |
|-----------|---------|
| Inferência streaming, um token por vez, memória constante | RNN ou modelo de espaço de estados (Mamba, RWKV) |
| Sequências muito longas (>1M tokens) onde memória da attention explode | Attention linear, Mamba 2, Hyena |
| Dispositivo de borda sem acelerador de matmul | RNN com separação por profundidade ainda ganha em FLOPs/watt |
| Qualquer outra coisa (treinamento, inferência em batch, contexto até 128K) | Transformer |

Modelos de espaço de estados (SSMs) como Mamba são basicamente RNNs com parametrização estruturada que dá o melhor dos dois mundos: memória de varredura `O(N)`, treinamento paralelo via scan seletivo. Recuperam 90% da qualidade de transformers com melhor escala de contexto longo. Em 2026 a maioria dos laboratórios de ponta treina modelos híbridos SSM+transformer (ex: Jamba, Samba) — recorrência não morreu, é um componente.

## Entregando

Veja `outputs/skill-architecture-picker.md`. A skill escolhe uma arquitetura para um novo problema de sequência dado comprimento, throughput e restrições de orçamento de treinamento. Deve sempre se recusar a recomendar uma RNN pura para execuções de treinamento acima de 1B tokens sem declarar o trade-off.

## Exercícios

1. **Fácil.** Pegue `rnn_style` de `code/main.py` e substitua o estado oculto escalar por um vetor de estados ocultos de tamanho 64. Refaça a medição. O overhead serial cresce quanto com a dimensionalidade do estado oculto?
2. **Médio.** Implemente uma soma de prefixos paralela (varredura Hillis-Steele) em Python puro. Verifique que produz a mesma saída numérica que uma varredura serial no comprimento 1024. Conte a profundidade.
3. **Difícil.** Porte a redução estilo attention para PyTorch na GPU. Cronometre ambas enquanto varia o comprimento da sequência de 64 a 65.536. Plote e explique a forma da curva.

## Termos-Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|------------------------|--------------------------|
| Recorrência | "RNNs são sequenciais" | Computação onde o passo `t` depende do passo `t-1`, forçando execução serial ao longo do eixo temporal. |
| Profundidade serial | "O quão profundo é o grafo" | Maior cadeia de operações dependentes; limita tempo mesmo com hardware infinito. |
| Attention | "Deixar tokens se olharem" | Soma ponderada `sum_j a_ij v_j` onde `a_ij` vem de um score de similaridade entre as posições i e j. |
| Janela de contexto | "Quanto o modelo vê" | Número de posições que uma camada de attention pode receber como entrada; custo de memória quadrático escala aqui. |
| Viés indutivo | "Pressupostos embutidos na arquitetura" | Prior sobre como os dados são; CNNs assume invariância de translação, RNNs assumem recência. |
| Modelo de espaço de estados | "RNN com álgebra por trás" | Recorrência parametrizada para treinamento paralelo via matrizes de espaço de estados estruturadas. |
| Gargalo quadrático | "Por que contexto custa tão caro" | Memória da attention = `O(N²)` no comprimento da sequência; Flash Attention esconde as constantes, não a escala. |

## Leituras Complementares

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — o paper que matou a recorrência no NLP mainstream.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — onde a attention nasceu, conectada a uma RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — o paper original da LSTM, por registro.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — resposta recorrente moderna aos transformers.
