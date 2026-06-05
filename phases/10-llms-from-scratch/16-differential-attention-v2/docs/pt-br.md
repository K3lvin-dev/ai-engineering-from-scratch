# Differential Attention (V2)

> A attention por softmax espalha um pouco de probabilidade sobre cada token que não casa. Em 100k tokens esse ruido se acumula e afoga o sinal. O Differential Transformer (Ye et al., ICLR 2025) corrige isso calculando attention como a diferença de dois softmaxes, subtraindo o chão de ruido compartilhado. DIFF V2 (Microsoft, janeiro de 2026) e a reescrita pra stack de produção: laténcia de decode compativel com baseline Transformer, sem kernels custom, compativel com FlashAttention. Esta aula vai de V1 a V2 de ponta a ponta, com uma implementação toy funcional da operação de diferença que você pode rodar em Python stdlib.

**Tipo:** Construir
**Linguagens:** Python (stdlib)
**Pré-requisitos:** Fase 7 · 02 (self-attention), Fase 7 · 15 (variantes de attention), Fase 10 · 14 (percurso arquitetonico)
**Tempo:** ~60 minutos

## Objetivos de Aprendizado

- Enunciar precisamente por que a attention por softmax tem um chão de ruido e por que ele cresce com o comprimento do contexto.
- Derivar a formula de differential attention e explicar por que a subtração cancela o componente de ruido compartilhado enquanto preserva o sinal.
- Percorrer o diff de V1 a V2: o que ficou mais rapido, o que ficou mais simples, o que ficou mais estavel, e por que cada mudanca foi necessaria pra pre-treinamento de produção.
- Implementar differential attention do zero em Python puro e verificar empiricamente a propriedade de cancelamento de ruido em um sinal sintetico sinal-mais-ruido.

## O Problema

A attention padrão por softmax tem uma propriedade matématica que vira uma dor de cabeca operacional em escala. Pra uma consulta `q`, os pesos de attention são `softmax(qK^T / sqrt(d))`. Softmax nunca produz zeros exatos -- cada token que não casa recebe massa positiva. Essa massa residual e ruido, e escala com o comprimento do contexto. Em 128k tokens, mesmo que cada token que não casa receba apenas 0.001% da probabilidade, 127.999 deles combinados contribuem com cerca de 12% do total. O modelo tem que aprender a contornar um chão de ruido que cresce com o contexto.

Empiricamente isso aparece como interferencia de heads de attention: citations alucinadas em RAG de contexto longo, falhas lost-in-the-middle em tasks de recuperação de 100k tokens, e degradação sútil de acuracia em benchmarks de agulha no palheiro além de 32k. O paper do Differential Transformer (arXiv:2410.05258, ICLR 2025) mediu a diferença: DIFF Transformers atingem perplexidade menor, acuracia de contexto longo maior e menos alucinações que baselines de mesmo tamanho.

DIFF V1 tinha três problemas que o mantiveram fora de pipelines de pre-treinamento de fronteira. Seu value cache tinha que ser carregado duas vezes por passo de decode, ele requeria kernels CUDA custom que quebravam a compatibilidade com FlashAttention, e seu RMSNorm por head desestabilizava treinamento de longa duração em escalas de 70B+. DIFF V2 (blog do Microsoft unilm, 20 de janeiro de 2026) corrigiu os três. Esta aula percorre ambas as versões, constrói o operador de diferença e faz benchmark de cancelamento de ruido em uma consulta toy.

## O Conceito

### O chão de ruido do softmax

Pra uma consulta `q` e keys `K = [k_1, ..., k_N]`, os pesos de attention sao:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Nenhum `w_i` e sempre zero. Se `k_i` e completamente irrelevante pra `q`, o score `q . k_i` não e 0 -- ele oscila em torno de zero com variancia `||q||^2 / d`. Após a normalização por softmax, cada token irrelevante ainda contribui `O(1/N)` pra soma ponderada. A contribuição total de tokens irrelevantes e `O((N-1)/N) = O(1)` -- não e uma quantidade pequena.

O que o modelo quer e algo como um hard top-k: peso alto nos tokens que casam, peso próximo de zero em todo o resto. Softmax e suave demais pra fazer isso diretamente.

### A ideia diferencial

Divida as projeções Q e K de cada head em duas: Q = (Q_1, Q_2) e K = (K_1, K_2). Calcule dois mapas de attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Saida:

```
DiffAttn = (A_1 - lambda * A_2) V
```

A subtração cancela qualquer distribuição de ruido que os dois mapas compartilham. Se ambos os mapas tem pesos aproximadamente uniformes nos 127k tokens irrelevantes (que e o que vão ter na inicialização aleatoria), esses cancelam. O sinal -- peso picado nos poucos tokens realmente relevantes -- so cancela se aparecer nos dois mapas na mesma magnitude, o que não acontece quando o modelo treina.

`lambda` e um escalar aprendivel por head, parametrizado como `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Pode ser negativo. `lambda_init` default pra um numero pequeno positivo como 0.8.

### Por que isso combina com cancelamento de ruido por head

Pense em dois microfones ruidosos gravando a mesma voz. Ambos captam o orador mais ruido de fundo correlacionado. Subtraia um do outro e o ruido compartilhado some. A voz sobrevive porque os dois sinais diferem em fase ou amplitude o suficiente pra impedir o cancelamento total. O `lambda` por head aprende exatamente esse equilibrio.

### V1 vs V2: o diff

V1 manteve a contagem de parâmetros igual ão baseline Transformer. Pra ter duas queries por head ele reduziu o tamanho do head pela metade. Isso custou expressividade do head e -- mais dolorosamente -- cortou o value cache por head pela metade. Decode teve que carregar o value cache duas vezes por etapa (uma por branch de softmax). Resultado: decode mais lento que o baseline apesar de combinar contagem de parâmetros.

V2 dobra o numero de Q heads e mantem os KV heads iguais (pegando parâmetros da projeção up). O tamanho do head continua o mesmo que o baseline. Após a subtração, a dimensão extra e projetada de volta pra combinar com a projeção O_W do baseline Transformer. Três coisas acontecem ão mesmo tempo:

1. Velocidade de decode combina com baseline (KV cache carregado uma vez).
2. FlashAttention roda inalterado (sem kernel custom).
3. Intensidade aritmetica no decode sobe (mais compute por byte carregado da HBM).

V2 também remove o RMSNorm por head que V1 usava pra estabilizar a subtração. Em escalas de pre-treinamento de 70B, esse RMSNorm desestabilizava o treinamento tardio. V2 substitui por um esquema de inicialização mais simples que mantem o treinamento estavel sem o modulo extra.

### Quando usar

| Carga de trabalho | Beneficio |
|-------------------|-----------|
| RAG de contexto longo (64k+) | Mapas de attention mais limpos, menos citations alucinadas |
| Benchmarks de agulha no palheiro | Ganho substancial de acuracia além de 32k |
| QA multi-documento | Menos interferencia entre documentos |
| Completude de codigo em 8k | Marginal, não vale a mudanca arquitetural |
| Chat curto (< 4k) | Essencialmente indistinguivel do baseline |

O valor cresce com o comprimento do contexto. Em 4k tokens o chão de ruido e pequeno o suficiente que attention padrão e fine. Em 128k ta te prejudicando.

### Como combina com outros botões de 2026

| Feature | Compativel com DIFF V2? |
|---------|------------------------|
| GQA | Sim (V2 aumenta Q heads, não KV heads) |
| MLA (DeepSeek) | Sim em principio, nenhum paper publicando a combinação |
| MoE | Sim (attention e independente do bloco MLP) |
| RoPE | Sim (inalterado) |
| YaRN / escalabilidade de contexto longo | Sim (exatamente onde DIFF mais ajuda) |
| FlashAttention | Sim em V2 (era não em V1) |
| Decodificação eespecificaçãoulativa | Sim (mudanca de attention e invisivel pro loop de decodificação eespecificaçãoulativa) |

## Construir

`code/main.py` implementa differential attention em Python puro. Uma consulta toy com estrutura conhecida de sinal-mais-ruido permite medir a taxa de cancelamento de ruido diretamente.

### Passo 1: attention padrão por softmax

Operações matriciais de stdlib: listas de listas, matmul manual, softmax com subtração de máximo pra estabilidade numérica.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Passo 2: dividir Q, K em duas metades

Estilo V1: reduzir o tamanho do head pela metade. Estilo V2: manter o tamanho do head e dobrar o numero de heads. A implementação toy usa V1 pra clareza pedagogica -- a matématica e identica, so muda a contabilidade.

### Passo 3: duas branches de softmax + subtração

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Nota: os pesos de saida podem ser negativos. Isso e ok -- o value cache ainda lida com contribuições com sinal. A projeção V subsequente absorve o sinal.

### Passo 4: medição de cancelamento de ruido

Construa uma sequencia sintetica de comprimento 1024. Coloque o token de sinal em uma posição conhecida, preencha o resto com ruido. Calcule (a) o peso padrão de attention por softmax na posição do sinal e (b) o peso de attention diferencial. Meça a razão sinal-ruido em cada uma. DIFF attention produz consistentemente uma razão sinal-ruido maior por um fator de 3x-10x dependendo de quanto as duas branches foram treinadas pra diferir.

### Passo 5: contabilidade de parâmetros V1 vs V2

Dada uma config (hidden=4096, heads=32, d_head=128), imprima:

- Baseline Transformer: Q, K, V cada um tamanho `hidden * hidden`, MLP em 4 * hidden.
- DIFF V1: Q, K cada um tamanho `hidden * hidden`, V tamanho `hidden * hidden` (inalterado), head dim pela metade internamente. Adiciona parâmetros `lambda` por head (O(heads * d_head)).
- DIFF V2: Q tamanho `2 * hidden * hidden`, K tamanho `hidden * hidden`, V tamanho `hidden * hidden`. Dim extra projetada de volta antes de O_W. Adiciona mesmos parâmetros `lambda`.

O toy mede o custo extra de parâmetros pra V2 (aproximadamente `hidden * hidden` extra por bloco de attention) e imprime.

## Usar

DIFF V2 ainda não esta sendo enviado em todo servidor de inferência de produção em abril de 2026, mas a integração esta em andamento no vLLM e SGLang. Enquanto isso o padrão aparece em:

- Modelos de produção de contexto longo internos da Microsoft.
- Replicações de pesquisa em varios runs de treinamento de modelos abertos almejando 256k+ de contexto.
- Arquiteturas hibridas que combinam DIFF attention com attention de sliding-window em camadas alternas.

Quando você usaria isso em 2026:

- Treinando um modelo novo do zero almejando 64k+ de contexto efetivo. Adicionar differential attention desde o comeco; retreinar depois e caro.
- Fine-tuning de um modelo de contexto longo onde falhas lost-in-the-middle dominam sua avaliação. Um LoRA nas projeções Q pode aproximar a estrutura DIFF.

Quando não usaria:

- Você esta servindo um modelo denso pre-treinado com desempenho de contexto longo estavel. O custo de retreinamento raramente compensa nos pesos existentes.
- Seu contexto e sempre abaixo de 16k. Chão de ruido e desprezivel.

## Entregar

Esta aula produz `outputs/skill-diff-attention-integrator.md`. Dada uma arquitetura de modelo, comprimento de contexto alvo, perfil de alucinação e orcamento de treinamento, produz um plano de integração pra adicionar differential attention a um novo run de pre-treinamento ou fine-tuning por LoRA.

## Exercicios

1. Rode `code/main.py`. Verifique que a razão sinal-ruido reportada pra differential attention e maior que a attention padrão por softmax na consulta sintetica. Varie a amplitude do ruido e mostre o ponto de cruzamento onde a attention padrão fica inútilizavel.

2. Calcule o delta de contagem de parâmetros do baseline pro DIFF V1 e do baseline pro DIFF V2 pra um modelo de classe 7B (hidden=4096, heads=32, d_head=128, 32 camadas). Mostre quais componentes ganharam parâmetros e quais permaneceram iguais.

3. Leia a Seção 3 do paper DIFF V1 (arXiv:2410.05258) e a Seção 2 do blog do Hugging Face do DIFF V2. Em duas frases, explique por que o RMSNorm por head do V1 era necessario e por que V2 conseguiu remove-lo sem causar divergencia de treinamento.

4. Implemente uma abalação: compute differential attention com `lambda = 0` (primeiro softmax puro) e `lambda = 1` (subtração total). Na consulta sintetica, meça como a razão sinal-ruido muda ão longo da varredura. Identifique o `lambda` que maximiza sinal-ruido.

5. Estenda o toy pra GQA + DIFF V2. Escolha 8 KV heads e 32 Q heads. Mostre que o tamanho do KV cache combina com um modelo GQA baseline com a mesma configuração (8, 32).

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Differential attention | "Dois softmaxes subtraindo um do outro" | Divide Q, K em duas metades, calcula dois mapas de softmax, subtrai o segundo (escalado por lambda) do primeiro, depois multiplica por V |
| Chão de ruido | "A cauda não-zero do softmax" | O peso O(1/N) que softmax coloca em cada token irrelevante, que soma O(1) em contextos longos |
| lambda | "A escala de subtração" | Escalar aprendivel por head parametrizado como `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; pode ser negativo |
| DIFF V1 | "A versão do ICLR 2025" | Differential Transformer original; reduz head dim pela metade pra preservar contagem de parâmetros, precisa kernel custom, decode mais lento |
| DIFF V2 | "A correção de janeiro de 2026" | Dobra Q heads mantendo KV heads; combina velocidade de decode do baseline e funciona com FlashAttention |
| RMSNorm por head | "O estabilizador do V1" | Norm extra que V1 aplicava após a diferença; V2 removeu pra evitar instabilidade no treinamento tardio |
| Razão sinal-ruido | "Quanta attention e desperdicada" | Razão do peso na posição do sinal verdadeiro pro peso medio em posições irrelevantes |
| Lost in the middle | "Modo de falha de contexto longo" | Fenomeno empirico onde a acuracia de recuperação cai pra documentos no meio de um contexto longo -- DIFF attention reduz isso |
| Intensidade aritmetica | "FLOPs por byte carregado" | Razão que V2 aumentou no decode por dobrar queries por carga KV; importante pra decode limitado por memoria |

## Leitura Complementar

- [Ye et al. -- Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) -- o paper original com teoria de cancelamento de ruido e abalações de contexto longo
- [Microsoft unilm -- Differential Transformer V2 (Hugging Face blog, janeiro 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) -- a reescrita pra stack de produção, compativel com decode do baseline, compativel com FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) -- análise teorica de por que a subtração recupera a estrutura de attention pre-treinada
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) -- variante de compartilhamento de parâmetros
- [Vaswani et al. -- Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) -- o baseline Transformer que DIFF subtrai
- [Liu et al. -- Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) -- o benchmark de contexto longo que DIFF attention foca
