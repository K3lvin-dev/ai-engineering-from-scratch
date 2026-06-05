# Multi-Token Prediction (MTP)

> Todo LLM autoregressivo do GPT-2 ão Llama 3 treina com uma loss por posição: predizer o próximo token. DeepSeek-V3 adicionou uma segunda loss por posição: predizer o token seguinte. Os 14B extras de parâmetros (num modelo de 671B) foram destilados de volta pro modelo principal através do fluxo de gradiente, e os heads MTP treinados foram reaproveitados na inferência como rascunhos de decodificação eespecificaçãoulativa com 80%+ de aceitação. 1.8x de throughput de geração veio gratis. Esta aula constroi o modulo MTP sequencial do relatorio tecnico da DeepSeek, calcula a loss e o layout de parâmetros do head compartilhado e explica por que MTP mantem a cadeia causal enquanto o MTP paralelo original de Gloeckle et al. quebrava.

**Tipo:** Construir
**Linguagens:** Python (stdlib)
**Pré-requisitos:** Fase 10 · 04 (pre-treinamento de mini GPT), Fase 10 · 15 (decodificação eespecificaçãoulativa)
**Tempo:** ~60 minutos

## Objetivos de Aprendizado

- Enunciar o objetivo de treinamento do MTP e derivar a loss conjunta através das profundezas de predição.
- Explicar a diferença entre os heads MTP paralelos de Gloeckle et al. (2024) e os modulos MTP sequenciais do DeepSeek-V3 e por que o design sequencial preserva a cadeia causal.
- Calcular o overhead de parâmetros e memoria de adicionar modulos MTP a um run de pre-treinamento.
- Implementar um modulo MTP do zero: o embedding compartilhado, o bloco transformer por profundez, a projeção e o head de saida compartilhado.

## O Problema

A predição do próximo token e o objetivo padrão de treinamento de LLMs. Cada hidden staté e supervisionado pra predizer exatamente uma coisa: o token imediatamente seguinte. Isso e um sinal surpreendentemente fraco. A maior parte da informação numa sequencia vai além de um token -- estrutura, çõesão, factualidade, fluidez aritmetica. O modelo tem que aprender tudo isso acumulando muitos sinais de um token ão longo de trilhões de tokens.

MTP pergunta: e se cada hidden staté fosse supervisionado pra predizer multiplos tokens futuros de uma vez? Gloeckle et al. (Meta, 2024) mostraram que isso ajuda. A implementação deles colocou varios output heads independentes no topo da backbone, cada um predizendo um offset diferente. Paralelo, simples, mas os heads viam o mesmo hidden staté sem nenhum refinamento hierarquico -- e as predições não encadeavam causalmente, então não podiam ser usadas pra decodificação eespecificaçãoulativa.

DeepSeek-V3 (dezembro de 2024) redesenhou o MTP como modulos sequenciais que mantem a cadeia causal em cada profundez de predição. O modelo prediz `t+1` de `h_i^(0)`, depois prediz `t+2` de um novo hidden staté `h_i^(1)` que combina `h_i^(0)` com o embedding `E(t+1)`, e assim por diante. Cada profundez e seu proprio bloco transformer pequeno. O embedding compartilhado e o output head compartilhado mantem o overhead de parâmetros moderado. Na escala do DeepSeek-V3, 14B de parâmetros extras nos modulos MTP além dos 671B de pesos do modelo principal. Esse overhead de 2% comprou sinais de treinamento mais densos E um rascunho pronto pra decodificação eespecificaçãoulativa na inferência.

Esta aula constrói um unico modulo MTP e a loss de profundez D do zero. A matématica e limpa. A implementação são 150 linhas.

## O Conceito

### A receita do MTP sequencial

DeepSeek-V3 adiciona `D` modulos MTP no topo do modelo principal. Cada modulo `k` (pra `k = 1..D`) prediz o token na profundez `k` -- ou seja, `t_{i+k}` dado um prefixo até a posição `i`.

O modulo `k` consiste em:

- Um bloco transformer `T_k` com sua propria attention e MLP.
- Uma matriz de projeção `M_k` que combina o hidden staté da profundez anterior com o embedding do token ground-truth da proxima profundez.
- O embedding compartilhado `E` (igual ão do modelo principal).
- O output head compartilhado `Out` (igual ão do modelo principal).

No treinamento, pra um prefixo até a posição `i`, o hidden staté por profundez e:

```
h_i^(0) = backbone do modelo principal na posição i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   pra k >= 1
```

A predição por profundez e:

```
logits_{i+k} = Out(h_i^(k-1))   pra k = 1..D
```

A loss por profundez e cross-entropy contra o ground-truth `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

A loss conjunta através das profundezes:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` e um fator de ponderação pequeno -- DeepSeek-V3 usa 0.3 nos primeiros 10% do treinamento e 0.1 depois. A loss total de treinamento e `L_main + L_MTP`.

### Por que sequencial, não paralelo

O MTP paralelo original de Gloeckle tinha D output heads, cada um diretamente aplicado a `h_i^(0)`. Cada head prediz `t_{i+k}` do mesmo hidden staté da backbone. Isso treina bem, mas as predições não são condicionadas umas nas outras. Você não pode usar a saida de `head_1` pra ajudar `head_2` -- os heads disparam em paralelo.

O design sequencial do DeepSeek-V3 constroi `h_i^(k)` a partir de `h_i^(k-1)` mais o embedding real do próximo token `E(t_{i+k})`. Isso preserva a cadeia causal: pra predizer `t_{i+k+1}`, o modulo na profundez `k+1` ve o que estava em `t_{i+k}`. Isso e estruturalmente identico a como um decoder autoregressivo consome sua propria saida -- tornando os modulos MTP diretamente usaveis como rascunhos de decodificação eespecificaçãoulativa.

Na inferência: alimente `h_i^(k-1)` e o rascunho `t_{i+k}` no modulo `k+1`, obtenha uma predição pra `t_{i+k+1}`. Repita. Isso e exatamente um rascunho estilo EAGLE, usando o modulo MTP treinado como rede de rascunho. DeepSeek-V3 reporta 80%+ de aceitação no primeiro modulo MTP e ~1.8x de speedup.

### Contabilidade de parâmetros

Pra um modelo com hidden `h` e vocabulario `V`:

- Modelo principal: bilhões de parâmetros, mais um output head de tamanho `V * h`.
- Output head compartilhado: reútiliza o head do modelo principal. Sem params extras.
- Embedding compartilhado: reútiliza o embedding do modelo principal. Sem params extras.
- Por modulo MTP:
  - Projeção `M_k`: `(2h) * h = 2h^2`.
  - Bloco transformer `T_k`: attention (`4h^2` pra MHA) mais MLP (tipicamente `8h^2` pra SwiGLU com razão 8/3). Cerca de `12h^2` por bloco.

Total extra por modulo: `~14h^2`. Pra `h = 7168` do DeepSeek-V3, D = 1 modulo: `~14 * 7168^2 = ~720M` parâmetros no papel. DeepSeek-V3 reporta 14B -- a diferença e majoritariamente layers de expert sendo MoE no modulo MTP também.

### O ganho da decodificação eespecificaçãoulativa

Durante pre-treinamento, os modulos MTP desaceleram o treinamento em cerca de 10% (mais compute no forward, loss extra). O ganho e duplo:

1. Sinais de treinamento mais densos. Cada hidden staté ve D+1 alvos de supervision. Efeito medido em MMLU, GSM8K, MATH, HumanEval: melhorias consistentes de poucos pontos percentuais nas abalações do DeepSeek-V3.

2. Rascunho gratis pra decodificação eespecificaçãoulativa na inferência. O modulo MTP ja esta treinado pra predizer os próximos poucos tokens. Reaproveitado como rede de rascunho, ele entrega taxas de aceitação de 80%+. Nesse nivel, N=3 ou N=5 de decodificação eespecificaçãoulativa da 1.8x de throughput. O custo de treinamento de 10% compensa na primeira vez que você roda inferência.

### Relação com EAGLE

EAGLE treina um modelo de rascunho pequeno SEPARADAMENTE após o pre-treinamento. MTP assa o rascunho dentro do pre-treinamento. As duas abordagens convergem em taxas de aceitação similares mas por pipelines diferentes:

| Dimensão | EAGLE-3 | MTP (DeepSeek-V3) |
|----------|---------|-------------------|
| Quando treinado | Pos-pre-treinamento | Durante pre-treinamento |
| Retrocompativel com pesos existentes | Sim | Não (precisa retreinar) |
| Params do rascunho | 1-2 camadas transformer | 1 bloco transformer + projeção |
| Taxa de aceitação | 0.88-0.92 | 0.80+ na profundez 1 |
| Beneficio além do speedup | So decodificação eespecificaçãoulativa | Sinais de treinamento mais densos + speedup |

## Construir

`code/main.py` constrói um modulo MTP unico de ponta a ponta: embedding compartilhado, projeção, bloco transformer, output head compartilhado. Depois calcula a loss de cross-entropy por profundez em uma sequencia sintetica curta e imprime a contagem de parâmetros por componente. Um vocabulario toy de 32 tokens mantem os numeros legiveis.

### Passo 1: tabela de embedding compartilhada

Uma unica tabela `vocab_size x hidden` e usada pelo modelo principal E por cada modulo MTP em cada profundez. Não e uma segunda copia -- literalmente o mesmo tensor.

### Passo 2: a combinação por profundez

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat na dim de features, depois projeta pra hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # substituto de adição vetorial
    projected = matvec(M_k, concat)
    return projected
```

O DeepSeek-V3 real concaténa os dois vetores RMSNormed em `[2h]` e projeta com uma matriz `h x 2h`. O toy usa adição vetorial pela brevidade do stdlib.

### Passo 3: o bloco transformer na profundez k

Self-attention mais MLP. No toy, um bloco de attention linear de uma camada e uma MLP SwiGLU mantem a estrutura visivel sem numpy.

### Passo 4: o output head compartilhado

Reútiliza a projeção de saida do modelo principal. Logits sobre o vocabulario.

### Passo 5: loss por profundez

Cross-entropy de softmax(logits) contra o token ground-truth no offset `k`. Agregue através das profundezes com o fator de escalamento `lambda / D`.

### Passo 6: contabilidade de parâmetros

Imprima a contagem total de parâmetros, a contagem compartilhada (embedding, head) e a contagem extra por modulo. Mostre a razão de extra MTP pro tamanho do modelo principal.

## Usar

MTP esta integrado no DeepSeek-V3 (dezembro de 2024) e na série DeepSeek-R1. Na inferência:

- A stack propria de servir da DeepSeek consome modulos MTP como decodificadores eespecificaçãoulativos nativamente.
- vLLM e SGLang tem caminhos de integração pro MTP do DeepSeek-V3 em abril de 2026.
- O tutorial ROCm da AMD mostra uma config eespecificaçãoifica de decodificação eespecificaçãoulativa MTP com speedup medido de 1.8x no checkpoint V3.

Quando usar MTP num novo run de pre-treinamento:

- Você controla o pipeline completo de pre-treinamento e quer bancar sinais de treinamento mais densos.
- Você sabe que vai servir o modelo em escala e quer decodificação eespecificaçãoulativa gratis.
- Seu hidden size e pelo menos 4096. Em escala 1B o overhead prejudica mais do que o ganho ajuda.

Quando não usar:

- Fine-tuning de um modelo denso pre-treinado existente. O modulo MTP não foi treinado.
- Modelos de pesquisa onde você quer um baseline limpo pra comparar. MTP muda a arquitetura.

## Entregar

Esta aula produz `outputs/skill-mtp-planner.md`. Dada uma eespecificaçãoificação de run de pre-treinamento (tamanho do modelo, dados, compute), retorna um plano pra integrar MTP: numero de profundezes D, schedule de `lambda`, overhead de memoria e a fiação de decodificação eespecificaçãoulativa na inferência.

## Exercicios

1. Rode `code/main.py`. Mostre que a loss por profundez diminui monotonicamente quando o sinal sintetico fortalece. Modifique o sintetico pra usar um padrão fixo e verifique que as losses de profundez 1 e profundez 2 convergem.

2. Calcule o overhead de parâmetros pra um modelo denso 70B (hidden 8192, 80 camadas) com D=1 modulo MTP. Compare com o overhead de 14B reportado pelo DeepSeek-V3. Explique por que o numero da DeepSeek e maior: o bloco transformer do MTP herda a mesma estrutura MoE, inflando a contagem de parâmetros por modulo.

3. Implemente D=2 no toy: adicione um segundo modulo MTP que recebe h^(1) e prediz `t_{i+2}`. Verifique que a loss conjunta e a contabilidade de parâmetros combinam com as equações 19-21 do paper da DeepSeek.

4. Mude o toy pra MTP paralelo (estilo Gloeckle): adicione D output heads no topo do hidden staté principal, cada um predizendo um offset diferente. Meça como as losses por profundez se comparam a versão sequencial no mesmo sinal sintetico. A versão sequencial deve produzir loss de profundez k menor pra k > 1 porque ela condiciona nas predições intermediarias.

5. Use o modulo MTP treinado como um rascunho estilo EAGLE: chame o modulo k pra propor `t_{i+k}` na inferência. Meça a taxa de aceitação desses tokens de rascunho contra as predições do modelo principal em uma sequencia de retenção. Se você chegar a 50%+ no toy, você reproduziu a propriedade empirica de MTP-como-rascunho.

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Modulo MTP | "Bloco de loss extra" | Um bloco transformer pequeno mais projeção que prediz um token `k` posições a frente do modelo principal |
| Profundez de predição | "Qual offset" | O inteiro `k` tal que o modulo `k` prediz `t_{i+k}` do prefixo até a posição `i` |
| MTP paralelo | "Estilo Gloeckle" | D heads independentes no mesmo hidden staté da backbone, sem cadeia condicional |
| MTP sequencial | "Estilo DeepSeek-V3" | Cada modulo condiciona no hidden staté da profundez anterior mais o embedding do próximo token; preserva cadeia causal |
| Output head compartilhado | "Reútilizar o head principal" | Os modulos MTP chamam o head LM do modelo principal, não uma projeção de saida separada |
| Embedding compartilhado | "Reútilizar a tabela principal" | A mesma tabela de embedding de vocabulario e usada em toda parte; sem parâmetros duplicados |
| Matriz de projeção M_k | "Combinar hidden + próximo-token" | Uma camada linear `h x 2h` que dobra o hidden staté anterior e o embedding do token alvo na entrada da proxima profundez |
| Loss conjunta L_MTP | "Losses extras media" | Media aritmetica das losses de cross-entropy por profundez, escalada por `lambda` |
| Taxa de aceitação na profundez 1 | "Quão seguido o rascunho MTP ta certo" | A taxa em que a predição top-1 do modulo MTP D=1 iguala a predição top-1 do modelo principal; 80%+ no DeepSeek-V3 |
| Ponderação lambda | "Importancia da loss extra" | Fator de escalamento por profundez; 0.3 no início do treinamento, 0.1 depois no DeepSeek-V3 |

## Leitura Complementar

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) -- a descrição completa do MTP sequencial (Seção 2.2), incluindo as equações de loss conjunta e o speedup de 1.8x na inferência
- [Gloeckle et al. -- Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) -- o baseline de MTP paralelo que o design da DeepSeek melhora
- [Model card do DeepSeek-V3 no Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) -- 685B total (671B principal + 14B MTP), notas de deploy
- [Leviathan et al. -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) -- o framework de decodificação eespecificaçãoulativa no qual MTP se encaixa
- [Li et al. -- EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) -- a arquitetura de rascunho 2025 do EAGLE, o concorrente que MTP disputa
