# Jamba -- Hibrido SSM-Transformer

> Modelos de espaco de estado (SSMs) e transformers querem coisas diferentes. Transformers compram qualidade via attention a custo quadratico. SSMs compram inferência em tempo linear e memoria constante via recorrencia mas perdem qualidade. Jamba da AI21 (marco de 2024) e Jamba 1.5 (agosto de 2024) colocam os dois no mesmo modelo: 1 camada Transformer pra cada 7 camadas Mamba, MoE em cada outro bloco e uma janela de contexto de 256k que cabe em uma GPU 80GB unica. Mamba-3 (ICLR 2026) aperta o lado do SSM com espacos de estado complexos e projeções MIMO. Esta aula le ambas as arquiteturas de ponta a ponta e explica por que a receita hibrida sobreviveu três anos de escalabilidade quando tentativas puras de SSM e puro Transformer de contexto longo não sobreviveram.

**Tipo:** Aprender
**Linguagens:** Python (stdlib, calculadora de mix de camadas)
**Pré-requisitos:** Fase 10 · 14 (arquiteturas de modelos abertos), Fase 10 · 17 (native sparse attention)
**Tempo:** ~60 minutos

## Objetivos de Aprendizado

- Explicar as três primitivas de um bloco Jamba -- camadas Transformer, camadas Mamba, MoE -- e a receita de intercalação 1:7:par.
- Enunciar como uma recorrencia de SSM parece em alto nivel e por que ela viabiliza inferência de memoria constante.
- Calcular o footprint do KV cache de um modelo Jamba em contexto 256k e comparar com o que um modelo puro-Transformer precisaria.
- Nomear as três inovações do Mamba-3 (discretização trapezoidal exponencial, atualização de estado complexa, MIMO) e o que cada uma foca.

## O Problema

Attention e quadratico no comprimento da sequencia. Modelos de espaco de estado são lineares. Essa diferença se acumula: em 256k tokens, um mapa de attention de Transformer e 65B entradas por head; o estado recorrente de um SSM e de tamanho fixo independente do comprimento da sequencia.

Modelos puramente SSM (Mamba, Mamba-2) combinam perplexidade de Transformer em escalas pequenas mas perdem em tarefas de rastreamento de estado e falham em catégorias de recuperação em contexto. A intuicao: SSMs comprimem historico em um estado fixo, e quando o historico e longo, informação vaza. Attention lembra tudo exatamente mas paga custo quadratico.

A correção obvia: use os dois. Coloque camadas Transformer onde recall exato importa. Use camadas SSM em outro lugar. Ajuste a razão. Jamba e o primeiro modelo de produção a enviar essa receita hibrida em escala (52B total, 12B ativos, 256k contexto, GPU 80GB unica). Jamba 1.5 estende a familia pra 398B total / 94B ativos. Mamba-3 (ICLR 2026) e o melhor baseline puramente SSM atual que hibridos podem ser reconstruidos ão redor.

Esta aula le os três papers e produz o modelo mental pra "escolher a razão certa."

## O Conceito

### Um SSM em uma pagina

Um modelo de espaco de estado processa uma sequencia `x_1, ..., x_N` via um estado `h` de tamanho fixo:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

A cada passo o estado evolui via uma dinâmica linear `A`, recebe a entrada `B x_t` e emite saida `C h_t`. `A, B, C` podem ser aprendidos. Note a propriedade critica: calcular `y_t` precisa apenas de `h_{t-1}` e `x_t`, não de nenhum `x` anterior. Memoria e constante. Inferência e O(1) por token.

O truque pra qualidade de modelagem e a estrutura de `A`. S4 (Gu 2021) usava uma matriz altamente estruturada que podia ser avaliada eficientemente como uma convolução longa durante treinamento. Mamba (Gu, Dão 2023) substituiu os `A, B, C` fixos por dependentes dos dados (a parte "seletiva"). Mamba-2 (2024) simplificou mais a estrutura. Mamba-3 (2026) readiciona complexidade em lugares eespecificaçãoificos.

A propriedade chave: pra um LLM decoder, uma camada SSM e um substituto direto pra uma camada de attention, com estado de tamanho fixo por camada ão inves de um KV cache que cresce.

### O bloco Jamba

Um bloco Jamba alterna camadas de acordo com dois numeros:

- `l`: a razão attention-Mamba. Jamba usa `l = 8`, ou seja, 1 camada Transformer pra cada 7 camadas Mamba (7 Mamba + 1 Attention = 8 camadas por grupo).
- `e`: a frequencia de MoE. Jamba usa `e = 2`, ou seja, cada outra camada aplica MoE.

A sequencia de camadas dentro de um bloco:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (onde | marca MoE aplicado)
```

Cada bloco Jamba e 8 camadas. Em 4 blocos de profundidade (32 camadas no total), você tem 28 Mamba e 4 Attention. 16 dessas usam MoE.

### Por que a razão 1:7

AI21 rodou abalações: que razão de attention-Mamba da melhor perplexidade-por-parâmetro E recall em contexto nas avaliações de contexto longo deles?

- Muita attention (1:1): qualidade sobe mas memoria e velocidade degradam.
- Pouca attention (1:15): memoria e otima mas recuperação em contexto falha.
- Ponto ideal: 1:7 ou 1:8.

A intuicao: as camadas Transformer lidam com recall exato e rastreamento de estado. As camadas Mamba lidam com o bulk barato do processamento.

### Encoding posicional

Camadas Mamba são elas mesmas consientes de posição (via recorrencia). Camadas de attention nos hibridos originais baseados em Mamba não usavam RoPE -- as camadas SSM forneciam a info de posição. Jamba 1.5 adiciona RoPE as camadas de attention pra generalização em contexto longo, um refinamento posterior baseado em avaliação empirica de contexto longo.

### O orcamento de memoria

Pra um formato Jamba-1 (32 camadas: 28 Mamba + 4 Attention, hidden 4096, 32 heads de attention):

- KV cache (so camadas de attention): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` em 256k BF16. So as 4 camadas de attention contribuem.
- Estado SSM: `28 * hidden * staté_size` por prefixo de token, mas isso e de tamanho fixo por camada, não escala com o comprimento da sequencia. Estado tipico Mamba de 16 por feature, hidden 4096: `28 * 4096 * 16 * 2 = 3.7 MB` total.

Compare com um Transformer puro de 32 camadas, mesmo hidden, MHA completo em 32 heads: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` em 256k BF16. Uma redução de 8x no KV cache. Mesmo contra o baseline GQA(8) que a maioria dos modelos de 2024 usa (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), o hibrido 1:7 de Jamba com 16 GB ainda e 2x menor.

E isso que AI21 quer dizer com "256k contexto em uma GPU 80GB unica." O KV cache de um Transformer puro de MHA completo não caberia; até um baseline GQA não deixa espaco pra pesos e ativações; o de Jamba deixa.

### Mamba-3: o baseline puramente SSM em 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) introduz três inovações no lado puramente SSM:

1. **Discretização trapezoidal exponencial.** Substitui a discretização pelo metodo de Euler no Mamba-2 por uma recorrencia mais expressiva. Operação similar a convolução aplicada no estado-entrada dentro da recorrencia central, ão inves de uma convolução externa em `x_t`.

2. **Atualização de estado complexa.** Mambas anteriores reduziram a matriz de estado de complexa (S4) pra real diagonal (Mamba) pra identidade escalada (Mamba-2). Mamba-3 readiciona valores complexos -- equivalente a um embedding rotacional dependente dos dados no estado. Isso restaura capacidades de rastreamento de estado que simplificações reais anteriores custaram.

3. **Projeções multi-entrada multi-saida (MIMO).** Ao inves de projeções escalares por feature, usar projeções matriciais. Melhora poder de modelagem e útilização de hardware na inferência sem aumentar laténcia de decode.

Em 1.5B de parâmetros, Mamba-3 melhora a acuracia downstream media em 0.6 pontos sobre Gatéd DeltaNet; a variante MIMO adiciona mais 1.2 pontos pra um total de 1.8 pontos de ganho. No mesmo tamanho de estado, Mamba-3 combina Mamba-2 com metade do estado.

Mamba-3 ainda não esta sendo enviado em um hibrido de produção em escala -- mas e o candidato obvio pro lado SSM do próximo modelo classe Jamba.

### Quando usar um hibrido

Hibridos ganham quando:

- Contexto e longo o suficiente que o KV cache de Transformer puro vira doloroso (64k+).
- Tarefas misturam estrutura de curto alcance (bom pra SSM) com recall de longo alcance (precisa de Transformer).
- Você quer implantação em orcamentos de memoria de GPU unica onde so o KV cache de Transformer não caberia.

Hibridos perdem quando:

- Contexto e curto (abaixo de 16k). O overhead de SSM e desperdicado; Transformer puro e fine.
- Tarefas precisam de attention em todo-lugar-pra-todo-lugar (raciocinio profundo, referencia cruzada multi-documento). A esparsidade das camadas de attention no hibrido prejudica.
- Você esta escalando pra modelos de fronteira de trilhões de parâmetros. Transformer puro + MLA + MoE (estilo DeepSeek-V3) atualmente ganha a corrida de capacidade.

### O cenario competitivo

| Modelo | Familia | Escala | Alegação unica |
|--------|---------|--------|---------------|
| Mamba-2 | puro SSM | 3B | tempo linear, memoria constante |
| Jamba | hibrido | 52B/12B | 256k em 80GB |
| Jamba 1.5 Large | hibrido | 398B/94B | contexto longo de nivel enterprise |
| Mamba-3 | puro SSM | 1.5B (paper) | rastreamento de estado restaurado |
| DeepSeek-V3 | Transformer puro + MoE | 671B/37B | capacidade de fronteira |

O cenario em 2026: Transformer puro MoE domina a fronteira, mas hibridos possuem o nicho de contexto 256k+. Os ganhos de rastreamento de estado do Mamba-3 podem empurrar razões de hibrido pra baixo (mais SSM, menos attention) na proxima geração.

## Usar

`code/main.py` e uma calculadora de memoria pra arquiteturas hibridas. Dada uma razão SSM-Transformer e uma config de hidden-size / contagem-de-camadas, calcula:

- KV cache no contexto alvo.
- Memoria de estado SSM.
- Memoria total no contexto N para uma variedade de formatos de modelo.

A calculadora suporta:

- Baseline de Transformer puro (KV cache cresce com N).
- Hibrido estilo Jamba 1:7.
- SSM puro (sem KV cache nenhum).

Os numeros são direto dos papers Jamba-1 e Jamba-1.5 pra formatos publicados e extrapolados pra variantes hipotheticas.

Considerações de integração pra implantação real:

- A maioria dos servidores de inferência de produção (vLLM, SGLang) suporta Jamba e Mamba. Verifique a versão eespecificaçãoifica.
- Em contexto 256k, a vantagem de memoria do Jamba aparece no throughput de requests concorrentes. Na mesma VRAM você caixa mais sequencias de Jamba que de Transformer.
- Mamba-3 como modelo standalone ainda não esta sendo enviado em produção -- preview de pesquisa em 1.5B.

## Entregar

Esta aula produz `outputs/skill-hybrid-picker.md`. Dada uma eespecificaçãoificação de carga de trabalho (perfil de contexto, mix de tarefas, orcamento de memoria), recomenda entre um Transformer puro, um hibrido estilo Jamba e um SSM puro, com raciocinio explicito sobre os tradeoffs de memoria e qualidade.

## Exercicios

1. Rode `code/main.py` pra calcular o KV cache em contexto 256k pra um Transformer puro de 32 camadas (hidden 4096, 32 heads) e pra um hibrido Jamba-1 do mesmo formato. Verifique a redução de ~8x de memoria que o paper da AI21 alega.

2. Modifique a calculadora pra modelar um hibrido 1:3 (4 Mamba : 1 Attention) e um hibrido 1:15 (14 Mamba : 1 Attention). Plote KV cache vs razão. Em que razão o KV cache iguala a memoria de estado SSM?

3. Leia a Seção 3 do paper do Jamba (arXiv:2403.19887). Explique por que AI21 usa Mamba-1 ão inves de Mamba-2 apesar de Mamba-2 ser mais rapido. Dica: a seção de abalação hibrida documenta isso.

4. Calcule o overhead de parâmetros do MoE em cada outra camada no Jamba 1.5 Large (398B total, 94B ativos). Compare a razão ativa com o DeepSeek-V3 (37B/671B) e explique por que a arquitetura do Jamba empurra a razão ativa pra cima.

5. Leia a Seção 3 do paper do Mamba-3 (arXiv:2603.15569). Explique em três frases por que uma atualização de estado complexa equivale a um embedding rotacional dependente dos dados. Ligue a resposta a derivação do RoPE da Aula 04 da Fase 7.

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Modelo de espaco de estado (SSM) | "Recorrencia com estado fixo" | Uma camada com recorrencia aprendida `h_t = A h_{t-1} + B x_t`; memoria constante por token |
| SSM seletivo | "O truque do Mamba" | Parâmetros A, B, C dependentes dos dados que dão ão modelo seletividade tipo gaté em tempo linear |
| Razão attention-Mamba | "Quantas camadas de attention" | No Jamba, `l = 8` significa 1 camada de attention pra cada 7 Mamba |
| Bloco Jamba | "O grupo de 8 camadas" | Um attention + sete Mamba + MoE em posições alternadas |
| Estado SSM | "O buffer oculto" | Estado de tamanho fixo por camada que substitui o KV cache nas camadas Mamba |
| Contexto 256k | "O numero principal do Jamba" | O comprimento de sequencia que Jamba-1 cabe em uma GPU 80GB unica; Transformer puro não consegue nesse tamanho |
| Mamba-3 | "SSM puro 2026" | A melhor arquitetura SSM atual com estado complexo + MIMO; o baseline que hibridos reconstruem ão redor |
| MIMO | "Multi-entrada multi-saida" | Inovação do Mamba-3 usando projeções matriciais ão inves de escalares por funcionalidade |
| Discretização trapezoidal exponencial | "A recorrencia do Mamba-3" | Recorrencia mais expressiva que subsume a discretização pelo metodo de Euler do Mamba-2 |
| Arquitetura hibrida | "Misturar attention e SSM" | Qualquer modelo que alterna camadas Transformer e SSM; Jamba e o arquetipo de produção |

## Leitura Complementar

- [Lieber et al. -- Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) -- o paper original do Jamba, abalações de razão, alegação de contexto 256k
- [AI21 -- Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) -- a familia escalada, releases publicos de 398B/94B e 12B/52B
- [Gu, Dão -- Mamba: Linear-Time Sequence Modeling with Selective Staté Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) -- o paper do SSM seletivo que Jamba se baseia
- [Dão, Gu -- Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) -- o sucessor estruturado-estado-espaco simplificado
- [Lahoti et al. -- Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) -- estado complexo, MIMO, a fronteira SSM pura 2026
- [Gu et al. -- Efficiently Modeling Long Sequences with Structured Staté Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) -- o paper S4, o ponto de partida da genealogia SSM pra LLMs
