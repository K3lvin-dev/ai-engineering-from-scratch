# DualPipe Parallelism

> DeepSeek-V3 foi treinado em 2.048 GPUs H800 com experts MoE espalhados entre nodes. A comunicação all-to-all entre nodes de experts custou 1 GPU-hora de comunicação pra cada 1 GPU-hora de compute. GPUs ficaram ociosas metade do tempo. DualPipe (DeepSeek, dez 2024) e um pipeline bidirecional que sobrepoe computação forward e backward com as comunicações all-to-all que elas disparam. Bolhas caem, throughput sobe e a manutenção de duas copias de parâmetros do modelo (o "dual" que da o nome) e barata uma vez que Expert Parallelism ja esta espalhando experts entre ranks de qualquer forma. Esta aula e um percurso do tipo Aprender sobre o que DualPipe realmente faz e por que o refinamento DualPipeV do Sea AI Lab reduz o custo 2x de parâmetros a expensa de uma bolha um pouco mais apertada.

**Tipo:** Aprender
**Linguagens:** Python (stdlib, simulador de schedule)
**Pré-requisitos:** Fase 10 · 05 (treinamento distribuido, FSDP, DeepSpeed), Fase 10 · 14 (arquiteturas de modelos abertos e MoE)
**Tempo:** ~60 minutos

## Objetivos de Aprendizado

- Nomear os quatro componentes de um chunk forward-backward do DualPipe e por que cada um tem sua propria janela de sobreposição.
- Explicar o problema de bolhas de pipeline em escala e o que "sem bolhas" significa na pratica vs. marketing.
- Rastrear um schedule do DualPipe na mão pra 8 ranks de PP e 16 micro-batches e confirmar que os fluxos forward e reverso preenchem os slots ociosos um do outro.
- Enunciar o tradeoff que o DualPipeV (Sea AI Lab, 2025) faz: reduz a replicação 2x de parâmetros a expensa de uma bolha levemente maior quando Expert Parallelism não esta ativo.

## O Problema

Treinar um modelo MoE de 671B em 2.000 GPUs H800 esbarra em três gargalos compostos:

1. **Pressão de memoria.** Cada GPU guarda um fatia do modelo. Memoria de ativação em sequencia 8k em 61 camadas em 128 heads e enorme.
2. **Bolhas de pipeline.** Pipeline parallelism tradicional (GPipe, 1F1B) deixa GPUs ociosas enquanto esperam pela entrada ou gradiente do seu estagio. Em 8 estagios, cerca de 12% do tempo de GPU pode ser bolha mesmo com scheduling 1F1B.
3. **All-to-all entre nodes.** MoE com expert parallelism espalha experts entre nodes. Cada forward pass dispara um all-to-all pra despachar tokens pros seus experts, e outro pra combinar. Em 2.000 GPUs isso facilmente vira uma razão de compute-comunicação de 1:1.

Cada um desses tem soluções separadas: gradient checkpointing pra memoria, Zero Bubble (Sea AI Lab, 2023) pra bolhas de pipeline, kernels de comunicação expert-parallel pro all-to-all. O que o DualPipe faz e faz-los jogar junto. O schedule sobrepoe compute e comunicação dentro de um unico chunk forward-backward, injeta micro-batches de ambas as pontas do pipeline simultaneamente e usa o schedule resultante pra esconder all-to-all dentro das janelas de compute.

Resultado reportado: eliminação quase total de bolhas de pipeline, mais de 95% de útilização de GPU no run de treinamento de 14.8T tokens do DeepSeek-V3.

## O Conceito

### Recapitulação de pipeline parallelism

Divida um modelo N-camadas em P dispositivos. O dispositivo `i` guarda as camadas `i * N/P .. (i+1) * N/P - 1`. Um micro-batch flui forward pelos dispositivos 0 até P-1, depois backward de P-1 até 0. Cada dispositivo so pode comecar seu estagio forward quando o dispositivo anterior envia sua saida, e so pode comecar backward quando o dispositivo downstream envia o gradiente upstream.

GPipe (Huang et al., 2019) agenda um micro-batch por vez, o que desperdicaria a maior parte do tempo de GPU. 1F1B (Narayanan et al., 2021) alterna passes forward e backward pra multiplos micro-batches. Zero Bubble (Qi et al., 2023) divide o passo backward em duas partes -- backward-pra-input (B) e backward-pra-pesos (W) -- e agenda pra preencher a bolha. Após Zero Bubble, o pipeline quase fica apertado.

DualPipe e o próximo passo. Ele adiciona duas ideias em cima:

### Ideia 1: decomposição em chunks

Cada chunk forward e dividido em quatro componentes:

- **Attention.** Projeções Q/K/V, attention, projeção de saida.
- **All-to-all despacho.** Comunicação entre nodes que envia tokens pros seus experts.
- **MLP.** Computação do expert MoE.
- **All-to-all combinação.** Comunicação entre nodes que traz as saidas dos experts de volta.

Um chunk backward adiciona versões de gradiente de cada um desses. DualPipe agenda para que o all-to-all de despacho aconteca em paralelo com a computação de attention do próximo chunk, e o all-to-all de combinação aconteca em paralelo com a computação do MLP do chunk seguinte.

### Ideia 2: scheduling bidirecional

A maioria dos schedules de pipeline injeta micro-batches do estagio 0 e flui até o estagio P-1. DualPipe injeta micro-batches de AMBAS as pontas. Estagio 0 ve micro-batches forward que se originam ali; estagio P-1 também ve micro-batches forward que se originam ali. Os dois fluxos se encontram no meio.

Pra isso funcionar, o dispositivo `i` precisa guardar AMBOS a camada `i` do início do pipeline E a camada `P - 1 - i` do fim do pipeline. Essa e a parte "dual" do DualPipe: cada dispositivo mantem duas copias das camadas do modelo que precisa servir (uma pra cada direcao). Na escala do DeepSeek-V3, isso e um custo de replicação de 2x de parâmetros. E acessavel porque Expert Parallelism ja espalha os experts MoE tão finos que replicar as camadas não-expert duas vezes e detalhe.

De forma crucial, o fluxo forward em uma direção e o fluxo backward na outra se sobrepem exatamente onde as bolhas estariam em um schedule de direção unica. As bolhas somem.

### Um schedule rastreado na mão

Considere P = 4 ranks, 8 micro-batches, divididos 4 forward / 4 reverso. Tempo vai da esquerda pra direita; linhas são os ranks dos dispositivos.

```
           Tempo →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Lendo a notação "F4/F5R": rank 1 esta rodando forward do micro-batch 4 (indo da esquerda pra direita no pipeline) E forward do micro-batch 5 (indo da direita pra esquerda) no mesmo slot de tempo. E o que "bidirecional" significa operacionalmente.

No rank 2 os fluxos cruzados se sobrepem mais cedo, no rank 0 e P-1 se sobrepem mais tarde. Na fase estavel do schedule, cada rank roda forward-de-X-direção sobreposto com backward-de-Y-direção. Compute ta ocupado. Despachos all-to-all do forward pass escondem dentro do compute backward. Combinações all-to-all escondem dentro do compute forward. As bolhas são comprimidas.

### Contabilidade de bolha

Bolha padrão de pipeline 1F1B (tempo desperdicado por rank):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

A refinação do Zero Bubble reduz mas não pra zero. DualPipe, na fase estavel, tem bolha zero se a contagem de micro-batches for divisivel por 2 vezes a profundidade do pipeline. Fora da fase estavel (aquecimento e resfriamento), tem alguma bolha mas ela não cresce com o numero de micro-batches -- uma propriedade chave que o paper destaca.

Em linguagem de marketing: "sem bolhas". Em termos tecnicos: bolhas não crescem com a contagem de micro-batches. A análise posterior do Sea AI Lab (DualPipeV / Cut-in-half) mostra a bolha zero completa so quando Expert Parallelism não e o gargalo; com all-to-all guiado por EP, algum compromisso de scheduling sempre esta presente.

### DualPipeV -- o refinamento

Sea AI Lab (2025) observou que a replicação 2x de parâmetros e desperdicio quando o overlap de comunicação EP não e o ponto. O schedule DualPipeV dobra a injeção bidirecional num schedule "em V" que roda em uma unica copia de parâmetros. A bolha e levemente maior que a do DualPipe, mas a economia de memoria e substancial. DeepSeek adotou DualPipeV na implementação open-source DualPipe como modo EP-off.

O tradeoff:

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Copias de params por dispositivo | 2 | 1 | 1 | 1 |
| Bolha vs micro-batches | constante | crescimento pequeno | cresce | cresce |
| Sobreposição compute-comunicação | total | parcial | mínima | parcial |
| Usar quando | MoE pesado em EP | denso ou leve em EP | baseline | qualquer pipeline |

### O que isso significa pra um run de 14.8T tokens

O pre-treinamento do DeepSeek-V3 consumiu 14.8T tokens em 2.048 GPUs H800 em cerca de 2.8M GPU-horas. Com 1F1B ingenuo, eles teriam perdido 12-15% disso pra bolhas de pipeline -- 340-420K GPU-horas, suficiente pra treinar um modelo 70B inteiro. DualPipe recuperou a maior parte. Quantificar a contribuição diretamente e dificil sem os logs internos, mas a alegação no paper e mais de 95% de útilização de GPU media ão longo do treinamento.

Pra runs menores (menos de 1k GPUs), DualPipe e excessivo -- bolhas de pipeline são menores em relação ão custo total e treinamento de modelos densos raramente atinge o gargalo de all-to-all. Pra treinamento MoE de fronteira em escala de milhares de GPUs, e praticamente obrigatorio.

### Onde se posiciona na stack

- Complementar ão **FSDP** (Fase 10 · 05). FSDP fragmenta os parâmetros do modelo entre ranks; DualPipe agenda o compute entre ranks. Eles combinam.
- Compativel com o fracionamento de gradientes **ZeRO-3**. A contabilidade da replicação de duas copias precisa cooperar com os gradientes fracionados do ZeRO.
- Requer **kernels all-to-all custom** ajustados pra topologia eespecificaçãoifica do cluster. Os kernels open-source da DeepSeek são a implementação de referencia.

## Usar

`code/main.py` e um simulador de schedule de pipeline. Ele recebe `(P, n_micro_batches, schedule)` e imprime a útilização na fase estavel pra cada um de 1F1B, Zero Bubble, DualPipe e DualPipeV. E uma ferramenta didatica -- os numeros combinam com as alegações qualitativas nos papers, não são alegação de speedup medido em produção.

O valor do simulador: rode com diferentes P e contagens de micro-batches e veja como a fração de bolha cresce pra 1F1B mas não pra DualPipe.

Considerações de integração pra um run de treinamento real:

- Escolha uma profundidade de pipeline parallelism que divida limpo na sua contagem de micro-batches.
- Garanta que sua malha de expert parallelism suporte all-to-all bidirecional. Os kernels da DeepSeek são referencia.
- Espere gastar uma semana de debug no schedule em si na primeira vez. A contabilidade e trabalhosa.
- Monitore útilização de GPU por rank, não so agrega. O beneficio do DualPipe vem de apertar os atrasados.

## Entregar

Esta aula produz `outputs/skill-dualpipe-planner.md`. Dada uma eespecificaçãoificação de cluster de treinamento (contagem de GPUs, topologia, interconnect, formato do modelo), recomenda uma estratégia de pipeline parallelism, o algoritmo de scheduling e a fração de bolha esperada na escala alvo.

## Exercicios

1. Rode `code/main.py` em `(P=8, micro_batches=16, schedule=dualpipe)` e `(P=8, micro_batches=16, schedule=1f1b)`. Calcule a diferença de útilização de GPU e expresse como GPU-horas recuperadas por milhão de tokens de treinamento.

2. Esboce a tabela do schedule pra `(P=4, micro_batches=8, schedule=dualpipe)` na mão. Marque cada slot de tempo com o ID do micro-batch e direção. Identifique o primeiro slot de tempo onde bolhas estão ausentes.

3. Leia a Figura 5 do relatorio tecnico do DeepSeek-V3 (arXiv:2412.19437). Identifique a janela de sobreposição do all-to-all de despacho dentro de um chunk forward do DualPipe. Explique como o schedule de compute esconde isso.

4. Calcule o overhead de 2x de parâmetros do DualPipe pra um modelo denso 70B com P=8 estagios de pipeline e um modelo MoE 671B com P=16 estagios de pipeline. Mostre por que o overhead do caso MoE e proporcionalmente menor (a maioria dos parâmetros são experts, fragmentados em um grande grupo EP).

5. Compare o DualPipe com o Chimera (um agendador bidirecional concorrente de 2021). Identifique as duas propriedades eespecificaçãoificas que o DualPipe adicionou que o Chimera não tinha, usando a Seção 3.4 do paper como referencia.

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Bolha de pipeline | "Tempo ocioso por rank" | Ciclos de GPU desperdicados porque um estagio de pipeline esta esperando sua entrada ou gradiente |
| 1F1B | "Schedule padrão de pipeline" | Um forward / um backward interleaved; o baseline que DualPipe supera |
| Zero Bubble | "Sea AI Lab 2023" | Divide backward em B (gradiente de input) e W (gradiente de peso); quase aperta o pipeline |
| DualPipe | "Schedule do DeepSeek-V3" | Pipeline bidirecional + sobreposição compute-comunicação; bolhas não crescem com contagem de micro-batches |
| DualPipeV | "Cut-in-half" | Refinamento em V que reduz a replicação 2x de parâmetros a expensa de bolhas levemente maiores |
| Chunk | "Unidade de trabalho do pipeline" | Um forward ou backward de um micro-batch através de um estagio de pipeline |
| All-to-all despacho | "Enviar tokens pros experts" | Comunicação entre nodes que roteia tokens pros seus experts MoE designados |
| All-to-all combinação | "Trazer saidas dos experts de volta" | Comunicação entre nodes que coleta as saidas dos experts após o MLP |
| Expert Parallelism (EP) | "Experts entre GPUs" | Fragmenta experts MoE entre ranks pra diferentes GPUs guardarem diferentes experts |
| Pipeline Parallelism (PP) | "Camadas entre GPUs" | Fragmenta camadas do modelo entre ranks; a dimensão que DualPipe agenda |
| Fração de bolha | "Tempo de GPU desperdicado" | (bubble_time / total_time); a fração que DualPipe empurra pra zero |

## Leitura Complementar

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437), Seção 3.3.2 e Figura 5](https://arxiv.org/abs/2412.19437) -- a referencia principal do DualPipe
- [DualPipe GitHub repository da DeepSeek](https://github.com/deepseek-ai/DualPipe) -- a implementação open-source de referencia, incluindo o modo DualPipeV (Cut-in-half)
- [Qi et al. -- Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) -- o predecessor Zero Bubble
- [Sea AI Lab -- DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) -- a análise do DualPipeV que informou o modo EP-off da DeepSeek
- [Narayanan et al. -- PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) -- o schedule 1F1B que DualPipe compara
- [Huang et al. -- GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) -- o paper original de pipeline parallelism e o problema de bolhas
