# Modelagem Autoregressiva Visual (VAR): Previsão de Próxima Escala

> Modelos de difusão amostram iterativamente no tempo (passos de denoising). VAR amostra iterativamente em escala — prevê um token 1x1, depois 2x2, depois 4x4, até a resolução final, cada escala condicionando na anterior. O paper de 2024 mostrou que VAR corresponde às leis de escala estilo GPT para geração de imagens e supera DiT no mesmo orçamento computacional. Esta aula constrói o mecanismo central.

**Tipo:** Construir
**Linguagens:** Python (com PyTorch)
**Pré-requisitos:** Fase 7 Aula 03 (Multi-Head Attention), Fase 8 Aula 06 (DDPM)
**Tempo:** ~90 minutos

## O Problema

Geração autoregressiva dominou modelagem de linguagem porque escala de forma previsível: mais computação, mais parâmetros, menor perplexidade, melhores saídas. Geração de imagens teve duas tentativas AR principais antes de 2024: PixelRNN/PixelCNN (pixel por pixel) e DALL-E 1 / Parti / MuseGAN (token por token em códigos VQ-VAE).

Ambas sofreram de um problema de ordem de geração. Pixels e tokens estão arranjados em um grid 2D, mas o modelo AR precisa visitá-los em uma ordem raster 1D. Um pixel de cedo não sabe o que a imagem eventualmente se torna. A qualidade de geração escalou pior que GPT em texto e nunca alcançou a qualidade de modelo de difusão em computação correspondente.

VAR corrige o problema de ordem de geração mudando o que está sendo gerado. Em vez de prever tokens de imagem um por um no espaço, VAR prevê uma imagem inteira em resoluções crescentes. Passo 1: prevê um token 1x1 (o "resumo" da imagem). Passo 2: prevê um grid 2x2 de tokens (features mais grosseiras). Passo 3: prevê um grid 4x4. Passo K: prevê o grid final (H/8)x(W/8).

Cada escala attende a todas as escalas anteriores (causalmente na "ordem de escala") e em paralelo dentro de sua própria escala. O problema de ordem desaparece: a imagem inteira na escala k é produzida em uma única passada de transformer.

## O Conceito

### Tokenizador Multi-Escala VQ-VAE

VAR precisa de um **tokenizador discreto multi-escala**. Para uma imagem x, ele produz uma sequência de grids de tokens de resolução progressivamente crescente:

```
x -> encoder -> latente f
f -> tokenizar em 1x1: grid de tokens z_1 de forma (1, 1)
f -> tokenizar em 2x2: grid de tokens z_2 de forma (2, 2)
...
f -> tokenizar em (H/p)x(W/p): grid de tokens z_K de forma (H/p, W/p)
```

Cada z_k usa o mesmo codebook (tamanho típico 4096-16384). A tokenização em cada escala não é independente — é treinada para que somar os resíduos em cada escala reconstrua f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Esta é uma variante de **VQ residual**. A escala k captura o que as escalas 1..k-1 perderam. O decoder pega a soma de todos os embeddings de escala e produz a imagem.

O tokenizador VQ multi-escala é treinado uma vez (como VQGAN) e depois congelado. Todo o trabalho gerativo é feito pelo modelo autoregressivo por cima.

### Previsão de Próxima Escala

O modelo gerativo é um transformer que vê tokens de todas as escalas anteriores e prevê os tokens da próxima escala.

Estrutura da sequência de entrada:
```
[START, tokens z_1, tokens z_2, tokens z_3, ..., tokens z_K]
```

As embeddings posicionais codificam tanto o índice da escala quanto a posição espacial dentro da escala. A attention é causal na ordem de escala: um token na escala k, posição (i, j), pode attendere a todos os tokens nas escalas 1..k e a tokens na própria escala k que vêm antes na ordem intra-escala usada (VAR usa attention posicional fixa sem causalidade intra-escala — todas as posições dentro de uma escala são previstas em paralelo).

Perda de treino: em cada escala k, prevê os tokens z_k dados todos os tokens de escalas anteriores. Perda de entropia cruzada nos códigos VQ discretos. Mesma estrutura que GPT exceto que a "sequência" agora é estruturada por escala.

### Geração

Em inferência:
```
gerar z_1 = amostrar de p(z_1)                    # 1 token
gerar z_2 = amostrar de p(z_2 | z_1)              # 4 tokens em paralelo
gerar z_3 = amostrar de p(z_3 | z_1, z_2)         # 16 tokens em paralelo
...
decodificar: f = soma de embed-e-upsample escalas 1..K
imagem = VAE_decoder(f)
```

Para K = 10 escalas, a geração são 10 passadas forward de transformer. Cada passada produz sua escala inteira em paralelo — sem autoregressão por token dentro da escala. Para uma imagem 256x256 isso são aproximadamente 10 passadas vs 28-50 do DiT.

### Por que Próxima Escala Vence Próximo-Token

Três vitórias estruturais:
1. **Grosseiro-fino alinha com estatísticas de imagem naturais.** Percepção visual humana e datasets de imagem ambos exibem regularidades dependentes de escala: estrutura de baixa frequência é estável e previsível; detalhe de alta frequência é condicional ao conteúdo de baixa frequência. Previsão de próxima escala explora isso.
2. **Geração paralela dentro da escala.** Diferente de AR por tokens estilo GPT, VAR produz todos os tokens de uma escala em um passo. Comprimento efetivo de geração é log-escalar em vez de linear.
3. **Sem viés de ordem de geração.** Tokens na escala k veem toda a escala k-1; não há viés "à esquerda" ou "acima" que force tokens cedo a se comprometerem antes que contexto tardio esteja disponível.

### Lei de Escala

Tian et al. demonstraram que VAR segue uma curva de escala por lei de potência para FID no ImageNet — assim como GPT faz para perplexidade. Dobrar parâmetros ou computação reduz o erro pela metade de forma confiável. Este foi o primeiro modelo gerativo de imagem a exibir esse comportamento de escala de forma tão limpa quanto modelos de linguagem. O resultado é que previsões de escala do VAR tornam-se previsíveis a partir da computação, não palpites empíricos por arquitetura.

### Relação com Difusão

VAR e difusão compartilham a mesma história de compressão de dados: ambos quebram o problema de geração em uma sequência de subproblemas mais fáceis.

- Difusão: adicione ruído gradualmente, aprenda a desfazer um passo.
- VAR: adicione resolução gradualmente, aprenda a prever a próxima escala.

São eixos diferentes através do problema. Ambos produzem distribuições condicionais tratáveis. Empiricamente VAR é mais rápido em inferência (menos passadas, todas paralelas dentro de uma escala) e corresponde ou supera DiT em ImageNet condicional por classe. VAR condicional por texto (VARclip, HART) é uma direção de pesquisa ativa.

## Construa

Em `code/main.py` você vai:
1. Construir um **tokenizador VQ multi-escala** minúsculo em dados sintéticos de "imagem" (anéis Gaussianos 2D).
2. Treinar um **transformer estilo VAR** para prever a próxima escala dos tokens.
3. Amostrar chamando o transformer 4 vezes (4 escalas) e decodificando.
4. Verificar que o treino ordenado por escala torna a geração paralela dentro de uma escala.

Esta é uma implementação de brinquedo. O ponto é ver a máscara de attention estruturada por escala e a geração paralela-dentro-de-escala funcionando.

## Entregue

Esta aula produz `outputs/skill-var-tokenizer-designer.md` — uma skill para projetar um tokenizador multi-escala: número de escalas, proporções de escala, tamanho do codebook, compartilhamento residual, arquitetura do decoder.

## Exercícios

1. **Ablação de contagem de escalas.** Treine VAR com 4, 6, 8, 10 escalas. Meça qualidade de reconstrução vs número de passadas autoregressivas. Mais escalas = resíduos mais finos = melhor qualidade mas mais passadas.
2. **Tamanho do codebook.** Treine tokenizadores com tamanhos de codebook 512, 4096, 16384. Codebooks maiores dão melhor reconstrução mas previsão mais difícil. Encontre o ponto de inflexão.
3. **Verificação de paralelismo dentro da escala.** Para um VAR treinado, meça o padrão de attention explicitamente. Dentro da escala k, o modelo attende a posições cross-escala mas não intra-escala? Verifique a implementação da máscara.
4. **VAR vs escala de DiT.** Para a mesma tarefa condicional por classe no ImageNet, treine VAR e DiT com orçamentos de parâmetros correspondidos (ex: 33M, 130M, 458M). Plote FID vs computação. VAR deveria se adiantar do DiT em cada tamanho — reproduza o resultado do paper em escala pequena.
5. **Condicionalização por texto.** Estenda VAR para receber um embedding de texto (CLIP pooled) como entrada condicional extra via adaLN. Esta é a receita HART. Quanto o FID melhora com amostragem alinhada a texto?

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|----------------|----------------------||
|| VAR | "AutoRegressivo Visual" | Geração de imagem por previsão de próxima escala sobre uma pirâmide de grids de tokens VQ ||
|| Previsão de próxima escala | "Prever grosseiro, depois fino" | O modelo prevê tokens em escalas de resolução crescente, condicionando em todas as escalas anteriores ||
|| Tokenizador VQ multi-escala | "VQ residual" | VQ-VAE que produz K grids de tokens de resolução crescente, com decoder somando todas as escalas ||
|| Escala k | "Nível k da pirâmide" | Um dos K níveis de resolução, de 1x1 em k=1 até (H/p)x(W/p) em k=K ||
|| Paralelo-dentro-de-escala | "Uma forward por escala" | Todos os tokens na escala k são previstos em uma única passada de transformer, não autoregressivamente ||
|| Causal-entre-escalas | "Attention ordenada por escala" | Token na escala k pode attendere a todas as escalas 1..k mas não escalas k+1..K ||
|| VQ residual | "Tokenização aditiva" | Tokens de cada escala codificam o resíduo deixado por escalas menores; decoder soma todos os embeddings de escala ||
|| Lei de escala VAR | "Escala de GPT para imagem" | FID segue uma lei de potência previsível em computação, como perplexidade de modelos de linguagem ||
|| HART | "VAR + texto híbrido" | Variante VAR condicional por texto combinando decodificação iterativa estilo MaskGIT com a estrutura de escala do VAR ||
|| Embedding posicional de escala | "Tríplade (escala, linha, col)" | Codificação posicional carrega tanto o índice da escala quanto coordenadas espaciais dentro da escala ||

## Leituras Complementares

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — o paper VAR, referência canônica
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, baseline de comparação com difusão
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, a família de tokenizadores que o tokenizador multi-escala do VAR estende
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, a base da tokenização discreta de imagem
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR condicional por texto
