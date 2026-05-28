# Inpainting, Outpainting e Edição de Imagem

> Texto-para-imagem faz coisas novas. Inpainting conserta coisas velhas. Em produção, 70% do trabalho faturável de imagem é edição — trocar fundo, remover logo, estender canvas, regenerar mão. Inpainting é onde difusão ganha seu_keep.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 8 · 07 (Difusão Latente), Fase 8 · 08 (ControlNet & LoRA)
**Tempo:** ~75 minutos

## O Problema

Um cliente envia uma foto de produto perfeita com um letreiro distraindo no fundo. Você quer apagar o letreiro e manter tudo o resto pixel-idêntico. Você não pode rodar texto-para-imagem do zero — o resultado terá cor diferente, iluminação diferente, ângulo do produto diferente. Você quer regenerar *apenas* a região mascarada, e quer que a regeneração respeite o contexto ao redor.

Isso é inpainting. Variantes:

- **Inpainting.** Regenera dentro de uma máscara, mantém pixels de fora.
- **Outpainting.** Regenera fora da máscara (ou além do canvas), mantém dentro.
- **Edição de imagem.** Regenera a imagem inteira mas mantém fidelidade semântica ou estrutural à original (SDEdit, InstructPix2Pix).

Toda pipeline de difusão em 2026 distribui um modo de inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Funcionam no mesmo princípio.

## O Conceito

![Inpainting: denoising awareness de máscara com re-injeção de contexto](../assets/inpainting.svg)

### A abordagem ingênua (e por que está errada)

Roda texto-para-imagem padrão com máscara. Em cada passo de amostragem, substitui a região não mascarada do latent ruidoso pela imagem limpa forward-difundida. Funciona... mal. Artefatos de borda vazam porque o modelo não tem informação do que está na região mascarada.

### O modelo de inpainting propriamente dito

Treina um U-Net modificado que recebe 9 canais de entrada em vez de 4:

```
input = concat([ latent_ruidoso (4ch), imagem_codificada (4ch), máscara (1ch) ], dim=channel)
```

Os canais extras são uma cópia da imagem fonte codificada pelo VAE mais uma máscara de canal único. No treinamento, você mascara regiões aleatoriamente e treina o modelo para denoiser apenas a região mascarada enquanto a região não mascarada é dada como sinal de condição limpo. Na inferência, o modelo "vê" o que cerca a região mascarada e produz completamentos coerentes.

SD-Inpaint, SDXL-Inpaint, Flux-Fill todos usam essa entrada de 9 canais (ou análoga). Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — edição livre

Adiciona ruído na imagem fonte até algum `t` intermediário, depois roda a cadeia reversa de `t` até 0 com um novo prompt. Sem retreinamento. A escolha do `t` inicial troca fidelidade por liberdade criativa:

- `t/T = 0.3` → quase idêntica à fonte, pequenas mudanças de estilo
- `t/T = 0.6` → edições moderadas, preserva estrutura grossa
- `t/T = 0.9` → gerada de puro ruído, preservação mínima da fonte

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune de um modelo de difusão em triples `(imagem_input, instrução, imagem_output)`. Na inferência, condiciona na imagem de entrada e em uma instrução textual ("faça pôr-do-sol", "adicione um dragão"). Duas escalas CFG: escala de imagem e escala de texto.

### RePaint (Lugmayr et al., 2022)

Mantém um modelo de difusão incondicional padrão. Em cada passo reverso, reamostra — pula de volta para um estado mais ruidoso periodicamente e regenera. Evita artefatos de borda. Usado quando você não tem um modelo de inpainting treinado.

## Construa

`code/main.py` implementa um esquema de inpainting 1-D de brinquedo em dados 5-dimensionais. Treinamos um DDPM em dados de mistura 5-D onde cada amostra são 5 floats de um de dois clusters. Na inferência, "mascaramos" 2 das 5 dimensões, injetamos a versão forward-noisy das três não mascaradas em cada passo, e regeneramos apenas as dimensões mascaradas.

### Passo 1: dados DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Passo 2: treina denoiser em todas as 5 dimensões

DDPM padrão. Rede sai previsão de ruído 5-D para input 5-D ruidoso.

### Passo 3: na inferência, reverso awareness de máscara

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # substitui dims não mascaradas com versão freshly noised da fonte limpa
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...depois roda o passo reverso normal em x_t
```

Essa é a abordagem ingênua e funciona em dados 1-D de brinquedo. Inpainting de imagem real usa entrada de 9 canais porque coesão de textura importa mais.

### Passo 4: outpainting

Outpainting é inpainting com máscara invertida: mascara o novo canvas (anteriormente inexistente), preenche o resto com o original. Mesmo objetivo de treinamento.

## Armadilhas

- **Emendas.** A abordagem ingênua deixa bordas visíveis porque informação de gradiente não flui através da máscara. Solução: dilate a máscara por 8-16 pixels, ou use um modelo de inpainting adequado.
- **Vazamento de máscara.** Se a região não mascarada da imagem de condição é de baixa qualidade ou ruidosa, polui a geração dentro da máscara. Desnoise ou borrar levemente.
- **CFG interage com tamanho da máscara.** CFG alto em máscara pequena = patch saturado. Reduza CFG para edições pequenas.
- **Penhasco de fidelidade SDEdit.** Ir de `t/T = 0.5` para `t/T = 0.6` pode perder a identidade do sujeito. Varre e faça checkpoint.
- **Incompatibilidade de prompt.** O prompt deve descrever a *imagem inteira*, não só o novo conteúdo. "Um gato sentado numa cadeira" não "um gato".

## Use

| Tarefa | Pipeline |
|------|----------|
| Remover objeto, máscara pequena | SD-Inpaint ou Flux-Fill, prompt padrão |
| Trocar céu | SD-Inpaint + "céu azul ao pôr-do-sol" |
| Estender canvas | Modo outpaint SDXL (feathering 8px) ou Flux-Fill com máscara de outpaint |
| Regenerar mão / rosto | SD-Inpaint com prompt re-descrevendo o sujeito + ControlNet-Openpose |
| Mudar estilo de uma região | SDEdit em `t/T=0.5` na região mascarada |
| "Faça pôr-do-sol" | InstructPix2Pix ou Flux-Kontext |
| Substituição de fundo | Máscara SAM → SD-Inpaint |
| Ultra-alta fidelidade | Flux-Fill ou GPT-Image (hospedado) para casos mais difíceis |

SAM (Segment Anything do Meta, 2023) + inpaint de difusão é a pipeline de 2026 para remoção de fundo. SAM 2 (2024) funciona em vídeo.

## Entregue

Salve `outputs/skill-editing-pipeline.md`. Skill recebe uma imagem original + descrição de edição + máscara opcional (ou prompt SAM) e gera: abordagem de geração de máscara, modelo base, escalas CFG (imagem + texto), modo SDEdit-t ou inpainting, e checklist de QA.

## Exercícios

1. **Fácil.** Em `code/main.py`, varie a fração de dimensões mascaradas de 0.2 a 0.8. Em qual fração a qualidade do inpaint (residual nas dims mascaradas) iguala geração incondicional?
2. **Médio.** Implemente RePaint: a cada 10º passo reverso, pule 5 passos (adicione ruído) e refaça o denoising. Meça se reduz o residual de borda na máscara.
3. **Difícil.** Use diffusers da Hugging Face para comparar: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill em 20 tarefas de regeneração de rosto. Pontue aderência de pose e preservação de identidade separadamente.

## Termos Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|------|-----------------|-----------------------|
| Inpainting | "Preencher o buraco" | Regenera dentro de uma máscara; mantém pixels de fora. |
| Outpainting | "Estender o canvas" | Regenera fora do canvas; mantém dentro. |
| U-Net de 9 canais | "Modelo adequado de inpainting" | U-Net com `ruidoso \\| codificado-fonte \\| máscara` como input. |
| SDEdit | "Img2img com nível de ruído" | Ruído até tempo `t`, denoisa com novo prompt. |
| InstructPix2Pix | "Edições só com texto" | Difusão fine-tuned em triples (imagem, instrução, output). |
| RePaint | "Sem retreinamento" | Re-ruidifica periodicamente durante reverso para reduzir emendas. |
| SAM | "Segment Anything" | Gerador de máscara por cliques ou caixas; combina com inpaint. |
| Flux-Kontext | "Edição com contexto" | Variante do Flux que aceita imagem referência + instrução para edições. |

## Nota de produção: pipelines de edição são sensíveis a latência

Usuários editando uma imagem esperam round-trips sub-5-segundos. Um SDXL-Inpaint de 30 passos em 1024² são 3-4 s num L4, mais geração de máscara SAM (~200 ms) e encode/decode VAE (~500 ms combinados). Em enquadramento de produção, isso é limitado por TTFT e não por throughput — batch 1, baixa concorrência, minimize cada estágio:

- **SAM-H é o mais lento.** SAM-H em 1024² são ~200 ms; SAM-ViT-B são ~40 ms com perda menor de qualidade. SAM 2 (vídeo) adiciona overhead temporal; não use para edições de imagem única.
- **Pule o encode quando possível.** `pipe.image_processor.preprocess(img)` codifica em latents. Se você tem os latents da geração anterior (típico em UIs de edição iterativa), passe diretamente via `latents=...` para pular um encode VAE.
- **Dilatação de máscara importa para throughput também.** Máscara pequena significa que a maior parte do forward pass do U-Net é desperdiçada (pixels não mascarados são clamped de qualquer forma). `StableDiffusionInpaintPipeline` do diffusers roda o U-Net inteiro; apenas as variantes de inpainting adequado de 9 canais exploram compute mascarado.
- **Flux-Kontext é a resposta de 2025.** Forward pass único sobre `(imagem_fonte, instrução)` — sem máscara separada, sem sweep de ruído SDEdit. Em H100 distribui uma edição em ~1.5 s. A lição arquitetural: colapsar os estágios.

## Leitura Adicional

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting sem treinamento.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — edição por instrução de texto.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, a fonte de máscaras.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — SAM para vídeo.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — edição no nível de attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — ferramentas 2024.
