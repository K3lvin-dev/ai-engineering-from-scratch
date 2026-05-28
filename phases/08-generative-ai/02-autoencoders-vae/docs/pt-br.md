# Autoencoders e Variational Autoencoders (VAE)

> Um autoencoder comum comprime e depois reconstrói. Ele memoriza. Ele não gera. Adiciona um truque — força o código parecer Gaussiano — e você ganha um sampler. Esse truque único, a reparametrização de `z = μ + σ·ε`, é porque todo modelo de imagem de difusão latente e flow matching que você usa em 2026 tem um VAE na entrada.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 3 · 02 (Backprop), Fase 3 · 07 (CNNs), Fase 8 · 01 (Taxonomia)
**Tempo:** ~75 minutos

## O Problema

Comprimir um dígito MNIST de 784 pixels num código de 16 números, depois reconstruir. Um autoencoder comum vai acertar o MSE de reconstrução mas o espaço do código é uma bagunça amorfa. Escolhe um ponto aleatório no espaço do código, decodifica, e ganha ruído. Não tem sampler. É um modelo de disfarçado.

O que você realmente quer é: (a) o espaço do código é uma distribuição limpa, suave, de onde você pode amostrar — digamos Gaussiana isotrópica `N(0, I)`, (b) decodificar qualquer amostra produz um dígito plausível, e (c) o encoder e decoder ainda comprimem bem. Três objetivos, uma arquitetura, uma loss.

O VAE do Kingma em 2013 resolve isso treinando o encoder para sair de uma *distribuição* `q(z|x) = N(μ(x), σ(x)²)`, puxando essa distribuição para o prior `N(0, I)` via penalidade KL, e depois amostrando `z` de `q(z|x)` antes de decodificar. Na inferência, descarta o encoder, amostra `z ~ N(0, I)`, decodifica. A penalidade KL é o que força o espaço do código ser estruturado.

Em 2026 VAEs raramente são distribuídos isoladamente — foram superados por difusão em qualidade bruta de imagem — mas são o encoder de escolha para todo modelo de difusão latente (SD 1/2/XL/3, Flux, AudioCraft). Aprenda o VAE e você aprende a primeira camada invisível de todo pipeline de imagem que você usa.

## O Conceito

![Autoencoder vs VAE: o truque da reparametrização](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Espaço do código não estruturado.

**Encoder do VAE.** Sai dois vetores: `μ(x)` e `log σ²(x)`. Esses definem `q(z|x) = N(μ, diag(σ²))`.

**Truque da reparametrização.** Amostrar de `q(z|x)` não é diferenciável. Reescreve a amostra como `z = μ + σ·ε` onde `ε ~ N(0, I)`. Agora `z` é uma função determinística de `(μ, σ)` mais ruído não-paramétrico — gradientes fluem por `μ` e `σ`.

**Loss.** Limite Inferior da Evidência (ELBO), dois termos:

```
loss = reconstrução + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstrução puxa `x̂` para `x`. KL puxa `q(z|x)` para o prior. Elas se balanceiam. β pequeno (<1) = amostras mais nítidas, espaço do código menos Gaussiano. β grande (>1) = espaço do código mais limpo, amostras mais borradas. β-VAE (Higgins 2017) tornou esse parâmetro famoso e iniciou a pesquisa de desentanglement.

**Amostragem.** Na inferência: desenha `z ~ N(0, I)`, passa pelo decoder. Um forward pass — sem amostragem iterativa como difusão.

## Construa

`code/main.py` implementa um VAE minúsculo sem numpy nem torch. Input são dados sintéticos 8-dimensionais desenhados de uma mistura gaussiana de dois componentes em 8-D. Encoder e decoder são MLPs de uma camada oculta. Implementamos ativação tanh, forward pass, loss, e um backward pass escrito à mão. Não é produção — é pedagogia.

### Passo 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"]))
    return mu, log_sigma2
```

`log σ²` em vez de `σ` para que a saída da rede seja irrestrita (softplus de σ é uma armadilha — gradientes morrem em σ ≈ 0).

### Passo 2: reparametrizar e decodificar

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Passo 3: o ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL exato em forma fechada porque ambas as distribuições são Gaussianas. Não integre numericamente. Gente ainda distribui código com estimativas KL por Monte Carlo em 2026 — é 3x mais lento sem motivo.

### Passo 4: gerar

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Esse é o modelo generativo. Cinco linhas.

## Armadilhas

- **Colapso posterior.** Termo KL força `q(z|x) → N(0, I)` tão agressivamente que `z` não carrega info sobre `x`. Solução: β-annealing (comece com β=0, suba para 1), free bits, ou pule o KL em dimensões inativas.
- **Amostras borradas.** A verossimilhança Gaussiana do decoder implica reconstrução MSE, que é Bayes-ótima para L2 (a média) — a média de vários dígitos plausíveis é um dígito borrado. Solução: decoder discreto (VQ-VAE, NVAE), ou use o VAE apenas como encoder e empilhe difusão nos latents (o que o Stable Diffusion faz).
- **β alto demais, cedo demais.** Veja colapso posterior. Comece com β≈0.01 e aumente.
- **Dimensão latent pequena demais.** 16-D funciona para MNIST, 256-D para ImageNet 256², 2048-D para ImageNet 1024². O VAE do Stable Diffusion comprime 512×512×3 → 64×64×4 (fator de downsample de 32x em área espacial, 32x em canais).

## Use

A stack de VAE em 2026:

| Situação | Escolha |
|-----------|------|
| Encoder de imagem-latent para difusão | VAE do Stable Diffusion (`sd-vae-ft-ema`) ou VAE do Flux |
| Encoder de áudio-latent | Encodec (Meta), SoundStream ou DAC (Descript) |
| Latents de vídeo | Patches espaciotemporais do Sora, Latte VAE, WAN VAE |
| Aprendizado de representação desentrelaçada | β-VAE, FactorVAE, TCVAE |
| Latents discretos (para modelagem transformer) | VQ-VAE, RVQ (ResidualVQ) |
| Latents contínuos para geração | VAE comum, depois condicione um modelo flow/difusão nesse espaço latent |

Um modelo de difusão latente é um VAE com um modelo de difusão vivendo entre encoder e decoder. O VAE faz a compressão bruta, o modelo de difusão faz o trabalho pesado. Mesmo padrão para vídeo (VAE + DiT de vídeo) e áudio (Encodec + transformer MusicGen).

## Entregue

Salve `outputs/skill-vae-trainer.md`.

A skill recebe: perfil do dataset + dimensão latent alvo + uso downstream (reconstrução, amostrada ou input para difusão latente) e gera: escolha da arquitetura (comum/β/VQ/RVQ), agendamento β, dimensão latent, verossimilhança do decoder (Gaussiana vs categórica), e plano de avaliação (reconstrução MSE, KL por dim., distância Fréchet entre `q(z|x)` e `N(0, I)`).

## Exercícios

1. **Fácil.** Mude `β` em `code/main.py` para `0.01`, `0.1`, `1.0`, `5.0`. Registre o MSE final de reconstrução e KL. Qual β é Pareto-ótimo para seus dados sintéticos?
2. **Médio.** Substitua a verossimilhança Gaussiana do decoder por uma verossimilhança Bernoulli (loss de entropia cruzada). Compare qualidade de amostra numa versão binarizada dos mesmos dados sintéticos.
3. **Difícil.** Estenda `code/main.py` para um mini VQ-VAE: substitua o `z` contínuo por uma busca de nearest-neighbour num codebook com K=32 entradas. Compare MSE de reconstrução e reporte quantas entradas do codebook são usadas (colapso de codebook é real).

## Termos Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|------|-----------------|-----------------------|
| Autoencoder | Rede encode-decode | `x → z → x̂`, aprende MSE. Não é generativo. |
| VAE | AE com sampler | Encoder sai uma distribuição, penalidade KL molda o espaço do código. |
| ELBO | Limite inferior da evidência | `log p(x) ≥ recon - KL[q(z\\|x) \\|\\| p(z)]`; apertado quando `q = p(z\\|x)`. |
| Reparametrização | `z = μ + σ·ε` | Reescreve nó estocástico como determinístico + puro ruído. Permite backprop através da amostragem. |
| Prior | `p(z)` | Distribuição alvo do latent, tipicamente `N(0, I)`. |
| Colapso posterior | "Termo KL vence" | Encoder ignora `x`, sai o prior; decoder precisa alucinar. |
| β-VAE | Peso KL ajustável | `loss = recon + β·KL`. β maior = mais desentrelaçado mas mais borrado. |
| VQ-VAE | Latent discreto | Substitui `z` contínuo por vetor mais próximo do codebook; permite modelagem transformer. |

## Nota de produção: o VAE é o caminho mais quente num servidor de difusão

Num pipeline de Stable Diffusion / Flux / SD3, o VAE é chamado duas vezes por request — uma para encode (se fizer img2img / inpainting) e uma para decode. Em 1024² o passo do decoder é frequentemente o maior pico de memória de ativação em todo o pipeline porque upsamples latents de `128×128×16` de volta para `1024×1024×3`. Duas consequências práticas:

- **Fatie ou divida o decode.** `diffusers` expõe `pipe.vae.enable_slicing()` e `pipe.vae.enable_tiling()`. Tiling troca um pequeno artefato de emenda por `O(tile²)` memória em vez de `O(H·W)`. Essencial para 1024²+ em GPUs de consumo.
- **Decoder bf16, numéricos fp32 para o resize final.** O VAE do SD 1.x foi lançado em fp32 e *produz NaN silenciosamente* quando convertido para fp16 em 1024²+. SDXL distribui `madebyollin/sdxl-vae-fp16-fix` — sempre prefira a variante fp16-fix ou use bf16.

## Leitura Adicional

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — o paper do VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — β-VAE desentrelaçado.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — VAE de imagem state-of-the-art.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE como encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, o padrão VAE de áudio.
