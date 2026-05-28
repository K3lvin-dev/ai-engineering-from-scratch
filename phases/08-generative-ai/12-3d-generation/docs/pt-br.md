# Geração 3D

> 3D é a modalidade onde a alavancagem de 2D-para-3D é mais forte. A ruptura de 2023 foi o 3D Gaussian Splatting. O impulso gerativo de 2024-2026 adiciona difusão multi-view + reconstrução 3D por cima para produzir objetos e cenas a partir de um único prompt ou foto.

**Tipo:** Aprender
**Linguagens:** Python
**Pré-requisitos:** Fase 4 (Visão), Fase 8 · 07 (Latent Diffusion)
**Tempo:** ~45 minutos

## O Problema

Conteúdo 3D é doloroso:

- **Representação.** Meshes, nuvens de pontos, grids de voxel, campos de distância assinados (SDFs), campos de radiância neurais (NeRFs), Gaussianas 3D. Cada um tem trade-offs.
- **Escassez de dados.** ImageNet tem 14M de imagens. O maior dataset 3D limpo (Objaverse-XL, 2023) tem ~10M objetos, a maioria de baixa qualidade.
- **Memória.** Um grid de voxel 512³ são 128M voxels; uma cena NeRF útil precisa de 1M amostras/ray. Geração é mais difícil que reconstrução.
- **Supervisão.** Para uma imagem 2D você tem os pixels. Para 3D geralmente temos poucas visões 2D e precisamos levantar para 3D.

O stack de 2026 separa os dois problemas. Primeiro, gere *imagens 2D multi-view* com um modelo de difusão. Segundo, ajuste uma *representação 3D* (geralmente Gaussian splatting) nessas imagens.

## O Conceito

![Geração 3D: difusão multi-view + reconstrução 3D](../assets/3d-generation.svg)

### Representação: 3D Gaussian Splatting (Kerbl et al., 2023)

Represente uma cena como uma nuvem de ~1M Gaussianas 3D. Cada uma tem 59 parâmetros: posição (3), covariância (6, ou quaternião 4 + escala 3), opacidade (1), cor harmônicos esféricos (48 no grau 3, 3 no grau 0).

Renderização = projeção + alpha-compositing. Rápido (~100 fps em 1080p em uma 4090). Diferenciável. Ajustado por descida do gradiente contra fotos ground-truth. Uma cena se ajusta em 5-30 minutos em uma GPU consumer.

Duas inovações de 2023-2024 por cima:
- **Gaussian splats generativos.** Modelos como LGM, LRM, InstantMesh predizem uma nuvem Gaussiana diretamente de uma ou poucas imagens.
- **4D Gaussian Splatting.** Gaussianas com offsets por frame para cenas dinâmicas.

### Difusão multi-view

Fine-tune um modelo de difusão de imagem pré-treinado para gerar múltiplas visões consistentes do mesmo objeto a partir de um prompt de texto ou imagem única. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Geralmente produzem 4-16 visões ao redor do objeto, levantadas para 3D via Gaussian splatting ou NeRF.

### Pipelines de texto-para-3D

|| Modelo | Entrada | Saída | Tempo ||
||-------|-------|--------|------||
|| DreamFusion (2022) | texto | NeRF via SDS | ~1 hora por asset ||
|| Magic3D | texto | mesh + textura | ~40 min ||
|| Shap-E (OpenAI, 2023) | texto | 3D implícito | ~1 min ||
|| SJC / ProlificDreamer | texto | NeRF / mesh | ~30 min ||
|| LRM (Meta, 2023) | imagem | triplane | ~5 s ||
|| InstantMesh (2024) | imagem | mesh | ~10 s ||
|| SV3D (Stability, 2024) | imagem | novel views | ~2 min ||
|| CAT3D (Google, 2024) | 1-64 imagens | NeRF 3D | ~1 min ||
|| TripoSR (2024) | imagem | mesh | ~1 s ||
|| Meshy 4 (2025) | texto + imagem | mesh PBR | ~30 s ||
|| Rodin Gen-1.5 (2025) | texto + imagem | mesh PBR | ~60 s ||
|| Tencent Hunyuan3D 2.0 (2025) | imagem | mesh | ~30 s ||

Direção 2025-2026: modelos diretos de texto-para-mesh com materiais PBR adequados para game engines. O passo intermediário de difusão multi-view ainda é a receita de melhor desempenho para objetos gerais.

### NeRF (para contexto)

Neural Radiance Field (Mildenhall et al., 2020). Um MLP minúsculo recebe `(x, y, z, direção de visão)` e produz `(cor, densidade)`. Renderize integrando ao longo de rays. Supera novel-view synthesis baseada em mesh em qualidade mas é 100-1000x mais lento para renderizar. Suplantado por Gaussian splatting na maioria dos usos em tempo real mas ainda domina na pesquisa.

## Construa

`code/main.py` implementa um "Gaussian splatting" 2D de brinquedo: represente uma imagem-alvo sintética (um gradiente suave) como uma soma de splats Gaussianos 2D. Otimize posições, cores e covariâncias por descida do gradiente para corresponder ao alvo. Você vê as duas operações centrais: renderização direta (splat + alpha-composite) e ajuste por descida do gradiente.

### Passo 1: splat Gaussiano 2D

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Passo 2: renderize somando splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Gaussian splatting 3D real ordena Gaussianas por profundidade e faz alpha-composite em ordem. Nosso brinquedo 2D apenas soma.

### Passo 3: ajuste por descida do gradiente

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Armadilhas

- **Inconsistência de visão.** Se você gera 4 visões independentemente e elas discordam sobre a estrutura do objeto, o ajuste 3D fica borrado. Solução: difusão multi-view com attention compartilhada.
- **Alucinação da parte de trás.** Imagem única → 3D precisa inventar o lado não visto. Qualidade varia drasticamente.
- **Explosão de Gaussian splats.** Treino sem restrição cresce para 10M de splats e sobreadapta. Heurísticas de densificação + poda (do paper original 3D-GS) são essenciais.
- **Problemas de topologia.** Meshes de campos implícitos (SDFs) frequentemente têm buracos ou auto-interseções. Rode um remesher (ex: voxel remesh do blender) antes de lançar.
- **Licença dos dados de treino.** Objaverse tem licenças mistas; uso comercial varia por modelo.

## Use

|| Tarefa | Escolha de 2026 ||
||------|-----------||
|| Reconstrução de cena a partir de fotos | Gaussian splatting (3DGS, Gsplat, Scaniverse) ||
|| Texto-para-3D para jogos | Meshy 4 ou Rodin Gen-1.5 (saída PBR) ||
|| Imagem-para-3D | Hunyuan3D 2.0, TripoSR, InstantMesh ||
|| Novel-view synthesis a partir de poucas imagens | CAT3D, SV3D ||
|| Reconstrução de cena dinâmica | 4D Gaussian Splatting ||
|| Avatar / humano vestido | Gaussian Avatar, HUGS ||
|| Pesquisa / SOTA | O que caiu semana passada ||

Para lançar 3D production em pipeline de jogo ou e-commerce: Meshy 4 ou Rodin Gen-1.5 produzem meshes PBR que vão direto para Unity / Unreal.

## Entregue

Salve `outputs/skill-3d-pipeline.md`. A skill recebe um briefing 3D (entrada: texto / uma imagem / poucas imagens; saída: mesh / splat / NeRF; uso: renderização / jogo / VR) e gera: pipeline (difusão multi-view + ajuste, ou modelo de mesh direto), modelo base, orçamento de iteração, pós-processamento de topologia, canais de material necessários.

## Exercícios

1. **Fácil.** Execute `code/main.py` com 4, 16, 64 Gaussianas. Reporte o MSE final vs alvo.
2. **Médio.** Estenda para Gaussianas coloridas (RGB). Confirme que a reconstrução corresponde ao padrão de cor do alvo.
3. **Difícil.** Usando gsplat ou Nerfstudio, reconstrua um objeto real a partir de 50 fotos. Reporte tempo de ajuste e SSIM final em visões de teste.

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|-----------------|-----------------------||
|| 3D Gaussian Splatting | "3DGS" | Cena como nuvem de Gaussianas 3D; renderização diferenciável por alpha-composite. ||
|| NeRF | "Campo de radiância neural" | MLP que produz cor + densidade em um ponto 3D; renderiza por integração de ray. ||
|| Triplane | "Três planos 2D" | Fatora 3D em três grids de features 2D alinhados aos eixos; mais barato que volumétrico. ||
|| SDS | "Score distillation sampling" | Treina modelo 3D usando o score de difusão 2D como pseudo-gradiente. ||
|| Difusão multi-view | "Muitas visões ao mesmo tempo" | Modelo de difusão que produz um batch de visões de câmera consistentes. ||
|| PBR | "Renderização baseada em física" | Material com canais albedo, rugosidade, metálico, normal. ||
|| Densificação | "Crescer splats" | Heurística de treino 3DGS: dividir / clonar splats em regiões de alto gradiente. |

## Nota de produção: 3D ainda não tem substrato compartilhado

Diferente de imagem (latent diffusion + DiT) e vídeo (DiT espaço-temporal), 3D não tem um runtime dominante único em 2026. A árvore de decisão de produção bifurca na representação:

- **NeRF / triplane.** Inferência é ray-marching + forward de MLP por amostra. Uma renderização 512² precisa de milhões de forwards de MLP. Faça batch dos samples de ray agressivamente; SDPA/xformers se aplica.
- **Difusão multi-view + reconstrução LRM.** Pipeline de dois estágios. Estágio 1 (DiT multi-view) é um servidor de difusão assim como a Aula 07. Estágio 2 (transformer LRM) é uma passada forward sobre as visões. O perfil de latência geral é "difusão + passada única" — escolha primitivas de serving por estágio de acordo.
- **SDS / DreamFusion.** Otimização por asset, não inferência. Trabalhos de build, não handlers de requisição.

Para a maioria dos produtos de 2026, a resposta certa é "rode um modelo de difusão multi-view sob demanda, reconstrua para 3DGS assincronamente, sirva o 3DGS para visualização em tempo real". Isso divide a carga de forma limpa entre um servidor de inferência GPU (rápido) e um otimizador offline (lento).

## Leituras Complementares

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — difusão multi-view.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
