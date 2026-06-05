# 3D Gaussian Splatting do Zero

> Uma cena é uma nuvem de milhões de Gaussianos 3D. Cada um tem uma posição, orientação, escala, opacidade e uma cor que depende da direção de visão. Rasterize-os, retropropague através da rasterização, pronto.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Phase 4 Lesson 13 (Visão 3D & NeRF), Phase 1 Lesson 12 (Operações com Tensores), Phase 4 Lesson 10 (Conceitos básicos de difusão opcional)
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Explicar por que 3D Gaussian Splatting substituiu NeRF como o padrão de produção para reconstrução 3D fotorrealista em 2026
- Declarar os seis parâmetros por Gaussian (posição, quatérnio de rotação, escala, opacidade, cor de harmônicos esféricos, característica opcional) e quantos floats cada um contribui
- Implementar um rasterizador de 2D Gaussian splatting do zero usando composição `alpha`, depois mostrar como o caso 3D se projeta para o mesmo loop
- Usar `nerfstudio`, `gsplat` ou `SuperSplat` para reconstruir uma cena a partir de 20-50 fotos e exportar para a extensão glTF `KHR_gaussian_splatting` ou o schema OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat`

## O Problema

Um NeRF armazena uma cena como os pesos de um MLP. Cada pixel renderizado é centenas de consultas MLP ao longo de um raio. O treinamento leva horas, a renderização leva segundos, e os pesos não podem ser editados — se você quiser mover uma cadeira dentro de uma cena, tem que retreinar.

3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) substituiu tudo isso. Uma cena é um conjunto explícito de Gaussianos 3D. A renderização é rasterização GPU a 100+ fps. O treinamento leva minutos. A edição é direta: traduza um subconjunto de Gaussianos e você moveu a cadeira. Em 2026, o Khronos Group ratificou uma extensão glTF para Gaussian splats, OpenUSD 26.03 oferece um schema de Gaussian splat, Zillow e Apartments.com renderizam imóveis com eles, e a maioria dos novos papers de pesquisa sobre reconstrução 3D são variantes da ideia central do 3DGS.

O modelo mental é simples, a matemática tem partes móveis suficientes que a maioria das introduções começa pela rasterização e pula as projeções e harmônicos esféricos. Esta lição constrói tudo — uma versão 2D primeiro, depois a extensão 3D.

## O Conceito

### O que um Gaussian carrega

Um Gaussian 3D é uma mancha paramétrica no espaço com estes atributos:

```
posição         mu         (3,)    centro em coordenadas mundiais
rotação         q          (4,)    quatérnio unitário codificando orientação
escala           s          (3,)    log-escalas por eixo (exponenciadas na renderização)
opacidade       alpha      (1,)    opacidade pós-sigmoid [0, 1]
Coeficientes SH c_lm       (3 * (L+1)^2,)   cor dependente da visão
```

Rotação + escala constroem uma covariância 3x3: `Sigma = R S S^T R^T`. Essa é a forma do Gaussian em 3D. Harmônicos esféricos permitem que a cor mude com a direção de visão — destaques especulares, brilho sutil, brilho dependente da visão — sem armazenar texturas por vista. Com grau SH 3 você obtém 16 coeficientes por canal de cor, 48 floats por Gaussian apenas para a cor.

Uma cena tipicamente tem 1-5 milhões de Gaussianos. Cada um armazena aproximadamente 60 floats (3 + 4 + 3 + 1 + 48 + misc). Isso é 240 MB para uma cena de cinco milhões de Gaussianos — muito menor que a nuvem de pontos equivalente com textura por ponto, e uma ordem de magnitude menor que os pesos de MLP de um NeRF re-renderizados em alta resolução.

### Rasterização, não ray marching

```mermaid
flowchart LR
    SCENE["Milhões de Gaussianos 3D<br/>(posição, rotação, escala,<br/>opacidade, cor SH)"] --> PROJ["Projetar para 2D<br/>(extrínsecos + intrínsecos da câmera)"]
    PROJ --> TILES["Atribuir a tiles<br/>(16x16 espaço de tela)"]
    TILES --> SORT["Ordenar por profundidade<br/>por tile"]
    SORT --> ALPHA["Composição alfa<br/>frente-para-trás"]
    ALPHA --> PIX["Cor do pixel"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

Cinco passos, todos amigáveis à GPU. Nenhuma consulta MLP por pixel. Uma única RTX 3080 Ti renderiza 6 milhões de splats a 147 fps.

### O passo de projeção

O Gaussian 3D na posição mundial `mu` com covariância 3D `Sigma` projeta para um Gaussian 2D na posição de tela `mu'` com covariância 2D `Sigma'`:

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = transformação de visão (rotação + translação da câmera)
J = Jacobiano da projeção perspectiva em mu'
```

A pegada do Gaussian 2D é uma elipse cujos eixos são os autovetores de `Sigma'`. Cada pixel dentro dessa elipse recebe a contribuição do Gaussian, ponderada por `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`.

### A regra de composição alfa

Para um pixel, os Gaussianos que o cobrem são ordenados de trás para frente (ou equivalentemente de frente para trás com fórmula invertida). A cor é composta com a mesma equação de todo rasterizador semi-transparente desde os anos 1980:

```
C_pixel = soma_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmitância até i
alpha_i = opacidade_i * exp(-0.5 * d^T Sigma'^-1 d)   contribuição local
c_i = avaliar_SH(SH_i, direção_visão)    cor dependente da visão
```

Esta é **a mesma equação da renderização volumétrica do NeRF**, apenas sobre um conjunto esparso explícito de Gaussianos em vez de amostras densas ao longo de um raio. Essa identidade é por que a qualidade renderizada corresponde ao NeRF — ambos estão integrando a mesma equação de campo de radiação.

### Por que isso é diferenciável

Cada passo — projeção, atribuição de tile, composição alfa, avaliação SH — é diferenciável com respeito aos parâmetros do Gaussian. Dada uma imagem de verdade, compute a loss do pixel renderizado, retropropague através do rasterizador, atualize todos `(mu, q, s, alpha, c_lm)` por descida de gradiente. Em ~30.000 iterações, os Gaussianos encontram suas posições, escalas e cores certas.

### Densificação e poda

Um conjunto fixo de Gaussianos não pode cobrir uma cena complexa. O treinamento inclui dois mecanismos adaptativos:

- **Clonar** um Gaussian em sua posição atual quando a magnitude de seu gradiente é alta mas sua escala é pequena — a reconstrução precisa de mais detalhes aqui.
- **Dividir** um Gaussian de grande escala em dois menores quando seu gradiente é alto — um Gaussian grande é muito suave para ajustar a região.
- **Podar** Gaussianos cuja opacidade cai abaixo de um limiar — eles não estão contribuindo.

A densificação roda a cada N iterações. Uma cena tipicamente cresce de ~100k Gaussianos iniciais (semeados a partir de pontos SfM) para 1-5M no final do treinamento.

### Harmônicos esféricos em um parágrafo

A cor dependente da visão é uma função `c(direção)` na esfera unitária. Harmônicos esféricos são a base de Fourier da esfera. Trunque no grau `L` e você obtém `(L+1)^2` funções de base por canal. Avaliar a cor para uma nova vista é um produto escalar entre os coeficientes SH aprendidos e a base avaliada na direção de visão. Grau 0 = um coeficiente = cor constante. Grau 3 = 16 coeficientes = suficiente para capturar sombreamento Lambertiano, especular e reflexão suave. Papers de 3D Gaussian Splatting usam grau 3 por padrão.

### O stack de produção 2026

```
1. Captura         smartphone / DJI drone / scanner de mão
2. SfM / MVS       COLMAP ou GLOMAP deriva poses de câmera + pontos esparsos
3. Treinar 3DGS    nerfstudio / gsplat / inria official / PostShot (~10-30 min em RTX 4090)
4. Editar          SuperSplat / SplatForge (limpar floaters, segmentar)
5. Exportar        .ply -> glTF KHR_gaussian_splatting ou .usd (OpenUSD 26.03)
6. Visualizar      Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### Variantes 4D e generativas

- **4D Gaussian Splatting** — Gaussianos são funções do tempo; usado para vídeo volumétrico (Superman 2026, A$AP Rocky "Helicopter").
- **Splats generativos** — modelos texto-para-splat (Marble by World Labs) que alucinam cenas inteiras.
- **3D Gaussian Unscented Transform** — variante da NVIDIA NuRec para simulação de direção autônoma.

## Construa

### Passo 1: Um Gaussian 2D

Primeiro construímos um rasterizador 2D. O caso 3D se reduz a ele após a projeção.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def avaliar_gaussiano_2d(means, covs, points):
    """
    means:  (G, 2)      centros
    covs:   (G, 2, 2)   matrizes de covariância
    points: (H, W, 2)   coordenadas de pixel
    retorna: (G, H, W)  densidade em cada pixel para cada Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    densidade = torch.exp(-0.5 * d)
    return densidade.view(G, H, W)
```

`einsum` faz a forma quadrática `diff^T Sigma^-1 diff` para cada par (Gaussian, pixel).

### Passo 2: Rasterizador de splatting 2D

Composição alfa frente-para-trás. A profundidade em 2D não tem significado, então usamos um escalar aprendido por Gaussian para ordenação.

```python
def rasterizar_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     em [0, 1]
    depths:    (G,)     escalar por Gaussian usado para ordenação
    image_size: (H, W)
    retorna:   (H, W, 3) imagem renderizada
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densidades = avaliar_gaussiano_2d(means, covs, points)
    alphas = opacities[:, None, None] * densidades
    alphas = alphas.clamp(0.0, 0.99)

    ordem = torch.argsort(depths)
    alphas = alphas[ordem]
    colours_sorted = colours[ordem]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

Não é rápido — uma implementação real usa kernels CUDA baseados em tiles — mas é exatamente a matemática certa e totalmente diferenciável.

### Passo 3: Uma cena de splat 2D treinável

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # ângulo único em 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colors = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterizar_2d(self.means, covs, colors, opacities, self.depth, image_size)
```

`log_scale`, `opacity_logit` e `colour_logits` são todos parâmetros sem restrições mapeados através da ativação certa no momento da renderização. Este é o padrão para toda implementação 3DGS.

### Passo 4: Ajustar Gaussianos 2D a uma imagem alvo

```python
import math
import numpy as np

def fazer_alvo(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Círculo vermelho
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Quadrado azul
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


alvo = fazer_alvo(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, alvo)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"passo {step:3d}  mse {loss.item():.4f}")
```

Em 200 passos, os 64 Gaussianos se acomodam nas duas formas. Essa é a ideia inteira — descida de gradiente em primitivas geométricas explícitas.

### Passo 5: De 2D para 3D

A extensão 3D mantém o mesmo loop. As adições:

1. Rotação por Gaussian é um quatérnio em vez de um ângulo único.
2. Covariância é `R S S^T R^T` com `R` construído a partir do quatérnio e `S = diag(exp(log_scale))`.
3. Projeção `(mu, Sigma) -> (mu', Sigma')` usa os extrínsecos da câmera e o Jacobiano da projeção perspectiva em `mu`.
4. Cor se torna uma expansão de harmônicos esféricos; avalie-a na direção de visão.
5. Ordenação por profundidade é a partir do z real do espaço da câmera em vez de um escalar aprendido.

Toda implementação de produção (`gsplat`, `inria/gaussian-splatting`, `nerfstudio`) faz exatamente isso na GPU com kernels CUDA baseados em tiles.

### Passo 6: Avaliação de harmônicos esféricos

A base SH até o grau 3 tem 16 termos por canal. Avaliação:

```python
def avaliar_sh_grau_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   última dim é canais RGB
    dirs:      (..., 3)       vetores unitários
    retorna:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # termos de grau 3 omitidos aqui por brevidade; versão completa de 16 coeficientes no arquivo de código
    return result
```

`sh_coeffs` aprendidos armazenam a "cor em toda direção" para aquele Gaussian. No momento da renderização, você avalia contra a direção de visão atual e obtém um RGB de 3 vetores.

## Use

Para trabalho 3DGS real, use `gsplat` (Meta) ou `nerfstudio`:

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` é o treinador 3DGS do nerfstudio. A execução leva 10-30 minutos em um RTX 4090 para uma cena típica.

Opções de exportação que importam em 2026:

- `.ply` — nuvem Gaussian bruta (portátil, maior arquivo).
- `.splat` — formato quantizado PlayCanvas / SuperSplat.
- glTF `KHR_gaussian_splatting` — padrão Khronos, portátil entre visualizadores (fev 2026 RC).
- OpenUSD `UsdVolParticleField3DGaussianSplat` — nativo USD, para pipelines NVIDIA Omniverse e Vision Pro.

Para cenas 4D / dinâmicas, `4DGS` e `Deformable-3DGS` estendem o mesmo maquinário com médias e opacidades variantes no tempo.

## Entregue

Esta lição produz:

- `outputs/prompt-3dgs-capture-planner.md` — um prompt que planeja uma sessão de captura (número de fotos, caminho da câmera, iluminação) para um dado tipo de cena.
- `outputs/skill-3dgs-export-router.md` — uma skill que escolhe o formato de exportação correto (`.ply` / `.splat` / glTF / USD) dado o visualizador ou motor downstream.

## Exercícios

1. **(Fácil)** Execute o treinador de splat 2D acima em uma imagem sintética diferente. Varie `num_splats` em `[16, 64, 256]` e plote MSE vs passo para cada um. Identifique o ponto de retornos decrescentes.
2. **(Médio)** Estenda o rasterizador 2D para suportar cores RGB por Gaussian que dependem de um escalar "ângulo de visão" através de um harmônico de grau 2. Treine em um par de imagens alvo e verifique que o modelo reconstrói ambas.
3. **(Difícil)** Clone `nerfstudio` e treine `splatfacto` em uma captura de 20 fotos de qualquer cena que você tenha (mesa, planta, rosto, sala). Exporte para glTF `KHR_gaussian_splatting` e abra em um visualizador (Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9). Reporte tempo de treino, número de Gaussianos e fps renderizado.

## Termos-Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|------------------------|---------------------------|
| 3DGS | "Gaussian splats" | Representação explícita de cena como milhões de Gaussianos 3D com posição, rotação, escala, opacidade, cor SH por Gaussian |
| Covariância | "Forma do Gaussian" | `Sigma = R S S^T R^T`; orientação e escala anisotrópica de um Gaussian |
| Composição alfa | "Mistura trás-para-frente" | Mesma equação da renderização volumétrica do NeRF, agora sobre um conjunto esparso explícito |
| Densificação | "Clonar e dividir" | Adição adaptativa de novos Gaussianos onde a reconstrução está subajustada |
| Poda | "Deletar baixa opacidade" | Remover Gaussianos que colapsaram para opacidade quase zero durante o treino |
| Harmônicos esféricos | "Cor dependente da visão" | Base de Fourier na esfera; armazena cor como uma função da direção de visão |
| Splatfacto | "3DGS do nerfstudio" | O caminho mais fácil para treinar 3DGS em 2026 |
| `KHR_gaussian_splatting` | "Padrão glTF" | Extensão Khronos 2026 que torna 3DGS portátil entre visualizadores e motores |

## Leitura Complementar

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — o paper original
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — rasterizador CUDA de qualidade de produção
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — receita de treinamento de referência
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — o formato portátil 2026
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — schema `UsdVolParticleField3DGaussianSplat`
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — visão geral da indústria
