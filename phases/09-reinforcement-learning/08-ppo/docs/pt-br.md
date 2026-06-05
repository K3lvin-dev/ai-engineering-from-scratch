# Proximal Policy Optimization (PPO)

> O A2C joga fora cada rollout após uma atualização. O PPO envolve o gradiente de política em uma razão de importância clipped para que você possa fazer 10+ épocas nos mesmos dados sem a política explodir. Schulman et al. (2017). Ainda o algoritmo de gradiente de política padrão em 2026.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 06 (REINFORCE), Fase 9 · 07 (Actor-Critic)
**Tempo:** ~75 minutos

## O Problema

O A2C (Lição 07) é on-policy: o gradiente `E_{π_θ}[A · ∇ log π_θ]` requer dados amostrados da *atual* `π_θ`. Dê uma atualização, e `π_θ` muda; os dados que você usou agora são off-policy. Reutilize-os e seu gradiente fica viesado.

Rollouts são caros. No Atari, um rollout com 8 ambientes × 128 passos = 1024 transições e uma dúzia de segundos de tempo de ambiente. Jogar isso fora após um passo de gradiente é desperdício.

Trust Region Policy Optimization (TRPO, Schulman 2015) foi a primeira correção: restrinja cada atualização para que a divergência KL entre a política antiga e a nova fique abaixo de `δ`. Teoricamente limpo, mas requer uma solução de gradiente conjugado por atualização. Ninguém roda TRPO em 2026.

O PPO (Schulman et al. 2017) substitui a restrição dura de trust region por um objetivo clipped simples. Uma linha extra de código. Dez épocas por rollout. Sem gradientes conjugados. Garantias teóricas boas o bastante. Nove anos depois, ainda é o algoritmo de gradiente de política padrão para tudo, desde MuJoCo até RLHF.

## O Conceito

![Objetivo surrogate clipped do PPO: clipping da razão em 1 ± ε](../assets/ppo.svg)

**A razão de importância.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Essa é a razão de verossimilhança da nova política vs a política que coletou os dados. `r_t = 1` significa nenhuma mudança. `r_t = 2` significa que a nova política tem o dobro de chance de tomar `a_t` que a antiga.

**O surrogate clipped.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Dois termos:

- Se a vantagem `A_t > 0` e a razão tenta crescer além de `1 + ε`, o clip achata o gradiente — não empurre uma ação boa além de `+ε` acima da probabilidade antiga.
- Se a vantagem `A_t < 0` e a razão tenta crescer além de `1 - ε` (ou seja, tornaríamos uma ação ruim mais provável em relação à sua redução clipped), o clip limita o gradiente — não empurre uma ação ruim abaixo de `-ε`.

O `min` cuida da outra direção: se a razão se moveu na direção *benéfica*, você ainda recebe o gradiente (sem clipping no lado que te prejudicaria).

`ε = 0.2` típico. Plote o objetivo em função de `r_t`: uma função linear por partes com um teto plano no "lado bom" e um piso plano no "lado ruim."

**A perda completa do PPO.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Mesma estrutura actor-critic do A2C. Três coeficientes, geralmente `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**O loop de treinamento.**

1. Colete `N × T` transições em `N` ambientes paralelos por `T` passos cada.
2. Compute vantagens (GAE), congele-as como constantes.
3. Congele `π_{θ_old}` como um snapshot da `π_θ` atual.
4. Para `K` épocas, para cada minibatch de `(s, a, A, V_target, log π_old(a|s))`:
   - Compute `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Aplique `L^{CLIP}` + perda de valor + entropia.
   - Passo de gradiente.
5. Descarte o rollout. Volte ao passo 1.

`K = 10` e minibatches de 64 é um conjunto de hiperparâmetros padrão. PPO é robusto: os números exatos raramente importam dentro de ±50%.

**Variante com penalidade KL.** O paper original propôs uma alternativa usando uma penalidade KL adaptativa: `L = L^{PG} - β · KL(π_θ || π_old)` com `β` ajustado baseado na KL observada. A versão com clipping se tornou dominante; a variante KL sobrevive em RLHF (onde KL para a política de referência é uma restrição separada que você sempre quer de qualquer forma).

## Construa

### Passo 1: capture `log π_old(a | s)` no momento do rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

O snapshot é tirado uma vez, no momento do rollout. Ele não muda durante as épocas de atualização.

### Passo 2: compute vantagens GAE (Lição 07)

Mesmo que no A2C. Normalize entre o lote.

### Passo 3: atualização surrogate clipped

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

O padrão "clipped → gradiente zero" é o coração do PPO. Se a nova política já se afastou demais na direção benéfica, a atualização para.

### Passo 4: valor e entropia

Adicione MSE padrão ao alvo do crítico e um bônus de entropia no ator, igual ao A2C.

### Passo 5: diagnósticos

Três coisas para observar a cada atualização:

- **KL média** `E[log π_old - log π_θ]`. Deve ficar em `[0, 0.02]`. Se passar de `0.1`, reduza `K_EPOCHS` ou `LR`.
- **Fração de clipping** — a fração de amostras cuja razão está fora de `[1-ε, 1+ε]`. Deve ser `~0.1-0.3`. Se `~0`, o clipping nunca dispara → aumente `LR` ou `K_EPOCHS`. Se `~0.5+`, você está sobre-ajustando o rollout → reduza-os.
- **Variância explicada** `1 - Var(V_target - V_pred) / Var(V_target)`. Métrica de qualidade do crítico. Deve subir em direção a 1 conforme o crítico aprende.

## Armadilhas

- **Coeficiente de clipping mal ajustado.** `ε = 0.2` é o padrão de facto. Ir para `0.1` deixa as atualizações muito tímidas; `0.3+` convida instabilidade.
- **Épocas demais.** `K > 20` desestabiliza rotineiramente porque a política se afasta muito de `π_old`. Limite as épocas, especialmente para redes grandes.
- **Sem normalização de recompensa.** Escalas grandes de recompensa invadem o intervalo de clipping. Normalize recompensas (desvio padrão corrente) antes de computar vantagens.
- **Esquecer a normalização da vantagem.** Normalização por lote (média zero, desvio um) é padrão. Pular isso quebra o PPO na maioria dos benchmarks.
- **Taxa de aprendizado sem decaimento.** PPO se beneficia de decaimento linear da LR até zero. LR constante é geralmente pior.
- **Erros de matemática da razão de importância.** Sempre use `exp(log_new - log_old)` para estabilidade numérica, não `new / old`.
- **Sinal do gradiente errado.** Maximizar o surrogate = *minimizar* `-L^{CLIP}`. Um sinal invertido é o bug mais comum do PPO.

## Use

O PPO é o algoritmo RL padrão de 2026 em um número surpreendente de domínios:

| Caso de uso | Variante do PPO |
|-------------|-----------------|
| MuJoCo / controle robótico | PPO com política Gaussiana, GAE(0.95) |
| Atari / jogos discretos | PPO com política categórica, rollouts de 128 passos |
| RLHF para LLMs | PPO com penalidade KL para o modelo de referência, recompensa do RM ao final da resposta |
| Agentes de jogos em larga escala | IMPALA + PPO (AlphaStar, OpenAI Five) |
| LLMs de raciocínio | GRPO (Lição 12) — variante do PPO sem crítico |
| Dados só de preferência | DPO — colapso do PPO+KL em forma fechada, sem amostragem online |

O *formato da perda* do PPO — surrogate clipped + valor + entropia — é o arcabouço para DPO, GRPO e quase todo pipeline de RLHF.

## Entregue

Salve como `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produza uma configuração de treinamento PPO e um plano de diagnóstico para um dado ambiente.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Dado um ambiente e orçamento de treinamento, entregue:

1. Tamanho do rollout. `N` ambientes × `T` passos.
2. Agenda de atualização. `K` épocas, tamanho do minibatch, esquema de LR.
3. Parâmetros do surrogate. `ε` (clip), `c_v`, `c_e`, normalização de vantagem ligada.
4. Vantagem. GAE(`λ`) com `γ` e `λ` explícitos.
5. Plano de diagnóstico. KL, fração de clipping, limites de variância explicada com alertas.

Recuse `K > 30` ou `ε > 0.3` (trust region insegura). Recuse qualquer execução PPO sem normalização de vantagem ou monitoramento de KL/clipping. Sinalize fração de clipping sustentada acima de 0.4 como desvio.
```

## Exercícios

1. **Fácil.** Rode PPO no GridWorld 4×4 com `ε=0.2, K=4`. Compare eficiência de amostras com A2C (uma época por rollout) com o mesmo número de passos de ambiente.
2. **Médio.** Varra `K ∈ {1, 4, 10, 30}`. Plote retorno vs passos de ambiente e acompanhe a KL média por atualização. Em qual `K` a KL explode nesta tarefa?
3. **Difícil.** Substitua o surrogate clipped por uma penalidade KL adaptativa (`β` dobra se `KL > 2·target`, reduz pela metade se `KL < target/2`). Compare retorno final, estabilidade e ausência de clipping.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|---------------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; desvio da política que coletou os dados. |
| Clipped surrogate | "O truque principal do PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; gradiente plano após o clip no lado benéfico. |
| Trust region | "Intenção do TRPO / PPO" | Limitar cada atualização KL para garantir melhoria monotônica. |
| KL penalty | "Trust region suave" | PPO alternativo: `L - β · KL(π_θ \|\| π_old)`. `β` adaptativo. |
| Clip fraction | "Com que frequência o clipping dispara" | Diagnóstico — deve ser 0.1-0.3; fora disso significa mal ajustado. |
| Multi-epoch training | "Reuso de dados" | K épocas em cada rollout; custo de variância trocado por eficiência de amostras. |
| On-policy-ish | "Quase on-policy" | PPO é nominalmente on-policy, mas K>1 épocas usa dados levemente off-policy com segurança. |
| PPO-KL | "O outro PPO" | Variante com penalidade KL; usada em RLHF onde KL-para-referência já é uma restrição. |

## Leitura Complementar

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — o paper.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, predecessor do PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — todos os hiperparâmetros do PPO ablacionados.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; a receita do PPO em RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — exposição moderna e limpa com PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — implementação PPO de referência em arquivo único usada por muitos papers.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — a receita de produção para PPO em modelos de linguagem; leia junto com a Lição 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — o paper das "37 otimizações em nível de código"; quais truques do PPO são essenciais e quais são folclore.
