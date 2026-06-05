# Actor-Critic — A2C e A3C

> O REINFORCE é barulhento. Adicione um crítico que aprende `V̂(s)`, subtraia do retorno, e você obtém uma vantagem com a mesma expectativa mas variância muito menor. Isso é o actor-critic. O A2C roda de forma síncrona; o A3C roda em múltiplas threads. Ambos são o modelo mental para todo método moderno de deep-RL.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 04 (TD Learning), Fase 9 · 06 (REINFORCE)
**Tempo:** ~75 minutos

## O Problema

O REINFORCE clássico funciona, mas a variância é terrível. Os retornos Monte Carlo `G_t` podem variar por um fator de 10 entre episódios. Multiplicar esse ruído por `∇ log π` e tirar a média produz um estimador de gradiente que precisa de milhares de episódios para mover a política a mesma distância que você moveria com bem menos atualizações de DQN.

A variância vem do uso de retornos brutos. Se você subtrai uma linha de base `b(s_t)` — qualquer função do estado, incluindo um valor aprendido — a expectativa não muda e a variância cai. A melhor linha de base tratável é `V̂(s_t)`. Agora a quantidade que multiplica `∇ log π` é a *vantagem*:

`A(s, a) = G - V̂(s)`

Uma ação é boa se gerou retorno acima da média; ruim se abaixo da média. REINFORCE com um crítico aprendido é *actor-critic*. O crítico dá ao ator um professor de baixa variância. Esse é todo método de deep-policy depois de 2015 (A2C, A3C, PPO, SAC, IMPALA).

## O Conceito

![Actor-critic: rede de política + rede de valor, residual TD como vantagem](../assets/actor-critic.svg)

**Duas redes, uma perda compartilhada:**

- **Ator** `π_θ(a | s)`: a política. Amostrada para agir. Treinada com gradiente de política.
- **Crítico** `V_φ(s)`: estima o retorno esperado a partir do estado. Treinado para minimizar `(V_φ(s) - alvo)²`.

**A vantagem.** Duas formas padrão:

- *Vantagem MC:* `A_t = G_t - V_φ(s_t)`. Não viesada, variância maior.
- *Vantagem TD:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Viesada (usa `V_φ`), variância muito menor. Também chamada de *residual TD* `δ_t`.

**Vantagem n-passos.** Interpola entre as duas:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` é TD puro. `n = ∞` é MC. A maioria das implementações usa `n = 5` para Atari, `n = 2048` para PPO no MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) propôs uma média ponderada exponencial sobre todas as vantagens n-passos:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

com `λ ∈ [0, 1]`. `λ = 0` é TD (baixa variância, alto viés). `λ = 1` é MC (alta variância, não viesado). `λ = 0.95` é o padrão em 2026 — ajuste até o dial viés/variância estar onde você quer.

**A2C: advantage actor-critic síncrono.** Colete `T` passos em `N` ambientes paralelos. Compute vantagens para cada passo. Atualize ator e crítico no lote combinado. Repita. O irmão mais simples e mais escalável do A3C.

**A3C: advantage actor-critic assíncrono.** Mnih et al. (2016). Dispara `N` threads trabalhadoras, cada uma rodando um ambiente. Cada trabalhadora computa gradientes localmente em seu próprio rollout, depois aplica assincronamente a um servidor de parâmetros compartilhado. Sem replay buffer — as trabalhadoras desc correlacionam rodando trajetórias diferentes. A3C provou que você podia treinar em CPUs em escala. Em 2026, A2C baseado em GPU (ambientes paralelos em lote) domina porque GPUs querem lotes grandes.

**A perda combinada.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Três termos: perda de gradiente de política, regressão de valor, bônus de entropia. `c_v ~ 0.5`, `c_e ~ 0.01` são pontos de partida canônicos.

## Construa

### Passo 1: um crítico

Crítico linear `V_φ(s) = w · features(s)` atualizado com MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Em um ambiente tabular o crítico converge em algumas centenas de episódios. No Atari, substitua o crítico linear por um tronco CNN compartilhado + cabeça de valor.

### Passo 2: vantagem n-passos

Dado um rollout de comprimento `T` e um `V(s_T)` final bootstrapado:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` é o alvo do crítico. `advantages` é o que multiplica `∇ log π`.

### Passo 3: atualização combinada

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, um rollout por atualização, taxas de aprendizado separadas para ator e crítico.

### Passo 4: paralelização (A3C vs A2C)

- **A3C:** inicie `N` threads. Cada uma roda seu próprio ambiente e seu próprio forward pass. Periodicamente envie atualizações de gradiente para um mestre compartilhado. Sem locks no mestre — corridas são ok, só adicionam ruído.
- **A2C:** rode `N` instâncias de ambiente em um único processo, empilhe observações em um lote `[N, obs_dim]`, forward pass em lote, backward pass em lote. Maior utilização de GPU, determinístico, mais fácil de raciocinar. O padrão em 2026.

Nosso código de brinquedo é single-threaded para clareza; reescrever para A2C em lote são três linhas de numpy.

## Armadilhas

- **Viés do crítico antes do gradiente do ator.** Se o crítico é aleatório, sua linha de base não é informativa e você está treinando em ruído puro. Aqueça o crítico por algumas centenas de passos antes de ligar o gradiente de política, ou use uma taxa de aprendizado lenta para o ator.
- **Normalização da vantagem.** Normalize vantagens para média zero / desvio padrão um por lote. Estabiliza o treinamento imensamente com custo quase zero.
- **Tronco compartilhado.** Use um extrator de features compartilhado para ator e crítico em entradas de imagem. Cabeças separadas. As features compartilhadas pegam carona em ambas as perdas.
- **Contrato on-policy.** A2C reusa dados para exatamente uma atualização. Mais do que isso e seu gradiente fica viesado (correção de importance sampling é o que o PPO adiciona).
- **Colapso de entropia.** Sem `c_e > 0`, a política fica quase determinística em algumas centenas de atualizações e para de explorar.
- **Escala da recompensa.** As magnitudes da vantagem dependem da escala da recompensa. Normalize recompensas (ex.: dividindo pelo desvio padrão corrente) para magnitudes de gradiente consistentes entre tarefas.

## Use

A2C/A3C raramente são a escolha final em 2026, mas são a arquitetura que tudo depois refina:

| Método | Relação com A2C |
|--------|----------------|
| PPO | A2C + razão de importância clipped para atualizações multi-época |
| IMPALA | A3C + correção off-policy V-trace |
| SAC (Fase 9 · 07) | A2C off-policy com um crítico de valor suave (próxima lição) |
| GRPO (Fase 9 · 12) | A2C sem o crítico — vantagem relativa ao grupo |
| DPO | A2C colapsado em uma perda de ranqueamento de preferência, sem amostragem |
| AlphaStar / OpenAI Five | A2C com treinamento em liga + pre-treinamento por imitação |

Se você vir "vantagem" em um artigo de 2026, pense actor-critic.

## Entregue

Salve como `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produza uma configuração A2C / A3C / GAE para um dado ambiente, com estimação de vantagem e pesos de perda especificados.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Dado um ambiente e orçamento de computação, entregue:

1. Paralelismo. A2C (GPU em lote) vs A3C (CPU assíncrono) e número de workers.
2. Comprimento do rollout T. Passos por ambiente por atualização.
3. Estimador de vantagem. n-passos ou GAE(λ); especifique λ.
4. Pesos da perda. `c_v` (valor), `c_e` (entropia), gradiente clip.
5. Taxas de aprendizado. Ator e crítico (separados se usando).

Recuse A2C com worker único em ambientes com horizonte > 1000 (muito on-policy, muito lento). Recuse entregar sem normalização de vantagem. Sinalize qualquer execução com `c_e = 0` e entropia observada < 0.1 como colapso de entropia.
```

## Exercícios

1. **Fácil.** Treine actor-critic com vantagem MC (`G_t - V(s_t)`) no GridWorld 4×4. Compare eficiência de amostras com o REINFORCE-com-linha-de-base-média-móvel da Lição 06.
2. **Médio.** Troque para vantagem por residual TD (`r + γ V(s') - V(s)`). Meça a variância dos lotes de vantagem. Quanto ela cai?
3. **Difícil.** Implemente GAE(λ). Varra `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Plote retorno final vs eficiência de amostras. Onde está o ponto ideal viés/variância para esta tarefa?

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|------|-------------|---------------------------|
| Actor | "A rede de política" | `π_θ(a\|s)`, atualizada por gradiente de política. |
| Critic | "A rede de valor" | `V_φ(s)`, atualizada por regressão MSE para retornos / alvos TD. |
| Advantage | "Quanto melhor que a média" | `A(s, a) = Q(s, a) - V(s)` ou seus estimadores. Multiplicador para `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; estimativa de vantagem de um passo. |
| GAE | "O knob de interpolação" | Soma ponderada exponencial de vantagens n-passos, parametrizada por `λ`. |
| A2C | "Actor-critic síncrono" | Em lote entre ambientes; um passo de gradiente por rollout. |
| A3C | "Actor-critic assíncrono" | Threads trabalhadoras empurram gradientes para um servidor de parâmetros compartilhado. Paper original; menos comum em 2026. |
| Bootstrap | "Use V no horizonte" | Trunque o rollout, adicione `γ^n V(s_{t+n})` para fechar a soma. |

## Leitura Complementar

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, o paper original de actor-critic assíncrono.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — fundamentos; combine com o Cap. 9 sobre aproximação de funções quando o crítico é uma rede neural.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — actor-critic distribuído escalável com correção off-policy V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — implementações de produção de A2C/PPO que valem a pena ler.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — o resultado fundamental de convergência para a decomposição actor-critic em duas escalas de tempo.
