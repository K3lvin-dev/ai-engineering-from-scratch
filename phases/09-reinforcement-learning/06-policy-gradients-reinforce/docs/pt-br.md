# Gradiente de Política — REINFORCE do Zero

> Pare de estimar valor. Parametrize a política diretamente, compute o gradiente do retorno esperado, suba a colina. Williams (1992) escreveu em um teorema. É por que PPO, GRPO e todo loop de RL de LLM existem.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 3 · 03 (Backpropagation), Fase 9 · 03 (Monte Carlo), Fase 9 · 04 (Aprendizado TD)
**Tempo:** ~75 minutos

## O Problema

Q-learning e DQN parametrizam a função de *valor*. Você escolhe ações por `argmax Q`. Isso é fine para ações discretas e estados discretos. Quebra quando ações são contínuas (qual `argmax` sobre um torque 10-dimensional?) ou quando você quer uma política estocástica (`argmax` é determinístico por construção).

Gradientes de política parametrizam a *política* em vez disso. `π_θ(a | s)` é uma rede neural que produz uma distribuição sobre ações. Amostra dela para agir. Compute o gradiente do retorno esperado com relação a `θ`. Suba a colina. Sem `argmax`. Sem recursão de Bellman. Apenas ascensão de gradiente em `J(θ) = E_{π_θ}[G]`.

O teorema REINFORCE (Williams 1992) diz que esse gradiente é computável: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Rode um episódio. Calcule o retorno. Multiplique por `∇ log π_θ(a | s)` em cada passo. Faça média. Ascensão de gradiente. Pronto.

Todo algoritmo LLM-RL em 2026 — PPO, DPO, GRPO — é um refinamento de REINFORCE. Entendê-lo nos seus dedos é o pré-requisito para o resto desta fase, e para a Fase 10 · 07 (implementação RLHF) e Fase 10 · 08 (DPO).

## O Conceito

![Gradiente de política: política softmax, gradiente log-π, atualização ponderada por retorno](../assets/policy-gradient.svg)

**O teorema de gradiente de política.** Para qualquer política `π_θ` parametrizada por `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

onde `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` é o retorno descontado a partir do passo `t`. A expectativa é sobre trajetórias completas `τ` amostradas de `π_θ`.

**A prova é curta.** Diferencie `J(θ) = Σ_τ P(τ; θ) G(τ)` sob a expectativa. Use `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (o truque do log-derivado). Fatore `log P(τ; θ) = Σ log π_θ(a_t | s_t) + termos do ambiente que não dependem de θ`. Os termos do ambiente anulam. Duas linhas de álgebra te dão o teorema.

**Truques de redução de variância.** REINFORCE puro tem variância assassina — retornos são ruidosos, `∇ log π` é ruidoso, o produto deles é muito ruidoso. Duas soluções padrão:

1. **Subtração de baseline.** Substitua `G_t` por `G_t - b(s_t)` para qualquer baseline `b(s_t)` que não dependa de `a_t`. Não-enviesado porque `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Escolha típica: `b(s_t) = V̂(s_t)` aprendido por um crítico → actor-critic (Aula 07).
2. **Retorno-a-partir-daqui.** Substitua `Σ_t G_t · ∇ log π_θ(a_t | s_t)` por `Σ_t G_t^{de t} · ∇ log π_θ(a_t | s_t)`. Apenas retornos futuros importam para uma dada ação — recompensas passadas contribuem ruído de média zero.

Combinados, você tem:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

que é REINFORCE com baseline — o ancestral direto de A2C (Aula 07) e PPO (Aula 08).

**Parametrização softmax da política.** Para ações discretas, a escolha padrão:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

onde `f_θ` é qualquer rede neural que produz um score por ação. O gradiente tem uma forma limpa:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

ou seja, o score da ação tomada menos seu valor esperado sob a política.

**Política Gaussiana para ações contínuas.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` tem forma fechada. É tudo que o SAC da Fase 9 · 07 precisa.

## Construa

### Passo 1: rede de política softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Use uma política linear (um vetor de pesos por ação) para um ambiente tabular. Para Atari, troque por CNN e mantenha a cabeça softmax.

### Passo 2: amostragem e log-probabilidade

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Passo 3: rollout com log-probs capturados

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Passo 4: atualização REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

O gradiente `∇ log π(a|s) = e_a - π(·|s)` (onehot de `a` menos probabilidades) é o cerne dos gradientes de política softmax. Grave na memória muscular.

### Passo 5: baselines

Uma média móvel de `G` sobre episódios recentes é redução de variância suficiente para rodar um GridWorld 4×4; leva ~500 episódios para convergir. Atualize o baseline para um `V̂(s)` aprendido e você ganha actor-critic.

## Armadilhas

- **Gradientes explodindo.** Retornos podem ser enormes. Sempre normalize `G` para `~N(0, 1)` no batch antes de multiplicar por `∇ log π`.
- **Colapso de entropia.** A política converge a uma ação quase-determinística cedo demais, para de explorar, fica presa. Solução: adicione bônus de entropia `β · H(π(·|s))` ao objetivo.
- **Alta variância.** REINFORCE puro precisa de milhares de episódios. Um baseline de crítico (Aula 07) ou a região de confiança de TRPO/PPO (Aula 08) é a solução padrão.
- **Ineficiência em amostras.** On-policy significa que você descarta cada transição após uma atualização. Correções off-policy via amostragem por importância recuperam dados, ao custo de variância (a razão do PPO é um peso IS com clip).
- **Gradientes não-estacionários.** O mesmo gradiente de 100 episódios atrás usa `π` antiga. Métodos on-policy atualizam a cada poucos rollouts por essa razão.
- **Atribuição de crédito.** Sem retorno-a-partir-daqui, recompensas passadas contribuem ruído. Sempre use retorno-a-partir-daqui.

## Use

Em 2026, REINFORCE raramente é rodado diretamente mas sua fórmula de gradiente está em todo lugar:

|| Caso de uso | Método derivado ||
||----------|---------------||
|| Controle contínuo | PPO / SAC com política Gaussiana ||
|| RLHF de LLM | PPO com penalidade KL, rodando em política a nível de token ||
|| Raciocínio de LLM (DeepSeek) | GRPO — REINFORCE com baseline relativo ao grupo, sem crítico ||
|| Multi-agente | REINFORCE com crítico centralizado (MADDPG, COMA) ||
|| Robótica com ações discretas | A2C, A3C, PPO ||
|| Apenas configurações de preferência | DPO — REINFORCE reescrito como perda de verossimilhança de preferência, sem amostragem ||

Quando você lê `loss = -advantage * log_prob` em um script de treino de 2026, isso é REINFORCE com baseline. Papers inteiros (DPO, GRPO, RLOO) são truques de redução de variância sobre essa linha.

## Entregue

Salve como `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produza uma configuração de treino REINFORCE / actor-critic / PPO para uma tarefa dada e diagnostique problemas de variância.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Dado um ambiente (ações discretas / contínuas, horizonte, estatísticas de recompensa), gere:

1. Cabeça da política. Softmax (discreto) ou Gaussiano (contínuo) com contagem de parâmetros.
2. Baseline. Nenhum (puro), média móvel, `V̂(s)` aprendido, ou crítico A2C.
3. Controles de variância. Retorno-a-partir-daqui ligado por padrão, normalização de retorno, valor de clip de gradiente.
4. Bônus de entropia. Coeficiente β e agenda de decaimento.
5. Tamanho do batch. Episódios por atualização; contrato de frescor de dados on-policy.

Recuse REINFORCE-sem-baseline em horizontes > 500 passos. Recuse controle de ação contínuas com cabeça softmax. Sinalize qualquer execução com `β = 0` e entropia observada da política < 0,1 como colapsada em entropia.
```

## Exercícios

1. **Fácil.** Implemente REINFORCE no GridWorld 4×4 com uma política linear softmax. Treine por 1.000 episódios sem baseline. Plote a curva de aprendizado; meça variância (std dos retornos).
2. **Médio.** Adicione um baseline de média móvel. Treine de novo. Compare eficiência de amostras e variância com a execução pura. Quanto o baseline reduz os passos para convergência?
3. **Difícil.** Adicione um bônus de entropia `β · H(π)`. Faça varredura de `β ∈ {0, 0,01, 0,1, 1,0}`. Plote retorno final e entropia da política. Onde fica o ponto ideal nessa tarefa?

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|-----------------|-----------------------||
|| Gradiente de política | "Treine a política diretamente" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; derivado do truque do log-derivado. ||
|| REINFORCE | "O algoritmo PG original" | Williams (1992); retornos Monte Carlo multiplicados por gradiente de log-política. ||
|| Truque do log-derivado" | "Estimador da função score" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; torna gradientes de expectativas tratáveis. ||
|| Baseline | "Redução de variância" | Qualquer `b(s)` subtraída de `G`; não-enviesada porque `E[b · ∇ log π] = 0`. ||
|| Retorno-a-partir-daqui | "Apenas retornos futuros contam" | `G_t^{de t}` em vez de `G_0` completo; correto e de menor variância. ||
|| Bônus de entropia | "Encoraje exploração" | Termo `+β · H(π(·|s))` mantém a política de colapsar. ||
|| On-policy | "Treine no que você acabou de ver" | A expectativa do gradiente é w.r.t. a política atual — não pode reutilizar dados antigos diretamente. ||
|| Vantagem | "Quão melhor que a média" | `A(s, a) = G(s, a) - V(s)`; a quantidade sinal que REINFORCE-com-baseline multiplica. |

## Leituras Complementares

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — o paper original REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — o teorema moderno de gradiente de política com aproximação por função.
- [Sutton & Barto (2018). Cap. 13 — Métodos de Gradiente de Política](http://incompleteideas.net/book/RLbook2020.pdf) — apresentação didática.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — exposição pedagógica clara com código PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — redução de variância e a visão de gradiente natural que conecta REINFORCE à família de regiões de confiança (TRPO, PPO).
