# Métodos Monte Carlo — Aprendendo de Episódios Completos

> Programação dinâmica precisa de um modelo. Monte Carlo não precisa de nada além de episódios. Rode a política, observe os retornos, faça a média. A ideia mais simples em RL — e a que destrava tudo que vem depois.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 01 (MDPs), Fase 9 · 02 (Programação Dinâmica)
**Tempo:** ~75 minutos

## O Problema

Programação dinâmica é elegante, mas pressupõe que você pode consultar `P(s' | s, a)` para cada estado e ação. Quase nada no mundo real funciona assim. Um robô não consegue computar analiticamente a distribuição sobre pixels da câmera após um torque de junta. Um algoritmo de precificação não consegue integrar sobre cada possível reação de cliente. Um LLM não consegue enumerar todas as continuações possíveis após um token.

Você precisa de um método que só precise da capacidade de *amostrar* do ambiente. Rode a política. Pegue uma trajetória `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Use-a para estimar valores. Isso é Monte Carlo.

A mudança de DP para MC é filosoficamente importante: passamos de *modelo conhecido + backup exato* para *rollouts amostrados + retorno médio*. A variância pula, mas a aplicabilidade explode. Todo algoritmo de RL depois desta aula — TD, Q-learning, REINFORCE, PPO, GRPO — é um estimador Monte Carlo no fundo, às vezes com bootstrap sobreposto.

## O Conceito

![Monte Carlo: rollout, calcule retornos, faça média; primeira-visita vs toda-visita](../assets/monte-carlo.svg)

**A ideia central, em uma linha:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` onde `G^{(i)}(s)` são retornos observados seguindo visitas a `s` sob a política `π`.

**Primeira-visita vs toda-visita MC.** Dado um episódio que visita o estado `s` múltiplas vezes, primeira-visita MC só conta o retorno da primeira visita; toda-visita MC conta todas. Ambos são não-enviesados no limite. Primeira-visita é mais simples de analisar (amostras iid). Toda-visita usa mais dados por episódio e geralmente converge mais rápido na prática.

**Média incremental.** Em vez de armazenar todos os retôrnos, atualize a média corrente:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Reorganize: `V_new = V_old + α · (target - V_old)` com `α = 1/n`. Troque `1/n` por uma taxa de passo constante `α ∈ (0, 1)` e você tem um estimador MC não-estacionário que rastreia mudanças em `π`. Esse movimento é o pulo inteiro de MC para TD para todo algoritmo moderno de RL.

**Exploração agora é um problema.** DP tocou cada estado por enumeração. MC só vê estados que a política visita. Se `π` é determinística, regiões inteiras do espaço de estado nunca são amostradas, e suas estimativas de valor ficam em zero para sempre. Três soluções, em ordem histórica:

1. **Inícios explorando.** Comece cada episódio de um par (s, a) aleatório. Garante cobertura; irrealista na prática (você não pode "resetar" um robô em um estado arbitrário).
2. **ε-guloso.** Aja guloso w.r.t. Q atual, mas com probabilidade `ε` escolha uma ação aleatória. Todos os pares estado-ação são amostrados assintoticamente.
3. **MC off-policy.** Colete dados sob uma política de comportamento `μ`, aprenda sobre a política alvo `π` via amostragem por importância. Alta variância, mas é a ponte para métodos com replay buffer como DQN.

**Controle Monte Carlo.** Avalie → melhore → avalie, assim como iteração de política, mas a avaliação é baseada em amostragem:

1. Rode `π`, pegue um episódio.
2. Atualize `Q(s, a)` dos retornos observados.
3. Torne `π` ε-gulosa w.r.t. `Q`.
4. Repita.

Converge a `Q*` e `π*` com probabilidade 1 sob condições suaves (todo par visitado infinitamente, `α` satisfaz Robbins-Monro).

## Construa

### Passo 1: rollout → lista de (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Sem modelo, apenas `env.reset()` e `env.step(s, a)`. Mesma interface de um ambiente gym mas simplificada.

### Passo 2: calcule retornos (passada reversa)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Uma passada, `O(T)`. A recorrência reversa `G_t = r_{t+1} + γ G_{t+1}` evita refazer somas.

### Passo 3: avaliação MC de primeira-visita

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Três linhas fazem o trabalho: marque estado como visto na primeira visita, incremente contagem, atualize média corrente.

### Passo 4: MC control ε-guloso (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Passo 5: compare com o padrão ouro de DP

Sua estimativa MC de `V^π` deve concordar com o resultado de DP da Aula 02 conforme episódios → ∞. Na prática: 50.000 episódios no GridWorld 4×4 te aproximam a `~0,1` da resposta de DP.

## Armadilhas

- **Episódios infinitos.** MC requer que episódios *terminem*. Se sua política pode rodar para sempre, limite `max_steps` e trate o limite como falha implícita. GridWorld com política aleatória frequentemente estoura o tempo — isso é normal, apenas certifique-se de contar corretamente.
- **Variância.** MC usa retornos completos. Em episódios longos, a variância é enorme — uma recompensa desafortunada no final muda `V(s_0)` pela mesma quantidade. Métodos TD (Aula 04) cortam isso com bootstrap.
- **Cobertura de estados.** MC guloso com Q novo e empates só tentará uma ação. Você *precisa* explorar (ε-guloso, inícios explorando, UCB).
- **Políticas não-estacionárias.** Se `π` muda (como em MC control), retôrnos antigos são de uma política diferente. MC de α constante lida com isso; MC de média-amostral não.
- **Importância amostral off-policy.** Os pesos `π(a|s)/μ(a|s)` multiplicam ao longo de uma trajetória. A variância explode com o horizonte. Limite com IS ponderado por decisão ou mude para TD.

## Use

O papel de 2026 dos métodos Monte Carlo:

|| Caso de uso | Por que MC ||
||----------|--------||
|| Jogos de curto horizonte (blackjack, pôquer) | Episódios terminam naturalmente; retornos são limpos. ||
|| Avaliação offline de uma política logada | Média de retornos descontados sobre trajetórias armazenadas. ||
|| Monte Carlo Tree Search (AlphaZero) | Rollouts MC das folhas da árvore guiam a seleção. ||
|| Avaliação de RL para LLM | Calcule recompensa média sobre conclusões amostradas para uma política dada. ||
|| Estimação de baseline em PPO | O alvo de vantagem `A_t = G_t - V(s_t)` usa um MC `G_t`. ||
|| Ensinar RL | Algoritmo mais simples que realmente funciona — remova bootstrap para ver o cerne. ||

Algoritmos modernos de RL profundo (PPO, SAC) interpolam entre MC puro (retornos completos) e TD puro (bootstrap de um passo) via retornos `n`-step ou GAE. Ambas as pontas são instâncias do mesmo estimador.

## Entregue

Salve como `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Avalie uma política via rollouts Monte Carlo e gere um relatório de convergência com comparação DP se disponível.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Dado um ambiente (episódico, com API de reset+step) e uma política, gere:

1. Método. Primeira-visita vs toda-visita MC. Razão.
2. Orçamento de episódios. Número-alvo, diagnóstico de variância, erro padrão esperado.
3. Plano de exploração. Agenda de ε (se necessário) ou inícios explorando.
4. Comparação padrão ouro. V* ótimo por DP se tabular; caso contrário uma cota de um baseline Q-learning / PPO.
5. Verificação de terminação. Limite de passos máximos, timeouts, tratamento de trajetórias não-terminantes.

Recuse rodar MC em tarefas não-episódicas sem um limite de horizonte finito. Recouse reportar estimativas de V^π com menos de 100 episódios por estado para tarefas tabulares. Sinalize qualquer política com ações de variância-zero como risco de exploração.
```

## Exercícios

1. **Fácil.** Implemente avaliação MC de primeira-visita da política aleatória uniforme no GridWorld 4×4. Rode 10.000 episódios. Plote `V(0,0)` como função do número de episódios contra a resposta de DP.
2. **Médio.** Implemente MC control ε-guloso com `ε ∈ {0,01, 0,1, 0,3}`. Compare retorno médio após 20.000 episódios. Como é a curva? Onde fica o tradeoff viés-variância?
3. **Difícil.** Implemente MC *off-policy* com amostragem por importância: colete dados sob política aleatória uniforme `μ`, estime `V^π` para a política ótima determinística `π`. Compare IS simples vs IS ponderado por decisão vs IS ponderado. Qual tem menor variância?

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|-----------------|-----------------------||
|| Monte Carlo | "Amostragem aleatória" | Estime expectativas fazendo média sobre amostras iid da distribuição. ||
|| Retorno `G_t` | "Recompensa futura" | Soma de recompensas descontadas do passo `t` ao fim do episódio: `Σ_{k≥0} γ^k r_{t+k+1}`. ||
|| MC de primeira-visita | "Conte cada estado uma vez" | Apenas a primeira visita em um episódio contribui para a estimativa de valor. ||
|| MC de toda-visita | "Use todas as visitas" | Toda visita contribui; ligeiramente enviesado mas mais eficiente em amostras. ||
|| ε-guloso | "Ruído de exploração" | Escolha ação gulosa com prob `1-ε`; ação aleatória com prob `ε`. ||
|| Amostragem por importância | "Corrigir por amostrar da distribuição errada" | Repese retornos por produtos de `π(a|s)/μ(a|s)` para estimar `V^π` a partir de dados de `μ`. ||
|| On-policy | "Aprenda dos meus próprios dados" | Política alvo = política de comportamento. MC puro, PPO, SARSA. ||
|| Off-policy | "Aprenda dos dados de outra pessoa" | Política alvo ≠ política de comportamento. MC com importância amostral, Q-learning, DQN. |

## Leituras Complementares

- [Sutton & Barto (2018). Cap. 5 — Métodos Monte Carlo](http://incompleteideas.net/book/RLbook2020.pdf) — o tratamento canônico.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — análise primeira-visita vs toda-visita.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — MC off-policy e controle de variância.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — estimadores IS de baixa variância modernos.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — a primeira demonstração empírica em larga escala de MC/TD auto-jogo convergindo a jogo supra-humano; precursor conceitual de todas as aulas na segunda metade desta fase.
