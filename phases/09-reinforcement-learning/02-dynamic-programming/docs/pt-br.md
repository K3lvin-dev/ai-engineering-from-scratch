# Programação Dinâmica — Iteração de Política e Iteração de Valor

> Programação dinâmica é RL com trapaça. Você já conhece as funções de transição e recompensa; apenas itera a equação de Bellman até `V` ou `π` parar de mudar. É o benchmark que todo método baseado em amostragem tenta se aproximar.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 01 (MDPs)
**Tempo:** ~75 minutos

## O Problema

Você tem um MDP com modelo conhecido: pode consultar `P(s' | s, a)` e `R(s, a, s')` para qualquer par estado-ação. Um gerente de estoque conhece a distribuição de demanda. Um jogo de tabuleiro tem transições determinísticas. Um gridworld são quatro linhas de Python. Você tem um *modelo*.

RL sem modelo (Q-learning, PPO, REINFORCE) foi inventado para o caso onde você não tem modelo — só pode amostrar do ambiente. Mas quando você tem um, existem métodos mais rápidos e melhores: programação dinâmica. Bellman os projetou em 1957. Ainda definem correção: quando as pessoas dizem "política ótima para este MDP", elas querem dizer a política que DP retornaria.

Você precisa deles em 2026 por três razões. Primeiro, todo ambiente tabular em pesquisa de RL (GridWorld, FrozenLake, CliffWalking) é resolvido com DP para produzir a política padrão ouro. Segundo, valores exatos permitem *debugar* métodos de amostragem: se a estimativa de Q-learning para `V*(s_0)` discorda da resposta de DP em 30%, seu Q-learning tem um bug. Terceiro, métodos modernos de RL offline e planejamento (MCTS, busca do AlphaZero, RL baseado em modelo na Fase 9 · 10) todos iteram um backup de Bellman sobre um modelo aprendido ou dado.

## O Conceito

![Iteração de política e iteração de valor, lado a lado](../assets/dp.svg)

**Dois algoritmos, ambos iteração de ponto fixo em Bellman.**

**Iteração de política.** Alterna dois passos até a política parar de mudar.

1. *Avaliação:* dada política `π`, compute `V^π` aplicando repetidamente `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` até convergir.
2. *Melhoria:* dado `V^π`, torne `π` gulosa w.r.t. `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Convergência é garantida porque (a) cada passo de melhoria ou mantém `π` igual ou aumenta estritamente `V^π` para algum estado, (b) o espaço de políticas determinísticas é finito. Geralmente converge em ~5-20 iterações externas mesmo para grandes espaços de estado.

**Iteração de valor.** Colapsa avaliação e melhoria em uma passada. Aplique a equação de *otimalidade* de Bellman:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Repita até `max_s |V_{new}(s) - V(s)| < ε`. Extraia a política no final tomando a ação gulosa. Estritamente mais rápida por iteração — sem loop interno de avaliação — mas geralmente precisa de mais iterações para convergir.

**Iteração de política generalizada (GPI).** O enquadramento unificador. Função de valor e política estão presas em um loop de melhoria bidirecional; qualquer método que conduza ambos à consistência mútua (iteração de valor asíncrona, iteração de política modificada, Q-learning, actor-critic, PPO) é uma instância de GPI.

**Por que `γ < 1` importa.** O operador de Bellman é uma `γ`-contração na norma sup: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Contração implica ponto fixo único e convergência geométrica. Remova `γ < 1` e você perde a garantia — precisa de um horizonte finito ou um estado terminal absorvente.

## Construa

### Passo 1: construa o modelo MDP do GridWorld

Use o mesmo GridWorld 4×4 da Aula 01. Adicionamos uma variante estocástica: com probabilidade `0,1` o agente desliza para uma direção perpendicular aleatória.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` retorna uma lista de `(s', r, p)`. Esse é todo o modelo.

### Passo 2: avaliação de política

Dada uma política `π(s) = {ação: prob}`, itere a equação de Bellman até `V` parar de mudar:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Passo 3: melhoria de política

Substitua `π` pela política gulosa w.r.t. `V`. Se `π` não mudou, retorne — estamos no ótimo.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Passo 4: junte tudo

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # início arbitrário
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Convergência típica em 4×4: 4-6 iterações externas. Produz `V*(0,0) ≈ -6` e uma política que reduz estritamente a contagem de passos.

### Passo 5: iteração de valor (a versão de um loop)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Mesmo ponto fixo, menos linhas de código.

## Armadilhas

- **Esquecer de tratar terminais.** Se você aplica Bellman a um estado absorvente, ele ainda escolhe uma "melhor ação" que não muda nada. Proteja com `if s == terminal: V[s] = 0`.
- **Convergência sup vs L2.** Use `max |V_new - V|`, não média. A garantia teórica é na norma sup.
- **Atualização in-place vs síncrona.** Atualizar `V[s]` in-place (Gauss-Seidel) converge mais rápido que um dict separado `V_new` (Jacobi). Código de produção usa in-place.
- **Empates de política.** Se duas ações têm valor Q igual, `argmax` pode quebrar empates diferente a cada iteração, fazendo a verificação de "política estável" oscilar. Use um desempate estável (primeira ação em ordem fixa).
- **Explosão de espaço de estados.** DP é `O(|S| · |A|)` por passada. Funciona até ~10⁷ estados. Acima disso, precisa de aproximação por função (Fase 9 · 05 em diante).

## Use

Em 2026, DP é o baseline de correção e o loop interno de planejadores:

|| Caso de uso | Método ||
||----------|--------||
|| Resolver um MDP tabular pequeno exatamente | Iteração de valor (mais simples) ou iteração de política (menos iterações externas) ||
|| Verificar uma implementação de Q-learning / PPO | Compare com V* ótimo por DP em um ambiente de brinquedo ||
|| RL baseado em modelo (Fase 9 · 10) | Backup de Bellman sobre um modelo de transição aprendido ||
|| Planejamento em AlphaZero / MuZero | Monte Carlo Tree Search = backup de Bellman asíncrono ||
|| RL offline (CQL, IQL) | Q-iteração conservadora — DP com penalidade em ações OOD ||

Cada vez que alguém diz "função de valor ótima", quer dizer "ponto fixo de DP". Quando você vê `V*` ou `Q*` em um paper, imagine esse loop.

## Entregue

Salve como `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Resolva um MDP tabular pequeno exatamente via iteração de política ou iteração de valor. Reporte comportamento de convergência.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Dado um MDP com modelo conhecido, gere:

1. Escolha. Iteração de política vs iteração de valor. Razão atrelada a |S|, |A|, γ.
2. Inicialização. V_0, política inicial. Sensibilidade à convergência.
3. Parada. Tolerância sup-número ε. Número esperado de passadas.
4. Verificação. V*(s_0) calculado exatamente. Política gulosa extraída.
5. Uso. Como esse baseline será usado para debugar/avaliar métodos baseados em amostragem.

Recuse rodar DP em espaços de estado > 10⁷. Recuse afirmar convergência sem verificação sup-número. Sinalize qualquer γ ≥ 1 em tarefa de horizonte infinito como violação de garantia.
```

## Exercícios

1. **Fácil.** Rode iteração de valor no GridWorld 4×4 com `γ ∈ {0,9, 0,99}`. Quantas passadas até `max |ΔV| < 1e-6`? Imprima `V*` como um grid 4×4.
2. **Médio.** Compare iteração de política vs iteração de valor no GridWorld *estocástico* (probabilidade de deslizar `0,1`). Conte: passadas, tempo de relógio, `V*(0,0)` final. Qual converge mais rápido em iterações? Em tempo de relógio?
3. **Difícil.** Construa iteração de política modificada: no passo de avaliação, rode apenas `k` passadas em vez de até convergência. Plote o erro de `V*(0,0)` vs `k` para `k ∈ {1, 2, 5, 10, 50}`. O que a curva te diz sobre o tradeoff avaliação/melhoria?

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|-----------------|-----------------------||
|| Iteração de política | "Algoritmo DP" | Alterna avaliação (`V^π`) e melhoria (gulosa `π` w.r.t. `V^π`) até a política parar de mudar. ||
|| Iteração de valor | "DP mais rápido" | Backup de otimalidade de Bellman aplicado em uma passada; converge a `V*` geometricamente. ||
|| Operador de Bellman | "A recursão" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; uma `γ`-contração na norma sup. ||
|| Contração | "Por que DP converge" | Qualquer operador `T` com `||T x - T y|| ≤ γ ||x - y||` tem um ponto fixo único. ||
|| GPI | "Tudo é DP" | Iteração de Política Generalizada: qualquer método que conduza `V` e `π` à consistência mútua. ||
|| Atualização síncrona | "Estilo Jacobi" | Use `V` antigo durante toda a passada; analisável mas mais lento. ||
|| Atualização in-place | "Estilo Gauss-Seidel" | Use `V` conforme está sendo atualizado; converge mais rápido na prática. |

## Leituras Complementares

- [Sutton & Barto (2018). Cap. 4 — Programação Dinâmica](http://incompleteideas.net/book/RLbook2020.pdf) — apresentação canônica de iteração de política e iteração de valor.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — tratamento rigoroso de argumentos de mapeamento de contração.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — iteração de política modificada e sua análise de convergência.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — o paper original de iteração de política.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — a ponte de DP para DP aproximado / RL profundo usada por todas as aulas seguintes.
