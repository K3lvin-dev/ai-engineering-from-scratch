# Diferença Temporal — Q-Learning e SARSA

> Monte Carlo espera até o episódio terminar. TD atualiza após cada passo fazendo bootstrap da estimativa de valor seguinte. Q-learning é off-policy e otimista; SARSA é on-policy e cauteloso. Ambos são uma linha de código. Ambos sustentam todo método de RL profundo nesta fase.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 01 (MDPs), Fase 9 · 02 (Programação Dinâmica), Fase 9 · 03 (Monte Carlo)
**Tempo:** ~75 minutos

## O Problema

Monte Carlo funciona mas tem duas demandas caras. Precisa de episódios que terminem, e só atualiza depois que o retorno final está no bolso. Se seu episódio tem 1.000 passos, MC espera 1.000 passos para atualizar qualquer coisa. É alta variância, baixo viés, e lento na prática.

Programação dinâmica tem o perfil oposto — backups bootstrapados de zero-varância — mas requer um modelo conhecido.

Aprendizado por diferença temporal (TD) divide a diferença. A partir de uma transição única `(s, a, r, s')`, forme um alvo de um passo `r + γ V(s')` e empurre `V(s)` na direção dele. Sem modelo. Sem episódios completos. Viés por usar um `V` aproximado no LHS, mas variância drasticamente menor que MC e atualizações online desde o primeiro passo.

Este é o pivô sobre o qual todo o RL moderno — DQN, A2C, PPO, SAC — gira. O resto da Fase 9 são camadas de aproximação por função e truques construídos sobre a atualização TD de um passo que você escreverá nesta aula.

## O Conceito

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**A atualização TD(0) para V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

A quantidade entre colchetes é o erro TD `δ = r + γ V(s') - V(s)`. É o análogo online de `G_t - V(s_t)` no MC. Convergência requer `α` satisfazendo Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) e todos os estados visitados infinitamente.

**Q-learning.** Um método TD off-policy para controle:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

O `max` assume que a política *gulosa` será seguida de `s'` em diante, independentemente de qual ação o agente realmente toma. Essa desacoplação faz o Q-learning aprender `Q*` enquanto o agente explora via ε-guloso. Mnih et al. (2015) converteu isso em Q-learning profundo no Atari (Aula 05).

**SARSA.** Um método TD on-policy:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

O nome é a tupla `(s, a, r, s', a')`. SARSA usa a ação `a'` que o agente *realmente* toma depois, não o `argmax` guloso. Converge a `Q^π` para qualquer ε-guloso `π` que esteja rodando, que no limite `ε → 0` se torna `Q*`.

**A diferença do cliff-walking.** Na tarefa clássica de cliff-walking (cair no precipício = recompensa -100), Q-learning aprende o caminho ótimo ao longo da borda do precipício mas ocasionalmente toma a penalidade durante exploração. SARSA aprende um caminho mais seguro um passo do precipício porque fatora o ruído de exploração em seu valor Q. Com treino, ambos atingem o ótimo em `ε → 0`. Na prática isso importa: quando a exploração realmente acontece em implantação, o comportamento do SARSA é mais conservador.

**SARSA esperado.** Substitua `Q(s', a')` pelo seu valor esperado sob `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Menor variância que SARSA (sem amostragem de `a'`), mesmo alvo on-policy. Geralmente o padrão em livros didáticos modernos.

**TD n-step e TD(λ).** Interpole entre TD(0) e MC esperando `n` passos antes de bootstrap. `n=1` é TD, `n=∞` é MC. TD(λ) faz média sobre todos os `n` com pesos geométricos `(1-λ)λ^{n-1}`. A maioria do RL profundo usa `n` entre 3 e 20.

## Construa

### Passo 1: SARSA na política ε-gulosa

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Oito linhas. A *única* diferença do Q-learning é a linha do alvo.

### Passo 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

O `max` desacopla o alvo do comportamento. Esse único símbolo é a diferença entre on-policy e off-policy.

### Passo 3: curvas de aprendizado

Rastreie retorno médio a cada 100 episódios. Q-learning converge mais rápido em GridWorld determinístico simples; SARSA é mais conservador no cliff-walking. No GridWorld 4×4 de `code/main.py`, ambos estão perto do ótimo após ~2.000 episódios com `α=0,1, ε=0,1`.

### Passo 4: compare com a verdade de DP

Rode iteração de valor (Aula 02) para obter `Q*`. Verifique `max_{s,a} |Q_aprendido(s,a) - Q*(s,a)|`. Um agente TD tabular saudável chega a `~0,5` no GridWorld 4×4 após 10.000 episódios.

## Armadilhas

- **Valores Q iniciais importam.** Inicialização otimista (`Q = 0` para uma tarefa de recompensa negativa) encoraja exploração. Inicialização pessimista pode prender uma política gulosa para sempre.
- **Agenda de α.** `α` constante é fine para problemas não-estacionários. Decaimento `α_n = 1/n` dá convergência na teoria mas é lento demais na pratique — fixe `α` em `[0,05, 0,3]` e monitore a curva de aprendizado.
- **Agenda de ε.** Comece alto (`ε=1,0`), decaia para `ε=0,05`. "GLIE" (guloso no limite com exploração infinita) é a condição de convergência.
- **Viés de máximo no Q-learning.** O operador `max` é enviesado para cima quando `Q` é ruidoso. Leva a superestimação — Double Q-learning de Hasselt (usado pelo DDQN na Aula 05) corrige isso com duas tabelas Q.
- **Episódios não-terminantes.** TD pode aprender sem terminais, mas você precisa ou limitar passos ou tratar bootstrap corretamente no limite. Padrão: trate o limite como não-terminal, continue bootstrap.
- **Hash de estados.** Se estados são tuplas/tensores, use uma chave hashável (tupla, não lista; tupla de floats arredondados, não bruta).

## Use

O cenário TD de 2026:

|| Tarefa | Método | Razão ||
||------|--------|--------||
|| Ambientes tabulares pequenos | Q-learning | Aprende política ótima diretamente ||
|| On-policy crítico em segurança | SARSA / SARSA esperado | Conservador durante exploração ||
|| Estado de alta dimensionalidade | DQN (Fase 9 · 05) | Função Q neural com replay e rede-alvo ||
|| Ações contínuas | SAC / TD3 (Fase 9 · 07) | Atualização TD em Q-network; rede de política emite ações ||
|| RL de LLM (baseado em modelo de recompensa) | PPO / GRPO (Fase 9 · 08, 12) | Actor-critic com vantagem estilo TD via GAE ||
|| RL offline | CQL / IQL (Fase 9 · 08) | Q-learning com regularização conservadora ||

Noventa por cento do "RL" que você lê em papers de 2026 é alguma elaboração de Q-learning ou SARSA. Entenda a atualização tabular nos seus dedos antes de ler mais fundo.

## Entregue

Salve como `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Escolha entre Q-learning, SARSA, SARSA esperado para uma tarefa tabular ou de pequenas features de RL.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Dado um ambiente tabular ou de pequenas features, gere:

1. Algoritmo. Q-learning / SARSA / SARSA esperado / variante n-step. Uma frase de razão ligada a on-policy vs off-policy e variância.
2. Hiperparâmetros. α, γ, ε, agenda de decaimento.
3. Inicialização. Valor Q_0 (otimista vs zero) e justificativa.
4. Diagnóstico de convergência. Curva de aprendizado alvo, verificação `|Q - Q*|` se DP for possível.
5. Ressalva de implantação. Como a exploração se comportará em inferência? A conservadorismo do SARSA é necessário?

Recuse aplicar TD tabular a espaços de estado > 10⁶. Recuse lançar um agente Q-learning sem ressalva de viés de máximo. Sinalize qualquer agente treinado com ε mantido em 1,0 o tempo todo (sem fase de exploração).
```

## Exercícios

1. **Fácil.** Implemente Q-learning e SARSA no GridWorld 4×4. Plote curvas de aprendizado (retorno médio a cada 100 episódios) por 2.000 episódios. Qual converge mais rápido?
2. **Médio.** Construa um ambiente de cliff-walking (4×12, última linha é o precipício com recompensa -100 e reset para o início). Compare as políticas finais de Q-learning e SARSA. Tire print dos caminhos que cada um toma. Qual fica mais perto do precipício?
3. **Difícil.** Implemente Double Q-learning. Em um GridWorld de recompensa ruidosa (ruído gaussiano σ=5 adicionado à recompensa por passo), mostre que Q-learning superestima `V*(0,0)` por uma quantidade significativa enquanto Double Q-learning não.

## Termos Chave

|| Termo | O que as pessoas dizem | O que realmente significa ||
||------|-----------------|-----------------------||
|| Erro TD | "O sinal de atualização" | `δ = r + γ V(s') - V(s)`, o resíduo bootstrapado. ||
|| TD(0) | "TD de um passo" | Atualização após cada transição usando apenas a estimativa do próximo estado. ||
|| Q-learning | "RL off-policy 101" | Atualização TD com `max` sobre ações do próximo estado; aprende `Q*` independente da política de comportamento. ||
|| SARSA | "Q-learning on-policy" | Atualização TD usando a próxima ação real; aprende `Q^π` para a ε-guloso π atual. ||
|| SARSA esperado | "O SARSA de baixa variância" | Substitua `a'` amostrado por sua expectativa sob π. ||
|| GLIE | "Agenda correta de exploração" | Guloso no Limite com Exploração Infinita; necessário para convergência de Q-learning. ||
|| Bootstrap | "Usar estimativa atual no alvo" | O que distingue TD de MC. Fonte de viuze mas redução massiva de variância. ||
|| Viés de maximização | "Q-learning superestima" | `max` sobre estimativas ruidosas é enviesado para cima; corrigido por Double Q-learning. |

## Leituras Complementares

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — o paper original e prova de convergência.
- [Sutton & Barto (2018). Cap. 6 — Aprendizado por Diferença Temporal](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, SARSA esperado.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — correção para viés de maximização.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — motivação do SARSA esperado.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — o paper que cunhou SARSA (então chamado "modified connectionist Q-learning").
- [Sutton & Barto (2018). Cap. 7 — Bootstrap n-step](http://incompleteideas.net/book/RLbook2020.pdf) — generaliza TD(0) para TD(n), o caminho de Q-learning para eligibility traces e, depois, GAE no PPO.
