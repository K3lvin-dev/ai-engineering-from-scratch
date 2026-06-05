# Multi-Agent RL

> RL de agente único assume que o ambiente é estacionário. Coloque dois agentes aprendendo no mesmo mundo e essa suposição quebra: cada agente é parte do ambiente do outro, e ambos estão mudando. Multi-agent RL é o conjunto de truques para fazer o aprendizado convergir quando a suposição de Markov não se sustenta mais.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 04 (Q-learning), Fase 9 · 06 (REINFORCE), Fase 9 · 07 (Actor-Critic)
**Tempo:** ~45 minutos

## O Problema

Um robô aprendendo a navegar uma sala é um problema de RL de agente único. Um time de futebol não é. AlphaStar vs oponentes do StarCraft não é. Um mercado de agentes de lances não é. Dois carros negociando uma parada de quatro vias não é. Problemas do mundo real com muitos agentes não são.

Em todo ambiente multi-agente, da perspectiva de qualquer agente, os outros agentes *são* parte do ambiente. Conforme eles aprendem e mudam seu comportamento, o ambiente se torna não-estacionário. A propriedade de Markov — "o próximo estado depende apenas do estado atual e da minha ação" — é violada porque o próximo estado também depende do que os *outros* agentes escolheram, e suas políticas são alvos móveis.

Isso quebra as provas de convergência tabulares (a garantia do Q-learning assume um ambiente estacionário). Quebra também o deep RL ingênuo: agentes se perseguem em loops, nunca convergem para uma política estável. Você precisa de técnicas específicas para multi-agente: treinamento centralizado / execução descentralizada, linhas de base contrafactuais, jogo em liga, self-play.

Aplicações em 2026: enxames de robôs, roteamento de tráfego, frotas de veículos autônomos, simuladores de mercado, sistemas multi-agente com LLMs (Fase 16) e qualquer jogo com mais de um jogador inteligente.

## O Conceito

![Quatro regimes de MARL: independente, crítico centralizado, self-play, liga](../assets/marl.svg)

**Formalismo: Markov Game.** Uma generalização de MDP: estados `S`, uma ação conjunta `a = (a_1, …, a_n)`, transição `P(s' | s, a)`, e recompensas por agente `R_i(s, a, s')`. Cada agente `i` maximiza seu próprio retorno sob sua própria política `π_i`. Se as recompensas são idênticas, é **totalmente cooperativo**. Se soma zero, é **adversarial**. Se misto, é **soma geral**.

**Desafios centrais:**

- **Não-estacionariedade.** `P(s' | s, a_i)` da visão do agente `i` depende de `π_{-i}`, que está mudando.
- **Atribuição de crédito.** Com recompensa compartilhada, qual agente a causou?
- **Coordenação de exploração.** Agentes devem explorar estratégias complementares, não explorar redundantemente o mesmo estado.
- **Escalabilidade.** O espaço de ação conjunto cresce exponencialmente em `n`.
- **Observabilidade parcial.** Cada agente vê apenas sua própria observação; o estado global está oculto.

**Quatro regimes dominantes:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Cada agente aprende seu próprio Q ou política, tratando os outros como parte do ambiente. Simples, às vezes funciona (especialmente com experience replay agindo como um truque de suavização para modelagem de agente). Convergência teórica: nenhuma. Na prática: bom para tarefas frouxamente acopladas, ruim para tarefas fortemente acopladas.

**2. Treinamento centralizado, execução descentralizada (CTDE).** Paradigma moderno mais comum. Cada agente tem sua própria *política* `π_i` que condiciona na observação local `o_i` — execução descentralizada padrão na implantação. Durante o *treinamento*, um crítico centralizado `Q(s, a_1, …, a_n)` condiciona no estado global completo e na ação conjunta. Exemplos:
- **MADDPG** (Lowe et al. 2017): DDPG com um crítico centralizado por agente.
- **COMA** (Foerster et al. 2017): linha de base contrafactual — pergunte "qual teria sido minha recompensa se eu tivesse tomado a ação `a'`?" — isola minha contribuição.
- **MAPPO** / **IPPO** com crítico compartilhado (Yu et al. 2022): PPO com uma função de valor centralizada. Dominante em 2026 para MARL cooperativo.
- **QMIX** (Rashid et al. 2018): decomposição de valor — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` com mistura monotônica.

**3. Self-play.** Duas cópias do mesmo agente jogam uma contra a outra. A política do oponente *é* minha política de um snapshot passado. AlphaGo / AlphaZero / MuZero. OpenAI Five. Funciona melhor para jogos de soma zero; o sinal de treinamento é simétrico.

**4. Jogo em liga (League play).** Uma extensão do self-play para ambientes de soma geral / adversarial: mantenha uma população de políticas passadas e atuais, amostre um oponente da liga, treine contra ele. Adiciona exploiters (especializados em vencer o melhor atual) e main exploiters (especializados em vencer exploiters). AlphaStar (StarCraft II). Necessário quando o jogo admite ciclos de estratégia "pedra-papel-tesoura".

**Comunicação.** Permita que agentes enviem mensagens aprendidas `m_i` uns aos outros. Funciona em ambientes cooperativos. Foerster et al. (2016) mostraram que a comunicação inter-agente diferenciável pode ser treinada de ponta a ponta. Hoje, sistemas multi-agente baseados em LLM (Fase 16) essencialmente se comunicam em linguagem natural.

## Construa

Esta lição usa um GridWorld 6×6 com dois agentes cooperativos. Eles começam em cantos opostos e devem alcançar um objetivo compartilhado. Recompensa compartilhada: `-1` por passo enquanto qualquer agente ainda estiver se movendo, `+10` quando ambos chegam. Veja `code/main.py`.

### Passo 1: o ambiente multi-agente

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

O espaço de ação *conjunto* é `|A|² = 16`. O estado global são duas posições.

### Passo 2: Independent Q-learning

Cada agente roda sua própria tabela Q indexada pelo estado conjunto. A cada passo: ambos escolhem ações ε-gulosas, coletam transição conjunta, cada um atualiza seu próprio Q com a recompensa compartilhada.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Funciona nesta tarefa porque as recompensas são densas e alinhadas. Falha em tarefas fortemente acopladas (ex.: onde um agente tem que *esperar* pelo outro).

### Passo 3: Q centralizado com atualização de valor decomposto

Use um Q sobre ações conjuntas `Q(s, a_1, a_2)`. Atualize a partir da recompensa compartilhada. Descentralize na execução marginalizando: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Troca o espaço de ação conjunto exponencial por uma visão global *correta*.

### Passo 4: self-play simples (adversarial 2 agentes)

Mesmo agente, dois papéis. Treine o agente A contra o agente B; após `K` episódios, copie os pesos de A em B. Treinamento simétrico, progresso consistente. A receita do AlphaZero em miniatura.

## Armadilhas

- **Replay não-estacionário.** Experience replay com agentes independentes é pior que com agente único porque transições antigas foram geradas por oponentes agora obsoletos. Correção: re-rotule ou pondere por recência.
- **Ambiguidade de atribuição de crédito.** Recompensa compartilhada após um episódio longo; nenhuma maneira clara de dizer qual agente contribuiu. Correção: linhas de base contrafactuais (COMA), ou modelagem de recompensa por agente.
- **Desvio de política / perseguição.** A melhor resposta de cada agente muda com a atualização do outro. Correção: crítico centralizado, taxas de aprendizado lentas, ou congele um de cada vez.
- **Reward hacking via coordenação.** Agentes encontram explorações coordenadas que o designer não antecipou. Agentes de leilão convergem para dar lance zero. Correção: design cuidadoso de recompensa, restrições comportamentais.
- **Redundância de exploração.** Ambos os agentes exploram os mesmos pares estado-ação. Correção: bônus de entropia por agente, ou condicionamento de papel.
- **Ciclos de liga.** Self-play puro pode ficar preso em um ciclo de dominância. Correção: jogo em liga com oponentes diversos.
- **Explosão de amostras.** `n` agentes × espaço de estados × ações conjuntas. Aproxime com aproximação de função; espaços de ação fatorados (uma cabeça de saída de política por agente).

## Use

O mapa de aplicações MARL em 2026:

| Domínio | Método | Notas |
|---------|--------|-------|
| Navegação / manipulação cooperativa | MAPPO / QMIX | CTDE; crítico compartilhado + atores descentralizados. |
| Jogos de dois jogadores (xadrez, Go, pôquer) | Self-play com MCTS (AlphaZero) | Soma zero; treinamento simétrico. |
| Multijogador complexo (Dota, StarCraft) | Jogo em liga + pre-treinamento por imitação | OpenAI Five, AlphaStar. |
| Frotas de veículos autônomos | CTDE MAPPO / PPO com attention | Obs. parcial; tamanhos de equipe variáveis. |
| Mercados de leilão | Equilíbrio de teoria dos jogos + RL | Mean-field RL quando `n` → ∞. |
| Sistemas multi-agente com LLM (Fase 16) | Comunicação em linguagem natural + condicionamento de papel | Loop RL na camada de planejamento do agente. |

Em 2026, a maior área de crescimento do MARL é baseada em LLM: enxames de agentes de modelo de linguagem negociando, debatendo, construindo software. O RL aparece como otimização de preferência em saídas em *nível de trajetória*, não de token (Fase 16 · 03).

## Entregue

Salve como `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Escolha o regime de RL multi-agente certo (IPPO, CTDE, self-play, liga) para uma dada tarefa.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Dada uma tarefa com `n` agentes, entregue:

1. Classificação do regime. Cooperativo / adversarial / soma geral. Justifique.
2. Algoritmo. IPPO / MAPPO / QMIX / self-play / liga. Razão ligada ao acoplamento e estrutura de recompensa.
3. Acesso à informação. Treinamento centralizado (qual informação global vai para o crítico)? Execução descentralizada?
4. Atribuição de crédito. Linha de base contrafactual, decomposição de valor, ou modelagem de recompensa.
5. Plano de exploração. Entropia por agente, treinamento baseado em população, ou liga.

Recuse independent Q-learning em tarefas cooperativas fortemente acopladas. Recuse recomendar self-play para soma geral com riscos de ciclo. Sinalize qualquer pipeline MARL sem uma avaliação com oponente fixo (números de self-play escolhidos a dedo são comuns).
```

## Exercícios

1. **Fácil.** Treine independent Q-learning no GridWorld cooperativo de 2 agentes. Quantos episódios até o retorno médio > 0? Plote a curva de aprendizado conjunta.
2. **Médio.** Adicione uma tarefa de "coordenação": o objetivo é alcançado apenas quando ambos os agentes pisam nele no mesmo turno. O independent Q ainda converge? O que quebra?
3. **Difícil.** Implemente um crítico centralizado para treinamento estilo MAPPO e compare a velocidade de convergência com IPPO na tarefa de coordenação.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|---------------------------|
| Markov game | "MDP multi-agente" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; cada agente tem sua própria recompensa. |
| CTDE | "Treinamento centralizado, execução descentralizada" | Crítico conjunto no treinamento; a política de cada agente usa apenas obs. local. |
| IPPO | "PPO independente" | Cada agente roda PPO separadamente. Linha de base simples; muitas vezes subestimado. |
| MAPPO | "PPO multi-agente" | PPO com uma função de valor centralizada condicionada ao estado global. |
| QMIX | "Decomposição de valor monotônica" | `Q_tot = f_monotone(Q_1, …, Q_n)` permite argmax descentralizado. |
| COMA | "Multi-agente contrafactual" | Vantagem = meu Q menos Q esperado marginalizando sobre minha ação. |
| Self-play | "Agente vs si mesmo no passado" | Agente único, dois papéis; padrão para jogos de soma zero. |
| League play | "Treinamento populacional" | Armazene políticas passadas, amostre oponentes do pool; lida com ciclos de estratégia. |

## Leitura Complementar

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE com um crítico centralizado.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — linhas de base contrafactuais para atribuição de crédito.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — decomposição de valor com monotonicidade.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO é surpreendentemente forte para MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — jogo em liga em escala.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — self-play puro em jogos de soma zero.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — inclui o breve tratamento do livro sobre ambientes multi-agente e o problema de não-estacionariedade que o CTDE foi projetado para resolver.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — pesquisa cobrindo MARL cooperativo, competitivo e misto com resultados de convergência.
