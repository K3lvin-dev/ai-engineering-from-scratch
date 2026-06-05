# RL para Jogos — AlphaZero, MuZero e a Era do Raciocínio com LLM

> 1992: TD-Gammon venceu campeões humanos no gamão com TD puro. 2016: AlphaGo venceu Lee Sedol. 2017: AlphaZero dominou xadrez, shogi e Go do zero. 2024: DeepSeek-R1 provou que a mesma receita, com GRPO substituindo PPO, funciona para raciocínio. Jogos são o benchmark que impulsiona cada avanço nesta fase.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 05 (DQN), Fase 9 · 08 (PPO), Fase 9 · 09 (RLHF), Fase 9 · 10 (MARL)
**Tempo:** ~120 minutos

## O Problema

Jogos têm tudo que o RL quer. Recompensa limpa (vitória/derrota). Episódios infinitos (self-play reseta). Simulação perfeita (o jogo *é* o simulador). Espaços de ação discretos ou contínuos pequenos. Estrutura multi-agente que força robustez adversarial.

E jogos são como todo grande avanço do RL foi testado. TD-Gammon (gamão, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (modelo aprendido, 2019). AlphaTensor (multiplicação de matrizes, 2022). AlphaDev (algoritmos de ordenação, 2023). DeepSeek-R1 (raciocínio matemático, 2025) — a mais recente demonstração de que técnicas de RL para jogos funcionam em texto.

Esta lição de encerramento examina as três arquiteturas marcantes — AlphaZero, MuZero e GRPO — através de uma lente unificadora: **self-play + busca + melhoria de política**. Cada uma generaliza a anterior; GRPO em particular é a receita do AlphaZero aplicada ao raciocínio de LLMs, com tokens como ações e verificação matemática como o sinal de vitória.

## O Conceito

![AlphaZero ↔ MuZero ↔ GRPO: mesmo loop, ambientes diferentes](../assets/rl-games.svg)

**O loop unificador.**

```
while True:
    trajectory = self_play(current_policy, search)     # joga contra si mesmo
    policy_target = search.improved_policy(trajectory) # busca melhora a política bruta
    policy_net.update(policy_target, value_target)     # supervisionado na saída da busca
```

**AlphaZero (2017).** Silver et al. Dado um jogo (xadrez, shogi, Go) com regras conhecidas:

- Rede de política-valor: uma torre `f_θ(s) → (p, v)`. `p` é uma prior sobre movimentos legais. `v` é o resultado esperado do jogo.
- Monte Carlo Tree Search (MCTS): a cada movimento, expanda uma árvore de continuações possíveis. Use `(p, v)` como a prior + bootstrap. Selecione nós por UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: jogue partidas agente-contra-agente. No movimento `t`, a distribuição de visitas do MCTS `π_t` se torna o alvo de treinamento da política.
- Perda: `L = (v - z)² - π · log p + c · ||θ||²`. `z` é o resultado do jogo (+1 / 0 / -1).

Zero conhecimento humano. Zero heurísticas artesanais. Uma única receita que dominou xadrez, shogi e Go após algumas dezenas de milhões de jogos de self-play cada.

**MuZero (2019).** Schrittwieser et al. Remove o requisito de que as regras sejam conhecidas.

- Em vez de um ambiente fixo, aprenda um *modelo de dinâmica latente* `(h, g, f)`:
  - `h(s)`: codifique observação para estado latente.
  - `g(s_latent, a)`: prediga próximo estado latente + recompensa.
  - `f(s_latent)`: prediga prior da política + valor.
- O MCTS roda no *espaço latente aprendido*. Mesma busca, mesmo loop de treinamento.
- Funciona em Go, xadrez, shogi *e* Atari — um algoritmo, sem conhecimento das regras.

**MuZero Estocástico (2022).** Adiciona dinâmica estocástica e nós de chance; estende para jogos da classe do gamão.

**Muesli, Gumbel MuZero (2022-2024).** Melhorias em eficiência de amostras e busca determinística.

**GRPO (2024-2025).** Receita do DeepSeek-R1. Mesmo loop em forma de AlphaZero, aplicado ao raciocínio de modelos de linguagem:

- "Jogo": responda um problema de matemática / programação / raciocínio. "Vitória" = verificador (teste passa, resposta numérica confere) retorna 1.
- Política: o LLM. Ações: tokens. Estado: prompt + resposta até agora.
- Sem crítico (V_φ estilo PPO). Em vez disso, para cada prompt, amostre `G` completações da política. Compute recompensa para cada uma. Use a **vantagem relativa ao grupo** `A_i = (r_i - mean_r) / std_r` como o sinal para atualização estilo REINFORCE.
- Penalidade KL para a política de referência para evitar desvio (como RLHF).
- Perda completa:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Sem modelo de recompensa, sem crítico, sem MCTS. A linha de base relativa ao grupo substitui todos os três. Iguala ou supera a qualidade do PPO-RLHF em benchmarks de raciocínio com uma fração do computação.

**A receita R1 completa.** DeepSeek-R1 (DeepSeek 2025) são dois modelos em um paper:

- **R1-Zero.** Comece do modelo base DeepSeek-V3. Sem SFT. Aplique GRPO diretamente com dois componentes de recompensa: *recompensa de acurácia* (baseada em regras — a resposta final parseou para o número correto? / o código passou nos testes unitários) e *recompensa de formato* (a completação envolveu sua cadeia de pensamento em tags `<think>…</think>`). Ao longo de milhares de passos, o comprimento médio da resposta cresce de ~100 para ~10.000 tokens e as pontuações em benchmarks de matemática sobem para perto do nível o1-preview. O modelo aprende a raciocinar do zero. O lado negativo: suas cadeias de pensamento são frequentemente ilegíveis, misturam línguas e carecem de polimento estilístico.
- **R1.** Corrige os problemas de legibilidade do R1-Zero com um pipeline de quatro estágios:
  1. **SFT de partida a frio.** Colete alguns milhares de demonstrações longas de CoT com formatação limpa. Faça fine-tuning supervisionado do modelo base nelas. Isso dá um ponto de partida legível.
  2. **GRPO orientado a raciocínio.** Aplique GRPO com as recompensas de acurácia+formato mais uma *recompensa de consistência de idioma* para evitar alternância de código.
  3. **Amostragem por rejeição + SFT rodada 2.** Amostre ~600K trajetórias de raciocínio do checkpoint RL, mantenha apenas aquelas com respostas finais corretas e CoT legível, e combine com ~200K exemplos SFT não relacionados a raciocínio (escrita, QA, autoconsciência). Faça fine-tuning da base novamente.
  4. **GRPO de espectro completo.** Mais uma rodada RL cobrindo tanto raciocínio (recompensas baseadas em regras) quanto alinhamento geral (recompensas baseadas em preferência de utilidade/inofensividade).

O resultado iguala o o1 no AIME e MATH-500 com pesos abertos, e é pequeno o suficiente para destilar. O mesmo paper também libera seis modelos densos destilados (Qwen-1.5B até Llama-70B) fazendo SFT nos traços de raciocínio do R1 — sem RL no aluno. Destilação de um professor RL forte consistentemente supera RL do zero na escala do aluno.

**Por que GRPO em vez de PPO para raciocínio.** Três razões no paper DeepSeekMath (fev 2024): (1) nenhuma rede de valor para treinar, reduzindo memória pela metade; (2) a linha de base do grupo naturalmente lida com a recompensa esparsa de final de trajetória que tarefas de raciocínio produzem; (3) a normalização por prompt torna as vantagens comparáveis entre problemas de dificuldade drasticamente diferente, o que o crítico único do PPO não consegue.

**Sem busca vs com busca.** Jogos se ramificaram:

- *Jogos de informação perfeita com horizontes longos* (Go, xadrez): ainda baseados em busca. AlphaZero / MuZero dominam.
- *Raciocínio com LLM*: sem MCTS ainda em produção; GRPO em rollouts completos, best-of-N para inferência. Modelos de recompensa de processo (PRMs) sugerem que busca em nível de passo pode ser adicionada de volta.

## Construa

O código em `code/main.py` implementa **GRPO em miniatura** — um bandido com múltiplos grupos de amostras. O algoritmo é o mesmo que em um LLM; só a política e o ambiente são mais simples. Ensina a *perda* e a *vantagem relativa ao grupo*, que é a inovação de 2025.

### Passo 1: um ambiente verificador minúsculo

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Em GRPO real o verificador roda testes unitários ou verifica igualdade matemática.

### Passo 2: política: softmax sobre K tokens de resposta por prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Equivalente à saída da última camada de um LLM condicionado a um prompt.

### Passo 3: amostragem em grupo e vantagem relativa ao grupo

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

A vantagem relativa ao grupo é o truque do DeepSeek de 2024. Nenhum crítico necessário. A "linha de base" é a média do grupo, e a normalização usa o desvio padrão do grupo.

### Passo 4: compare com linha de base REINFORCE (sem valor)

Mesma configuração, mesma computação, REINFORCE simples. GRPO converge mais rápido e mais estavelmente.

### Passo 5: observe entropia e KL

Mesmos diagnósticos do RLHF: KL média para a referência, entropia da política, recompensa ao longo do tempo. Uma vez que estabilizam, o treinamento terminou.

## Armadilhas

- **Reward hacking via exploração do verificador.** GRPO herda o risco do RLHF: se o verificador está errado ou é explorável, o LLM vai encontrar a exploração. Verificadores robustos (múltiplos casos de teste, provas formais) importam.
- **Tamanho do grupo muito pequeno.** A variância da linha de base do grupo é `~1/√G`. Abaixo de `G = 4`, o sinal de vantagem é ruidoso; escolha padrão é `G = 8` a `64`.
- **Viés de comprimento.** Completações de LLM de diferentes comprimentos têm diferentes log-probabilidades. Normalize por contagem de tokens, ou use log-prob em nível de sequência, ou trunque para comprimento máximo.
- **Ciclos de self-play puro.** Treinamento estilo AlphaZero pode ficar preso em loops de dominância em jogos de soma geral. Mitigado por pools de oponentes diversos (jogo em liga, Lição 10).
- **Incompatibilidade busca-política.** AlphaZero treina a política para imitar a saída da busca. Se a rede de política é pequena demais para representar a distribuição da busca, o treinamento estagna.
- **Piso de computação.** MuZero / AlphaZero precisam de computação massiva. Uma única ablação é frequentemente centenas de GPU-horas. Demonstrações em miniatura existem (ex.: AlphaZero no Connect Four) para aprendizado.
- **Cobertura do verificador.** Testes unitários que passam para uma solução bugada reforçam o bug. Projete verificadores que pegam casos de borda.

## Use

O panorama de RL para jogos em 2026, por domínio:

| Domínio | Método dominante |
|---------|------------------|
| Jogos de tabuleiro dois jogadores soma zero (Go, xadrez, shogi) | AlphaZero / MuZero / KataGo |
| Jogos de cartas com informação imperfeita (pôquer) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / jogos de pixel | Muesli / MuZero / IMPALA-PPO |
| Estratégia multijogador grande (Dota, StarCraft) | PPO + self-play + liga (OpenAI Five, AlphaStar) |
| Raciocínio matemático/código com LLM | GRPO (DeepSeek-R1, Qwen-RL, replicações abertas) |
| Alinhamento de LLM | DPO / RLHF-PPO (não GRPO; verificador é preferência, não verificável) |
| Robótica | PPO + DR (não é RL de jogos, mas usa as mesmas ferramentas de gradiente de política) |
| Problemas combinatórios | Variantes AlphaZero (AlphaTensor, AlphaDev) |

A *receita* — self-play, melhoria aumentada por busca, destilação de política — abrange texto, pixels e controle físico. GRPO é a instância mais jovem; mais estão por vir.

## Entregue

Salve como `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Projete um pipeline de treinamento RL para jogos ou raciocínio (AlphaZero / MuZero / GRPO) para um dado domínio.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Dado um alvo (jogo de informação perfeita / informação imperfeita / Atari / raciocínio LLM / combinatório), entregue:

1. Adequação do ambiente. Regras conhecidas? Markov? Estocástico? Multi-agente? Informa AlphaZero vs MuZero vs GRPO.
2. Estratégia de busca. MCTS (PUCT com prior aprendida), amostrado por Gumbel, best-of-N, ou nenhum.
3. Plano de self-play. Self-play simétrico / liga / dados offline / gerado por verificador.
4. Sinal alvo. Resultado do jogo / recompensa do verificador / preferência / modelo aprendido. Inclua plano de robustez.
5. Diagnósticos. Taxa de vitória vs linha de base, curva ELO, taxa de aprovação do verificador, KL para referência.

Recuse AlphaZero em jogos de informação imperfeita (encaminhe para CFR). Recuse GRPO sem um verificador confiável. Recuse qualquer pipeline RL para jogos sem um conjunto fixo de oponentes de linha de base (ELO de self-play não é calibrado de outra forma).
```

## Exercícios

1. **Fácil.** Implemente o bandido GRPO em `code/main.py`. Treine em 2 prompts × 4 tokens de resposta cada. Convirja em < 1.000 atualizações com `G=8`.
2. **Médio.** Adicione PPO (clipped) e REINFORCE clássico. Compare eficiência de amostras e variância da recompensa com GRPO no mesmo bandido.
3. **Difícil.** Estenda para uma "cadeia de raciocínio" de comprimento 2: o agente emite dois tokens e o verificador recompensa o par. Meça como o GRPO lida com a atribuição de crédito em sequências de dois passos. (Dica: compute vantagem do grupo por *sequência completa*, propague para ambas as posições de token.)

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|---------------------------|
| MCTS | "Busca em árvore com rede aprendida" | Monte Carlo Tree Search; seleção UCB1/PUCT com priors `(p, v)` aprendidos. |
| AlphaZero | "Self-play + MCTS" | Rede de política-valor treinada para igualar visitas MCTS e resultado do jogo. |
| MuZero | "AlphaZero com modelo aprendido" | Mesmo loop mas em espaço latente via dinâmica aprendida. |
| GRPO | "PPO sem crítico" | Group Relative Policy Optimization; REINFORCE com linha de base média do grupo + KL. |
| PUCT | "O UCB do AlphaZero" | `Q + c · p · √N / (1 + N_a)` — equilibra estimativa de valor com prior. |
| Self-play | "Agente vs si mesmo no passado" | Padrão para soma zero; sinal de treinamento simétrico. |
| League play | "Self-play baseado em população" | Passado + atual + exploiters amostrados como oponentes. |
| Verifier reward | "RL verificável" | Recompensa vem de um verificador determinístico (testes passam, resposta confere). |
| Process reward | "PRM" | Pontua cada passo de raciocínio, não apenas a resposta final. |

## Leitura Complementar

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — o paper que introduziu GRPO e a linha de base relativa ao grupo.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — a receita completa R1 de quatro estágios mais a ablação R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning em escala.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — o paper que começou tudo.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — a referência de produção para aplicar GRPO com funções de recompensa personalizadas.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — replicação aberta da receita R1 em múltiplas escalas.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — o enquadramento do livro para self-play, busca e "recompensa projetada" que o R1 instancia em escala LLM.
