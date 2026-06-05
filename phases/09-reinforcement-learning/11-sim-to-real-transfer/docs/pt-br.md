# Sim-to-Real Transfer

> Uma política treinada em um simulador que falha no hardware real é uma política que memorizou o simulador. Domain randomization, domain adaptation e system identification são as três ferramentas para fazer controladores aprendidos atravessarem o reality gap.

**Tipo:** Learn
**Linguagens:** Python
**Pré-requisitos:** Fase 9 · 08 (PPO), Fase 2 · 10 (Viés/Variância)
**Tempo:** ~45 minutos

## O Problema

Treinar um robô real é lento, perigoso e caro. Um bípede leva milhões de episódios de treinamento para aprender a andar; um bípede real que cai uma vez quebra hardware. A simulação te dá resetes ilimitados, reprodutibilidade determinística, ambientes paralelos e nenhum dano físico.

Mas simuladores são errados. Rolamentos têm mais atrito que os modelos do MuJoCo. Câmeras têm distorção de lente que o simulador não inclui. Motores têm atrasos, folga e saturação que 99% dos modelos de simulação pulam. Vento, poeira e iluminação variável sabotam uma política treinada em renderização estéril. O **reality gap** — diferença sistemática entre a distribuição da simulação e a distribuição real — é o problema central do RL implantado para robótica.

Você precisa de uma política que seja *robusta ao desvio de distribuição sim-para-real*. Três abordagens históricas: randomize o simulador (domain randomization), adapte a política com um pouco de dados reais (domain adaptation / fine-tuning), ou identifique os parâmetros do sistema real e os iguale (system identification). Em 2026 a receita dominante combina todas as três com simulação paralela massiva (Isaac Sim, Isaac Lab, Mujoco MJX em GPU).

## O Conceito

![Três regimes sim-to-real: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. Durante o treinamento, randomize todo parâmetro da simulação que possa diferir no robô real: massas, coeficientes de atrito, ganhos PD do motor, ruído de sensor, posição da câmera, iluminação, texturas, modelos de contato. A política aprende uma distribuição condicional sobre "em qual simulação estou hoje" e generaliza através do espectro inteiro. Se o robô real cai dentro do envelope de treinamento, a política funciona.

- **Lado bom:** nenhum dado real necessário. Uma receita, muitos robôs.
- **Lado ruim:** treinamento super-randomizado produz uma política "universal" mas excessivamente cautelosa. Ruído demais ≈ regularização demais.

**System Identification (SI).** Ajuste os parâmetros do simulador para dados do mundo real antes do treinamento. Se você consegue medir o atrito da junta do braço no robô real, coloque isso na simulação. Depois treine uma política que espera esses valores. Precisa de acesso ao sistema real mas reduz o reality gap diretamente.

- **Lado bom:** alvo de treinamento preciso, baixo ruído.
- **Lado ruim:** erro residual do modelo é invisível para a política; pequenos efeitos não identificados (ex.: zona morta do motor) ainda quebram a implantação.

**Domain Adaptation.** Treine na simulação, faça fine-tuning com uma pequena quantidade de dados reais. Duas variações:

- **Real2Sim2Real:** aprenda um simulador residual `f(s, a, z) - f_sim(s, a)` usando rollouts reais, treine na simulação corrigida. Fecha a lacuna sem muitos dados reais.
- **Adaptação de observação:** treine uma política que mapeia observação real → observação tipo simulação via um extrator de features aprendido (ex.: GAN pixel-a-pixel). O controlador fica na simulação.

**Aprendizagem privilegiada / professor-aluno.** Miki et al. 2022 (ANYmal quadrúpede). Treine um *professor* em simulação que tem acesso a informação privilegiada (atrito real, altura do terreno, deriva da IMU). Destile um *aluno* que só vê observações de sensores reais. O aluno aprende a inferir features privilegiadas do histórico, robusto através de parâmetros físicos.

**Simulação massivamente paralela.** 2024–2026. Isaac Lab, Mujoco MJX, Brax todos rodam milhares de robôs paralelos em uma única GPU. PPO com 4.096 humanoides paralelos coleta anos de experiência em horas. O "reality gap" encolhe conforme a distribuição de treinamento se alarga; DR se torna quase grátis quando cada um desses 4.096 ambientes tem parâmetros randomizados diferentes.

**A receita real de 2026 (exemplo de caminhada quadrúpede):**

1. Simulação massivamente paralela com domain randomization de gravidade, atrito, ganhos do motor, carga.
2. Política professora treinada com informação privilegiada (mapa do terreno, velocidade real do corpo).
3. Política aluna destilada da professora usando apenas propriocepção (codificadores das juntas das pernas).
4. Adaptação de observação opcional via autoencoder na IMU real.
5. Implante. Zero-shot em 10+ ambientes. Se falhar, faça minutos de fine-tuning no mundo real com PPO com restrição de segurança.

## Construa

O código desta lição é uma demonstração minúscula de domain randomization em um GridWorld com transições *ruidosas*. Treinamos uma política que experimenta probabilidades de derrapagem randomizadas na "simulação" e avaliamos no "real" com um nível de derrapagem que ela nunca viu durante o treinamento. A forma mapeia diretamente para transferência MuJoCo-para-hardware.

### Passo 1: simulação parametrizada

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` é um parâmetro que o simulador expõe. Em robótica real poderia ser atrito, massa, ganho do motor — qualquer coisa que mude entre simulação e real.

### Passo 2: treine com DR

No início de cada episódio, amostre `slip ~ Uniform[0.0, 0.4]`. Treine PPO / Q-learning / qualquer coisa. Faça isso por muitos episódios.

### Passo 3: avalie zero-shot em derrapagens "reais"

Avalie em `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Os quatro primeiros estão dentro do suporte de treinamento; `0.5` e `0.7` estão fora. Uma política treinada com DR deve permanecer quase ótima dentro do suporte e degradar graciosamente fora. Uma política treinada com derrapagem fixa será frágil fora de sua derrapagem de treinamento.

### Passo 4: compare com treinamento estreito

Treine uma segunda política com `slip = 0.0` apenas. Avalie na mesma varredura de `slip`. Você deve ver uma queda catastrófica assim que a derrapagem real > 0.

## Armadilhas

- **Randomização demais.** Treine em `slip ∈ [0, 0.9]` e sua política fica tão avessa a risco que nunca tenta o caminho ótimo. Combine com a distribuição *esperada* do mundo real, não "qualquer coisa pode acontecer."
- **Randomização de menos.** Treine em uma fatia fina e a política não consegue generalizar. Use um currículo adaptativo (Automatic Domain Randomization) que alarga a distribuição conforme a política melhora.
- **Espaço de parâmetros mal identificado.** Randomize a coisa errada (matiz da câmera quando a lacuna real é atraso do motor) e DR não ajuda. Perfile o robô real primeiro.
- **Vazamento de informação privilegiada.** Um professor que usa estado global para ações, não apenas observações, pode produzir um aluno que não consegue alcançar. Garanta que a política do professor seja realizável pelo aluno dado o histórico de observações.
- **Falha de transferência sim-para-sim.** Se sua política não é robusta para uma variante de simulação mais difícil, ela não será robusta para o mundo real também. Sempre teste em uma variante de simulação reservada antes de implantar.
- **Sem envelope de segurança no mundo real.** Uma política que funciona na simulação e "funciona no real" sem um escudo de segurança de baixo nível ainda pode quebrar hardware. Adicione limites de taxa, limites de torque, limites de junta em um controlador não aprendido.

## Use

A pilha sim-to-real de 2026:

| Domínio | Pilha |
|---------|-------|
| Locomoção com pernas (ANYmal, Spot, humanoide) | Isaac Lab + DR + professor/aluno privilegiado |
| Manipulação (mãos dextras, pick-and-place) | Isaac Lab + DR + DR-GAN para visão |
| Direção autônoma | CARLA / NVIDIA DRIVE Sim + DR + fine-tuning real |
| Corrida de drones | RotorS / Flightmare + DR + adaptação online |
| Manipulação em-dedo/mão | OpenAI Dactyl (DR em escala sem precedentes) |
| Braços industriais | MuJoCo-Warp + SI + pequeno fine-tuning real |

Para controle em todas as escalas, o fluxo de trabalho é consistente: ajuste a simulação o melhor que puder, randomize o que não puder ajustar, treine políticas enormes, destile, implante com um escudo de segurança.

## Entregue

Salve como `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Planeje um pipeline de transferência sim-to-real para um dado robô + tarefa, cobrindo DR, SI e segurança.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Dada uma plataforma robótica, uma tarefa e acesso a tempo de hardware real, entregue:

1. Inventário do reality gap. Fontes suspeitas ranqueadas por impacto esperado (contato, sensor, atraso de atuação, visão).
2. Parâmetros de DR. Lista exata, intervalos, distribuição. Justifique cada intervalo contra medições reais.
3. Passos de SI. Quais parâmetros medir; método de medição.
4. Divisão professor/aluno. Qual informação privilegiada o professor usa; quais observações o aluno usa.
5. Envelope de segurança. Limites de baixo nível, paradas de emergência, controlador de backup.

Recuse implantar sem (a) um teste zero-shot em variante de simulação, (b) um escudo de segurança, (c) um plano de reversão. Sinalize qualquer intervalo de DR maior que 3× a variabilidade real medida como provavelmente super-randomizado.
```

## Exercícios

1. **Fácil.** Treine um agente Q-learning no GridWorld com derrapagem fixa (slip=0.0). Avalie em slip ∈ {0.0, 0.1, 0.3, 0.5}. Plote retorno vs slip.
2. **Médio.** Treine um agente Q-learning com DR amostrando `slip ~ Uniform[0, 0.3]`. Avalie a mesma varredura. Quanto o DR compra em slip=0.5 (fora da distribuição)?
3. **Difícil.** Implemente um currículo: comece com slip=0.0, alargue o intervalo de DR a cada vez que a política atinge 90% do ótimo. Meça o total de passos de ambiente para alcançar slip=0.3 zero-shot vs uma linha de base DR fixa.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|---------------------------|
| Reality gap | "Diferença sim-para-real" | Desvio de distribuição entre física/sensor de treinamento e implantação. |
| Domain randomization (DR) | "Treine através de simulações aleatórias" | Randomize parâmetros da simulação durante o treinamento para a política generalizar. |
| System identification (SI) | "Meça o real e ajuste a simulação" | Estime parâmetros físicos reais; configure a simulação para igualar. |
| Domain adaptation | "Fine-tuning em dados reais" | Pequeno fine-tuning no mundo real após treino em simulação; pode adaptar observação ou dinâmica. |
| Privileged info | "Informação real para o professor" | Informação que só a simulação tem; o aluno deve inferi-la do histórico de observações. |
| Teacher/student | "Destile privilegiado -> observável" | Professor treinado com atalhos; aluno aprende a imitar sem eles. |
| ADR | "Domain Randomization Automática" | Currículo que alarga os intervalos de DR conforme a política melhora. |
| Real2Sim | "Feche a lacuna com dados reais" | Aprenda um residual para fazer a simulação imitar rollouts reais. |

## Leitura Complementar

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — o paper original de DR (visão para robótica).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR para dinâmica, locomoção quadrúpede.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR em escala.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — professor-aluno para ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — a simulação massivamente paralela que impulsiona implantações de 2025-2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — método de currículo ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — o enquadramento Dyna (use um modelo para planejamento + rollouts) que sustenta pipelines modernos sim-to-real.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taxonomia de métodos sim-to-real com resultados de benchmark.
