# Reward Modeling & RLHF

> Humanos não sabem escrever uma função de recompensa para "resposta de assistente boa", mas conseguem comparar duas respostas e escolher a melhor. Ajuste um modelo de recompensa a essas comparações, depois aplique RL no modelo de linguagem contra ele. Christiano 2017. InstructGPT 2022. A receita que transformou GPT-3 em ChatGPT. Em 2026 está sendo majoritariamente substituída por DPO — mas o modelo mental permanece.

**Tipo:** Build
**Linguagens:** Python
**Pré-requisitos:** Fase 5 · 05 (Sentiment), Fase 9 · 08 (PPO)
**Tempo:** ~45 minutos

## O Problema

Você treinou um modelo de linguagem no objetivo de predição do próximo token. Ele escreve inglês gramatical. Também mente, divaga e se recusa a recusar. Você não consegue consertar isso com mais pré-treinamento — texto da web é o problema, não a cura.

Você quer uma *recompensa escalar* que diga "a resposta A é melhor que a resposta B para a instrução X." Escrever essa função de recompensa manualmente é impossível. "Utilidade" não é uma expressão de forma fechada sobre tokens. Mas humanos podem comparar duas saídas e marcar uma preferência. Isso é barato de coletar em escala.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) converte preferências em um modelo de recompensa, depois otimiza o LM via PPO contra essa recompensa. Em três etapas: SFT → RM → PPO. É a receita que entregou ChatGPT, Claude, Gemini e todos os outros LLMs alinhados em 2023–2025.

Em 2026 a etapa PPO é majoritariamente substituída por DPO (Fase 10 · 08) porque é mais barata e quase tão boa para ajuste de alinhamento. Mas a peça do *modelo de recompensa* ainda sustenta todo amostrador Best-of-N, todo pipeline de RL a partir de recompensas verificáveis e todo modelo de raciocínio que usa um modelo de recompensa de processo. Entenda RLHF e você entende toda a pilha de alinhamento.

## O Conceito

![RLHF em três estágios: SFT, treinamento do RM em preferências pareadas, PPO com penalidade KL](../assets/rlhf.svg)

**Estágio 1: Supervised Fine-Tuning (SFT).** Comece de um modelo base pré-treinado. Faça fine-tuning em demonstrações escritas por humanos do comportamento desejado (respostas seguindo instruções, respostas úteis, etc.). Resultado: um modelo `π_SFT` que é *viesado para comportamento bom* mas ainda tem um espaço de ação ilimitado.

**Estágio 2: Treinamento do Modelo de Recompensa.**

- Colete pares de respostas `(y_+, y_-)` para prompts `x`, rotulados por humanos como "y_+ é preferido sobre y_-."
- Treine um modelo de recompensa `R_φ(x, y)` para atribuir pontuações mais altas a `y_+`.
- Perda: o **logístico pareado Bradley-Terry**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ é a sigmoide. A diferença na recompensa implica uma log-odds de preferência. BT é o padrão desde 1952 (Bradley-Terry) e é a escolha dominante em RLHF moderno.

- `R_φ` geralmente é inicializado a partir do modelo SFT com uma cabeça escalar no topo. Mesmo backbone transformer; uma camada linear simples produz a recompensa.

**Estágio 3: PPO contra o RM com penalidade KL.**

- Inicialize a política treinável `π_θ` a partir de `π_SFT`. Mantenha uma *referência* congelada `π_ref = π_SFT`.
- Recompensa ao final de uma resposta `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  A penalidade KL evita que `π_θ` se afaste arbitrariamente de `π_SFT` — é um *regularizador*, não uma trust region rígida. `β` tipicamente `0.01`-`0.05`.
- Rode PPO (Lição 08) com esta recompensa. As vantagens são computadas na trajetória em nível de token, mas o RM pontua apenas a resposta completa.

**Por que a KL?** Sem ela, o PPO vai alegremente encontrar estratégias de recompensa adulterada — o RM foi treinado apenas em completações dentro da distribuição. Uma resposta fora da distribuição pode pontuar mais que qualquer resposta escrita por humano. A KL mantém `π_θ` perto da variedade onde o RM foi treinado. É o knob mais importante do RLHF.

**Status em 2026:**

- **DPO** (Rafailov 2023): álgebra de forma fechada colapsa os Estágios 2+3 em uma única perda supervisionada sobre dados de preferência. Sem RM, sem PPO. Mesma qualidade em benchmarks de alinhamento por uma fração do computação. Coberto na Fase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO com uma linha de base relativa ao grupo em vez de um crítico, recompensa de um *verificador* (código executa / resposta matemática confere) em vez de um RM treinado por humanos. Dominante para modelos de raciocínio. Coberto na Fase 9 · 12.
- **Process reward models (PRMs):** pontuam soluções parciais (cada passo de raciocínio), usados em variantes de RLHF e GRPO para raciocínio.
- **Constitutional AI / RLAIF:** usam um LLM alinhado para gerar preferências em vez de humanos. Escala o orçamento de preferência.

## Construa

Esta lição usa "prompts" e "respostas" sintéticos minúsculos representados como strings. O RM é um pontuador linear sobre uma representação bag-of-tokens. Sem LLM real — a *forma* do pipeline importa, não a escala. Veja `code/main.py`.

### Passo 1: dados sintéticos de preferência

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Em RLHF real isso é substituído por rotuladores humanos. A forma — `(prompt, resposta_preferida, resposta_rejeitada)` — é idêntica.

### Passo 2: modelo de recompensa Bradley-Terry

Pontuação linear: `R(x, y) = w · bag(y)`. Treine para minimizar a log-perda pareada BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Após algumas centenas de atualizações, `w` atribui pesos positivos a tokens de palavras boas e negativos a ruins.

### Passo 3: política estilo PPO sobre o RM

Nossa política de brinquedo produz um único token de um vocabulário. Pontuamos o token sob o RM, computamos `log π_θ(token | prompt)`, adicionamos uma penalidade KL-para-referência e aplicamos o surrogate PPO clipped.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Passo 4: monitore a KL

Acompanhe a KL média `KL(π_θ || π_ref)` a cada atualização. Se passar de `~5-10` a política se afastou muito de `π_SFT` — `β` baixo ou reward hacking começando. Esse é o principal diagnóstico em RLHF real.

### Passo 5: a receita de produção com TRL

Depois que você entende o pipeline de brinquedo, aqui está o mesmo loop como um usuário de biblioteca real escreve. O [TRL](https://huggingface.co/docs/trl) da Hugging Face é a implementação de referência — `RewardTrainer` para o Estágio 2 e `PPOTrainer` (com KL-para-referência embutida) para o Estágio 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Três coisas que a biblioteca faz por você. `adap_kl_ctrl=True` implementa o esquema de β adaptativo: se a KL observada excede `target_kl`, β dobra; se abaixo da metade, reduz pela metade. O modelo de referência é congelado por convenção — você não deve acidentalmente compartilhar parâmetros com `policy`. E a cabeça de valor vive no mesmo backbone que a política (`AutoModelForCausalLMWithValueHead` anexa uma cabeça MLP escalar), que é por que o TRL relata `policy/kl` e `value/loss` separadamente.

## Armadilhas

- **Sobre-otimização / reward hacking.** O RM é imperfeito; `π_θ` encontra completações adversariais que pontuam alto mas são ruins. Sintomas: recompensa sobe indefinidamente enquanto a avaliação humana estagna ou cai. Correção: pare cedo, aumente `β`, amplie os dados de treino do RM.
- **Length hacking.** RMs treinados em respostas úteis implicitamente recompensam comprimento. A política aprende a encher respostas. Remediação: recompensa normalizada por comprimento, ou RLAIF com um RM ciente de comprimento.
- **RM muito pequeno.** O RM precisa ser pelo menos tão grande quanto a política. Um RM minúsculo não consegue pontuar fielmente as saídas da política.
- **Ajuste de KL.** β muito baixo → desvio e reward hacking. β muito alto → política quase não muda. O truque padrão é um β *adaptativo* que mira uma KL fixa por passo.
- **Ruído nos dados de preferência.** ~30% dos rótulos humanos são ruidosos ou ambíguos. Calibre treinando o RM em dados filtrados por concordância ou use uma temperatura no BT.
- **Problemas off-policy.** Os dados do PPO são levemente off-policy após a primeira época. Monitore a fração de clipping como na Lição 08.

## Use

O RLHF em 2026 é em camadas:

| Camada | Alvo | Método |
|--------|------|--------|
| Seguir instruções, utilidade, inofensividade | Alinhamento | DPO (Fase 10 · 08) preferível sobre RLHF-PPO. |
| Correção de raciocínio (matemática, código) | Capacidade | GRPO com recompensa de verificador (Fase 9 · 12). |
| Tarefas multi-passos de longo horizonte | Agente | PPO / GRPO com modelos de recompensa de processo sobre passos. |
| Comportamento de segurança / recusa | Segurança | RLHF-PPO com RM de segurança separado, ou Constitutional AI. |
| Best-of-N na inferência | Alinhamento rápido | Use RM no tempo de decodificação; sem treino de política. |
| Destilação de recompensa | Computação de inferência | Treine uma pequena "cabeça de recompensa" no topo de um LM congelado. |

RLHF foi *o* método em 2022–2024. Em 2026, pipelines de alinhamento de produção são DPO-primeiro, PPO-apenas para os passos intensivos em RM ou críticos para segurança.

## Entregue

Salve como `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Projete um pipeline de alinhamento RLHF / DPO / GRPO para um modelo de linguagem, incluindo RM, KL e estratégia de dados.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Dado um LM base, um comportamento alvo (alinhamento / raciocínio / recusa / agente) e um orçamento de preferência ou verificador, entregue:

1. Estágio. SFT? RM? DPO? GRPO? Com justificativa.
2. Fonte de preferência ou verificador. Humanos, feedback de IA, baseado em regras, teste-unitário-pass, ou destilação de recompensa.
3. Estratégia de KL. β fixo, β adaptativo, ou DPO (KL implícita).
4. Diagnósticos. KL média, estabilidade da recompensa, guarda contra sobre-otimização (avaliação humana holdout).
5. Portão de segurança. Conjunto red-team, taxa de recusa, RM de segurança separado do RM de utilidade.

Recuse entregar RLHF-PPO sem um monitor de KL. Recuse usar um RM menor que a política alvo. Recuse recompensas só de comprimento. Sinalize qualquer pipeline que não segure um conjunto de avaliação humana cega como sem proteção contra sobre-otimização.
```

## Exercícios

1. **Fácil.** Treine o modelo de recompensa Bradley-Terry em `code/main.py` em 500 pares sintéticos de preferência. Meça a acurácia pareada em 100 pares reservados. Deve exceder 90%.
2. **Médio.** Rode o loop RLHF-PPO de brinquedo com `β ∈ {0.0, 0.1, 1.0}`. Para cada um, plote pontuação do RM vs KL-para-referência ao longo das atualizações. Quais execuções fazem reward hacking?
3. **Difícil.** Implemente DPO (perda de verossimilhança de preferência em forma fechada) nos mesmos dados de preferência e compare ao pipeline RLHF-PPO em computação usada e pontuação RM final alcançada.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|---------------------------|
| RLHF | "RL de alinhamento" | Pipeline de três estágios SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "A rede de pontuação" | Função escalar aprendida ajustada a preferências pareadas via Bradley-Terry. |
| Bradley-Terry | "Perda logística pareada" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; o objetivo padrão do RM. |
| KL penalty | "Fique perto da referência" | `β · KL(π_θ \|\| π_ref)` na recompensa; o regularizador anti-reward-hacking. |
| Reward hacking | "Lei de Goodhart" | Política explora falhas do RM; sintomas: recompensa sobe, avaliação humana estagna. |
| RLAIF | "Preferências rotuladas por IA" | RLHF onde os rótulos vêm de outro LM em vez de humanos. |
| PRM | "Process Reward Model" | Pontua passos parciais de raciocínio; usado em pipelines de raciocínio. |
| Constitutional AI | "Método da Anthropic" | Preferências geradas por IA guiadas por regras explícitas. |

## Leitura Complementar

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — o paper que começou o RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — a receita por trás do ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — RLHF anterior para sumarização.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; o padrão pós-RLHF em 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF e loop de autocrítica.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — o paper HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — `RewardTrainer` e `PPOTrainer` de produção. Leia o código fonte do trainer para detalhes do KL adaptativo e cabeça de valor.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) por Lambert, Castricato, von Werra, Havrilla — a explicação canônica do pipeline de três estágios com diagramas.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — a biblioteca; `examples/` tem scripts RLHF completos para Llama, Mistral e Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — a visão da hipótese de recompensa; pré-requisito essencial para pensar sobre reward hacking.
