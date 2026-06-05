# Prompt Caching e Context Caching

> Seu system prompt tem 4.000 tokens. Seu contexto RAG tem 20.000 tokens. Você envia ambos em toda requisição. E paga por ambos — toda vez. Prompt caching permite que o provedor mantenha esse prefixo quente do lado deles e cobre 10% da taxa normal na reutilização. Usado corretamente, reduz custo de inferência em 50-90% e latência de primeiro token em 40-85%.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Fase 11 · 01 (Prompt Engineering), Fase 11 · 05 (Context Engineering), Fase 11 · 11 (Caching and Cost)
**Tempo:** ~60 minutos

## O Problema

Um coding agente envia o mesmo system prompt de 15.000 tokens ao Claude em cada turno de uma conversa. Vinte turnos a $3/M tokens de entrada = $0,90 só de custo de entrada — antes de qualquer mensagem real do usuário. Multiplique por 10.000 conversas diárias e a conta chega a $9.000/dia para texto que nunca muda.

Você não pode encolher o prompt sem prejudicar a qualidade. Não pode evitar enviá-lo — o modelo precisa dele em todo turno. A única jogada é parar de pagar preço cheio por um prefixo que o provedor já viu.

Essa jogada é prompt caching. Anthropic lançou em agosto de 2024 (com uma variante de TTL estendido de 1 hora em 2025), OpenAI automatizou no final daquele ano, Google lançou context caching explícito junto com Gemini 1.5, e todos os três agora oferecem como recurso de primeira classe em seus modelos frontier.

## O Conceito

### O Mecanismo

Quando o prefixo de uma requisição corresponde a um de uma requisição recente, o provedor serve o KV-cache da execução anterior em vez de re-encodar os tokens. Você paga um pequeno premium de escrita na primeira vez e um grande desconto de leitura todas as vezes seguintes.

### Três Estilos de Provedor em 2026

| Provedor | Estilo API | Desconto em hit | Premium de write | TTL padrão | Mínimo cacheável |
|----------|-----------|-----------------|-------------------|------------|-------------------|
| Anthropic | Marcadores `cache_control` explícitos | 90% off entrada | 25% extra | 5 min (extensível 1h) | 1.024 tokens (Sonnet/Opus), 2.048 (Haiku) |
| OpenAI | Detecção de prefixo automática | 50% off entrada | nenhum | Até 1 hora (best-effort) | 1.024 tokens |
| Google (Gemini) | API `CachedContent` explícita | Leitura a ~25% da taxa normal | Taxa de armazenamento por token·hora | Configurável (padrão 1h) | 4.096 tokens (Flash), 32.768 (Pro) |

### O Invariante

Todos os três fazem cache apenas de prefixos. Se qualquer token difere entre requisições, tudo após o primeiro token diferente é miss. Coloque as partes *estáveis* no topo, as *variáveis* no fundo.

### O Layout Amigável a Cache

```
[system prompt]          <-- cacheie isto
[definições de tools]    <-- cacheie isto
[exemplos few-shot]      <-- cacheie isto
[documentos recuperados] <-- cacheie se reutilizado, senão não
[histórico da conversa]  <-- cacheie até último turno
[mensagem do usuário]    <-- nunca cacheie (diferente toda vez)
```

Viole a ordem — coloque a mensagem do usuário acima do system prompt, intercale recuperações dinâmicas entre exemplos few-shot — e o cache nunca acerta.

### O Cálculo de Break-Even

O premium de write de 25% da Anthropic significa que um bloco cacheado precisa ser lido pelo menos duas vezes para economizar líquido. 1 escrita + 1 leitura custa em média 0.675x por requisição (economiza 32%); 1 escrita + 10 leituras custa em média 0.205x (economiza 80%). Regra geral: cacheie qualquer coisa que você espera reutilizar pelo menos 3 vezes dentro do TTL.

## Construa

### Passo 1: Anthropic prompt caching com marcadores explícitos

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "Você é um revisor Python sênior. Siga a rubrica exatamente.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

O marcador `cache_control` diz à Anthropic para armazenar o bloco por 5 minutos. Reuso dentro dessa janela acerta; reuso após expira escreve novamente.

**Campos de uso na resposta:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # pago a 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # pago a 0.1x
```

Verifique ambos campos em CI — se `cache_read_input_tokens` fica em zero entre requisições, suas chaves de cache estão à deriva.

### Passo 2: TTL estendido de uma hora

Para jobs batch de longa duração, o padrão de 5 minutos expira entre jobs. Defina `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL de 1 hora custa 2x o premium de write (50% sobre a linha base em vez de 25%) mas se paga rápido em qualquer batch que reutiliza o prefixo mais de 5 vezes.

### Passo 3: OpenAI caching automático

OpenAI não dá nada para configurar. Qualquer prefixo acima de 1.024 tokens que corresponda a uma requisição recente recebe 50% de desconto automaticamente.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # longo e estável
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # a porção com desconto
```

A mesma regra de layout amigável a cache se aplica. Duas coisas matam o cache da OpenAI que não matam o da Anthropic: mudar o campo `user` (usado como componente de chave de cache) e reordenar tools.

### Passo 4: Gemini context caching explícito

Gemini trata o cache como um objeto de primeira classe que você cria e nomeia:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Revise este código:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini cobra armazenamento por token·hora enquanto o cache viver, e leituras a ~25% da taxa normal de entrada. Este é o formato certo quando você reutiliza o mesmo prompt gigante entre muitas sessões ao longo de dias.

### Passo 5: Medindo hit rate em produção

Veja `code/main.py` para um contador simulado de três provedores que rastreia contagens de write/read/miss e computa custo combinado por 1K requisições. Gate deploys numa taxa de hit alvo — a maioria dos setups Anthropic de produção deve ver >80% de fração de leitura após warmup.

## Pitfalls que ainda aparecem em 2026

- **Timestamps dinâmicos no topo.** `"Hora atual: 2026-04-22 15:30:02"` no topo do system prompt. Toda requisição erra. Mova timestamps para abaixo do breakpoint de cache.
- **Reordenação de tools.** Serialize tools em ordem estável — um reembaralhamento de dict entre deploys quebra todo hit.
- **Quase-duplicatas de texto livre.** "Você é útil." vs "Você é um assistente útil." — um byte de diferença = miss completo.
- **Blocos pequenos demais.** Anthropic impõe um piso de 1.024 tokens (2.048 para Haiku). Blocos menores silenciosamente não são cacheados.
- **Dashboards de custo cegos.** Divida "tokens de entrada" em cacheados vs não cacheados. Caso contrário, uma queda de tráfego parece uma vitória de cache.

## Use

A stack de caching em 2026:

| Situação | Escolha |
|----------|---------|
| Agente com system prompt estável de 10k+, muitos turnos | Anthropic `cache_control` com TTL 5 min |
| Job batch reutilizando um prefixo por 30+ minutos | Anthropic com `ttl: "1h"` |
| Endpoints serverless no GPT-5, sem infra customizada | OpenAI automático (apenas torne seu prefixo estável e longo) |
| Reuso multi-dia de um corpus gigante de código/docs | Gemini explícito `CachedContent` |
| Fallback entre provedores | Mantenha o layout do prefixo cacheável idêntico entre provedores |

Combine com semantic caching (Fase 11 · 11) para a camada de mensagem do usuário: prompt caching lida com reuso *token-idêntico*, semantic caching lida com reuso *significado-idêntico*.

## Entregue

Salve `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Dado um prompt (system + tools + few-shot + retrieval + history + user) e um perfil de uso (requisições por hora, TTL necessário, provedor), produza:

1. Layout. Seções reordenadas com um único breakpoint de cache marcado; explique quais seções são estáveis, quais são voláteis.
2. Modo de provedor. Anthropic cache_control, OpenAI automático ou Gemini CachedContent. Justifique a partir de TTL e padrão de reuso.
3. Break-even. Leituras esperadas por escrita dentro do TTL; custo líquido vs sem-cache com matemática.
4. Plano de verificação. Asserção em CI de que cache_read_input_tokens > 0 na segunda requisição idêntica; dashboard dividido por tokens cacheados vs não cacheados.
5. Modos de falha. Liste as três razões mais prováveis pelas quais o cache vai errar neste setup e como você vai prevenir cada uma.

Recuse-se a enviar um plano de cache que coloque um campo dinâmico acima do breakpoint. Recuse-se a habilitar TTL de 1h sem uma contagem de reuso que faça o premium de write 2x se pagar.
```

## Exercícios

1. **Fácil.** Rode uma conversa de 10 turnos com system prompt de 5.000 tokens contra Claude. Sem `cache_control` e depois com. Reporte o custo de tokens de entrada de cada.
2. **Médio.** Escreva um test harness que, dado um template de prompt e um log de requisições, calcula a taxa de hit esperada e economia em dólar por provedor (Anthropic 5m, Anthropic 1h, OpenAI automático, Gemini explícito).
3. **Difícil.** Construa um otimizador de layout: dado um prompt e uma lista de campos marcados `stable=True/False`, reescreva o prompt colocando um único breakpoint de cache na posição máxima amigável sem perder informação. Verifique num endpoint Anthropic real.

## Termos-chave

| Termo | O que o pessoal diz | O que realmente significa |
|-------|---------------------|--------------------------|
| Prompt caching | "Torna prompts longos baratos" | Reutilizar KV-cache do lado do provedor para prefixos correspondentes; 50-90% de desconto em tokens de entrada repetidos |
| `cache_control` | "O marcador Anthropic" | Atributo de content-block que declara "tudo até aqui é cacheável"; `{"type": "ephemeral"}` |
| Cache write | "Pagando o premium" | Primeira requisição que popula o cache; cobrado a ~1.25x taxa de entrada na Anthropic, grátis na OpenAI |
| Cache read | "O desconto" | Requisições seguintes que batem no prefixo; cobrado a 10% (Anthropic), 50% (OpenAI), ~25% (Gemini) |
| TTL | "Quanto tempo vive" | Segundos que o cache fica quente; Anthropic 5m padrão (extensível 1h), OpenAI best-effort até 1h, Gemini configurável |
| Extended TTL | "Cache de 1 hora Anthropic" | `{"type": "ephemeral", "ttl": "1h"}`; 2x premium de write mas vale a pena para reuso em batch |
| Prefix match | "Por que meu cache errou" | Caches só acertam quando cada token desde o início até o breakpoint é byte-idêntico |
| Context caching (Gemini) | "O explícito" | Objeto de cache nomeado e cobrado por armazenamento do Google; melhor para reuso multi-dia de corpora grandes |

## Leitura Adicional

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, TTL 1h, tabelas break-even
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — correspondência automática de prefixo
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — API `CachedContent` e pricing de armazenamento
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — post original com números de latência
- Fase 11 · 05 (Context Engineering) — onde cortar o prompt para o cache poder pousar
- Fase 11 · 11 (Caching and Cost) — combine prompt caching com um cache semântico em mensagens do usuário
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — o modelo de memória KV-cache que prompt caching expõe aos usuários
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill é a fase que prompt caching acelera
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching ao lado de speculative decoding, Flash Attention e MQA/GQA
