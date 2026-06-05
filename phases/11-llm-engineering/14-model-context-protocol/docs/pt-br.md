# Model Context Protocol (MCP)

> Toda aplicação LLM construída antes de 2025 inventou seu próprio schema de tools. Depois a Anthropic lançou MCP, Claude adotou, OpenAI adotou, e em 2026 é o formato padrão para conectar qualquer LLM a qualquer tool, fonte de dados ou agent. Escreva um MCP server e todo host fala com ele.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Fase 11 · 09 (Function Calling), Fase 11 · 03 (Structured Outputs)
**Tempo:** ~75 minutos

## O Problema

Você envia um chatbot que precisa de três tools: consulta ao banco, API de calendário e leitor de arquivos. Escreve três JSON schemas para Claude. Depois vendas quer as mesmas ferramentas no ChatGPT — reescreve para o parâmetro `tools` da OpenAI. Depois adiciona Cursor, Zed e Claude Code — mais três reescritas, cada uma com convenções JSON sutilmente diferentes. Uma semana depois, a Anthropic adiciona um campo novo; você atualiza seis schemas.

Esta era a realidade pré-2025. Todo host (a coisa rodando um LLM) e todo server (a coisa expondo ferramentas e dados) enviavam protocolos feitos sob medida. Escalar significava uma matriz de integração N×M.

Model Context Protocol colapsa essa matriz. Uma especificação baseada em JSON-RPC. Um server expõe tools, resources e prompts. Qualquer host compatível — Claude Desktop, ChatGPT, Cursor, Claude Code, Zed e uma longa cauda de frameworks de agentes — pode descobrir e chamar sem cola customizada.

## O Conceito

### Os Três Primitivos

Um MCP server expõe exatamente três coisas.

1. **Tools** — funções que o modelo pode chamar. Análogo dos `tools` da OpenAI ou `tool_use` da Anthropic. Cada uma tem nome, descrição, JSON Schema de entrada e um handler.
2. **Resources** — conteúdo read-only que o modelo ou usuário pode requisitar (arquivos, linhas de DB, respostas de API). Endereçados por URI.
3. **Prompts** — templates reutilizáveis que o usuário pode invocar como atalhos.

### O Formato de Transmissão

JSON-RPC 2.0 sobre stdio, WebSocket ou streamable HTTP. Cada mensagem é `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`. Métodos de descoberta: `tools/list`, `resources/list`, `prompts/list`. Métodos de invocação: `tools/call`, `resources/read`, `prompts/get`.

**Host vs client vs server.** O host é a aplicação LLM (Claude Desktop). O client é um subcomponente do host que fala com exatamente um server. O server é seu código. Um host pode montar muitos servers simultaneamente.

### O Handshake

Toda sessão abre com `initialize`. O client envia versão do protocolo e suas capacidades. O server responde com sua versão, nome e o conjunto de capacidades que suporta (`tools`, `resources`, `prompts`, `logging`, `roots`). Tudo depois é negociado contra essas capacidades.

### O Que MCP Não É

- Não é uma API de recuperação. RAG (Fase 11 · 06) ainda decide o que buscar; MCP é o transporte para expor resultados de recuperação como resources.
- Não é um framework de agentes. MCP é o encanamento; frameworks como LangGraph, PydanticAI e OpenAI Agents SDK ficam acima dele.
- Não é vinculado à Anthropic. A spec e implementações de referência são open source sob a organização `modelcontextprotocol`.

## Construa

### Passo 1: Um MCP server mínimo

O SDK oficial Python é `mcp`. O helper de alto nível `FastMCP` decora handlers.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Três decorators registram os três primitivos. As dicas de tipo se tornam o JSON Schema que o host vê. Rode sob Claude Desktop ou Claude Code.

### Passo 2: Chamando um MCP server de um host

O client Python oficial fala JSON-RPC. Com o SDK Anthropic leva dúzias de linhas.

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` retorna o mesmo schema que o LLM verá. Hosts de produção injetam esses schemas em cada turno para que o modelo possa emitir um bloco `tool_use` que o client então encaminha ao server.

### Passo 3: Transporte streamable HTTP

Stdio é bom para dev local. Para ferramentas remotas, use streamable HTTP — um POST por requisição, opcional Server-Sent Events para progresso.

```python
# Dentro do entrypoint do server
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Config de host (Claude Desktop `mcp.json` ou Claude Code `~/.mcp.json`):

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

O server mantém os mesmos decorators; só o transporte muda.

### Passo 4: Escopo e Segurança

Uma tool MCP é código arbitrário rodando no limite de confiança de outra pessoa. Três padrões obrigatórios:

- **Allowlists de capacidade.** Hosts expõem uma capacidade `roots` para que o server veja apenas caminhos permitidos. Reforce nos handlers de tool; não confie em caminhos fornecidos pelo modelo.
- **Humano no loop para mutação.** Tools read-only podem auto-executar. Tools write/delete devem exigir confirmação — hosts mostram uma UI de aprovação quando o server define `destructiveHint: true` nos metadados da tool.
- **Defesa contra tool poisoning.** Um resource malicioso pode conter instruções ocultas de injeção de prompt. Trate conteúdo de resource como dados não confiáveis; nunca deixe cruzar para território de system message.

## Pitfalls que ainda aparecem em 2026

- **Schema drift.** O modelo viu `tools/list` no turno 1. O conjunto de tools muda no turno 5. O modelo invoca uma tool que não existe mais. Hosts devem re-listar em `notifications/tools/list_changed`.
- **Blobs grandes de resource.** Despejar um arquivo de 2MB como resource desperdiça contexto. Pagine ou sum rootize server-side.
- **Muitos servers.** Montar 50 MCP servers estoura o orçamento de tools. A maioria dos modelos frontier degrada após ~40 tools.
- **Version skew.** Revisões da spec (2024-11, 2025-03, 2025-06, 2025-12) introduzem campos que quebram compatibilidade. Fixe a versão do protocolo em CI.
- **Deadlocks stdio.** Servers que logam para stdout corrompem o fluxo JSON-RPC. Log apenas para stderr.

## Use

| Situação | Escolha |
|----------|---------|
| Dev local, ferramentas single-user | Python `FastMCP`, transporte stdio |
| Tools remotas / SaaS | Streamable HTTP, OAuth 2.1 |
| Host TypeScript | `@modelcontextprotocol/sdk` |
| Servidor high-throughput | Rust SDK (`modelcontextprotocol/rust-sdk`) |
| Explorar ecossistema | `modelcontextprotocol/servers` monorepo |

Regra geral: se uma tool é read-only, cacheável e chamada por dois ou mais hosts, empacote como MCP server. Se é lógica inline única, mantenha como função local.

## Entregue

Salve `outputs/skill-mcp-server-designer.md`:

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Dado um domínio (API interna, banco de dados, fonte de arquivo) e os hosts que montarão o server, produza:

1. Mapa de primitivos. Quais capacidades viram `tools` (ação), quais viram `resources` (dados read-only), quais viram `prompts` (templates invocáveis pelo usuário). Uma linha por primitivo.
2. Plano de auth. Stdio (local confiável), streamable HTTP com chave API, ou OAuth 2.1 com PKCE. Escolha e justifique.
3. Rascunho de schema. JSON Schema para cada parâmetro de tool, com campos `description` ajustados para seleção de tool pelo modelo.
4. Lista de ações destrutivas. Toda tool que muta estado; exija `destructiveHint: true` e aprovação humana.
5. Plano de teste. Por tool: um teste de schema, um teste de round-trip através de um client MCP, um caso de red-team de injeção de prompt.

Recuse-se a enviar um server que escreve em disco ou chama APIs externas sem caminho de aprovação. Recuse-se a expor mais de 20 tools num server; divida em servers por domínio.
```

## Exercícios

1. **Fácil.** Estenda o `demo-server` com uma ferramenta `subtract`. Conecte do Claude Desktop. Confirme que o host capta a nova tool sem reiniciar emitindo `tools/list_changed`.
2. **Médio.** Adicione um `resource` expondo as últimas 100 linhas de `/var/log/app.log`. Reforce uma allowlist de roots para bloquear `../etc/passwd` mesmo se o modelo pedir.
3. **Difícil.** Construa um proxy MCP que multiplexa três servers upstream (Filesystem, GitHub, Postgres) em uma superfície agregada. Lidere colisões de nome e encaminhe `notifications/tools/list_changed` limpa mente.

## Termos-chave

| Termo | O que o pessoal diz | O que realmente significa |
|-------|---------------------|--------------------------|
| MCP | "Protocolo de ferramentas para LLMs" | Spec JSON-RPC 2.0 para expor tools, resources e prompts a qualquer host LLM |
| Host | "Claude Desktop" | A aplicação LLM — é dona do modelo e UI do usuário, monta um ou mais clients |
| Client | "Conexão" | Uma conexão por server dentro do host que fala JSON-RPC com exatamente um server |
| Server | "O thing com as tools" | Seu código; anuncia tools/resources/prompts e lida com a invocação deles |
| Tool | "Chamada de função" | Ação invocável pelo modelo com entrada JSON Schema e resultado texto/JSON |
| Resource | "Dados read-only" | Conteúdo endereçado por URI (arquivo, linha, resposta de API) que o host pode requisitar |
| Prompt | "Prompt salvo" | Template invocável pelo usuário (frequentemente com argumentos) exposto como comando |
| Transporte stdio | "Modo dev local" | Host pai executa o server como processo filho; JSON-RPC sobre stdin/stdout |
| Streamable HTTP | "O transporte remoto 2025-06" | POST para requisições, SSE opcional para mensagens iniciadas pelo server |

## Leitura Adicional

- [Model Context Protocol specificação](https://modelcontextprotocol.io/specification) — referência canônica, versionada por data
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — servers de referência (Filesystem, GitHub, Postgres, Slack, Puppeteer)
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — post de lançamento com fundamentos do design
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — SDK oficial usado nesta aula
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots, destructive hints, tool poisoning
- [Google A2A specificação](https://google.github.io/A2A/) — protocolo Agent2Agent; o padrão irmão para comunicação agente-agente
- [Anthropic — Building effective agents (Dez 2024)](https://www.anthropic.com/research/building-effective-agents) — onde MCP se encaixa no padrão mais amplo para design de agentes
