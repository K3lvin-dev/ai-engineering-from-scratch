# LangGraph — Máquinas de Estado para Agentes

> Um loop ReAct escrito à mão é um `while True`. Um loop ReAct escrito em LangGraph é um gráfico que você pode checkpointar, interromper, ramificar e viajar no tempo. O agente não mudou. A infraestrutura ao redor dele mudou.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Fase 11 · 09 (Chamada de Função), Fase 11 · 14 (Model Context Protocol)
**Tempo:** ~75 minutos

## O Problema

Você entrega um agente com chamada de função. Funciona por três turnos, então algo dá errado: o modelo tenta uma ferramenta que retorna 500, o usuário muda de ideia no meio da tarefa, ou o agente decide reembolsar um pedido sem aprovação humana. O loop `while True:` não tem ganchos. Você não consegue pausá-lo, não consegue rebobiná-lo, e não consegue ramificar para "e se o modelo tivesse escolhido a outra ferramenta." No momento em que você entrega isso além de uma demo, o agente vira uma caixa preta que ou funcionou ou não.

O próximo passo é óbvio quando você vê. O agente já é uma máquina de estado — prompt de sistema mais histórico de mensagens mais chamadas de ferramenta pendentes mais a próxima ação. Torne a máquina de estado explícita: nós para "o modelo pensa," "uma ferramenta executa," "um humano aprova," e arestas para as transições condicionais entre eles. Uma vez que o gráfico é explícito, a infraestrutura ganha quatro coisas de graça: checkpointing (salvar estado entre passos), interrupções (pausar para um humano), streaming (transmitir tokens e eventos intermediários), e viagem no tempo (rebobinar para um estado anterior e tentar uma ramificação diferente).

LangGraph é a biblioteca que entrega essa abstração. Não é um framework de agentes no sentido do LangChain ("aqui está um AgentExecutor, boa sorte"). É um runtime de gráficos com estado de primeira classe, persistência de primeira classe e interrupções de primeira classe. O loop do agente é algo que você desenha, não algo que você escreve à mão.

## O Conceito

![StateGraph do LangGraph: nós, arestas e o checkpointer](../assets/langgraph-stategraph.svg)

Um `StateGraph` tem três coisas.

1. **Estado.** Um dicionário tipado (TypedDict ou modelo Pydantic) que flui pelo gráfico. Cada nó recebe o estado completo e retorna uma atualização parcial, que o LangGraph mescla usando um *redutor* por campo — `operator.add` para listas que devem acumular, sobrescrita por padrão.
2. **Nós.** Funções Python `state -> partial_state`. Cada um é um passo discreto: "chamar o modelo," "executar ferramentas," "sumarizar."
3. **Arestas.** Transições entre nós. Arestas estáticas vão para um lugar. Arestas condicionais usam uma função roteadora `state -> nome_do_próximo_nó` para que o gráfico possa ramificar baseado na saída do modelo.

Você compila o gráfico. Compilar vincula a topologia, anexa um checkpointer (opcional mas essencial para produção) e retorna algo executável. Você o invoca com um estado inicial e um `thread_id`. Cada passo de execução persiste um checkpoint chaveado por `(thread_id, checkpoint_id)`.

### Os quatro superpoderes

**Checkpointing.** Cada transição de nó escreve o novo estado em um armazenamento (em memória para testes, Postgres/Redis/SQLite para produção). Retome chamando o gráfico novamente com o mesmo `thread_id`. O gráfico continua de onde parou.

**Interrupções.** Marque um nó com `interrupt_before=["human_review"]` e a execução para antes desse nó executar. O estado persiste. Sua API responde ao usuário com "aguardando aprovação." Uma requisição posterior ao mesmo `thread_id` com `Command(resume=...)` retoma a execução.

**Streaming.** `graph.stream(state, mode="updates")` produz deltas de estado conforme acontecem. `mode="messages"` transmite os tokens do LLM dentro dos nós do modelo. `mode="values"` produz instantâneos completos. Você escolhe o que mostrar na sua interface.

**Viagem no tempo.** `graph.get_state_history(thread_id)` retorna o registro completo de checkpoints. Passe qualquer `checkpoint_id` anterior para `graph.invoke` e você bifurca a partir daquele ponto. Ótimo para depuração ("e se o modelo tivesse escolhido a ferramenta B?") e para testes de regressão que reproduzem traços de produção.

### Redutores são o ponto

Cada campo de estado tem um redutor. A maioria dos padrões é suficiente — um novo valor sobrescreve o antigo. Mas listas de mensagens precisam de `operator.add` para que novas mensagens sejam anexadas em vez de substituir. Arestas paralelas mesclam suas atualizações através do redutor. Se dois nós ambos atualizam `messages` e você esqueceu o `Annotated[list, add_messages]`, o segundo vence silenciosamente e você perde metade do turno. O redutor é a única coisa sutil na biblioteca; acerte ele e o resto se compõe.

### O gráfico ReAct em quatro nós

Um agente ReAct de produção tem quatro nós e duas arestas:

1. `agent` — chama o LLM com o histórico de mensagens atual. Retorna a mensagem do assistente (que pode conter tool_calls).
2. `tools` — executa quaisquer tool_calls na última mensagem do assistente, anexa os resultados das ferramentas como mensagens de ferramenta.
3. Uma aresta condicional de `agent` que roteia para `tools` se a última mensagem tiver tool_calls, senão para `END`.
4. Uma aresta estática de `tools` de volta para `agent`.

É isso. Você obtém o loop ReAct completo (Pensamento → Ação → Observação → Pensamento → ...) com checkpointing, interrupções e streaming, em aproximadamente 40 linhas de código.

### StateGraph vs Send (fanout)

`Send(node_name, state)` permite que um nó despache subgráficos paralelos. Exemplo: o agente decide consultar três recuperadores de uma vez. Cada `Send` spawna uma execução paralela do nó alvo; suas saídas se mesclam através do redutor de estado. É assim que o LangGraph expressa o padrão orquestrador-trabalhadores sem primitivas de threading.

### Subgráficos

Um gráfico compilado pode ser um nó em outro gráfico. O gráfico externo vê um único nó; o gráfico interno tem seu próprio estado e seus próprios checkpoints. É assim que equipes constroem agentes supervisor-trabalhador: o gráfico supervisor roteia a intenção do usuário para um subgráfico trabalhador por domínio.

## Construa

### Passo 1: estado e nós

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` é o redutor que faz a lista de mensagens acumular em vez de sobrescrever. Esquecê-lo é o bug mais comum do LangGraph.

### Passo 2: execute com uma thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("encontre o endereço da sede da Anthropic")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Cada atualização é um dict `{node_name: state_delta}`. Seu frontend pode transmitir estas para a UI para que usuários vejam "agente está pensando… chamando search_web… obteve resultado… respondendo."

### Passo 3: adicione uma interrupção humano-no-loop

Marque um nó para que a execução pause antes dele executar.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pausar antes de cada chamada de ferramenta
)

state = app.invoke({"messages": [HumanMessage("delete o banco de dados de produção")]}, config)
# state["__interrupt__"] está definido. Inspecione as chamadas de ferramenta propostas.
# Se aprovado:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# Se negado: escreva uma mensagem de rejeição e retome
app.update_state(config, {"messages": [AIMessage("Bloqueado pelo revisor humano.")]})
```

O estado, o checkpoint e a thread persistem durante a interrupção. Nada fica na memória exceto durante a execução.

### Passo 4: viagem no tempo para depuração

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Bifurque a partir de um checkpoint anterior
target = history[3].config  # três passos atrás
for event in app.stream(None, target, stream_mode="values"):
    pass  # reproduzir daquele ponto em diante
```

Passar `None` como entrada reproduz a partir do checkpoint dado; passar um valor o anexa como uma atualização ao estado daquele checkpoint antes de retomar. É assim que você reproduz uma execução ruim do agente sem reexecutar a conversa inteira.

### Passo 5: troque o checkpointer para produção

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis e Postgres são fornecidos. `MemorySaver` é para testes. Qualquer coisa que persista entre reinicializações quer um armazenamento real.

## A Habilidade

> Você constrói agentes como gráficos, não como loops `while True`.

Antes de buscar o LangGraph, faça um design de 60 segundos:

1. **Nomeie os nós.** Toda decisão discreta ou ação com efeito colateral é um nó. "Agente pensa," "ferramenta executa," "revisor aprova," "resposta flui." Se você não consegue listá-los, a tarefa ainda não tem formato de agente.
2. **Declare o estado.** TypedDict mínimo com um redutor para cada campo de lista. Não enfie tudo em `messages`; eleve campos específicos da tarefa (um `plan` de trabalho, um contador de `budget`, uma lista de `retrieved_docs`) para o nível superior.
3. **Desenhe as arestas.** Estáticas a menos que o próximo passo dependa da saída do modelo. Toda aresta condicional precisa de uma função roteadora com ramificações nomeadas.
4. **Escolha um checkpointer antecipadamente.** `MemorySaver` para testes, Postgres/Redis/SQLite para qualquer outra coisa. Não entregue sem um — sem checkpointer não há retomada, não há interrupção, não há viagem no tempo.
5. **Decida interrupções antes das ferramentas executarem, não depois.** Aprovações vão na aresta que entra em um nó com efeito colateral para que você possa cancelar antes do dano; validação vai na aresta que sai do modelo para que você possa rejeitar chamadas ruins barato.
6. **Faça streaming por padrão.** `mode="updates"` para a UI, `mode="messages"` para streaming em nível de token dentro de nós do modelo, `mode="values"` para instantâneos completos durante avaliação.

Recuse-se a entregar um agente LangGraph que não tem checkpointer. Recuse-se a entregar um que interrompe *após* o efeito colateral. Recuse-se a entregar um campo `messages` sem `add_messages` como seu redutor.

## Exercícios

1. **Fácil.** Implemente o gráfico ReAct de quatro nós acima com uma ferramenta de calculadora e uma ferramenta de busca web. Verifique se `list(app.get_state_history(config))` retorna pelo menos quatro checkpoints para uma conversa de dois turnos.
2. **Médio.** Adicione um nó `planner` que executa antes de `agent` e escreve um `plan: list[str]` estruturado no estado. Faça com que `agent` marque passos do plano como concluídos. Falhe o teste se `plan` for perdido após uma retomada de checkpoint (redutor errado).
3. **Difícil.** Construa um gráfico supervisor que roteia entre três subgráficos (`researcher`, `writer`, `reviewer`) usando `Send`. Cada subgráfico tem seu próprio estado e checkpointer. Adicione um `interrupt_before=["writer"]` no gráfico externo para que um humano possa aprovar o resumo da pesquisa. Confirme que a viagem no tempo a partir de um checkpoint anterior reexecuta apenas a ramificação bifurcada.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|--------------------------|
| StateGraph | "O gráfico do LangGraph" | O objeto construtor ao qual você adiciona nós e arestas antes de compilar. |
| Redutor | "Como o campo mescla" | Uma função `(antigo, novo) -> mesclado` aplicada quando um nó retorna uma atualização para aquele campo; padrão é sobrescrever, `add_messages` anexa. |
| Thread | "Um ID de conversa" | Uma string `thread_id` que escopa todos os checkpoints para uma sessão. |
| Checkpoint | "Um estado pausado" | Um instantâneo persistido do estado completo do gráfico após uma transição de nó, chaveado por `(thread_id, checkpoint_id)`. |
| Interrupção | "Pausar para um humano" | `interrupt_before` / `interrupt_after` para a execução em um limite de nó; retome com `Command(resume=...)`. |
| Viagem no tempo | "Bifurcar a partir de um passo anterior" | `graph.invoke(None, config_with_old_checkpoint_id)` reproduz a partir daquele checkpoint em diante. |
| Send | "Despacho paralelo de subgráfico" | Um construtor que um nó pode retornar para spawnar N execuções paralelas de um nó alvo. |
| Subgráfico | "Um gráfico compilado como nó" | Um StateGraph compilado usado como nó em outro gráfico; preserva seu próprio escopo de estado. |

## Leitura Adicional

- [Documentação do LangGraph](https://langchain-ai.github.io/langgraph/) — referência canônica para StateGraph, redutores, checkpointers e interrupções.
- [Conceitos do LangGraph: estado, redutores, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — o modelo mental que esta lição usa, direto da fonte.
- [Persistência e Checkpoints no LangGraph](https://langchain-ai.github.io/langgraph/concepts/persistence/) — os detalhes sobre armazenamentos Postgres/SQLite/Redis, namespaces de checkpoint e IDs de thread.
- [Humano-no-loop no LangGraph](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)` e o padrão de editar estado.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — o padrão que todo agente LangGraph implementa; leia para entender a lógica do traço de raciocínio.
- [Anthropic — Building effective agents (Dez 2024)](https://www.anthropic.com/research/building-effective-agents) — quais formatos de gráfico (cadeia, roteador, orquestrador-trabalhadores, avaliador-otimizador) preferir e quando.
- Fase 11 · 09 (Chamada de Função) — a primitiva de chamada de ferramenta que todo nó de agente LangGraph reutiliza.
- Fase 11 · 14 (Model Context Protocol) — descoberta externa de ferramentas que se conecta a um `ToolNode` do LangGraph via adaptador MCP.
- Fase 11 · 17 (Compensações de frameworks de agentes) — quando escolher LangGraph sobre CrewAI, AutoGen ou Agno.
