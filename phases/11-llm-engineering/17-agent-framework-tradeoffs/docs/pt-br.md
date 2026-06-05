# Compensações de Frameworks de Agentes — LangGraph vs CrewAI vs AutoGen vs Agno

> Todo framework vende a mesma demo (agente de pesquisa constrói um relatório) e esconde o mesmo bug (o esquema de estado briga com a camada de orquestração). Escolha o framework cujas abstrações combinam com a forma do seu problema; todo o resto é cola que você escreve duas vezes.

**Tipo:** Aprendizado
**Linguagens:** Python
**Pré-requisitos:** Fase 11 · 09 (Chamada de Função), Fase 11 · 16 (LangGraph)
**Tempo:** ~45 minutos

## O Problema

Você tem uma tarefa que precisa de mais de uma chamada de LLM. Talvez seja um fluxo de pesquisa (planejar, buscar, sumarizar, citar). Talvez seja um pipeline de revisão de código (analisar diff, criticar, corrigir, validar). Talvez seja um assistente de múltiplos turnos que reserva voos, escreve e-mails e arquiva relatórios de despesas. Você escolhe um framework.

Três dias depois, você descobre que as abstrações do framework vazam. CrewAI te dá papéis mas briga com você quando o "pesquisador" precisa passar um plano estruturado para o "escritor." AutoGen te dá chat entre agentes mas não tem estado de primeira classe, então seu checkpoint é um pickle de um log de conversa. LangGraph te dá um gráfico de estado mas força você a nomear toda transição antes de saber o que o agente vai fazer. Agno te dá uma abstração de agente único que grita quando você tenta expandir para três trabalhadores concorrentes.

A solução não é "escolher o melhor framework." É combinar a abstração central do framework com a forma do seu problema. Esta lição desenha esse mapa.

## O Conceito

![Matriz de frameworks de agentes: abstração central vs forma do problema](../assets/framework-matrix.svg)

Quatro frameworks dominam o cenário de 2026. Suas abstrações centrais não são as mesmas.

| Framework | Abstração central | Melhor ajuste | Pior ajuste |
|-----------|-------------------|---------------|-------------|
| **LangGraph** | `StateGraph` — estado tipado, nós, arestas condicionais, checkpointer. | Fluxos de trabalho com estado explícito e interrupções humano-no-loop; agentes de produção precisando de depuração com viagem no tempo. | Brainstorming solto, orientado a papéis, onde a topologia é desconhecida. |
| **CrewAI** | `Crew` — papéis (objetivo, história), tarefas, processo (sequencial ou hierárquico). | Fluxos de trabalho orientados a papéis ou persona com um plano linear/hierárquico curto. | Qualquer coisa com estado além do histórico de turnos da equipe; ramificação complexa. |
| **AutoGen** | Par `ConversableAgent` — dois ou mais agentes que falam em turnos até uma condição de saída. | *Diálogo* multi-agente (professor-aluno, proponente-crítico, ator-revisor) onde o pensamento emerge do chat. | Fluxos de trabalho determinísticos com um DAG conhecido; qualquer coisa que precise de estado durável entre reinicializações. |
| **Agno** | `Agent` — um único LLM + ferramentas + memória, composível em equipes. | Agentes únicos de construção rápida e equipes leves; forte multimodalidade e drivers de armazenamento embutidos. | Gráficos profundos com ramificações explícitas e redutores customizados. |

### O que "abstração" realmente significa

A abstração central de um framework é a coisa que você desenha no quadro branco quando apresenta a arquitetura.

- **LangGraph** → você desenha um gráfico. Nós são passos, arestas são transições, e o objeto de estado em cada ponto é tipado. O modelo mental é uma máquina de estados.
- **CrewAI** → você desenha um organograma. Cada papel tem uma descrição de cargo e um gerente roteia tarefas. O modelo mental é uma pequena equipe de especialistas.
- **AutoGen** → você desenha um direct do Slack. Dois agentes trocam mensagens; um terceiro entra se você precisar de um moderador. O modelo mental é chat.
- **Agno** → você desenha uma caixa única com ferramentas penduradas nela. Coloque caixas lado a lado para formar uma equipe. O modelo mental é "agente com baterias incluídas."

### A questão do estado

Estado é onde a maioria das escolhas de frameworks quebra em produção.

- **LangGraph.** Estado tipado (`TypedDict` ou modelo Pydantic), redutores por campo, checkpointer de primeira classe (SQLite/Postgres/Redis). Retomar, interromper e viajar no tempo são gratuitos. *(Veja Fase 11 · 16.)*
- **CrewAI.** Estado flui como strings entre tarefas via campo `context`, ou estruturado através de `output_pydantic`. Sem armazenamento durável por equipe pronto para uso; você adapta o seu se a equipe precisar sobreviver a uma reinicialização.
- **AutoGen.** Estado é o histórico do chat e qualquer `context` definido pelo usuário. Transcrições de conversa persistem; estado de fluxo de trabalho arbitrário não, a menos que você escreva adaptadores.
- **Agno.** Drivers de armazenamento embutidos (SQLite, Postgres, Mongo, Redis, DynamoDB) anexados a um `Agent` via `storage=` — sessões de conversa e memórias de usuário persistem automaticamente. Não é um checkpointer de gráfico completo; é um armazenamento de sessão.

### A questão da ramificação

Todo agente não trivial ramifica. Quem decide a ramificação importa.

- **LangGraph** — você decide, via arestas condicionais. Roteamento é uma função Python com ramificações nomeadas. Ramificações são de primeira classe no gráfico compilado; o checkpointer registra qual ramificação foi tomada.
- **CrewAI** — o gerente decide no modo hierárquico; no modo sequencial você decide no momento da construção. Roteamento é implícito na lista de tarefas; não há um "se" de primeira classe fora do prompt do gerente.
- **AutoGen** — os agentes decidem via chat. Ramificação emerge de quem fala em seguida. `GroupChatManager` seleciona o próximo falante; você pode escrever um `speaker_selection_method` manualmente, mas o padrão é orientado por LLM.
- **Agno** — o agente decide por qual ferramenta chamar em seguida. Equipes têm um modo coordenador/roteador/colaborador; ramificação além disso é responsabilidade do desenvolvedor.

### A questão da observabilidade

- **LangGraph** — OpenTelemetry via LangSmith ou qualquer exportador OTel. Cada transição de nó é um span de rastreamento; checkpoints dobram como rastros reproduzíveis. LangSmith é a opção de primeira parte; Langfuse/Phoenix também têm adaptadores.
- **CrewAI** — OpenTelemetry de primeira classe desde o final de 2025; integrações com Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — Integração OpenTelemetry via `autogen-core`; AgentOps e Opik têm conectores. Granularidade de rastreamento é por mensagem de agente, não por nó.
- **Agno** — Flag `monitoring=True` embutida mais exportadores OpenTelemetry; integração estreita com Langfuse para rastros de sessão.

### Custo e latência

Todos os quatro frameworks adicionam sobrecarga por chamada (lógica do framework, validação, serialização). Ordem aproximada de sobrecarga crescente: Agno ≈ LangGraph < CrewAI ≈ AutoGen. A diferença é dominada por quanto roteamento extra de LLM o framework faz. O gerente hierárquico do CrewAI gasta tokens decidindo quem vai em seguida; o `GroupChatManager` do AutoGen também. LangGraph só gasta tokens onde você escreve `llm.invoke`. O caminho de agente único do Agno é enxuto.

Quando o custo por execução importa, prefira roteamento explícito (arestas do LangGraph, `speaker_selection_method` do AutoGen) em vez de roteamento selecionado por LLM.

### Interoperabilidade

- **LangGraph** ↔ **LangChain** ferramentas, recuperadores, LLMs. Adaptador MCP de primeira classe (ferramentas importadas como servidores MCP).
- **CrewAI** ↔ ferramentas herdam de `BaseTool`; ferramentas LangChain, ferramentas LlamaIndex e ferramentas MCP todas se adaptam. Delegação equipe-a-equipe via `allow_delegation=True`.
- **AutoGen** → `FunctionTool` envolve qualquer chamável Python; adaptador MCP disponível. Acoplamento estreito ao ecossistema AG2 para padrões agente-a-agente.
- **Agno** → Decorador `@tool` ou subclasse BaseTool; adaptador MCP; ferramentas podem ser compartilhadas entre agentes e equipes.

## A Habilidade

> Você consegue explicar, em uma frase, por que um dado framework é certo para um dado problema de agente.

Lista de verificação pré-construção:

1. **Desenhe a forma.** Isso é um gráfico (estado tipado, transições nomeadas)? Uma interpretação de papéis (especialistas passam trabalho)? Um chat (agentes conversam até terminar)? Um agente único com ferramentas?
2. **Decida quem ramifica.** Ramificação decidida pelo desenvolvedor → LangGraph. Ramificação decidida pelo gerente → CrewAI hierárquico. Ramificação emergente do chat → AutoGen. Ramificação decidida por chamada de ferramenta → Agno.
3. **Verifique o orçamento de estado.** Você precisa de retomada a partir de checkpoint? Viagem no tempo? Interrupções humanas no meio da execução? Se sim, LangGraph é o padrão; sessões do Agno cobrem estado no escopo da conversa.
4. **Verifique o orçamento de custo.** Roteamento selecionado por LLM custa tokens extras por turno. Se o agente executa milhares de vezes por dia, prefira roteamento explícito.
5. **Orçamente a sobrecarga do framework.** Todo framework é outra dependência. Se a tarefa é duas chamadas de LLM e uma ferramenta, escreva 30 linhas de Python puro; nenhum framework é mais barato que nenhum framework.

Recuse-se a buscar um framework antes de conseguir desenhar o gráfico, o organograma, o chat ou a caixa do agente. Recuse-se a escolher um que force você a lutar contra seu modelo de estado para a coisa que você realmente precisa.

## A Matriz de Decisão

| Forma do problema | Framework preferido | Por quê |
|-------------------|---------------------|---------|
| DAG de fluxo de trabalho com estado tipado, aprovações humanas, longa duração | LangGraph | Estado de primeira classe, checkpointer, interrupções, viagem no tempo. |
| Pipeline de pesquisa / escrita com papéis distintos | CrewAI (sequencial) ou subgráficos LangGraph | Papel-por-tarefa é barato de expressar no CrewAI; escale com LangGraph quando a ramificação ficar complexa. |
| Diálogo proponente-crítico ou professor-aluno | AutoGen | Chat entre dois agentes é sua forma nativa. |
| Agente único com ferramentas, sessões, memória | Agno | Configuração mais enxuta, armazenamento e memória embutidos. |
| Milhares de fanouts paralelos com redutores | LangGraph + `Send` | O único com uma API de despacho paralelo de primeira classe. |
| Protótipo rápido, sem compromisso com framework | Python puro + SDK do provedor | Nenhum framework é o framework mais rápido. |

## Exercícios

1. **Fácil.** Pegue a mesma tarefa — "pesquise a sede da Anthropic, escreva um resumo de 200 palavras, cite fontes" — e implemente em LangGraph (quatro nós: planejar, buscar, escrever, citar) e em CrewAI (três papéis: pesquisador, escritor, editor). Reporte custo de token por execução e linhas de código.
2. **Médio.** Construa a mesma tarefa em AutoGen (chat pesquisador ↔ escritor, editor entra via `GroupChat`) e Agno (um único agente com `search_tools` e `write_tools`, mais um armazenamento de sessão). Classifique as quatro implementações em (a) custo por execução, (b) capacidade de retomar após uma queda, (c) capacidade de injetar uma aprovação humana antes do passo de escrita.
3. **Difícil.** Construa um script de árvore de decisão `pick_framework.py` que recebe uma descrição curta do problema (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) e retorna uma recomendação com justificativa de uma frase. Verifique em seis casos que você mesmo projetar.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|--------------------------|
| Orquestração | "Como os agentes coordenam" | A camada que decide qual nó/papel/agente executa em seguida. |
| Estado durável | "Retomar após uma reinicialização" | Estado que sobrevive à morte do processo, anexado a um checkpoint ou armazenamento de sessão. |
| Roteamento selecionado por LLM | "Deixar o modelo decidir" | Um LLM planejador escolhe o próximo passo a cada turno; flexível mas paga tokens em cada decisão. |
| Roteamento explícito | "Desenvolvedor decide" | Uma função Python ou aresta estática escolhe o próximo passo; barato e auditável. |
| Crew | "Uma equipe do CrewAI" | Papéis + tarefas + processo (sequencial ou hierárquico) vinculados em um único executável. |
| GroupChat | "Chat multi-agente do AutoGen" | Uma conversa gerenciada entre N agentes com um seletor de falante. |
| Team (Agno) | "Agno multi-agente" | Modo de rotear / coordenar / colaborar sobre um conjunto de agentes. |
| StateGraph | "O gráfico do LangGraph" | Abstração de estado tipado, nó, aresta condicional, checkpointer. |

## Leitura Adicional

- [Documentação do LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, interrupções, viagem no tempo.
- [Documentação do CrewAI](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes.
- [Documentação do AutoGen](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools.
- [Documentação do Agno](https://docs.agno.com/) — Agent, Team, Workflow, storage, memory.
- [Anthropic — Building effective agents (Dez 2024)](https://www.anthropic.com/research/building-effective-agents) — biblioteca de padrões (encadeamento de prompt, roteamento, paralelização, orquestrador-trabalhadores, avaliador-otimizador) independente de framework.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — o loop que todo framework enfeita.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — paper de design do AutoGen.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — fundamento de interpretação de papéis sobre o qual as pilhas de persona estilo CrewAI se constroem.
- Fase 11 · 16 (LangGraph) — o framework contra o qual esta lição faz benchmark.
- Fase 11 · 19 (Reflexion) — um padrão que mapeia limpo para LangGraph mas estranho para CrewAI.
- Fase 11 · 22 (Observabilidade em produção) — como instrumentar qualquer framework que você escolher.
