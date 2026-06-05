# Async e Hogwild! Inference

> Decodificação eespecificaçãoulativa (Fase 10, Aula 15) paraleliza tokens dentro de uma sequencia. Frameworks multi-agente paralelizam sequencias inteiras mas forcam coordenação explicita (votação, divisão de sub-tarefas). Hogwild! Inference (Rodionov et al., arXiv:2504.06261) faz algo diferente: roda N instancias do mesmo LLM em paralelo contra um KV cache COMPARTILHADO. Cada worker ve imediatamente os tokens gerados por todos os outros workers. Modelos de raciocinio modernos -- QwQ, DeepSeek-R1 -- conseguem se auto-coordenar por meio desse cache compartilhado sem nenhum fine-tuning. A abordagem e experimental mas abre um eixo totalmente novo de paralelismo de inferência que fica ortogonal a decodificação eespecificaçãoulativa. Esta aula implementa um simulador Hogwild! de dois workers em Python stdlib e explica por que a colaboração por cache compartilhado emerge das habilidades de raciocinio do modelo existente.

**Tipo:** Construir
**Linguagens:** Python (stdlib)
**Pré-requisitos:** Fase 10 · 12 (otimização de inferência), Fase 10 · 15 (decodificação eespecificaçãoulativa)
**Tempo:** ~60 minutos

## Objetivos de Aprendizado

- Descrever as três topologias comuns de LLM paralelo (votação, sub-tarefa, Hogwild!) e nomear quais problemas cada uma foca.
- Enunciar o setup central do Hogwild! multiplos workers, um KV cache compartilhado, coordenação emergente via auto-prompting.
- Calcular o speedup de tempo real do Hogwild! como função do numero de workers `N`, paralelismo por tarefa `p` e overhead de coordenação `c`.
- Implementar um simulador Hogwild! de dois workers em uma tarefa toy e observar a divisão emergente de tarefas.

## O Problema

LLMs modernos resolvem problemas dificeis produzindo cadeias longas de raciocinio -- 5000 tokens de lógica passo-a-passo e comum, dezenas de milhares de tokens acontecem em problemas de matématica profundos. A 35 tokens/segundo de decode num modelo 70B, 50k tokens e 24 minutos. Interativo o modelo não e.

Decodificação eespecificaçãoulativa (Fase 10, Aula 15) te da 3-5x de speedup paralelizando dentro de uma sequencia. Além disso a dependencia sequencial da decodificação autoregressiva e o teto rigido. Cada token novo depende de todos os anteriores.

A pergunta obvia: podemos paralelizar entre sequencias? Rodar multiplos copias do mesmo modelo no mesmo problema, deixar cooperar, dividir o trabalho?

Trabalhos anteriores: ensambles de votação (rodar N modelos, escolher a resposta majoritaria), arvore de pensamento (ramificar caminhos de raciocinio e recombinar) e frameworks multi-agente (designar cada agente uma sub-tarefa, usar um coordenador). Todos ajudam em dominios de tarefas eespecificaçãoificos. Todos também introduzem maquinario explicito de coordenação -- regras de votação, lógica de ramificar-e-podar, protocolos de mensagem agente-a-agente.

Hogwild! Inference usa uma abordagem diferente. N workers compartilham um unico KV cache. Cada worker ve imediatamente os tokens gerados por todos os outros workers, como se fossem seu proprio contexto. Os workers -- sem nenhum treinamento ou fine-tuning -- descobrem como dividir o trabalho. Modelos de raciocinio modernos (QwQ, DeepSeek-R1, modo de raciocinio da familia Claude) podem ler o cache compartilhado e dizer coisas como "vejo que o worker 2 ja cuidou do caso base, então vou trabalhar no passo indutivo."

O speedup depende da carga de trabalho e e experimental em abril de 2026. Mas a ideia vale conhecer porque abre um novo eixo de paralelismo de inferência.

## O Conceito

### O setup

Inicialize N processos worker, todos rodando o mesmo LLM. Ao inves de caches KV por worker, mantenha UM cache compartilhado. Quando o worker `i` gera o token `t_j`, o token e escrito no cache compartilhado na proxima posição. Quando o worker `k` faz seu próximo passo, ele le o estado atual do cache (que inclui tudo que todos os N workers geraram até agora).

No momento do passo, workers correm pra escrever tokens. Não tem índice de posição por worker -- o cache e uma sequencia unica crescendo. A ordem e determinada pela hora de chegada da escrita.

### Por que coordenação emerge

Os workers compartilham um prompt. Tipicamente algo como "Você e uma de N instancias trabalhando juntas neste problema. Cada instancia le a memoria compartilhada e pode ver o que outras instancias escreveram. Evite trabalho redundante." O prompt mais o cache compartilhado e suficiente. Modelos de raciocinio leem o cache, notam quais partes do problema ja foram tentadas e (frequentemente mas não sempre) mudam pra partes inexploradas.

O paper Hogwild! (Rodionov et al., 2025) reporta observações como:

- Workers formulam planos e comunicam pra outros workers via o cache.
- Workers notam erros no raciocinio de outros workers e apontam.
- Workers adaptam quando um plano falha e propoem alternativas.
- Quando promptados pra checar redundancia, workers detectam e mudam.

Nenhum disso requer fine-tuning. O comportamento emergente vem das capacidades de raciocinio que o modelo ja tem.

### A nomenclatura

O nome do paper faz referencia a Hogwild! SGD (Recht et al., 2011), um otimizador de atualização assincrona. A analogia: workers assincronos do SGD todos escrevem em um vetor de parâmetros compartilhado; workers do Hogwild! Inference todos escrevem em um KV cache compartilhado. Ambos dependem de convergencia empirica ão inves de garantias de sincronização.

### RoPE viabiliza isso

Rotary Position Embeddings (RoPE, Su et al. 2021) codificam informação posicional via rotação nos vetores Q e K. Como posições são rotações e não offsets fixos, a posição de um token pode mudar sem recomputar a entrada do KV cache. Quando o worker `i` escreve no cache compartilhado na posição `p`, outros workers lendo aquela posição podem usar a entrada em cache diretamente -- sem necessidade de re-rotação.

Em um modelo de posição aprendida ou posição absoluta, Hogwild! precisaria invalidação de cache em cada escrita concorrente. RoPE deixa o cache estavel.

### Matématica de tempo real

Seja `T_serial` o tempo pra um worker resolver o problema sozinho. Seja `p` a fração paralelizavel por tarefa. Seja `c` o overhead de coordenação por passo (ler o cache estendido, decidir o que escrever).

Tempo com um worker: `T_serial`.
Tempo com N workers Hogwild!, se coordenação for gratis: `T_serial * ((1 - p) + p / N)`. Amdahl classico.
Com overhead de coordenação: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Pra um worker ser produtivo, `c` precisa ser pequeno em relação ão tempo de decode por passo. Em modelos de raciocinio gerando 5k+ tokens, os workers podem aguentar centenas de tokens de overhead de coordenação e ainda sair na frente. Em tarefas de chat curtas, coordenação domina e Hogwild! e pior que serial.

### Exemplo concreto

Problema de raciocinio: 10k tokens de cadeia de raciocinio. Suponha que o problema tenha `p = 0.7` de conteudo paralelizavel (estratégias de prova diferentes, análises de caso diferentes) e `c = 200` tokens de overhead de coordenação por worker. Com `N = 4` workers:

- Tempo serial: 10000 passos de decode.
- Tempo Hogwild!: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 passos de decode.
- Speedup: 10000 / 5550 = 1.8x.

Isso e moderado. Mas em problemas de raciocinio mais longos (50k tokens), o overhead de coordenação se amortiza e o speedup chega a 2.5-3x. Hogwild! e o equivalente de inferência do paralelismo a nivel de threads em uma linguagem que deixa você escrever codigo multi-threaded naturalmente.

### Quando usar Hogwild!

- Problemas de raciocinio longos (milhares de tokens) onde a tarefa pode ser paralelizada entre sub-objetivos independentes.
- Modelos de raciocinio que foram treinados pra pensar passo a passo. Modelos não-raciocinio não se auto-coordenam bem.
- Deploy em node unico com VRAM suficiente pra armazenar o cache compartilhado mais N processos worker. O cache e compartilhado, mas cada worker tem sua propria memoria de ativação.

### Quando não usar

- Chat interativo curto. Overhead de coordenação domina.
- Tarefas que não paralelizam (prova linear unica, compilação unica). N=1 e o máximo.
- Modelos não-raciocinio. Nenhuma coordenação emerge.
- Deploy multi-node. O cache compartilhado precisa sincronização muito rapida entre workers. Intra-node e fine; entre nodes e desastre de laténcia.

### O status experimental

Em abril de 2026, Hogwild! e um metodo de pesquisa com implementação open-source em PyTorch. Adoção de produção não aconteceu. Três bloqueios:

1. Gerenciamento de KV cache compartilhado entre processos concorrentes e engenharia não-trivial.
2. Coordenação emergente depende da tarefa; benchmarks ainda estão sendo construidos.
3. Os speedups são moderados comparados ão que decodificação eespecificaçãoulativa ja entrega, e os dois podem ser combinados mas a engenharia combinada e outra camada.

Vale conhecer. Vale experimentar. Ainda não vale apóstar um produto.

## Construir

`code/main.py` implementa um simulador toy Hogwild!:

- Dois processos worker, cada um um "LLM" deterministico que produz uma de varias catégorias de token (work-token, observe-token, coordinaté-token) com probabilidades conhecidas.
- Um cache compartilhado (so uma lista de tokens) que ambos os workers leem e escrevem.
- Uma lógica simples de coordenação: quando um worker ve que o outro ja produziu tokens de trabalho suficientes numa catégoria, ele escolhe uma catégoria diferente.

O simulador roda por um orcamento fixo de passos e reporta:

- Total de work-tokens produzidos.
- Tempo real total (numero de passos de worker).
- Speedup efetivo sobre um worker unico.
- Um rastro de qual worker escreveu qual token.

### Passo 1: o cache compartilhado

Uma lista que ambos os workers adicionam. Trancamento simples (Python `threading.Lock`) em implementação real; simulamos com um contador.

### Passo 2: o loop do worker

Cada worker, a cada passo:

- Le o cache compartilhado atual.
- Decide que catégoria de token escrever baseado no que ja esta la.
- Escreve um token.

### Passo 3: a heuristica de coordenação

Se a catégoria X ja tem K tokens no cache e a catégoria pretendida do worker e X, o worker muda pra catégoria Y. Isso e um substituto toy do comportamento de "note que isso ja esta coberto, faca outra coisa" dos modelos de raciocinio.

### Passo 4: speedup medido

Rode o simulador com N=1 worker e com N=2 workers, mesmo orcamento total de passos. Conte work-tokens produzidos. N=2 deve produzir cerca de 1.5-1.8x mais work-tokens por causa da divisão de tarefas guiada por coordenação.

### Passo 5: estrêssar a coordenação

Reduza a sensibilidade da heuristica de coordenação. Rode de novo. Observe que sem boa coordenação, N=2 produz redundante os mesmos tokens e o speedup cai abaixo de 1. Isso combina com a observação do paper: o truque so funciona se os workers tem a capacidade de raciocinio pra se auto-coordenarem.

## Usar

Integração do Hogwild! em produção em abril de 2026 e de nivel de pesquisa. A implementação de referencia do Yandex/HSE/IST e baseada em PyTorch e foca em setups de node unico multi-processo em modelos DeepSeek-R1 e QwQ.

Caminho pragmatico de adocao:

1. Profile da sua carga de trabalho de raciocinio. Meça a fração de tokens que são exploratorios (multiplos estratégias, análises de caso, busca) vs lineares.
2. Se exploração domina, rode um experimento Hogwild! de dois workers. Meça a melhoria de tempo real.
3. Se a melhoria for abaixo de 1.3x, você esta no regime dominado por coordenação. Volte pra um worker.
4. Se a melhoria for acima de 1.5x, va pra N=4 e meça de novo. Retornos decrescentes tipicamente batém em torno de N=4-8.

Combine com decodificação eespecificaçãoulativa: cada worker Hogwild! pode usar independentemente decodificação eespecificaçãoulativa. Os dois speedups se multiplicam (aproximadamente), levando um 3x de decodificação eespecificaçãoulativa e 1.8x de Hogwild! a um efetivo 5.4x sobre decodificação ingenua de um worker.

## Entregar

Esta aula produz `outputs/skill-parallel-inference-router.md`. Dado um perfil de carga de trabalho de raciocinio (orcamento de tokens, perfil de paralelismo de tarefas, familia de modelo, alvo de deploy), rota entre votação, arvore de pensamento, multi-agente, Hogwild! e estratégias de decodificação eespecificaçãoulativa.

## Exercicios

1. Rode `code/main.py` com as configurações padrão. Confirme que a configuração Hogwild! de N=2 produz mais work-tokens que o baseline de N=1 no mesmo tempo real.

2. Reduza a forca da heuristica de coordenação (set `coordination_weight=0.1`). Rode de novo. Mostre que o speedup despenca. Explique por que: os workers duplicam esforco quando não conseguem coordenar.

3. Calcule o speedup esperado do Hogwild! pra uma tarefa de raciocinio de 50k tokens com `p=0.8, c=500` e N=4 workers. Faca o mesmo pra uma tarefa de chat de 1k tokens com `p=0.3, c=200` e N=4. Por que uma e ganho e a outra e perda?

4. Leia a Seção 4 do paper Hogwild! (avaliação preliminar). Identifique os dois modos de falha que os autores reportam. Descreva como um prompt de coordenação melhor poderia mitigar cada um.

5. Combine Hogwild! com decodificação eespecificaçãoulativa no toy: cada worker usa uma decodificação eespecificaçãoulativa de 2 tokens internamente. Reporte o speedup multiplicativo. Que problema de contabilidade surge quando dois workers querem ambos estender o mesmo prefixo do cache compartilhado?

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Hogwild! | "Workers paralelos, cache compartilhado" | N instancias do mesmo LLM rodando concorrentemente com um KV cache compartilhado; coordenação emergente via auto-prompting |
| KV cache compartilhado | "O meio de coordenação" | Um unico buffer KV crescendo que todos os workers leem e escrevem; viabiliza visibilidade instantanea de tokens entre workers |
| Coordenação emergente | "Não precisa treinamento" | LLMs com capacidade de raciocinio podem ler o cache compartilhado e dividir trabalho sem nenhum fine-tuning ou protocolo explicito |
| Overhead de coordenação (c) | "Tokens gastos se orientando" | O custo por worker de ler o cache estendido e decidir o que fazer; precisa ficar pequeno vs tempo total de decode |
| Fração paralelizavel (p) | "O que pode rodar em paralelo" | Paralelismo a nivel de tarefa: a fração do trabalho total que não e intrinsecamente sequencial |
| RoPE viabiliza Hogwild! | "Posições rotacionais são shift-invariant" | Como posições são rotações, escrever em um cache compartilhado não requer recomputar tokens anteriores |
| Ensemble de votação | "Rodar N, escolher a maioria" | A topologia mais simples de inferência paralela; útil pra classificação, menos pra raciocinio longo |
| Arvore de pensamento | "Ramificar e podar" | Estratégia de raciocinio que explora multiplos ramos e poda; lógica explicita de coordenação |
| Framework multi-agente | "Designar sub-tarefas" | Cada agente ganha um papel; um coordenador orquestra; overhead pesado de protocolo |

## Leitura Complementar

- [Rodionov et al. -- Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) -- o paper Hogwild!, avaliação preliminar em QwQ e DeepSeek-R1
- [Recht, Re, Wright, Niu -- Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) -- o original Hogwild!, a origem do nome
- [Su et al. -- RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) -- RoPE, a propriedade que torna inferência de cache compartilhado viavel
- [Yão et al. -- Tree of Thoughts: Deliberaté Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) -- a estratégia de raciocinio em arvore que Hogwild! fica ortogonal a
- [Leviathan et al. -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) -- decodificação eespecificaçãoulativa, o paralelismo intra-sequencia que Hogwild! combina com
- [Implementação de referencia Hogwild! em PyTorch](https://github.com/eqimp/hogwild_llm) -- a unica fonte de verdade pros experimentos do paper
