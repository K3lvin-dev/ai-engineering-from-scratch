# AI Engineering Glossary

## A

### Agent
- **What people say:** "Um AI autônomo que pensa e age por conta própria"
- **What it actually means:** Um loop while onde um LLM decide qual ferramenta chamar em seguida, executa, vê o resultado e repete
- **Por que é chamado assim:** Emprestado da filosofia — um "agente" é qualquer coisa que pode agir no mundo. Em AI, significa apenas "LLM + ferramentas + loop"

### Attention
- **What people say:** "Como o AI foca nas partes importantes"
- **What it actually means:** Um mecanismo onde cada token calcula uma soma ponderada dos valores de todos os outros tokens, com pesos determinados por quão relevantes eles são (via dot product dos vetores query e key)
- **Por que é chamado assim:** O artigo de 2017 "Attention Is All You Need" nomeou assim por analogia à atenção seletiva humana

### Alignment
- **What people say:** "Tornando o AI seguro"
- **What it actually means:** O desafio técnico de fazer o comportamento de um sistema de AI corresponder às intenções, valores e preferências humanas, incluindo casos extremos que o designer não antecipou

### Autoregressive
- **What people say:** "O AI gera uma palavra de cada vez"
- **What it actually means:** Um modelo que prevê o próximo token condicionado em todos os tokens anteriores, então alimenta essa previsão de volta como entrada para o próximo passo. GPT, LLaMA e Claude são todos autoregressivos.

### Activation Function
- **What people say:** "A coisa não linear entre as camadas"
- **What it actually means:** Uma função aplicada após cada camada linear que introduz não linearidade. Sem ela, empilhar qualquer número de camadas lineares colapsa em uma única transformação linear. ReLU, GELU e SiLU são as mais comuns. A escolha afeta diretamente se os gradientes fluem durante o treinamento.

### Adam (Optimizer)
- **What people say:** "O optimizer padrão"
- **What it actually means:** Adaptive Moment Estimation. Combina momentum (primeiro momento) com taxas de aprendizado adaptativas por parâmetro (segundo momento). Tem correção de viés para passos iniciais. Funciona bem na maioria das tarefas sem muito ajuste.

### AdamW
- **What people say:** "Adam, mas melhor"
- **What it actually means:** Adam com weight decay desacoplado. No Adam padrão, a regularização L2 é escalada pela taxa de aprendizado adaptativa por parâmetro, o que não é o desejado. AdamW aplica weight decay diretamente aos pesos, independente das estatísticas do gradiente. O optimizer padrão para treinar transformers.

### Autograd
- **What people say:** "Gradientes automáticos"
- **What it actually means:** Um sistema que registra operações em tensores e calcula automaticamente gradientes via diferenciação em modo reverso. O autograd do PyTorch constrói um grafo de computação em tempo real (grafo dinâmico), enquanto JAX usa transformações de funções (grad). É isso que torna a backpropagation prática — você escreve o forward pass, e o framework calcula todas as derivadas.

## B

### Batch Size
- **What people say:** "Quantos exemplos de uma vez"
- **What it actually means:** O número de exemplos de treinamento processados em um único forward/backward pass antes de atualizar os pesos. Lotes maiores dão estimativas de gradiente mais estáveis, mas usam mais memória. Valores típicos: 32-512 para treinamento, maior para inferência. Batch size interage com a learning rate — dobre o batch, dobre a LR (linear scaling rule).

### Backpropagation
- **What people say:** "Como as redes neurais aprendem"
- **What it actually means:** Um algoritmo que calcula o quanto cada peso contribuiu para o erro aplicando a regra da cadeia de trás para frente na rede, então ajusta os pesos proporcionalmente
- **Por que é chamado assim:** Erros se propagam para trás, da saída para a entrada, camada por camada

## C

### Context Window
- **What people say:** "Quanto o AI consegue lembrar"
- **What it actually means:** O número máximo de tokens (entrada + saída) que cabem em uma única chamada de API. Não é memória — é um buffer de tamanho fixo que reset a cada chamada

### Chain of Thought (CoT)
- **What people say:** "Fazendo o AI pensar passo a passo"
- **What it actually means:** Uma técnica de prompt onde você pede ao modelo para mostrar seus passos de raciocínio, o que melhora a precisão em problemas de múltiplas etapas porque cada passo condiciona a geração do próximo token

### CNN (Convolutional Neural Network)
- **What people say:** "AI de imagens"
- **What it actually means:** Uma rede neural que usa operações de convolução (filtros deslizantes sobre a entrada) para detectar padrões locais. Empilhar convoluções detecta características cada vez mais complexas: bordas, texturas, objetos.

### CUDA
- **What people say:** "Programação de GPU"
- **What it actually means:** Plataforma de computação paralela da NVIDIA. Permite rodar operações matriciais em milhares de núcleos de GPU simultaneamente. PyTorch e TensorFlow usam CUDA internamente.

### Chunking
- **What people say:** "Dividindo documentos em pedaços"
- **What it actually means:** Quebrar texto em segmentos antes de embedding para recuperação. O tamanho do chunk determina a granularidade dos resultados de busca. Muito pequeno: perde contexto. Muito grande: dilui a relevância. Estratégias comuns: tamanho fixo com sobreposição, baseado em sentenças, ou divisão semântica. Tamanho típico de chunk: 256-512 tokens com 10-20% de sobreposição.

### Contrastive Learning
- **What people say:** "Aprendendo por comparação"
- **What it actually means:** Treinamento aproximando pares similares e afastando pares dissimilares no espaço de embedding. CLIP usa isso: combinando pares imagem-texto vs pares não correspondentes.

### Cosine Similarity
- **What people say:** "O quão similares dois vetores são"
- **What it actually means:** O cosseno do ângulo entre dois vetores: dot(a, b) / (||a|| * ||b||). Varia de -1 (oposto) a 1 (direção idêntica). Ignora magnitude, só considera direção. A métrica de similaridade padrão para embeddings e busca semântica.

### Cross-Entropy
- **What people say:** "A loss de classificação"
- **What it actually means:** Mede a diferença entre duas distribuições de probabilidade. Para classificação: -sum(y_true * log(y_pred)). Para modelos de linguagem: a log-probabilidade negativa do próximo token correto. Menor é melhor. Perplexity é apenas exp(cross-entropy).

## D

### Data Augmentation
- **What people say:** "Criando mais dados de treinamento"
- **What it actually means:** Criar cópias modificadas de dados existentes (rotacionar imagens, adicionar ruído, parafrasear texto) para aumentar a diversidade do conjunto de treinamento sem coletar novos dados. Reduz overfitting.

### Decoder
- **What people say:** "A parte de saída"
- **What it actually means:** Em transformers, um decoder usa self-attention causal (mascarada) para que cada posição só possa atender a posições anteriores. GPT é decoder-only. BERT é encoder-only. T5 é encoder-decoder.

### Diffusion Model
- **What people say:** "AI que gera imagens a partir de ruído"
- **What it actually means:** Um modelo treinado para reverter um processo gradual de adição de ruído — ele aprende a prever e remover ruído, e no momento da geração começa a partir de ruído puro e iterativamente remove o ruído

### DPO (Direct Preference Optimization)
- **What people say:** "Um RLHF mais simples"
- **What it actually means:** Um método de treinamento que pula o reward model completamente — ele otimiza diretamente o modelo de linguagem para preferir a melhor resposta em pares de preferências humanas

### Dropout
- **What people say:** "Desligando neurônios aleatoriamente"
- **What it actually means:** Durante o treinamento, definir aleatoriamente uma fração das ativações para zero. Força a rede a não depender de nenhum neurônio específico. Desligado durante a inferência. Regularização simples, mas eficaz.

## E

### Eigenvalue
- **What people say:** "Uma coisa matemática para PCA"
- **What it actually means:** Para uma matriz A, um eigenvalue lambda satisfaz Av = lambda*v para algum vetor v. Ele diz o quanto a matriz escala vetores naquela direção. Eigenvalues grandes = direções de alta variância nos seus dados.

### Embedding
- **What people say:** "Uma magia de AI que transforma palavras em números"
- **What it actually means:** Um mapeamento aprendido de itens discretos (palavras, imagens, usuários) para vetores densos em um espaço contínuo, onde itens similares ficam próximos
- **Por que é chamado assim:** Os itens são "incorporados" (embedded) em um espaço geométrico onde a distância tem significado

### Encoder
- **What people say:** "A parte de entrada"
- **What it actually means:** Em transformers, um encoder usa self-attention bidirecional para que cada posição possa atender a todas as posições. BERT é encoder-only. Bom para tarefas de compreensão (classificação, NER), mas não para geração.

### Epoch
- **What people say:** "Uma passada pelos dados"
- **What it actually means:** Exatamente isso. Uma passada completa por cada exemplo no conjunto de treinamento. Múltiplas epochs = ver os dados várias vezes. Mais epochs podem melhorar o aprendizado, mas aumentam o risco de overfitting.

## F

### Feature
- **What people say:** "Uma coluna nos seus dados"
- **What it actually means:** Uma propriedade mensurável individual dos dados. Em ML clássico, você engenha features manualmente. Em deep learning, a rede aprende features automaticamente a partir dos dados brutos.

### Few-Shot
- **What people say:** "Dê alguns exemplos para o AI primeiro"
- **What it actually means:** Incluir um pequeno número de exemplos de entrada-saída no prompt antes de pedir ao modelo para executar uma tarefa. Tipicamente 3-5 exemplos. O modelo faz pattern-matching nesses exemplos para entender o formato e comportamento desejados. Contrasta com zero-shot (sem exemplos) e fine-tuning (milhares de exemplos incorporados nos pesos).

### Fine-tuning
- **What people say:** "Treinando o AI nos seus dados"
- **What it actually means:** Começar com os pesos de um modelo pré-treinado e continuar o treinamento em um conjunto de dados menor e específico para a tarefa. Apenas atualiza pesos existentes, não adiciona novos conhecimentos do zero

### Function Calling
- **What people say:** "AI que pode usar ferramentas"
- **What it actually means:** Uma forma estruturada de LLMs solicitarem a execução de funções externas. Você define ferramentas com descrições JSON Schema, o modelo gera um objeto JSON estruturado especificando qual função chamar e com quais argumentos, seu código executa e o resultado volta para o modelo. Não é a mesma coisa que agents — function calling é o mecanismo, agents são o loop.

## G

### Guardrails
- **What people say:** "Filtros de segurança para AI"
- **What it actually means:** Camadas de validação de entrada/saída em torno de um LLM que detectam e bloqueiam conteúdo prejudicial, tentativas de prompt injection, vazamento de PII ou respostas fora do tópico. Tipicamente um pipeline: filtro de entrada -> LLM -> filtro de saída. Pode ser baseado em regras (regex, listas de palavras-chave) ou baseado em modelo (classificador que avalia segurança).

### GPT
- **What people say:** "ChatGPT" ou "O AI"
- **What it actually means:** Generative Pre-trained Transformer — uma arquitetura específica que prevê o próximo token usando um transformer decoder-only treinado em grandes corpora de texto
- **Por que é chamado assim:** Generative (produz texto), Pre-trained (treinado uma vez em grandes dados, depois adaptado), Transformer (a arquitetura)

### GAN (Generative Adversarial Network)
- **What people say:** "Dois AIs lutando entre si"
- **What it actually means:** Uma rede geradora tenta criar dados realistas enquanto uma rede discriminadora tenta distinguir o real do falso. Elas treinam juntas: o gerador melhora em enganar o discriminador, e o discriminador melhora em detectar falsificações.

### Gradient
- **What people say:** "A inclinação"
- **What it actually means:** Um vetor de derivadas parciais apontando na direção de maior aumento. Em ML, você vai na direção oposta ao gradiente (gradient descent) para minimizar a loss.

### Gradient Descent
- **What people say:** "Como o AI melhora"
- **What it actually means:** Um algoritmo de otimização que ajusta parâmetros na direção que mais reduz a função de loss, como descer uma colina em uma paisagem de alta dimensionalidade

## H

### Hyperparameter
- **What people say:** "Configurações que você ajusta"
- **What it actually means:** Valores definidos antes do treinamento que controlam o próprio processo de treinamento: learning rate, batch size, número de camadas, dropout rate. Diferente dos parâmetros do modelo (pesos), estes não são aprendidos a partir dos dados.

### Hallucination
- **What people say:** "O AI está mentindo" ou "inventando coisas"
- **What it actually means:** O modelo gera texto que parece plausível, mas não tem base em seus dados de treinamento ou no contexto fornecido — ele está completando padrões, não recuperando fatos

## I

### Inference
- **What people say:** "Rodando o AI"
- **What it actually means:** Usar um modelo treinado para fazer previsões em novos dados. Nenhuma atualização de pesos ocorre. É isso que você faz em produção: enviar entrada, obter saída.

### Inductive Bias
- **What people say:** Nunca ouviu falar
- **What it actually means:** As suposições embutidas na arquitetura de um modelo. CNNs assumem que padrões locais importam (convolução). RNNs assumem que ordem importa (processamento sequencial). Transformers assumem que tudo pode se relacionar com tudo (attention). O viés certo ajuda o modelo a aprender mais rápido com menos dados.

### JAX
- **What people say:** "O framework de ML do Google"
- **What it actually means:** Uma biblioteca compatível com NumPy que adiciona diferenciação automática (grad), compilação JIT (jit), vetorização automática (vmap) e paralelismo multi-dispositivo (pmap). Diferente do estilo orientado a objetos do PyTorch, JAX é puramente funcional — sem estado oculto, sem mutação in-place. Usado pelo Google DeepMind para AlphaFold, Gemini e pesquisa em larga escala.

## K

### KV Cache
- **What people say:** "Torna a inferência mais rápida"
- **What it actually means:** Durante a geração autoregressiva, armazenar em cache as matrizes key e value de tokens anteriores para não precisar recalculá-las a cada passo. Troca memória por velocidade. Essencial para inferência rápida de LLMs.

## L

### Latent Space
- **What people say:** "A representação oculta"
- **What it actually means:** Um espaço de representação comprimido e aprendido onde entradas similares mapeiam para pontos próximos. Autoencoders, VAEs e diffusion models todos trabalham em latent space. É de menor dimensão que a entrada, mas captura a estrutura importante.

### Learning Rate
- **What people say:** "Quão rápido o AI aprende"
- **What it actually means:** Um escalar que controla o tamanho do passo durante gradient descent. Muito alto: ultrapassa o mínimo e diverge. Muito baixo: converge muito lentamente ou fica preso. O hiperparâmetro mais importante.

### LLM (Large Language Model)
- **What people say:** "AI" ou "o cérebro"
- **What it actually means:** Uma rede neural baseada em transformer treinada para prever o próximo token em uma sequência, com bilhões de parâmetros, treinada em dados de texto em escala de internet

### LoRA (Low-Rank Adaptation)
- **What people say:** "Fine-tuning eficiente"
- **What it actually means:** Em vez de atualizar todos os pesos, inserir pequenas matrizes de baixo posto junto aos pesos originais. Apenas essas matrizes pequenas são treinadas, reduzindo a memória em 10-100x

### Loss Function
- **What people say:** "O quão errado o AI está"
- **What it actually means:** Uma função que mede a diferença entre a saída prevista e a real. O treinamento minimiza esta função. MSE para regressão, cross-entropy para classificação, contrastive loss para embeddings. A escolha da loss function define o que "bom" significa para o modelo.

## M

### Mixed Precision
- **What people say:** "Truque de treinamento para velocidade"
- **What it actually means:** Usar float16 para o forward pass e a maioria das operações (mais rápido, menos memória), mas manter float32 para acumulação de gradiente e atualização de pesos (mais preciso). Obtém 2x de aceleração com perda insignificante de precisão.

### MoE (Mixture of Experts)
- **What people say:** "Apenas parte do modelo roda"
- **What it actually means:** Um modelo com muitas sub-redes "especialistas" onde um mecanismo de roteamento envia cada entrada para apenas alguns especialistas. O modelo completo é enorme, mas cada forward pass é barato porque a maioria dos especialistas é ignorada. Mixtral e GPT-4 usam isso.

### MCP (Model Context Protocol)
- **What people say:** "Um jeito de AI usar ferramentas"
- **What it actually means:** Um protocolo aberto (JSON-RPC sobre stdio/HTTP) que padroniza como aplicações de AI se conectam a fontes de dados e ferramentas externas, com schemas tipados para ferramentas, recursos e prompts

## N

### NaN (Not a Number)
- **What people say:** "O treinamento quebrou"
- **What it actually means:** Um valor de ponto flutuante indicando resultados indefinidos (0/0, inf-inf). Em treinamento, NaN loss geralmente significa: learning rate muito alta, gradientes explosivos, log de zero ou divisão por zero. Sempre a primeira coisa a verificar quando o treinamento falha.

### Normalization
- **What people say:** "Escalando os dados"
- **What it actually means:** Ajustar valores para uma faixa padrão. Batch normalization normaliza através de um batch. Layer normalization normaliza através das features. Ambos estabilizam o treinamento e permitem learning rates mais altas.

## O

### Overfitting
- **What people say:** "O modelo memorizou os dados"
- **What it actually means:** O modelo tem bom desempenho nos dados de treinamento, mas ruim em dados não vistos. Ele aprendeu o ruído, não o sinal. Corrigir com: mais dados, regularização (dropout, weight decay), early stopping, data augmentation, modelo mais simples.

### Optimizer
- **What people say:** "A coisa que atualiza os pesos"
- **What it actually means:** Um algoritmo que usa gradientes para atualizar os parâmetros do modelo. SGD é o mais simples. Adam é o mais comum. Cada optimizer tem propriedades diferentes: velocidade de convergência, uso de memória, sensibilidade a hiperparâmetros.

## P

### Parameter
- **What people say:** "Tamanho do modelo"
- **What it actually means:** Um valor aprendível no modelo, tipicamente um peso ou viés. "7B parâmetros" significa 7 bilhões de números aprendíveis. Cada parâmetro float32 ocupa 4 bytes, então 7B parâmetros = 28GB de memória só para os pesos.

### Perplexity
- **What people say:** "O quão confuso o modelo está"
- **What it actually means:** O exponencial da cross-entropy loss média. Menor é melhor. Uma perplexity de 10 significa que o modelo está tão incerto quanto se estivesse escolhendo uniformemente entre 10 tokens a cada passo.

### Precision & Recall
- **What people say:** "Métricas de acurácia"
- **What it actually means:** Precision = dos itens que você marcou, quantos estavam corretos. Recall = de todos os itens corretos, quantos você encontrou. Eles se compensam: pegar todo spam (alto recall) significa mais falsos alarmes (baixa precision). F1 score é a média harmônica deles. Use precision quando falsos positivos são custosos, recall quando falsos negativos são custosos.

### Prompt Engineering
- **What people say:** "Falar com AI do jeito certo"
- **What it actually means:** Projetar o texto de entrada para produzir saídas desejadas de forma confiável — incluindo system prompts, exemplos few-shot, instruções de formato e gatilhos de chain-of-thought

### Prompt Injection
- **What people say:** "Hackeando o AI com palavras"
- **What it actually means:** Um ataque onde texto malicioso na entrada sobrescreve o system prompt ou as instruções. Injeção direta: usuário digita "Ignore instruções anteriores." Injeção indireta: um documento recuperado contém instruções ocultas. O equivalente a SQL injection para LLMs. Não existe solução completa — a defesa é camadas de validação de entrada, filtragem de saída e separação de privilégios.

## Q

### QLoRA
- **What people say:** "LoRA, mas mais barato"
- **What it actually means:** LoRA quantizado. Mantém os pesos congelados do modelo base em precisão de 4 bits (formato NF4) enquanto treina os adaptadores LoRA em 16 bits. Reduz a memória em mais 3-4x comparado ao LoRA padrão. Um modelo de 7B que precisa de 14GB com LoRA cabe em 4-6GB com QLoRA. A qualidade fica dentro de 1% do fine-tuning completo na maioria dos benchmarks.

## R

### RAG (Retrieval-Augmented Generation)
- **What people say:** "AI que pode pesquisar"
- **What it actually means:** Um padrão onde você recupera documentos relevantes de uma base de conhecimento (usando similaridade de embeddings), coloca-os no prompt e deixa o LLM responder com base nesse contexto
- **Por que é chamado assim:** Retrieval (encontrar documentos) + Augmented (adicionar ao prompt) + Generation (LLM escreve a resposta)

### RLHF (Reinforcement Learning from Human Feedback)
- **What people say:** "Como eles fazem o AI ser útil"
- **What it actually means:** Um pipeline de treinamento: (1) coletar preferências humanas sobre saídas do modelo, (2) treinar um reward model nessas preferências, (3) usar PPO para otimizar o LLM a produzir saídas com maior recompensa

### Quantization
- **What people say:** "Tornando o modelo menor"
- **What it actually means:** Reduzir a precisão dos pesos do modelo de float32 (4 bytes) para int8 (1 byte) ou int4 (0,5 bytes). Troca um pequeno valor de precisão por 4-8x menos memória e inferência mais rápida. GPTQ, AWQ e GGUF são formatos comuns.

### ReLU
- **What people say:** "Função de ativação"
- **What it actually means:** Rectified Linear Unit: f(x) = max(0, x). A ativação não linear mais simples. Rápida de computar, não satura para valores positivos. Usada em todo lugar porque funciona e é barata. Variantes: LeakyReLU, GELU, SiLU.

### ROUGE
- **What people say:** "Métrica de sumarização"
- **What it actually means:** Recall-Oriented Understudy for Gisting Evaluation. Mede a sobreposição entre texto gerado e texto de referência. ROUGE-1 conta correspondências de unigramas, ROUGE-2 conta correspondências de bigramas, ROUGE-L encontra a maior subsequência comum. Barato de computar, mas mede apenas similaridade superficial — duas frases com o mesmo significado, mas palavras diferentes, pontuam mal.

## S

### Semantic Search
- **What people say:** "Busca inteligente que entende o significado"
- **What it actually means:** Encontrar documentos pelo significado em vez de correspondência de palavras-chave. Embedding da consulta e de todos os documentos no mesmo espaço vetorial, depois retornar documentos cujos embeddings estão mais próximos do embedding da consulta. "pagamento falhou" encontra "transação recusada" mesmo sem compartilhar palavras. Alimentado por modelos de embedding + bancos de dados vetoriais.

### Streaming
- **What people say:** "Vendo a resposta aparecer palavra por palavra"
- **What it actually means:** O LLM envia os tokens conforme são gerados, em vez de esperar a resposta completa. Usa protocolos Server-Sent Events (SSE) ou WebSocket. Reduz a latência percebida de segundos para milissegundos para o primeiro token. Essencial para interfaces de chat em produção. Cada chunk contém um delta (token parcial ou palavra).

### Self-Attention
- **What people say:** "Como o modelo decide no que focar"
- **What it actually means:** Cada token computa vetores query, key e value. O peso de atenção entre dois tokens = dot product de suas queries e keys, escalado e softmax. Saída = soma ponderada dos vetores value. Permite que cada token veja todos os outros tokens.

### SFT (Supervised Fine-Tuning)
- **What people say:** "Ensinando o modelo a seguir instruções"
- **What it actually means:** Fine-tuning de um modelo pré-treinado em pares (instrução, resposta). O modelo aprende a gerar a resposta dada a instrução. É isso que transforma um modelo base em um modelo de chat.

### Softmax
- **What people say:** "Transforma números em probabilidades"
- **What it actually means:** softmax(x_i) = exp(x_i) / sum(exp(x_j)). Transforma um vetor de números reais arbitrários em uma distribuição de probabilidade (tudo positivo, soma 1). Usado em cabeças de classificação, pesos de atenção e em qualquer lugar que você precisar de probabilidades.

### Swarm
- **What people say:** "Um monte de agents de AI trabalhando juntos como abelhas"
- **What it actually means:** Múltiplos agents compartilhando estado e coordenando através de troca de mensagens, com comportamento emergente surgindo de regras individuais simples em vez de controle central

## T

### System Prompt
- **What people say:** "As instruções do AI"
- **What it actually means:** Uma mensagem especial no início de uma conversa que define o comportamento, persona e restrições do modelo. Processada antes das mensagens do usuário. Não visível ao usuário na maioria das UIs. Define o que o modelo deve e não deve fazer, seu tom, preferências de formato e foco de domínio. Diferente de user prompts — system prompts são definidos pelo desenvolvedor.

### Tensor
- **What people say:** "Um array multi-dimensional"
- **What it actually means:** A estrutura de dados fundamental em frameworks de deep learning. Um tensor 0D é um escalar, 1D é um vetor, 2D é uma matriz, 3D+ é um tensor. Em PyTorch e JAX, tensores rastreiam seu histórico de computação para diferenciação automática e podem estar em CPU ou GPU. Todas as entradas, saídas, pesos e gradientes de redes neurais são tensores.

### Token
- **What people say:** "Uma palavra"
- **What it actually means:** Uma unidade de subpalavra (tipicamente 3-4 caracteres em inglês) produzida por um tokenizer como BPE. "inacreditável" pode ser 3 tokens: "in" + "acredit" + "ável"

### Temperature
- **What people say:** "Configuração de criatividade"
- **What it actually means:** Um escalar que divide os logits antes do softmax. Temperature=1 é o padrão. Mais alta = distribuição mais plana = saídas mais aleatórias. Mais baixa = distribuição mais acentuada = mais determinístico. Temperature=0 é argmax (sempre escolher o token mais provável).

### Transfer Learning
- **What people say:** "Usando um modelo pré-treinado"
- **What it actually means:** Pegar um modelo treinado em uma tarefa e adaptá-lo para uma tarefa diferente. As camadas iniciais aprendem características gerais (bordas, padrões sintáticos) que são transferíveis. Apenas as camadas finais precisam de treinamento específico da tarefa. É por isso que você pode fazer fine-tuning de BERT para qualquer tarefa de NLP.

### Transformer
- **What people say:** "A arquitetura por trás da AI moderna"
- **What it actually means:** Uma arquitetura de rede neural que processa sequências usando self-attention (permitindo que cada posição atenda a todas as outras) em vez de recorrência, possibilitando paralelização massiva
- **Por que é chamado assim:** Transforma representações de entrada em representações de saída através de camadas de atenção

## U

### Underfitting
- **What people say:** "O modelo não está aprendendo"
- **What it actually means:** O modelo é simples demais para capturar os padrões nos dados. A loss de treinamento permanece alta. Corrigir com: mais parâmetros, mais camadas, treinamento mais longo, menos regularização, melhores features.

## V

### VAE (Variational Autoencoder)
- **What people say:** "Um modelo generativo"
- **What it actually means:** Um autoencoder que aprende um latent space suave forçando a saída do encoder a seguir uma distribuição Gaussiana. Você pode amostrar desta distribuição e decodificar para gerar novos dados. O truque da reparametrização o torna treinável via backpropagation.

### Vector Database
- **What people say:** "Um banco de dados especial para AI"
- **What it actually means:** Um banco de dados otimizado para armazenar vetores (arrays densos de floats) e realizar busca aproximada do vizinho mais próximo rápida. A operação central em busca por similaridade, RAG e sistemas de recomendação.

## W

### Weight
- **What people say:** "O que o modelo aprendeu"
- **What it actually means:** Um número único na matriz de parâmetros de um modelo. Uma camada linear com tamanho de entrada 768 e tamanho de saída 3072 tem 768*3072 = 2.359.296 pesos. O treinamento ajusta cada peso para minimizar a função de loss.

### Weight Decay
- **What people say:** "Regularização"
- **What it actually means:** Adicionar uma penalidade proporcional à magnitude dos pesos à função de loss. Equivalente à regularização L2. Impede que os pesos cresçam demais. Valor típico: 0,01-0,1.

## Z

### Zero-Shot
- **What people say:** "Sem necessidade de treinamento"
- **What it actually means:** Usar um modelo em uma tarefa para a qual ele não foi explicitamente treinado, sem exemplos específicos da tarefa no prompt. O modelo generaliza a partir do pré-treinamento. Funciona porque modelos grandes viram variedade suficiente para lidar com novos formatos de tarefa.
