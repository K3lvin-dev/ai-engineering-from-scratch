# Decodificação Eespecificaçãoulativa e EAGLE-3

> A Fase 7, Aula 16 provou a matématica: a rejeição de Leviathan preserva a distribuição do verificador exatamente. Esta aula e a visão da stack de treinamento da decodificação eespecificaçãoulativa de produção em 2026. EAGLE-3 transformou o modelo de rascunho de uma aproximação barata numa rede minuscula construida pra isso, treinada nos proprios hidden statés do verificador, e adicionou um loop de teste durante treinamento que alinha suas distribuções de treinamento e inferência. Resultado: 3x a 6.5x de speedup de ponta a ponta, taxas de aceitação por token acima de 0.9 em chat, sem troca de distribuição. Toda stack de inferência de produção em 2026 envia isso por padrão.

**Tipo:** Construir
**Linguagens:** Python (stdlib)
**Pré-requisitos:** Fase 7 · 16 (matématica de decodificação eespecificaçãoulativa), Fase 10 · 12 (otimização de inferência)
**Tempo:** ~75 minutos

## Objetivos de Aprendizado

- Enunciar o teorema de Leviathan em uma frase e provar que o loop eespecificaçãoulativo produz amostras identicamente distribuidas ão verificador.
- Percorrer a progressão de dois anos desde a decodificação eespecificaçãoulativa simples (Leviathan 2023) passando por EAGLE, EAGLE-2 e EAGLE-3 e nomear a limitação exata que cada etapa removeu.
- Calcular o speedup esperado a partir da taxa de aceitação `alpha` e da razão de custo rascunho-verificador `c`, e escolher o comprimento ótimo de rascunho `N` pra cada regime.
- Implementar o loop eespecificaçãoulativo completo do zero: rascunhar, verificar, reamostrar do resido, desfazer o KV cache na rejeição, emitir o token bonus na aceitação total.

## O Problema

A decodificação autoregressiva num modelo 70B roda talvez 35 tokens por segundo num H100. A GPU não esta nem perto de saturada. A largura de banda de memoria e o teto: cada token carrega 70B de pesos da HBM, faz uma etapa de aritmetica e produz uma unidade float. As unidades de compute ficam ociosas a maior parte do tempo.

A decodificação eespecificaçãoulativa transforma isso num problema de throughput que você consegue resolver de verdade. Um rascunho barato propoe `N` tokens em `N` forward passes pequenos. O verificador roda uma vez no prefixo mais todos os `N` rascunhos. Se a distribuição do verificador na posição `i` concorda com o rascunho (num sentido estatistico que vamos tornar preciso), aceitamos; se não, rejeitamos e amostramos uma correção da distribuição residual. Um unico forward do modelo grande produz até `N+1` tokens aceitos ão inves de um.

O teorema que importa e Leviathan, Kalman, Matias (ICML 2023): a distribuição de saida e identica ão que amostrar direto do verificador produziria. Não aproximadamente. Identiquement. Essa e a razão inteira pra decodificação eespecificaçãoulativa ser aceitavel em produção -- e uma otimização pura de laténcia sem troca de qualidade.

O que a Fase 7, Aula 16 te deu foi a matématica. O que esta aula te da e a stack de treinamento. Um bom rascunho vale 2x mais speedup que um rascunho barato. EAGLE, EAGLE-2 e EAGLE-3 (Li et al., 2024-2025) transformaram "rascunho = versão menor do mesmo modelo" em uma disciplina de engenharia precisa. Servidores de inferência de produção em 2026 vem com EAGLE-3 por padrão.

## O Conceito

### O invariante: reamostragem por rejeição de Leviathan

Seja `p(t)` a distribuição do rascunho pro próximo token dado um prefixo, e `q(t)` a do verificador. Amostre um token de rascunho `d ~ p`. Aceite com probabilidade `min(1, q(d) / p(d))`. Na rejeição, amostrar da distribuição residual `(q - p)_+ / ||(q - p)_+||_1`. As amostragens resultantes são distribuidas de acordo com `q`. Isso e verdade não importa quão ruim `p` e -- quanto pior, mais você rejeita, mas a saida continua exata.

Empilhe `N` dessas chamadas uma atras da outra usando um forward pass do verificador em `prefix + d_1 + ... + d_N`. O verificador retorna `q_1, q_2, ..., q_{N+1}` simultaneamente. Percorra da esquerda pra direita. Na primeira rejeição na posição `j`, amostrar de `residual(q_j, p_j)` e parar. Na aceitação total, amostrar um token bonus de `q_{N+1}`.

### O que determina o speedup

Seja `alpha` a taxa de aceitação esperada por token rascunhado. Seja `c = cost(draft) / cost(verifier)` a razão de custo. O numero esperado de tokens aceitos por forward do verificador e:

```
E[accepted] = (1 - alpha^(N+1)) / (1 - alpha)
```

O tempo real total esperado por token aceito e `(N * c + 1) / E[accepted]`. Minimize isso em relação a `N` e você tem o ponto ideal. Pra `alpha = 0.8, c = 0.05`: `N` ótimo e por volta de 5-7, speedup e 3.2x. Pra `alpha = 0.95, c = 0.02`: `N` ótimo e por volta de 8-10, speedup chega a 5x.

A maior alavanca e `alpha`. Ir de `alpha = 0.6` (rascunho simples) pra `alpha = 0.9` (EAGLE-3) com `N = 5` fixo te leva de 2.2 tokens aceitos esperados por forward do verificador pra 4.1. Quase 2x mais throughput do mesmo verificador.

### A progressão de dois anos

**Eespecificaçãoulativa simples (Leviathan, 2023).** Modelo de rascunho e um LLM menor treinado independentemente da mesma familia. Facil de conectar, `alpha` aproximado 0.6, speedup por volta de 2x no máximo.

**EAGLE-1 (Li et al., 2024).** Rascunho e um transformer minusculo -- tipicamente uma ou duas camadas -- que pega o hidden staté da última camada do verificador como entrada e prediz o próximo token diretamente. Como o rascunho ve a representação de features do verificador, sua distribuição e muito mais proxima da do verificador. `alpha` sobe pra 0.7-0.8.

**EAGLE-2 (Li et al., 2024).** Adiciona uma arvore dinâmica de rascunho: ão inves de propor uma unica sequencia de `N` tokens, propor uma arvore pequena de candidatos, pontuar cada um com o verificador em um forward pass (tree attention), e seguir o caminho de maior probabilidade. Comprimento do rascunho vira adaptativo por etapa. `alpha` por token do caminho aceito sobe acima de 0.85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Duas mudancas a mais. Primeiro, dropa a loss de predição de features completamente -- EAGLE-1/2 treinava o rascunho pra casar com os hidden statés do verificador, o que limita quanto dados ajudam. EAGLE-3 treina diretamente em predição de token. Segundo, teste durante treinamento (TTT): durante o treinamento do rascunho, alimentar as proprias predições anteriores do rascunho como entradas por varias etapas, da mesma forma que opera na inferência. Isso alinha as distribuções de treinamento e teste e para a acumulação de erros. Speedup medido: até 6.5x em chat, melhoria de 38% de throughput em batch 64 no SGLang num H100.

### Rollback do KV cache

A verificação estende o KV cache do verificador por `N` entradas em uma passada. Se a rejeição acontece na posição `j`, os conteudos do cache além da posição `j-1` agora estão errados. Duas implementações comuns: escrever num buffer temporario e commitar na aceitação (vLLM, TensorRT-LLM), ou manter um KV cache fisico mais um comprimento logico e truncar na rejeição. De qualquer forma, o custo do rollback e bytes por camada por head, que e desprezivel comparado ão custo do forward pass.

Pra busca em arvore do EAGLE-2, o verificador roda attention com uma mascara não-causal que respeita a topologia da arvore. A engenharia e trabalhosa mas o calculo e uma chamada padrão de flash-attention com uma mascara custom.

### Arquiteturas de rascunho em 2026

| Estratégia | Tipo de rascunho | `alpha` | Speedup | Custo de treinamento |
|-----------|-----------------|---------|---------|---------------------|
| Simples | LLM menor separado | 0.55-0.70 | 1.8-2.3x | Nenhum (reútiliza modelo menor existente) | 
| Medusa | Heads LM extras no verificador | 0.65-0.75 | 2-3x | ~1B tokens SFT |
| EAGLE-1 | Transformer 1-camada em hidden statés | 0.70-0.80 | 2.5-3x | ~60B tokens |
| EAGLE-2 | EAGLE-1 + arvore dinâmica de rascunho | 0.80-0.88 | 3-4x | ~60B tokens |
| EAGLE-3 | Fusão de features multi-camada + TTT | 0.88-0.92 | 3.5-6.5x | ~60-200B tokens |
| Lookahead | Sem rascunho (iteração Jacobi) | N/A | 1.3-1.6x | Nenhum |

Em produção em 2026: vLLM e SGLang vem com EAGLE-3 por padrão quando disponivel, EAGLE-2 caso contrario. TensorRT-LLM tem o caminho Medusa mais rapido pra modelos publicos da Meta e NVIDIA. llama.cpp envia rascunho simples pra implantação em CPU.

## Construir

Veja `code/main.py`. Este e o loop eespecificaçãoulativo completo de Leviathan com todas as pecas: rascunho de N, passada paralela do verificador, rejeição por posição, amostragem residual, token bonus, rollback do KV e verificação empirica de que a distribuição de saida combina com amostragem direta de `q`.

### Passo 1: a regra de rejeição

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Passo 2: distribuição residual

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Passo 3: um passo eespecificaçãoulativo completo

A função `especificação_step` rascunha `N` tokens de `p`, depois verifica todos de uma vez em uma avaliação paralela de `q`. Para cada token rascunhado aplica a regra de rejeição, e na primeira rejeição amostra a correção do residual. Se tudo aceitar, emite um token bonus de `q_{N+1}`.

### Passo 4: contabilidade do rollback do KV

O simulador rastreia um `kv_length` logico por worker. Na aceitação de `k` rascunhos, `kv_length += k`. Na rejeição na posição `j`, o cache ja foi escrito além de `j`, mas o comprimento logico e setado pra `prefix_length + j + 1` -- um além do token de correção. Leituras subsequentes truncam pro comprimento logico.

### Passo 5: a verificação de Leviathan

Rode 50.000 passos eespecificaçãoulativos. Conte a distribuição empirica de tokens aceitos. Compare com 50.000 amostragens diretas de `q`. A estatistica chi-quadrado deve estar bem abaixo do valor critico. O teorema passa na pratica.

### Passo 6: speedup vs. alpha

Varie a qualidade do rascunho perturbando `p` em relação a `q` em diferentes amplitudes. Meça `alpha`, depois plote tokens esperados por chamada do verificador como função de `alpha` e `N`. O codigo imprime uma tabela mostrando como qualidade de rascunho classe EAGLE-3 (`alpha` aproximado 0.9) desbloqueia 4-5 tokens por chamada do verificador.

## Usar

`vllm serve` em nivel de produção com EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --especificaçãoulative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_especificaçãoulative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang com EAGLE-3 em batch 64 num H100: aproximadamente 1.38x mais throughput que decodificação simples em batch 64, de acordo com o paper EAGLE-3.

Quando usar decodificação eespecificaçãoulativa:

- Qualquer carga de chat interativo onde laténcia p50 importa mais que throughput pico.
- Geração de codigo e saidas estruturadas (JSON, SQL). `alpha` e acima de 0.9 porque a distribuição alvo e altamente previsivel.
- Geração de texto longo (milhares de tokens). O speedup amortizado continua pagando.

Quando não usar:

- Modelos muito pequenos (< 3B). O rascunho não e tão mais barato que o verificador.
- Deploy em CPU batch-1 minusculo. O overhead de memoria do modelo de rascunho pode não valer.
- Amostragem criativa em temperatura onde `alpha` despenca.

## Entregar

Esta aula produz `outputs/skill-eagle3-tuner.md`. Dada uma carga de inferência (modelo, batch size, laténcia alvo, perfil de tarefa), recomenda uma estratégia de decodificação eespecificaçãoulativa e parâmetros de ajuste (familia do rascunho, `N`, profundidade da arvore, troca consciente de temperatura).

## Exercicios

1. Rode `code/main.py`. Confirme que a estatistica chi-quadrado na verificação da distribuição de Leviathan fica abaixo do valor critico de 95% em 50.000 amostras.

2. Varie `N` de 1 a 10 com `alpha` fixo em 0.9 e `c` fixo em 0.04. Plote tokens esperados por chamada do verificador e tempo real por token. Encontre o `N` que minimiza o tempo real. Explique a forma da curva.

3. Modifique o codigo pra simular busca em arvore EAGLE-2: a cada etapa, o rascunho propoe uma arvore de formato `[2, 2, 2]` (oito caminhos candidatos). O verificador roda uma vez e o caminho aceito de maior probabilidade vence. Calcule `alpha` por folha e tokens totais por chamada do verificador. Compare com decodificação eespecificaçãoulativa em cadeia linear no mesmo compute.

4. Implemente um simulador de rollback KV em batch pra duas sequencias concorrentes. Sequencia A tem todos os rascunhos aceitos; sequencia B rejeita na posição 2. Mostre que o `kv_length` correto e atualizado por sequencia e que nenhum trabalho e desperdicado.

5. Leia a Seção 4 do paper EAGLE-3 (Training-Time Test). Explique em duas frases por que o treinamento de rascunho ingenuo sem TTT sofre de vies de exposição, e por que alimentar o rascunho com suas proprias predições durante treinamento corrige isso. Conecte isso a literatura de scheduled sampling em seq2seq.

## Termos Principais

| Termo | O que a gente diz | O que realmente significa |
|-------|-------------------|--------------------------|
| Regra de Leviathan | "min(1, q sobre p)" | Bernoulli aceitar/rejeitar com probabilidade `min(1, q(d)/p(d))`, preserva a distribuição do verificador exatamente quando você amostra do residual na rejeição |
| Distribuição residual | "(q menos p) mais, normalizado" | `(q - p)_+` truncado em zero e renormalizado -- a distribuição correta pra amostrar na rejeição |
| Taxa de aceitação alpha | "quão seguido o rascunho ta certo" | Probabilidade esperada de sucesso de Bernoulli por token sob a regra de rejeicao; governa toda a matématica de speedup |
| EAGLE-1 | "rascunho de hidden staté" | Rascunho transformer minusculo condicionado no hidden staté da última camada do verificador (Li et al., 2024) |
| EAGLE-2 | "arvore dinâmica de rascunho" | EAGLE-1 mais uma arvore de continuações candidatas pontuadas com tree attention em uma passada do verificador |
| EAGLE-3 | "teste durante treinamento" | Dropa a loss de predição de features, treina em predição direta de token com o rascunho recebendo seus proprios outputs durante treinamento |
| Teste durante treinamento (TTT) | "correção de vies de exposição" | Rodar o rascunho autoregressivamente durante treinamento pra que as distribuições de entrada de treino e teste combinem -- analogo direto do scheduled sampling |
| Rollback do KV | "desfazer rascunho rejeitado" | Contabilidade que reseta o cache KV do verificador pro comprimento do prefixo aceito após uma rejeição |
| Token bonus | "o gratis" | Quando todos os `N` rascunhos aceitam, amostrar um extra de `q_{N+1}` sem custo adicional de verificador |
| Tree attention | "verificar varios candidatos de uma vez" | Attention com mascara não-causal que respeita a topologia de uma arvore de rascunho; calcula `q_i` pra cada no na arvore em um forward pass |

## Leitura Complementar

- [Leviathan, Kalman, Matias -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) -- o paper fundamental e teorema de equivalencia
- [Chen et al. -- Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) -- introdução independente concorrente com prova limpa
- [Li et al. -- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) -- EAGLE-1, rascunho condicionado em hidden staté
- [Li et al. -- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) -- busca em arvore dinâmica
- [Li et al. -- EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) -- o padrão de produção em 2026
- [Cai et al. -- Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) -- abordagem alternativa sem rascunho
- [Documentação de Decodificação Eespecificaçãoulativa do vLLM](https://docs.vllm.ai/en/latést/features/especificação_decode.html) -- referencia de produção canonica com todas as estratégias conectadas
