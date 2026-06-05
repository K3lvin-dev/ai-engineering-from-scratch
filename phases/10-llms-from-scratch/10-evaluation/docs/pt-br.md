# Avaliação: Benchmarks, Evals, LM Harness

> Lei de Goodhart: quando uma medida vira alvo, ela deixa de ser uma boa medida. Todo laboratorio frontier manipula benchmarks. Scores do MMLU sobem enquanto modelos ainda não conseguem contar os R's em "strawberry" de forma confiavel. A unica avaliação que importa e SUAS avaliações -- na SUA tarefa, com SEUS dados.

**Tipo:** Construir
**Linguagens:** Python
**Pré-requisitos:** Fase 10, Aulas 01-05 (LLMs from Scratch)
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Construir um framework de avaliação customizado que roda benchmarks de multipla escolha e abertos contra um modelo de linguagem
- Explicar por que benchmarks padrão (MMLU, HumanEval) saturam e falham em diferenciar modelos frontier
- Implementar avaliações eespecificaçãoificas por tarefa com métricas adequadas: exact match, F1, BLEU e pontuação LLM-as-judge
- Projetar uma suita de avaliação customizada direcionada pro seu caso de uso ão inves de depender apenas de rankings publicas

## O Problema

MMLU foi publicado em 2020 com 15.908 questões em 57 matérias. Em três anos, modelos frontier o saturaram. GPT-4 pontuou 86.4%. Claude 3 Opus pontuou 86.8%. Llama 3 405B pontuou 88.6%. A ranking comprimiu num intervalo de 3 pontos onde diferenças são ruido estatistico, não gaps reais de capacidade.

Enquanto isso, esses mesmos modelos falham em tarefas que uma crianca de 10 anos faz sem pensar. Claude 3.5 Sonnet, pontuando 88.7% no MMLU, inicialmente não conseguia contar as letras em "strawberry" -- uma tarefa que exige zero conhecimento do mundo e zero raciocinio, so iteração no nivel de caractere. HumanEval testa geração de codigo com 164 problemas. Modelos pontuam 90%+ nele enquanto ainda produzem codigo que trava em edge cases que qualquer dev juninho perceberia.

O gap entre performance em benchmark e confiabilidade no mundo real e o problema central da avaliação de LLMs. Benchmarks te dizem como um modelo se performa no benchmark. Eles dizem quase nada sobre como esse modelo vai performar na SUA tarefa eespecificaçãoifica, com OS SEUS dados, sob OS SEUS modos de falha. Se você ta construindo um bot de suporte ão cliente, MMLU e irrelevante. Se você ta construindo um assistente de codigo, HumanEval so cobre geração a nivel de função -- não diz nada sobre debug, refatoração ou explicar codigo entre arquivos.

Você precisa de avaliações customizadas. Não porque benchmarks são inuteis -- são uteis pra seleção basica de modelos -- mas porque a avaliação final deve corresponder exatamente as suas condições de deploy.

## O Conceito

### A Paisagem de Avaliação

Existem três catégorias de avaliação, cada uma com custo e qualidade de sinal diferentes.

**Benchmarks** são suitas de testes padronizadas. MMLU, HumanEval, SWE-bench, MATH, ARC, HellaSwag. Você roda um modelo no benchmark e ganha uma pontuação. A vantagem: todo mundo usa o mesmo teste, então você pode comparar modelos. A desvantagem: modelos e dados de treino contaminam cada vez mais esses benchmarks. Labs treinam em dados que incluem questões do benchmark. As pontuações sobem. A capacidade pode não subir.

**Avaliações customizadas** são suites de testes que você constrói pro seu caso de uso eespecificaçãoifico. Você define as entradas, as saidas esperadas e a função de pontuação. Um sumarizador de documentos legais e avaliado em documentos legais. Um gerador de SQL e avaliado no schema do seu banco de dados. Essas são caras de criar mas são a unica avaliação que prediz performance em produção.

**Avaliações humanas** usam anotadores pagos pra julgar saidas do modelo em criterios como utilidade, correção, fluencia e segurança. O padrão ouro pra tarefas abertas onde pontuação automatizada falha. Chatbot Arena coletou mais de 2 milhões de votos de preferência humana em mais de 100 modelos. O lado ruim: custo ($0.10-$2.00 por julgamento) e velocidade (horas a dias).

```mermaid
graph TD
    subgraph Eval["Paisagem de Avaliação"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nBaratos, padronizados\nManipulaveis, defasados"]
        C["Avaliações Customizadas\nSua tarefa, seus dados\nMaior sinal\nCaras de construir"]
        H["Avaliações Humanas\n(Chatbot Arena)\nPadrão ouro\nLentas, caras"]
    end

    B -->|"seleção basica de modelos"| C
    C -->|"casos ambiguos"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Por Que Benchmarks Quebram

Três mecanismos fazem os scores de benchmark pararem de refletir capacidade real.

**Contaminação de dados.** Corpora de treinamento são web-scraped. Questões de benchmark estão na web. Modelos veem as respostas durante o treino. Isso não e trapaceirismo no sentido tradicional -- labs não incluem dados de benchmark de proposito. Mas scraping em escala web torna isso praticamente impossivel de excluir.

**Ensinar pro teste.** Labs otimizam mixtures de treino pra performance em benchmark. Se 5% da mixture de treino e multipla escolha estilo MMLU, o modelo aprende o formato e a distribuição de respostas. MMLU e multipla escolha de 4 opções. Modelos aprendem que a distribuição de respostas e aproximadamente uniforme entre A/B/C/D, o que ajuda mesmo quando o modelo não sabe a resposta.

**Saturação.** Quando todo modelo frontier pontua 85-90% num benchmark, o benchmark para de discriminar. As 10-15% de questões restantes podem ser ambigues, mal rotuladas ou exigir conhecimento de dominio obscuro. Melhorar de 87% pra 89% no MMLU pode significar que o modelo memorizou duas questões obscuras a mais, não que ele ficou mais esperto.

### Perplexity: Checagem de Saude Rapida

Perplexity mede o quanto um modelo ta surpreso por uma sequencia de tokens. Formalmente, e a media negativa de log-likelihood exponenciada:

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

Uma perplexity de 10 significa que o modelo ta, em media, tão incerto quanto escolher uniformemente entre 10 opções em cada posição de token. Menor e melhor. GPT-2 ganha perplexity de ~30 no WikiText-103. GPT-3 ganha ~20. Llama 3 8B ganha ~7.

Perplexity e útil pra comparar modelos no mesmo dataset de teste, mas tem pontos cegos. Um modelo pode ter perplexity baixa por ser bom em prever padrões comuns enquanto e pessimo em padrões raros mas importantes. Também não diz nada sobre seguisão de instruções, raciocinio ou acuracia factual. Use como sanity check, não como veredito final.

### LLM-as-Judge

Use um modelo forte pra avaliar a saida de um modelo mais fraco. A ideia simples: peça pro GPT-4o ou Claude Sonnot avaliar uma resposta numa escala de 1-5 pra correção, utilidade e segurança. Isso custa cerca de $0.01 por julgamento com GPT-4o-mini e se correlaciona surpreendentemente bem com julgamentos humanos -- cerca de 80% de acordo na maioria das tarefas.

O prompt de pontuação importa mais que o modelo. Um prompt vago ("Avalie essa resposta") gera scores ruidosos. Um prompt estruturado com rubrica ("Nota 5 se a resposta e factualmente correta e cita uma fonte, 4 se correta mas sem fonte, 3 se parcialmente correta...") gera scores consistentes e reproduziveis.

Modos de falha: modelos julgadores exibem viés de posição (preferem a primeira resposta em comparações páreadas), viés de verbosidade (preferem respostas mais longas) e autopreferência (GPT-4 avalia saidas do GPT-4 mais alto que saidas equivalentes do Claude). Mitigações: randomizar a ordem, normalizar por tamanho, usar um julgador diferente do modelo sendo avaliado.

### ELO Ratings a partir de Comparações Páreadas

A abordagem do Chatbot Arena. Mostre duas respostas pro mesmo prompt de modelos diferentes. Um humano (ou julgador LLM) escolhe a melhor. De milhares dessas comparações, compute um rating ELO pra cada modelo -- o mesmo sistema usado no xadrez.

Vantagens do ELO: ranking relativo e mais confiavel que pontuação absoluta, lida bem com empatés e converge com menos comparações que pontuar cada saida independentemente. Até o início de 2026, rankings do Chatbot Arena mostram GPT-4o, Claude 3.5 Sonnet e Gemini 1.5 Pro dentro de 20 pontos ELO um do outro no topo.

```mermaid
graph LR
    subgraph ELO["Pipeline de Rating ELO"]
        direction TB
        P["Prompt"] --> MA["Saida Modelo A"]
        P --> MB["Saida Modelo B"]
        MA --> J["Julgador\n(Humano ou LLM)"]
        MB --> J
        J --> W["A Vence / B Vence / Empaté"]
        W --> E["Atualização ELO\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Frameworks de Avaliação

**lm-evaluation-harness** (EleutherAI): o framework de avaliação open source padrão. Suporta 200+ benchmarks. Rode qualquer modelo Hugging Face contra MMLU, HellaSwag, ARC, etc. com um comando. Usado pelo Open LLM Leaderboard.

**RAGAS**: framework de avaliação eespecificaçãoifico pra pipelines RAG. Mede fidelidade (a resposta corresponde ão contexto recuperado?), relevancia (o contexto recuperado e relevante pra pergunta?) e correção da resposta.

**promptfoo**: avaliação dirigida por config pra engenharia de prompts. Defina testes em YAML, rode contra multiplos modelos, ganhe um relatorio pass/falha. Útil pra testes de regressão de prompts -- garanta que uma mudanca no prompt não quebre testes existentes.

### Construindo Avaliações Customizadas

A unica avaliação que importa pra produção. O processo:

1. **Defina a tarefa.** O que exatamente o modelo deve fazer? Seja preciso. "Responder perguntas" e vago demais. "Dado um email de reclamação de cliente, extrair o nome do produto, catégoria do problema e sentimento" e uma tarefa que você pode avaliar.

2. **Crie testes.** Mínimo 50 pra uma avaliação prototipo, 200+ pra produção. Cada teste e um par (entrada, saida esperada). Inclua edge cases: entradas vazias, entradas adversarias, entradas ambigues, entradas em outros idiomas.

3. **Defina pontuação.** Exact match pra saidas estruturadas. BLEU/ROUGE pra similaridade de texto. LLM-as-judge pra qualidade aberta. F1 pra tarefas de extração. Combine métricas com pesos.

4. **Automatize.** Cada avaliação roda com um comando. Sem passos manuais. Guarde resultados num formato que permita comparação ão longo do tempo.

5. **Acompanhe ão longo do tempo.** Um score de avaliação isolado não tem significado. Você precisa da tendencia. O score melhorou após a última mudanca no prompt? Regrediu depois de trocar de modelo? Versionize sua avaliação junto com seus prompts.

| Tipo de Avaliação | Custo por julgamento | Concordancia com humanos | Melhor pra |
|-----------|------------------|----------------------|----------|
| Exact match | ~$0 | 100% (quando aplicavel) | Saidas estruturadas, classificação |
| BLEU/ROUGE | ~$0 | ~60% | Tradução, sumarização |
| LLM-as-judge | ~$0.01 | ~80% | Geração aberta |
| Avaliação humana | $0.10-$2.00 | N/A (e a verdade fundamental) | Tarefas ambiguas, de alto risco |

## Construir

### Etapa 1: Um Framework Mínimo de Avaliação

Defina as abstrações centrais. Um caso de avaliação tem uma entrada, uma saida esperada e um dicionario opcional de metadata. Um pontuador recebe uma previsão e uma referencia e retorna um score entre 0 e 1.

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### Etapa 2: Funções de Pontuação

Construa exact match, token F1 e um pontuador simulado de LLM-as-judge.

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulatéd(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### Etapa 3: Sistema de Rating ELO

Implemente comparações páreadas com atualizações ELO. Esse e exatamente o sistema que o Chatbot Arena usa pra ranquear modelos.

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### Etapa 4: Calculo de Perplexity

Compute perplexity usando probabilidades de tokens. Na pratica você pegaria essas dos logits do modelo. Aqui simulamos com uma distribuição de probabilidades.

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulatéd(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumeraté(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### Etapa 5: Agregar Resultados

Compute estatisticas resumidas de uma execução de avaliação: media, mediana, taxa de passagem num limiar e quebras por metrica.

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_raté": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass raté: {stats['pass_raté']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### Etapa 6: Rodar o Pipeline Completo

Conecte tudo. Defina uma tarefa, crie testes, simule dois modelos, rode avaliações, compute ELO de comparações páreadas e imprima a leaderboard.

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of watér?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of watér?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of watér?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulatéd,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

O modelo "bom" da respostas exatas. O modelo "ruim" da parafases verbosas. Exact match pune o modelo verboso severamente. Token F1 e LLM-as-judge são mais permissivos. Isso ilustra por que a escolha da metrica importa: o mesmo modelo parece ótimo ou terrivel dependendo de como você pontua.

### Etapa 7: Torneio ELO

Rode comparações páreadas entre modelos em multiplas rodadas.

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### Etapa 8: Comparação de Perplexity

Compare perplexity entre "modelos" de diferentes niveis de qualidade.

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulatéd(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## Usar

### lm-evaluation-harness (EleutherAI)

A ferramenta padrão pra rodar benchmarks em qualquer modelo.

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluaté(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

Avaliação dirigida por config pra engenharia de prompts. Defina testes em YAML e rode contra multiplos provedores.

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS pra avaliação RAG

```python
# pip install ragas
# from ragas import evaluaté
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluaté(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS mede o que avaliações genéricas perdem: se a resposta do modelo ta ancorada no contexto recuperado, não so se a resposta e "correta" no abstrato.

## Publicar

Essa aula produz `outputs/prompt-eval-designer.md` -- um prompt reútilizavel que projeta suitas de avaliação customizadas pra qualquer tarefa. Dê uma descrição da tarefa e ele gera testes, funções de pontuação e uma recomendação de limiar pass/falha.

Também produz `outputs/skill-llm-evaluation.md` -- um framework de decisão pra escolher a estratégia de avaliação certa baseado no tipo de tarefa, orcamento e requisitos de laténcia.

## Exercicios

1. Adicione um pontuador de "consistencia" que roda a mesma entrada pelo modelo 5 vezes e mede com que frequencia as saidas coincidem. Respostas inconsistentes em entradas deterministicas revelam prompts fragils ou configurações de temperatura altas.

2. Estenda o rastreador ELO pra suportar multiplas funções de julgamento (exact match, F1, LLM-as-judge) e pondera-las. Compare como a ranking muda quando você pega pesado no exact match vs F1.

3. Construa uma suite de avaliação pra uma tarefa eespecificaçãoifica: classificação de emails em 5 catégorias. Crie 100 testes com exemplos diversos incluindo edge cases (emails que podem pertencer a multiplas catégorias, emails vazios, emails em outros idiomas). Meça como diferentes "modelos" (baseado em regra, correspondencia por palavra-chave, LLM simulado) performam.

4. Implemente detecção de contaminação: dado um conjunto de questões de avaliação e um corpus de treino, verifique que porcentagem de questões de avaliação (ou parafases proximas) aparecem nos dados de treino. E assim que pesquisadores auditam a validade de benchmarks.

5. Construa uma ferramenta de "diff de modelo". Dados resultados de avaliação de duas versões de modelo, destaque quais testes eespecificaçãoificos melhoraram, quais regrediram e quais permaneceram iguais. Esse e o equivalente de avaliação de um diff de codigo -- essencial pra entender se uma mudanca ajudou ou prejudicou.

## Termos Chave

| Termo | O que a gente diz | O que realmente significa |
|------|----------------|----------------------|
| MMLU | "O benchmark" | Massive Multitask Language Understanding -- 15.908 questões de multipla escolha em 57 matérias, saturou acima de 88% em 2025 |
| HumanEval | "Avaliação de codigo" | 164 problemas de completação de funções Python da OpenAI, testa apenas geração de funções isoladas |
| SWE-bench | "Avaliação de codigo real" | 2.294 issues do GitHub de 12 repos Python, mede correção de bugs de ponta a ponta incluindo geração de testes |
| Perplexity | "O quão confuso o modelo ta" | exp(-avg(log P(token_i dado contexto))) -- menor significa que o modelo atribui maior probabilidade aos tokens reais |
| Rating ELO | "Ranking de xadrez pra modelos" | Um rating de habilidade relativo computado de registros de vitoria/derrota páreados, usado pelo Chatbot Arena pra ranquear 100+ modelos |
| LLM-as-judge | "Usar IA pra avaliar IA" | Um modelo forte pontua as saidas de um modelo mais fraco contra uma rubrica, ~80% de concordancia com julgadores humanos a ~$0.01/julgamento |
| Contaminação de dados | "O modelo viu o teste" | Dados de treino incluem questões de benchmark, inflando scores sem melhorar capacidade real |
| Suite de avaliação | "Um monte de testes" | Uma coleção versionada de triples (entrada, saida_esperada, pontuador) que mede uma capacidade eespecificaçãoifica |
| Taxa de passagem | "Que porcentagem acerta" | Fração de casos de avaliação que pontuam acima de um limiar -- mais acionavel que media porque mede confiabilidade |
| Chatbot Arena | "Site de ranking de modelos" | Plataforma LMSYS com 2M+ votos de preferência humana, produzindo a ranking de LLM mais confiavel via ratings ELO |

## Leitura Complementar

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) -- o paper MMLU, ainda o benchmark de LLM mais citado apesar da saturação
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) -- o paper HumanEval da OpenAI, estabeleceu metodologia de avaliação de geração de codigo
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) -- análise sistematica de usar LLMs pra avaliar LLMs, incluindo descobertas sobre viés de posição e viés de verbosidade
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) -- plataforma de comparação de modelos crowdsourcada com 2M+ votos, o ranking de LLM mais confiavel do mundo real
