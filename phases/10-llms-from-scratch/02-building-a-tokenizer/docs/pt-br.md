# Construindo um Tokenizer do Zero

> A Lição 01 te deu um brinquedo. Esta lição te dá uma arma.

**Tipo:** Construção
**Linguagens:** Python
**Pré-requisitos:** Fase 10, Lição 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Construir um tokenizer BPE de nível profissional que lida com Unicode, normalização de espaços e tokens especiais
- Implementar fallback no nível de byte para que o tokenizer consiga codificar qualquer entrada (incluindo emoji, CJK e código) sem tokens desconhecidos
- Adicionar padrões regex de pré-tokenização que dividem o texto nos limites das palavras antes de aplicar os merges BPE
- Treinar um tokenizer customizado num corpus e avaliar sua taxa de compressão comparada ao tiktoken em texto multilíngue

## O Problema

Seu tokenizer BPE da Lição 01 funciona em texto em inglês. Agora joga japonês nele. Ou emoji. Ou código Python com tabs e espaços misturados.

Ele quebra.

Não porque o BPE esteja errado — porque a implementação está incompleta. Um tokenizer de produção lida com bytes brutos em qualquer codificação, normaliza Unicode antes de dividir, gerencia tokens especiais que nunca são mesclados, encadeia pré-tokenização com divisão em subpalavras, e faz tudo isso rápido o suficiente para não gargalar um pipeline de treinamento processando 15 trilhões de tokens.

O tokenizer do GPT-2 tem 50.257 tokens. O Llama 3 tem 128.256. O GPT-4 tem aproximadamente 100.000. Esses não são números de brinquedo. As tabelas de merge por trás desses vocabulários foram treinadas em centenas de gigabytes de texto, e a maquinaria ao redor — normalização, pré-tokenização, injeção de tokens especiais, formatação de template de chat — é o que separa um tokenizer que lida com "hello world" de um que lida com a internet inteira.

Você vai construir essa maquinaria.

## O Conceito

### O Pipeline Completo

Um tokenizer de produção não é um algoritmo só. É um pipeline de cinco estágios, cada um resolvendo um problema diferente.

```mermaid
graph LR
    A[Texto Bruto] --> B[Normalizar]
    B --> C[Pré-Tokenizar]
    C --> D[Merge BPE]
    D --> E[Tokens Especiais]
    E --> F[IDs dos Tokens]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Cada estágio tem um trabalho específico:

| Estágio | O Que Faz | Por Que Importa |
|---------|-----------|-----------------|
| Normalizar | NFKC Unicode, minúsculas opcionais, remover acentos opcionais | A ligadura "fi" (U+FB01) vira "fi" (dois caracteres). Sem isso, a mesma palavra ganha tokens diferentes. |
| Pré-Tokenizar | Divide o texto em pedaços antes do BPE | Impede o BPE de mesclar entre limites de palavras. "the cat" nunca deve produzir um token "e c". |
| Merge BPE | Aplica regras de merge aprendidas em sequências de bytes | A compressão central. Transforma bytes brutos em tokens de subpalavras. |
| Tokens Especiais | Injeta [BOS], [EOS], [PAD], marcadores de template de chat | Esses tokens têm IDs fixos. Nunca participam dos merges BPE. O modelo precisa deles para estrutura. |
| Mapeamento de IDs | Converte strings de tokens em IDs inteiros | O modelo vê inteiros, não strings. |

### BPE em Nível de Byte

O tokenizer da Lição 01 operava em bytes UTF-8. Foi a decisão certa. Mas pulamos algo importante: o que acontece quando esses bytes não são UTF-8 válido?

BPE em nível de byte resolve isso tratando cada valor de byte possível (0-255) como um token válido. Seu vocabulário base tem exatamente 256 entradas. Qualquer arquivo — texto, binário, corrompido — pode ser tokenizado sem produzir um token desconhecido.

O GPT-2 adicionou um truque: mapear cada byte para um caractere Unicode imprimível para o vocabulário continuar legível para humanos. O byte 0x20 (espaço) vira o caractere "G" no mapeamento deles. Isso é puramente cosmético. O algoritmo não liga.

O poder real: BPE em nível de byte lida com todas as línguas da Terra. Caracteres chineses são 3 bytes UTF-8 cada. Japonês pode ser 3-4 bytes. Árabe, devanágari, emoji — tudo sequência de bytes. O algoritmo BPE encontra padrões nessas sequências de bytes exatamente da mesma forma que encontra padrões em bytes ASCII de inglês.

### Pré-Tokenização

Antes do BPE tocar no seu texto, você precisa dividi-lo em pedaços. Isso impede o algoritmo de merge de criar tokens que atravessam limites de palavras.

O GPT-2 usa um padrão regex para dividir o texto:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Esse padrão divide em contrações ("don't" vira "don" + "'t"), palavras com espaços opcionais à esquerda, números, pontuação e espaços em branco. O espaço à esquerda fica preso à palavra — então "the cat" vira [" the", " cat"], não ["the", " ", "cat"].

O Llama usa SentencePiece, que pula o regex completamente. Ele trata o fluxo bruto de bytes como uma sequência longa e deixa o algoritmo BPE descobrir os limites. Isso é mais simples, mas dá mais liberdade ao BPE para criar tokens que cruzam palavras.

A escolha importa. O regex do GPT-2 impede que o tokenizer aprenda que "the" no fim de uma palavra e "the" no começo da próxima devem se mesclar. O SentencePiece permite, o que às vezes produz compressão mais eficiente mas tokens menos interpretáveis.

### Tokens Especiais

Todo tokenizer de produção reserva IDs de tokens para marcadores estruturais:

| Token | Propósito | Usado Por |
|-------|-----------|-----------|
| `[BOS]` / `<s>` | Início da sequência | Llama 3, GPT |
| `[EOS]` / `</s>` | Fim da sequência | Todos os modelos |
| `[PAD]` | Preenchimento para alinhamento de lote | BERT, T5 |
| `[UNK]` | Token desconhecido (BPE em nível de byte elimina isso) | BERT, WordPiece |
| `<\|im_start\|>` | Início de limite de mensagem do chat | ChatGPT, Qwen |
| `<\|im_end\|>` | Fim de limite de mensagem do chat | ChatGPT, Qwen |
| `<\|user\|>` | Marcador de turno do usuário | Llama 3 |
| `<\|assistant\|>` | Marcador de turno do assistente | Llama 3 |

Tokens especiais nunca são divididos pelo BPE. Eles são identificados exatamente antes do algoritmo de merge rodar, substituídos por seu ID fixo, e o texto ao redor é tokenizado normalmente.

### Templates de Chat

É aqui que a maioria das pessoas se confunde e a maioria das implementações quebra.

Quando você envia mensagens para um modelo de chat, a API aceita uma lista de mensagens:

```
[
  {"role": "system", "content": "Você é útil."},
  {"role": "user", "content": "Olá"},
  {"role": "assistant", "content": "Oi!"}
]
```

O modelo não vê JSON. Ele vê uma sequência plana de tokens. O template de chat converte as mensagens nessa sequência plana usando tokens especiais. Cada modelo faz isso de forma diferente:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Você é útil.<|eot_id|><|start_header_id|>user<|end_header_id|>

Olá<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Oi!<|eot_id|>

ChatGPT:
<|im_start|>system
Você é útil.<|im_end|>
<|im_start|>user
Olá<|im_end|>
<|im_start|>assistant
Oi!<|im_end|>
```

Se você errar o template, o modelo produz lixo. Ele foi treinado num formato exato. Qualquer desvio — uma quebra de linha faltando, um token trocado, um espaço extra — coloca a entrada fora da distribuição de treinamento.

### Velocidade

Python é lento demais para tokenização em produção.

O tiktoken (OpenAI) é escrito em Rust com bindings Python. O tokenizers da HuggingFace também é Rust. O SentencePiece é C++. Eles alcançam aceleração de 10 a 100x comparado ao Python puro.

Para contexto: tokenizar 15 trilhões de tokens para o pré-treinamento do Llama 3 a 1 milhão de tokens por segundo (Python rápido) levaria 174 dias. A 100 milhões de tokens por segundo (Rust), leva 1,7 dias.

Você está construindo em Python para entender o algoritmo. Em produção, você usaria uma implementação compilada e só tocaria no wrapper Python.

## Construa

### Passo 1: Codificação em Nível de Byte

A base. Converta qualquer string em uma sequência de bytes, mapeie cada byte para um caractere imprimível para exibição, e reverta o processo.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Teste em texto multilíngue para ver as contagens de bytes:

```python
texts = [
    ("Inglês", "hello"),
    ("Chinês", "你好"),
    ("Emoji", "🔥"),
    ("Misto", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" tem 5 bytes. "你好" tem 6 bytes (3 por caractere). O emoji de fogo tem 4 bytes. O tokenizer em nível de byte não liga para qual língua é. Bytes são bytes.

### Passo 2: Pré-Tokenizador com Regex

Divida o texto em pedaços usando o padrão regex do GPT-2. Cada pedaço é tokenizado independentemente pelo BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

O módulo `regex` suporta escapes de propriedades Unicode (`\p{L}` para letras, `\p{N}` para números). O módulo padrão `re` não suporta, então caímos de volta para classes de caracteres ASCII. Para tokenizers multilíngues de produção, instale `regex`.

Teste:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

O espaço à esquerda permanece preso à palavra. Contrações dividem no apóstrofo. Pontuação vira seu próprio pedaço. O BPE nunca vai mesclar tokens através desses limites.

### Passo 3: BPE em Sequências de Bytes

O algoritmo central da Lição 01, mas agora operando em pedaços pré-tokenizados independentemente.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Passo 4: Manipulação de Tokens Especiais

Tokens especiais precisam de correspondência exata e IDs fixos. Eles bypassam o BPE completamente.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Passo 5: Classe Tokenizer Completa

Encadeie tudo: normalize, divida em tokens especiais, pré-tokenize, faça merge BPE, mapeie para IDs.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Passo 6: Teste Multilíngue

O teste de verdade. Jogue inglês, chinês, emoji e código nele.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Caracteres chineses produzem 3 bytes cada. O emoji produz 4 bytes. Nada disso quebra o tokenizer. Nenhum produz tokens desconhecidos. Esse é o poder do BPE em nível de byte.

## Use

### Comparando Tokenizers Reais

Carregue os tokenizers reais do Llama 3, GPT-4 e Mistral. Veja como cada um lida com o mesmo parágrafo multilíngue.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Você vai ver contagens de tokens diferentes para o mesmo texto. Llama 3 com vocabulário de 128K é mais agressivo ao mesclar padrões comuns. GPT-4 com 100K fica no meio. Mistral com 32K produz mais tokens mas tem uma camada de embedding menor.

A compensação é sempre a mesma: vocabulário maior significa sequências mais curtas, mas mais parâmetros.

## Entregue

Esta lição produz um prompt para construir e depurar tokenizers de produção. Veja `outputs/prompt-tokenizer-builder.md`.

## Exercícios

1. **Fácil:** Adicione um método `get_token_bytes(id)` que mostra os bytes brutos para qualquer ID de token. Use-o para inspecionar o que seus tokens mesclados mais comuns realmente representam.
2. **Médio:** Implemente o pré-tokenizador estilo Llama que divide em espaços e dígitos mas mantém espaços à esquerda. Compare seu vocabulário com a abordagem do regex do GPT-2 no mesmo corpus.
3. **Difícil:** Adicione um método de template de chat que recebe uma lista de mensagens `{"role": ..., "content": ...}` e produz a sequência de tokens correta para o formato de chat do Llama 3. Teste contra a implementação da HuggingFace.

## Termos-Chave

| Termo | O que dizem | O que realmente significa |
|-------|-------------|--------------------------|
| BPE em nível de byte | "Tokenizer que funciona em bytes" | BPE com um vocabulário base de 256 valores de byte — lida com qualquer entrada sem tokens desconhecidos |
| Pré-tokenização | "Dividir antes do BPE" | Divisão baseada em regex ou regras que impede o BPE de mesclar entre limites de palavras |
| Normalização NFKC | "Limpeza de Unicode" | Decomposição canônica seguida de composição de compatibilidade — ligadura "fi" vira "fi", "A" largo vira "A" |
| Template de chat | "Como mensagens viram tokens" | O formato exato para converter uma lista de mensagens role/content em uma sequência plana de tokens — específico do modelo e deve corresponder ao formato de treinamento |
| Tokens especiais | "Tokens de controle" | IDs de token reservados que bypassam o BPE — [BOS], [EOS], [PAD], marcadores de chat — identificados exatamente antes do merge |
| Fertilidade | "Tokens por palavra" | Razão entre tokens de saída e palavras de entrada — 1,3 para inglês no GPT-4, 2-3 para coreano, maior significa contexto desperdiçado |
| tiktoken | "Tokenizer da OpenAI" | Implementação BPE em Rust com bindings Python — 10-100x mais rápido que Python puro |
| Tabela de merge | "O vocabulário" | Lista ordenada de merges de pares de bytes aprendidos durante o treinamento — isso É o conhecimento aprendido do tokenizer |

## Leitura Adicional

- [Código fonte do tiktoken (OpenAI)](https://github.com/openai/tiktoken) — Implementação BPE em Rust usada pelo GPT-3.5/4
- [Tokenizers da HuggingFace](https://github.com/huggingface/tokenizers) — Biblioteca de tokenizers em Rust suportando BPE, WordPiece, Unigram
- [Paper do Llama 3 (Meta, 2024)](https://arxiv.org/abs/2407.21783) — detalhes sobre vocabulário de 128K e treinamento do tokenizer
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) — tokenização agnóstica a idioma
- [Código fonte do tokenizer do GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — o mapeamento original de byte para Unicode
