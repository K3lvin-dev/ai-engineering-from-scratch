# Normas e Distâncias

> Sua função de distância define o que "similar" significa. Escolha errado e tudo quebra.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 01 (Intuição de Álgebra Linear), 02 (Vetores, Matrizes & Operações)
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Implementar funções de distância L1, L2, cosseno, Mahalanobis, Jaccard e edição do zero
- Selecionar a métrica de distância apropriada para uma tarefa de ML e explicar por que alternativas falham
- Conectar normas L1 e L2 a regularização LASSO e Ridge e suas regiões de restrição geométrica
- Demonstrar como o mesmo dataset produz diferentes vizinhos mais próximos sob diferentes métricas

## O Problema

Você tem dois vetores. Talvez sejam word embeddings. Talvez sejam perfis de usuários. Talvez sejam arrays de pixels. Você precisa saber: o quão perto eles estão?

A resposta depende inteiramente de qual função de distância você escolhe. Dois pontos de dados podem ser vizinhos mais próximos sob uma métrica e distantes sob outra. Seu classificador KNN, seu motor de recomendações, seu banco de dados vetorial, seu algoritmo de clustering, sua função de perda -- tudo depende dessa escolha. Errando, seu modelo otimiza para a coisa errada.

Não existe distância universal melhor. L2 funciona para dados espaciais. Similaridade cosseno domina NLP. Jaccard lida com conjuntos. Distância de edição lida com strings. Mahalanobis leva em conta correlações. Wasserstein move massa de probabilidade. Cada uma codifica uma suposição diferente sobre o que "similar" significa.

Esta lição constrói cada função de distância principal do zero, mostra quando cada uma é a ferramenta certa, e demonstra como os mesmos dados produzem vizinhos completamente diferentes dependendo da métrica.

## O Conceito

### Normas: medindo magnitude de vetor

Uma norma mede o "tamanho" de um vetor. Toda função de distância entre dois vetores pode ser escrita como a norma da diferença: d(a, b) = ||a - b||. Então entender normas é entender distâncias.

### Norma L1 (Distância de Manhattan)

A norma L1 soma os valores absolutos de todos os componentes.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Chamada de distância de Manhattan porque mede o quão longe você anda em uma grade de cidade onde só pode se mover ao longo dos eixos. Sem diagonais.

```
Ponto A = (1, 1)
Ponto B = (4, 5)

Distância L1 = |4-1| + |5-1| = 3 + 4 = 7

Em uma grade, você anda 3 quarteirões leste e 4 quarteirões norte.
```

Quando usar L1:
- Dados esparsos de alta dimensão (features de texto, one-hot encodings)
- Quando você quer robustez a outliers (uma diferença enorme não domina)
- Problemas de seleção de features (regularização L1 promove esparsidade)

Conexão com regularização L1 (Lasso): adicionar ||w||_1 à sua função de perda penaliza a soma dos valores absolutos dos pesos. Isso empurra pesos pequenos para exatamente zero, realizando seleção automática de features. A penalidade L1 cria regiões de restrição em forma de diamante no espaço de pesos, e os cantos dos diamantes estão nos eixos onde alguns pesos são zero.

Conexão com funções de perda: Mean Absolute Error (MAE) é a distância L1 média entre previsões e alvos. Ela penaliza todos os erros linearmente, tornando-se robusta a outliers comparada ao MSE.

### Norma L2 (Distância Euclidiana)

A norma L2 é a distância em linha reta. Raiz quadrada da soma dos componentes ao quadrado.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Esta é a distância que você aprendeu na geometria. Pitágoras em n dimensões.

```
Ponto A = (1, 1)
Ponto B = (4, 5)

Distância L2 = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

A linha reta, cortando diagonalmente pela grade.
```

Quando usar L2:
- Dados contínuos de baixa a média dimensão
- Quando as escalas das features são comparáveis
- Distâncias físicas (dados espaciais, leituras de sensores)
- Similaridade de imagem no nível de pixel

Conexão com regularização L2 (Ridge): adicionar ||w||_2^2 à sua função de perda penaliza pesos grandes. Diferente de L1, não empurra pesos para zero. Encolhe todos os pesos proporcionalmente. A penalidade L2 cria regiões de restrição circulares, então não há cantos nos eixos. Pesos ficam pequenos mas raramente exatamente zero.

Conexão com funções de perda: Mean Squared Error (MSE) é a média das distâncias L2 ao quadrado. Elevar ao quadrado penaliza erros grandes mais fortemente que erros pequenos.

```
MAE (perda L1):  |y - y_hat|         Penalidade linear. Robusta a outliers.
MSE (perda L2):  (y - y_hat)^2       Penalidade quadrática. Sensível a outliers.
```

### Normas Lp: a família geral

L1 e L2 são casos especiais da norma Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Diferentes valores de p produzem "bolas unitárias" de formas diferentes (o conjunto de todos os pontos a distância 1 da origem):

```
p=1:    Forma de diamante      (cantos nos eixos)
p=2:    Círculo/esfera         (a bola redonda usual)
p=3:    Superelipse            (quadrado arredondado)
p=inf:  Quadrado/hipercubo    (lados planos ao longo dos eixos)
```

### Norma L-infinito (Distância de Chebyshev)

Conforme p se aproxima do infinito, a norma Lp converge para o componente absoluto máximo.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

A distância entre dois pontos é determinada pela única dimensão onde eles mais diferem. Todas as outras dimensões são ignoradas.

```
Ponto A = (1, 1)
Ponto B = (4, 5)

Distância L-inf = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Quando usar L-infinito:
- Quando o pior desvio em qualquer dimensão importa
- Tabuleiros de jogos (um rei no xadrez se move em L-infinito: um passo em qualquer direção custa 1)
- Tolerâncias de fabricação (cada dimensão deve estar dentro das especificações)

### Similaridade Cosseno e Distância Cosseno

A similaridade cosseno mede o ângulo entre dois vetores, ignorando suas magnitudes.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Varia de -1 (direções opostas) a +1 (mesma direção). Vetores perpendiculares têm similaridade cosseno 0.

Distância cosseno converte em distância: distancia_cosseno = 1 - similaridade_cosseno. Varia de 0 (direção idêntica) a 2 (direção oposta).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Por que cosseno domina NLP e embeddings: em texto, o comprimento do documento não deve afetar a similaridade. Um documento sobre gatos que é duas vezes mais longo que outro documento sobre gatos ainda deve ser "similar". Similaridade cosseno ignora magnitude (comprimento) e só se importa com direção. Dois documentos com a mesma distribuição de palavras mas comprimentos diferentes apontam na mesma direção e recebem similaridade cosseno 1.0.

Quando usar similaridade cosseno:
- Similaridade de texto (vetores TF-IDF, word embeddings, sentence embeddings)
- Qualquer domínio onde magnitude é ruído e direção é sinal
- Sistemas de recomendação (vetores de preferência de usuário)
- Busca por embeddings (bancos de dados vetoriais quase sempre usam cosseno ou produto escalar)

### Produto Escalar vs Similaridade Cosseno

O produto escalar de dois vetores é:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(ângulo)
```

Similaridade cosseno é o produto escalar normalizado por ambas as magnitudes. Quando ambos os vetores já estão normalizados (magnitude = 1), produto escalar e similaridade cosseno são idênticos.

```
Se ||a|| = 1 e ||b|| = 1:
    a . b = cos(ângulo entre a e b)
```

Quando eles diferem: o produto escalar inclui informação de magnitude. Um vetor com magnitude maior recebe uma pontuação de produto escalar maior. Isso importa em alguns sistemas de recuperação onde você quer que itens "populares" apareçam mais. A magnitude age como um sinal implícito de qualidade ou importância.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Ambos concordam na direção, mas produto escalar também reflete magnitude.
```

Na prática:
- Use similaridade cosseno quando quiser similaridade puramente direcional
- Use produto escalar quando magnitudes carregam informação significativa
- Muitos bancos de dados vetoriais (Pinecone, Weaviate, Qdrant) deixam você escolher
- Se seus embeddings são normalizados por L2, a escolha não importa

### Distância de Mahalanobis

A distância euclidiana trata todas as dimensões igualmente. Mas se suas features são correlatas ou têm escalas diferentes, L2 dá resultados enganosos.

A distância de Mahalanobis leva em conta a estrutura de covariância dos dados.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

onde S é a matriz de covariância dos dados.

Intuitivamente: a distância de Mahalanobis primeiro descorrelaciona e normaliza os dados (whitening), depois computa distância L2 nesse espaço transformado. Se S é a matriz identidade (features não correlatas, variância unitária), a distância de Mahalanobis se reduz à distância euclidiana.

```
Exemplo: altura e peso são correlacionados.
Alguém com 1,88m e 82kg não é incomum.
Alguém com 1,52m e 82kg é incomum.

Distância euclidiana pode dizer que ambos estão igualmente distantes da média.
Distância Mahalanobis identifica corretamente o segundo como outlier
porque leva em conta a correlação altura-peso.
```

Quando usar distância Mahalanobis:
- Detecção de outliers (pontos com grande distância Mahalanobis da média são outliers)
- Classificação quando features têm diferentes escalas e correlações
- Quando você tem dados suficientes para estimar uma matriz de covariância confiável
- Controle de qualidade na manufatura (monitoramento multivariado de processos)

### Similaridade Jaccard (para conjuntos)

Similaridade Jaccard mede sobreposição entre dois conjuntos.

```
J(A, B) = |A intersect B| / |A union B|
```

Varia de 0 (sem sobreposição) a 1 (conjuntos idênticos). Distância Jaccard = 1 - Similaridade Jaccard.

```
A = {gato, cachorro, peixe}
B = {gato, pássaro, peixe, cobra}

Interseção = {gato, peixe}          tamanho = 2
União = {gato, cachorro, peixe, pássaro, cobra}  tamanho = 5

Similaridade Jaccard = 2/5 = 0.4
Distância Jaccard = 0.6
```

Quando usar Jaccard:
- Comparar conjuntos de tags, categorias ou features
- Similaridade de documentos baseada em presença de palavras (não frequência)
- Detecção de quase-duplicatas (aproximação MinHash de Jaccard)
- Comparar vetores de features binárias (dados de presença/ausência)
- Avaliar modelos de segmentação (Intersection over Union = Jaccard)

### Distância de Edição (Levenshtein)

A distância de edição conta o número mínimo de operações de caractere único necessárias para transformar uma string em outra. As operações são: inserir, deletar ou substituir.

```
"kitten" -> "sitting"

kitten -> sitten  (substituir k -> s)
sitten -> sittin  (substituir e -> i)
sittin -> sitting (inserir g)

Distância de edição = 3
```

Computado usando programação dinâmica. Preencha uma matriz onde a entrada (i, j) é a distância de edição entre os primeiros i caracteres da string A e os primeiros j caracteres da string B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Quando usar distância de edição:
- Verificação e correção ortográfica
- Alinhamento de sequências de DNA (com operações ponderadas)
- Correspondência fuzzy de strings
- Deduplicação de dados de texto bagunçados

### Divergência KL (não é distância, mas é usada como uma)

A divergência KL mede como uma distribuição de probabilidade difere de outra. Visto na Lição 09, mas pertence a esta discussão porque as pessoas a usam como "distância" apesar de não ser.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Propriedade crítica: divergência KL NÃO é simétrica.

```
D_KL(P || Q) != D_KL(Q || P)
```

Isso significa que ela falha no requisito básico de uma métrica de distância. Também não satisfaz a desigualdade triangular. É uma divergência, não uma distância.

KL direta (D_KL(P || Q)) é "buscadora da média": Q tenta cobrir todos os modos de P.
KL reversa (D_KL(Q || P)) é "buscadora de modo": Q foca em um único modo de P.

Quando você vê divergência KL:
- VAEs (o termo KL no ELBO empurra a distribuição latente para uma prior)
- Destilação de conhecimento (o estudante tenta igualar a distribuição do professor)
- RLHF (a penalidade KL mantém o modelo ajustado próximo ao modelo base)
- Métodos de gradiente de política (restringindo atualizações de política)

### Distância de Wasserstein (Distância Earth Mover)

A distância de Wasserstein mede o mínimo de "trabalho" necessário para transformar uma distribuição de probabilidade em outra. Pense assim: se uma distribuição é uma pilha de terra e a outra é um buraco, quanta terra você tem que mover e quão longe?

```
W(P, Q) = inf sobre todos os planos de transporte gamma de E[d(x, y)]
```

Para distribuições 1D, simplifica-se para a integral da diferença absoluta das funções de distribuição acumulada:

```
W_1(P, Q) = integral |FDP_P(x) - FDP_Q(x)| dx
```

Por que Wasserstein importa:
- É uma métrica verdadeira (simétrica, satisfaz desigualdade triangular)
- Fornece gradientes mesmo quando distribuições não se sobrepõem (divergência KL vai para infinito)
- Esta propriedade a tornou central para Wasserstein GANs (WGANs), que resolveram a instabilidade de treino das GANs originais

```
Distribuições sem sobreposição:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

Divergência KL: infinito (log de zero)
Wasserstein: 4 (mover toda massa 4 bins)

Wasserstein dá um gradiente significativo. KL não.
```

Quando usar Wasserstein:
- Treino de GAN (WGAN, WGAN-GP)
- Comparar distribuições que podem não se sobrepor
- Problemas de transporte ótimo
- Recuperação de imagens (comparando histogramas de cor)

### Por que Tarefas Diferentes Precisam de Distâncias Diferentes

| Tarefa | Melhor distância | Por quê |
|--------|-----------------|---------|
| Similaridade de texto | Cosseno | Magnitude é ruído, direção é significado |
| Comparação pixel de imagem | L2 | Relações espaciais importam, features têm escala comparável |
| Features esparsas de alta dim | L1 | Robusta, não amplifica diferenças raras grandes |
| Sobreposição de conjuntos (tags, categorias) | Jaccard | Dados são naturalmente conjuntos, não vetoriais |
| Correspondência de string | Distância de edição | Operações mapeiam para intuição humana de edição |
| Detecção de outlier | Mahalanobis | Leva em conta correlações e escalas de features |
| Comparação de distribuições | Divergência KL | Mede informação perdida ao usar Q em vez de P |
| Treino de GAN | Wasserstein | Dá gradientes mesmo quando distribuições não se sobrepõem |
| Embeddings (banco vetorial) | Cosseno ou produto escalar | Embeddings são treinados para codificar significado na direção |
| Recomendação | Produto escalar | Magnitude pode codificar popularidade ou confiança |
| Sequências de DNA | Distância de edição ponderada | Custos de substituição variam por par de nucleotídeos |
| Controle de qualidade fabril | L-infinito | Pior desvio em qualquer dimensão importa |

### Conexão com Funções de Perda

Funções de perda são funções de distância aplicadas a previsões vs alvos.

```
Função de perda       Distância usada       Comportamento
MSE                   L2 ao quadrado        Penaliza erros grandes pesadamente
MAE                   L1                    Penaliza todos erros igualmente
Huber                 L1 para erros grandes,  Melhor dos dois mundos: robusto a outliers,
                      L2 para erros pequenos gradiente suave perto de zero
Cross-entropy         Divergência KL         Mede incompatibilidade de distribuição
Hinge                 max(0, margem - d)     Penaliza só abaixo da margem
Triplet               L2 (tipicamente)       Aproxima positivos, afasta negativos
Contrastive           L2                     Pares similares perto, dissimilares além da margem
```

### Conexão com Regularização

Regularização adiciona uma penalidade de norma sobre os pesos à função de perda.

```
Regularização L1 (Lasso):   perda + lambda * ||w||_1
  -> Pesos esparsos. Alguns pesos tornam-se exatamente zero.
  -> Seleção automática de features.
  -> Solução tem cantos (não-diferenciável em zero).

Regularização L2 (Ridge):   perda + lambda * ||w||_2^2
  -> Pesos pequenos. Todos os pesos encolhem para zero.
  -> Sem seleção de features (nada vai a exatamente zero).
  -> Solução suave em todo lugar.

Elastic Net:                  perda + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combina esparsidade de L1 com estabilidade de L2.
  -> Grupos de features correlacionadas são mantidos ou removidos juntos.
```

Por que L1 produz esparsidade mas L2 não: imagine a região de restrição no espaço de pesos 2D. L1 é um diamante, L2 é um círculo. Os contornos da função de perda (elipses) têm maior probabilidade de tocar o diamante em um canto, onde um peso é zero. Eles tocam o círculo em um ponto suave, onde ambos os pesos são não-zero.

### Busca por Vizinhos Mais Próximos

Toda função de distância implica um problema de busca pelo vizinho mais próximo: dado um ponto de consulta, encontre os pontos mais próximos em um dataset.

A busca exata pelo vizinho mais próximo é O(n * d) por consulta em um dataset de n pontos com d dimensões. Para datasets grandes, isso é muito lento.

Algoritmos de Vizinhos Mais Próximos Aproximados (ANN) trocam uma pequena quantidade de precisão por enormes ganhos de velocidade:

```
Algoritmo         Abordagem                      Usado por
KD-trees          Partição do espaço alinhada    scikit-learn (baixa dim)
                  aos eixos
Ball trees        Hiperesferas aninhadas          scikit-learn (média dim)
LSH               Projeções hash aleatórias       Detecção de quase-duplicatas
HNSW              Grafo small-world               FAISS, Qdrant, Weaviate
                  navegável hierárquico
IVF               Índice de arquivo invertido     FAISS (escala bilhão)
                  com busca baseada em cluster
Quantização       Comprimir vetores, buscar       FAISS (memória restrita)
produto           no espaço comprimido
```

HNSW (Hierarchical Navigable Small World) é o algoritmo dominante em bancos de dados vetoriais modernos. Ele constrói um grafo multicamadas onde cada nó se conecta a seus vizinhos mais próximos aproximados. A busca começa na camada superior (esparsa, saltos longos) e desce até a camada inferior (densa, saltos curtos).

## Construa

### Passo 1: Todas as funções de norma e distância

Consulte `code/distances.py` para a implementação completa. Cada função é construída do zero usando apenas matemática básica do Python.

### Passo 2: Mesmos dados, distâncias diferentes, vizinhos diferentes

A demonstração em `distances.py` cria um dataset, escolhe um ponto de consulta, e mostra como o vizinho mais próximo muda dependendo da métrica de distância. O ponto que é "mais próximo" sob L1 pode não ser o mais próximo sob L2 ou cosseno.

### Passo 3: Busca de similaridade de embeddings

O código inclui uma busca simulada de similaridade de embeddings que encontra os "documentos" mais similares a uma consulta usando similaridade cosseno vs distância L2, mostrando que os rankings podem diferir.

## Use

O uso prático mais comum: encontrar itens similares em um banco de dados vetorial.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 mais similares ao item 0: {top_k}")
print(f"Similaridades: {similarities[top_k]}")
```

Quando você chama `model.encode(texto)` e depois busca em um banco de dados vetorial, isso é o que acontece por baixo dos panos. O modelo de embedding mapeia texto para vetores. O banco de dados vetorial computa similaridade cosseno (ou produto escalar) entre seu vetor de consulta e cada vetor armazenado, usando algoritmos ANN para evitar verificar todos eles.

## Exercícios

1. Calcule as distâncias L1, L2 e L-infinito entre (1, 2, 3) e (4, 0, 6). Verifique que L-inf <= L2 <= L1 sempre vale para qualquer par de pontos. Prove por que essa ordem é garantida.

2. Crie dois vetores onde a similaridade cosseno é alta (> 0.9) mas a distância L2 é grande (> 10). Explique geometricamente o que está acontecendo. Depois crie dois vetores onde a similaridade cosseno é baixa (< 0.3) mas a distância L2 é pequena (< 0.5).

3. Implemente uma função que recebe um dataset e um ponto de consulta e retorna o vizinho mais próximo sob L1, L2, cosseno e distância Mahalanobis. Encontre um dataset onde todos os quatro discordam sobre qual ponto é o mais próximo.

4. Calcule a distância de Wasserstein entre [0.5, 0.5, 0, 0] e [0, 0, 0.5, 0.5] manualmente usando o método da CDF. Depois calcule entre [0.25, 0.25, 0.25, 0.25] e [0, 0, 0.5, 0.5]. Qual é maior e por quê?

5. Implemente MinHash para similaridade Jaccard aproximada. Gere 100 conjuntos aleatórios, calcule Jaccard exato para todos os pares e compare com aproximação MinHash usando 50, 100 e 200 funções hash. Plote o erro de aproximação.

## Termos-Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|----------------------|--------------------------|
| Norma | "Tamanho de um vetor" | Função que mapeia um vetor para um escalar não-negativo, satisfazendo desigualdade triangular, homogeneidade absoluta, e zero apenas para o vetor zero |
| Norma L1 | "Distância de Manhattan" | Soma dos valores absolutos dos componentes. Produz esparsidade em otimização. Robusta a outliers |
| Norma L2 | "Distância Euclidiana" | Raiz quadrada da soma dos componentes ao quadrado. A distância em linha reta no espaço euclidiano |
| Norma Lp | "Norma generalizada" | A raiz p-ésima da soma das p-ésimas potências dos componentes absolutos. L1 e L2 são casos especiais |
| Norma L-infinito | "Norma máxima" ou "Distância de Chebyshev" | O valor absoluto máximo dos componentes. O limite de Lp quando p tende ao infinito |
| Similaridade cosseno | "Ângulo entre vetores" | Produto escalar normalizado por ambas as magnitudes. Varia de -1 a +1. Ignora comprimento do vetor |
| Distância cosseno | "1 menos similaridade cosseno" | Converte similaridade cosseno em distância. Varia de 0 a 2 |
| Produto escalar | "Cosseno não-normalizado" | Soma dos produtos componente a componente. Igual à similaridade cosseno vezes ambas as magnitudes |
| Distância Mahalanobis | "Distância consciente de correlação" | Distância L2 em um espaço que foi whitened (descorrelacionado e normalizado) usando a matriz de covariância dos dados |
| Similaridade Jaccard | "Sobreposição de conjuntos" | Tamanho da interseção dividido pelo tamanho da união. Para conjuntos, não vetores |
| Distância de edição | "Distância Levenshtein" | Mínimo de inserções, deleções e substituições para transformar uma string em outra |
| Divergência KL | "Distância entre distribuições" | Não é uma distância verdadeira (não simétrica). Mede bits extras ao usar Q para codificar P |
| Distância de Wasserstein | "Distância Earth Mover" | Trabalho mínimo para transportar massa de uma distribuição para outra. Uma métrica verdadeira |
| Vizinho mais próximo aproximado | "Busca ANN" | Algoritmos (HNSW, LSH, IVF) que encontram pontos aproximadamente mais próximos muito mais rápido que busca exata |
| HNSW | "O algoritmo de banco vetorial" | Grafo Hierárquico Navigable Small World. Grafo multicamadas para busca rápida de vizinhos mais próximos aproximados |
| Regularização L1 | "Lasso" | Adicionar a norma L1 dos pesos à perda. Leva pesos a zero (esparsidade) |
| Regularização L2 | "Ridge" ou "decaimento de pesos" | Adicionar a norma L2 ao quadrado dos pesos à perda. Encolhe pesos para zero sem esparsidade |
| Elastic Net | "L1 + L2" | Combina regularização L1 e L2. Lida com grupos de features correlacionadas melhor que qualquer um sozinho |

## Leitura Adicional

- [FAISS: Biblioteca para Busca de Similaridade Eficiente](https://github.com/facebookresearch/faiss) - Biblioteca da Meta para busca ANN em escala de bilhões
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - o artigo que introduziu distância Earth Mover em GANs
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - algoritmo ANN fundamental
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, onde similaridade cosseno se tornou o padrão para embeddings
- [Documentação sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html) - guia prático para métricas de distância e algoritmos de vizinhança no scikit-learn
