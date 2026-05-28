# Sistemas Lineares

> Resolver Ax = b é o problema mais antigo da matemática que ainda roda sua rede neural.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 01 (Intuição de Álgebra Linear), 02 (Vetores & Matrizes), 03 (Transformações de Matriz)
**Tempo:** ~120 minutos

## Objetivos de Aprendizado

- Resolver Ax = b usando eliminação gaussiana com pivoteamento parcial e substituição regressiva
- Fatorar matrizes com decomposições LU, QR e Cholesky e explicar quando cada uma é apropriada
- Derivar as equações normais para mínimos quadrados e conectá-las à regressão linear e ridge
- Diagnosticar sistemas mal-condicionados usando o número de condição e aplicar regularização

## O Problema

Toda vez que você treina uma regressão linear, resolve um sistema linear. Toda vez que calcula um ajuste de mínimos quadrados, resolve um sistema linear. Toda vez que uma camada de rede neural computa `y = Wx + b`, está avaliando um lado de um sistema linear.

A equação Ax = b aparece em todo lugar. A é uma matriz de coeficientes conhecidos. b é um vetor de saídas conhecidas. x é o vetor de incógnitas que você quer encontrar.

## O Conceito

### O que Ax = b significa geometricamente

Um sistema de equações lineares tem uma interpretação geométrica. Cada equação define um hiperplano. A solução é o ponto onde todos os hiperplanos se intersectam.

### Eliminação Gaussiana

Transforma Ax = b em um sistema triangular superior Ux = custo c que você resolve por substituição regressiva.

```
1. Para cada coluna k (coluna pivô):
   a. Encontre a maior entrada na coluna k (pivoteamento parcial).
   b. Troque essa linha com a linha k.
   c. Para cada linha i abaixo de k:
      - Compute multiplicador m = A[i][k] / A[k][k]
      - Subtraia m vezes a linha k da linha i.
2. Substituição regressiva: resolva de baixo para cima.
```

Custo: O(n^3) operações.

### Decomposição LU

Fatora A em uma matriz triangular inferior L e uma matriz triangular superior U: A = LU.

```
Ax = b
LUx = b
Seja y = Ux:
  Ly = b    (substituição progressiva, O(n^2))
  Ux = y    (substituição regressiva, O(n^2))
```

O custo O(n^3) é pago uma vez. Cada resolução subsequente é O(n^2).

### Decomposição QR

Fatora A em uma matriz ortogonal Q e uma matriz triangular superior R: A = QR.

Mais numericamente estável que LU para problemas de mínimos quadrados.

### Decomposição Cholesky

Quando A é simétrica e definida positiva: A = L L^T. Duas vezes mais rápida que LU.

### Mínimos Quadrados

Quando Ax = b não tem solução exata (sistema sobredeterminado), minimize o erro quadrático:

```
minimize ||Ax - b||^2

Equações normais:
A^T A x = A^T b
```

### Conexão com ML

**Regressão linear:** A solução fechada resolve as equações normais X^T X w = X^T y.

**Regressão ridge:** Adiciona lambda * I a X^T X. O sistema regularizado sempre pode ser resolvido via Cholesky.

**Processos gaussianos:** A média preditiva requer resolver K alpha = y.

## Construa

### Passo 1: Eliminação gaussiana

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]
        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]
    return x
```

### Passo 2: Decomposição LU

### Passo 3: Decomposição Cholesky

### Passo 4: Mínimos quadrados via equações normais

### Passo 5: Número de condição

## Entregue

Esta lição produz:
- `code/linear_systems.py` com implementações do zero
- Demonstração de que equações normais e sklearn produzem os mesmos pesos

## Termos-Chave

| Termo | O que as pessoas dizem | O que realmente significa |
|-------|----------------------|--------------------------|
| Sistema linear | "Resolver para x" | Conjunto de equações lineares Ax = b |
| Eliminação gaussiana | "Redução de linha" | Zerar entradas abaixo da diagonal usando operações de linha |
| Decomposição LU | "Fatorar em triângulos" | A = LU onde L é triangular inferior e U é superior |
| Decomposição QR | "Fatoração ortogonal" | A = QR onde Q tem colunas ortonormais |
| Cholesky | "Raiz quadrada de matriz" | Para A simétrica definida positiva, A = LL^T |
| Mínimos quadrados | "Melhor ajuste quando exato é impossível" | Minimizar soma dos resíduos ao quadrado |
| Equações normais | "Atalho do cálculo" | A^T A x = A^T b |
| Número de condição | "Quão confiável é essa resposta" | kappa = sigma_max / sigma_min |

## Leitura Adicional

- [MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang)
- [3Blue1Brown: Inverse Matrices](https://www.3blue1brown.com/lessons/inverse-matrices)
