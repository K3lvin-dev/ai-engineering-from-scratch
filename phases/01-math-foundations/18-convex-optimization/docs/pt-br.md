# Otimização Convexa

> Problemas convexos têm um vale. Redes neurais têm milhões. Saber a diferença importa.

**Tipo:** Construção
**Idioma:** Python
**Pré-requisitos:** Fase 1, Lições 04 (Cálculo para ML), 08 (Otimização)
**Tempo:** ~90 minutos

## Objetivos de Aprendizado

- Testar se uma função é convexa usando definição, segunda derivada e critério Hessiano
- Implementar método de Newton e comparar sua convergência quadrática com descida do gradiente
- Resolver problemas de otimização com restrição usando multiplicadores de Lagrange e interpretar condições KKT
- Explicar por que paisagens de perda de redes neurais são não-convexas mas SGD ainda encontra boas soluções

## O Problema

Lição 08 ensinou descida do gradiente, momentum e Adam. Esses otimizadores caminham colina abaixo em qualquer superfície. Mas não trazem garantias. Descida do gradiente em paisagem não-convex pode cair em um mau mínimo local, ficar preso em um ponto de sela, ou oscillar para sempre.

Mas muitos problemas em ML são convexos. Regressão linear, regressão logística, SVMs, LASSO, ridge. Para esses, algo mais forte existe: otimização com garantias matemáticas.

## O Conceito

### Conjuntos Convexos

Um conjunto S é convexo se para quaisquer dois pontos em S, o segmento de reta entre eles também está inteiramente em S.

### Funções Convexas

Uma função f é convexa se para quaisquer dois pontos x, y no domínio:

```
f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)
```

Geometricamente: o segmento entre quaisquer dois pontos no gráfico está acima ou no gráfico.

### Por que Convexidade Importa

**Teorema central:** Para uma função convexa, todo mínimo local é mínimo global.

Isso significa que descida do gradiente não pode ficar presa. Qualquer caminho leva à mesma resposta.

### Hessiana

A Hessiana H de uma função f é a matriz de segundas derivadas parciais:

```
H[i][j] = d^2 f / (dx_i dx_j)
```

Autovalores todos positivos: convexo. Sinais mistos: ponto de sela.

### Método de Newton

Usa informação de segunda ordem (a Hessiana). Ajusta uma aproximação quadrática e pula direto para o mínimo.

```
x_new = x - H^(-1) * gradiente
```

Convergência quadrática perto do mínimo. Mas custa O(n^3) para inverter a Hessiana.

### Otimização com Restrição

**Multiplicadores de Lagrange** convertem problema com restrição em sem restrição:

```
L(x, lambda) = f(x) + lambda * g(x)
```

**Condições KKT** estendem para restrições de desigualdade:

```
1. Estacionaridade
2. Viabilidade primal
3. Viabilidade dual
4. Folga complementar
```

### Regularização como Otimização com Restrição

L2: restrição circular (encolhe pesos para zero).
L1: restrição de diamante (seleciona features, alguns pesos ficam zero).

### Dualidade

Todo problema primal tem um problema dual. Para convexos, ambos têm o mesmo valor ótimo. SVMs são resolvidas na forma dual (truque do kernel).

### Por que Deep Learning Funciona Apesar da Não-Convexidade

1. A maioria dos mínimos locais é boa o suficiente
2. Pontos de sela, não mínimos locais, são o obstáculo real
3. Superparametrização suaviza a paisagem
4. Ruído estocástico age como regularização implícita

## Construa

```python
def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    violations = 0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)
        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        if f(mid) > t * f(x) + (1 - t) * f(y) + 1e-10:
            violations += 1
    return violations == 0, violations
```

## Termos-Chave

| Termo | Significado |
|-------|-------------|
| Conjunto convexo | Segmento entre quaisquer dois pontos fica dentro |
| Função convexa | Hessiana é semi-definida positiva em todo lugar |
| Hessiana | Matriz de segundas derivadas parciais |
| Método de Newton | Otimizador de segunda ordem usando Hessiana inversa |
| Multiplicador de Lagrange | Variável para converter restrição em não-restrição |
| KKT | Condições necessárias para optimalidade com restrições |
| Dualidade | Todo problema primal tem um dual companion |
| Ponto de sela | Gradiente zero mas mínimo em algumas direções e máximo em outras |

## Leitura Adicional

- [Boyd & Vandenberghe: Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- [Nocedal & Wright: Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)
