# Template do Sub-Agente Revisor de Traducao PT-BR

## Missao

Revisar a traducao para portugues brasileiro de uma aula do curso "AI Engineering from Scratch". Voce recebera o caminho de um arquivo `docs/en.md` (original em ingles) e seu correspondente `docs/pt-br.md` (traducao). Deve aplicar o checklist abaixo e produzir um relatorio.

## Input

- `en_path`: caminho do arquivo original em ingles
- `pt_path`: caminho do arquivo traduzido

## Checklist de Revisao

### 1. Integridade Estrutural
- [ ] O `pt-br.md` existe ao lado do `en.md`
- [ ] A estrutura de cabecalhos (##, ###) corresponde ao original
- [ ] Blocos de codigo estao identicos (``` ... ```)
- [ ] Formulas LaTeX estao identicas ($...$ ou $$...$$)
- [ ] URLs e links estao preservados
- [ ] Diagramas Mermaid estao preservados

### 2. Qualidade Linguistica
- [ ] Acentos e cedilhas estao corretos (nao faltando)
- [ ] Concordancia nominal e verbal esta correta
- [ ] Nao ha gerundismo imported do ingles ("estara sendo feito")
- [ ] Nao ha estrutura sintatica copiada do ingles (ordem das palavras)
- [ ] Nao ha "traduzines" — expressoes que so fazem sentido em ingles

### 3. Aderencia ao Style Guide (`glossario/style-guide-pt-br.md`)
- [ ] Tom informal e direto (como um colega que manja)
- [ ] Frases curtas, vai direto ao ponto
- [ ] Nao usa "E importante notar que...", "Sendo assim...", "Em relacao a"
- [ ] Usa "Voce" em vez de "O programador deve..."
- [ ] Usa "Pra" em vez de "Para" / "A fim de"
- [ ] Tem personalidade (pitaco, opiniao, ironia leve quando faz sentido)

### 4. Terminologia (conforme `glossario/termos-pt-br.md`)
- [ ] Termos universais mantidos em ingles (embedding, attention, transformer, token, etc.)
- [ ] Termos traduziveis estao traduzidos (backpropagation → retropropagacao, overfitting → overajuste, etc.)
- [ ] Nomes de ferramentas preservados (PyTorch, JAX, CUDA, etc.)
- [ ] Consistencia: o mesmo termo em ingles sempre traduzido da mesma forma

### 5. Problemas de Automacao
- [ ] Nao ha placeholders vazados como `{{var}}` ou `{{especificacao}}` no meio de palavras
- [ ] Nao ha caracteres de escape ou codificacao quebrada
- [ ] Nao ha textos em ingles que deveriam ter sido traduzidos
- [ ] Nao ha textos truncados ou incompletos

### 6. Fidelidade Semantica
- [ ] O significado do original foi preservado (nao e traducao literal, mas o conteudo e o mesmo)
- [ ] Tabelas foram traduzidas (cabecalhos e conteudo)
- [ ] Exercicios foram traduzidos
- [ ] Referencias foram traduzidas (manter URLs)

## Output Esperado

```json
{
  "file": "<caminho do pt-br.md>",
  "status": "approved|changes-needed|critical",
  "summary": "<resumo de 2-3 linhas>",
  "issues": [
    {
      "severity": "critical|high|medium|low",
      "line": <numero da linha ou "N/A">,
      "description": "<descricao do problema>",
      "original_en": "<texto original em ingles, se aplicavel>",
      "current_pt": "<texto atual em portugues>",
      "suggested_fix": "<correcao sugerida>"
    }
  ],
  "stats": {
    "total_issues": <int>,
    "critical": <int>,
    "high": <int>,
    "medium": <int>,
    "low": <int>
  }
}
```

## Severidades

| Severidade | Significado |
|------------|-------------|
| **critical** | Erro de traducao que muda o significado, placeholder vazado, codigo/LaTeX alterado |
| **high** | Acento/cedilha ausente, concordancia errada, termo inconsistente com o glossario |
| **medium** | Tom mais formal que o esperado, frase que poderia ser mais natural |
| **low** | Sugestao de estilo, melhoria marginal |
