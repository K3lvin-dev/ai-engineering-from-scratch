# Kanban — Revisao da Traducao PT-BR

> Acompanhamento do progresso da revisao completa da localizacao para portugues brasileiro.

## Colunas

| Backlog | Em Revisao | Aguardando Correcao | Corrigido | Aprovado |
|---------|------------|---------------------|-----------|----------|

---

## Seed — Problemas Conhecidos (Triagem Inicial)

### :critical: Criticos

| Item | Arquivo | Problema |
|------|---------|----------|
| content-translations.js | `site/content-translations.js` | ~536 entradas sem acentos/cedilhas (ex: "Configuracao", "Nucleo", "Avancado") |
| Placeholders vazados | `phases/03-deep-learning-core/03-backpropagation/docs/pt-br.md` | "Gradientees" na tabela de termos-chave |
| Placeholders vazados | `phases/10-llms-from-scratch/01-tokenizers/docs/pt-br.md` | `{{especificacao}}` vazado dentro de palavras em varias linhas |

### :high: Alto

| Item | Arquivo | Problema |
|------|---------|----------|
| Glossario incompleto | `glossario/termos-pt-br.md` | Faltam: precision, recall, F1, AUC, goodput, throughput, cold start |
| Revisao de tom | Todas as `docs/pt-br.md` | Verificar aderencia ao style guide (informal/direto) |

### :medium: Medio

| Item | Arquivo | Problema |
|------|---------|----------|
| UI strings | `site/header.js` | ~100 strings — conferir consistencia terminologica |
| Quiz translations | `site/quiz-translations.js` | ~1844 perguntas — verificar completude e indices `correct` |

---

## Sprints

- [ ] **Sprint 0**: Infraestrutura (kanban, template revisor, checklist)
- [ ] **Sprint 1**: Aulas de fundamentos (fases 00-06, ~114 aulas)
- [ ] **Sprint 2**: Arquitetura e geracao (fases 07-12, ~115 aulas)
- [ ] **Sprint 3**: Ferramentas, agentes, producao (fases 13-19, ~206 aulas)
- [ ] **Sprint 4**: Quiz translations (1844 perguntas)
- [ ] **Consolidacao**: Script de verificacao e fechamento

---

## Log de Revisoes

| Data | Fase | Revisor | Status | Observacoes |
|------|------|---------|--------|-------------|
| — | — | — | — | — |
