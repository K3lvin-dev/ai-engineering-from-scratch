# Translation Review Log

| Date | Phase | Batch | Critical | High | Medium | Low | Status |
|------|-------|-------|----------|------|--------|-----|--------|
| 2025-01 | Fase 00 | Setup & Tooling (7 aulas) | 3 | 4 | 7 | 3 | ✅ Fixed |
| 2025-01 | Fase 01 | Math Foundations (12 aulas) | 7 | 5 | 8 | 3 | 🔧 Fixed |
| 2025-01 | Fase 03 | Deep Learning Core (5 aulas) | 0 | 5 | 2 | 2 | 🔧 Fixed |
| 2025-01 | Fase 05 | NLP Foundations (29 aulas) | 0 | 1 | 4 | 3 | 🟡 Open |
| 2025-01 | Fase 06 | Speech & Audio (17 aulas) | 0 | 2 | 3 | 12 | 🔧 Fixed |
| 2025-01 | Fase 07-09 | Transformers/GenAI/RL | — | — | — | — | 🔄 In review |
| 2025-01 | Fase 10-12 | LLMs/Multimodal | — | — | — | — | 🔄 In review |
| 2025-01 | Fase 13-19 | Tools/Agents/Infra/Ethics/Capstone | — | — | — | — | 🔄 In review |

## Summary of Systematic Fixes Applied

| Issue Type | Scope | Files Affected |
|------------|-------|---------------|
| `{{especificação}}` → correct words (especificação, específico, especiais, espectrograma, etc.) | All phases | 255+ |
| `Gradientees` → `Gradientes` | Fase 03 | 2 |
| Acentos ausentes em content-translations.js | Site | 1 |
| Acentos ausentes em header.js | Site | 1 |
| `O Problemo` → `O Problema` | Fase 01 (9 lessons) | 9 |
| `AGORA` → `AND` (logic gate) | Fase 03 | 1 |
| `Eespecificaçãotrograma` → `Espectrograma` | Fase 06, 12 | 40+ |
| `retroespecificaçãotiva` → `retrospectiva` | Fases 09, 13, 15 | 3 |
| `Autonomo` → `Autônomo` | Site + Fase 15 | Several |
| Seções omitidas adicionadas (Key Terms, pyproject.toml, Processes, Focal Loss, Triplet Loss, Loss Landscape) | Fase 00, 03 | 5 lessons |

## Remaining Work

| Category | Count | Action Needed |
|----------|-------|---------------|
| Incomplete translations (< 40%) | 60 | Complete translation |
| Partial translations (40-70%) | 25 | Complete missing sections |
| Placeholder not started | 11 | Translate from scratch |
| Code block mismatches | 102 | Sync code blocks between EN and PT |
