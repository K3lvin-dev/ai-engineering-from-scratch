#!/usr/bin/env python3
"""Fix corrupted URLs and text in pt-br.md files."""
import glob

corrections = {
    # URL corruptions in URLs
    "https://modelcontextprotocol.io/especificaçãoification/": "https://modelcontextprotocol.io/specification/",
    "https://json-schema.org/especificaçãoification.html": "https://json-schema.org/specification.html",
    "https://a2a-protocol.org/latest/especificaçãoification/": "https://a2a-protocol.org/latest/specification/",
    "https://c2pa.org/especificaçãoifications/especificaçãoifications/": "https://c2pa.org/specifications/specifications/",
    "https://opentelemetry.io/docs/especificaçãos/": "https://opentelemetry.io/docs/specs/",
    "https://www.statsig.com/perespecificaçãotives/": "https://www.statsig.com/perspectives/",
    # Semantic corruptions
    "especificaçãos": "especificações",
    "Eespecificação": "Especificação",
    "Perespecificaçãotive": "Perspective",
    "proespecificaçãot": "prospect",
    "introespecificaçãoção": "introspection",
    "Tipo: Learn": "Tipo: Aprender",
    "Objetivos de Aprendizagem": "Objetivos de Aprendizado",
    "Fase 18 · 01 (InstructGPT)": "Fase 10 · 06 (SFT)",
    "llm-conformidade-guide-iso-42001": "llm-compliance-guide-iso-42001",
    "ai-conformidade/": "ai-compliance/",
}

count = 0
for fname in glob.glob("phases/**/docs/pt-br.md", recursive=True):
    with open(fname, "r") as f:
        content = f.read()
    orig = content
    for wrong, correct in corrections.items():
        content = content.replace(wrong, correct)
    if content != orig:
        with open(fname, "w") as f:
            f.write(content)
        count += 1

print(f"Fixed {count} files")
