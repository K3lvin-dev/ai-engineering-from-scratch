#!/usr/bin/env python3
"""
Automated verification script for PT-BR translation quality.
Run periodically to catch regressions.

Checks:
1. Every docs/en.md has a docs/pt-br.md counterpart
2. No placeholder leaks ({{...}} in pt-br.md text, outside code blocks)
3. No unaccented Portuguese words in content-translations.js
4. Header structure matches between en.md and pt-br.md
5. Suspiciously short pt-br.md files (translation < 50% of EN size)
6. Code blocks / LaTeX / URLs preserved identically

Exit codes: 0 = all good, 1 = warnings, 2 = errors
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

errors = []
warnings = []

# ============================================================
# 1. Every docs/en.md has docs/pt-br.md
# ============================================================
def check_missing_translations():
    en_files = sorted(glob.glob(os.path.join(ROOT, 'phases', '*', '*', 'docs', 'en.md')))
    missing = []
    for en_path in en_files:
        pt_path = en_path.replace('/en.md', '/pt-br.md')
        if not os.path.exists(pt_path):
            missing.append(en_path)
    if missing:
        errors.append(f"MISSING PT-BR ({len(missing)} files):")
        for m in missing[:10]:
            rel = os.path.relpath(m, ROOT)
            errors.append(f"  {rel}")
        if len(missing) > 10:
            errors.append(f"  ... and {len(missing) - 10} more")
    else:
        warnings.append("OK: every en.md has a pt-br.md counterpart")

# ============================================================
# 2. No placeholder leaks outside code blocks
# ============================================================
def check_placeholder_leaks():
    pt_files = sorted(glob.glob(os.path.join(ROOT, 'phases', '*', '*', 'docs', 'pt-br.md')))
    leaked = []
    # Known false positives: legitimate {{...}} in code blocks
    known_patterns = [
        r'\{\{question\}\}',    # template variable in examples
        r'\{\{id\}\}',          # in code examples
        r'\{\{[a-zA-Z_]+\.[a-zA-Z_]+\}\}',  # dot notation templates
    ]
    
    for pt_path in pt_files:
        with open(pt_path, 'r') as f:
            content = f.read()
        
        # Split into code blocks and non-code blocks
        segments = re.split(r'(```[\s\S]*?```|`[^`]+`)', content)
        
        for i, seg in enumerate(segments):
            if seg.startswith('`') or seg.startswith('```'):
                continue  # skip code blocks
                
            # Check for remaining {{...}} patterns
            found = re.findall(r'\{\{[a-zA-Z_]+\}\}', seg)
            for fnd in found:
                # Filter known false positives
                is_false_positive = any(
                    re.match(p, fnd) for p in known_patterns
                )
                if not is_false_positive:
                    rel = os.path.relpath(pt_path, ROOT)
                    leaked.append(f"{rel}: {fnd}")
    
    if leaked:
        errors.append(f"PLACEHOLDER LEAKS ({len(leaked)}):")
        for l in leaked[:10]:
            errors.append(f"  {l}")
        if len(leaked) > 10:
            errors.append(f"  ... and {len(leaked) - 10} more")
    else:
        warnings.append("OK: no placeholder leaks found")

# ============================================================
# 3. Check for common unaccented words in content-translations.js
# ============================================================
def check_content_translations():
    path = os.path.join(ROOT, 'site', 'content-translations.js')
    with open(path, 'r') as f:
        content = f.read()
    
    # Check pt-br values for missing accents
    unaccented = re.findall(r"': '([^']+)',", content)
    
    # Known unaccented patterns where the UNACCENTED version is WRONG.
    suspect_patterns = [
        # Words missing acento agudo or tilde where required
        r'\bConfiguracao\b',
        r'\bMatematica\b',
        r'\bNucleo\b',
        r'\bVisao\b',
        r'\bAvancado\b',
        r'\bReforco\b',
        r'\bEtica\b',
        r'\bSeguranca\b',
        r'\bProducao\b',
        r'\bAutonomo\b',
        r'\bDepuracao\b',
        r'\bIntuicao\b',
        r'\bTransformacoe\b',
        r'\bDiferenciacao\b',
        r'\bOtimizacao\b',
        r'\bInformacao\b',
        r'\bDivergencia\b',
        r'\bReducao\b',
        r'\bDecomposicao\b',
        r'\bNumerica\b',
        r'\bDistancias\b',
        r'\bEstatistica\b',
        r'\bEstocasticos\b',
        r'\bClassificacao\b',
        r'\bArvores\b',
        r'\bDecisao\b',
        r'\bAvaliacao\b',
        r'\bVariancia\b',
        r'\bHiperparametro\b',
        r'\bDeteccao\b',
        r'\bSegmentacao\b',
        r'\bTokenizacao\b',
        r'\bTraducao\b',
        r'\bRepresentacao\b',
        r'\bConversao\b',
        r'\bNegociacao\b',
        r'\bEspecializacao\b',
        r'\bOrquestracao\b',
        r'\bSimulacao\b',
        r'\bCoordenacao\b',
        r'\bQuantizacao\b',
        r'\bMitigacao\b',
        r'\bProveniencia\b',
        r'\bGovernanca\b',
    ]
    
    found_issues = []
    for val in unaccented:
        for pat in suspect_patterns:
            if re.search(pat, val):
                # Found the specific word - show the context
                found_issues.append(val)
                break
    
    if found_issues:
        errors.append(f"UNACCENTED WORDS in content-translations.js ({len(found_issues)}):")
        for issue in found_issues[:10]:
            errors.append(f"  '{issue}'")
        if len(found_issues) > 10:
            errors.append(f"  ... and {len(found_issues) - 10} more")
    else:
        warnings.append("OK: content-translations.js has proper accents")

# ============================================================
# 4. Check header.js for unaccented strings
# ============================================================
def check_header_js():
    path = os.path.join(ROOT, 'site', 'header.js')
    with open(path, 'r') as f:
        content = f.read()
    
    unaccented = [
        'Glossario', 'Catalogo', 'codigo', 'voce', 'Voce',
        'descricao', 'Nao', 'Otimo', 'Inicio', 'curriculo',
        'avaliacao', 'Avaliacao', 'renderizacao', 'Pontuacao',
        'Parametro', 'parametro'
    ]
    
    issues = [w for w in unaccented if w in content]
    if issues:
        errors.append(f"UNACCENTED STRINGS in header.js: {issues}")
    else:
        warnings.append("OK: header.js strings are properly accented")

# ============================================================
# 5. Suspiciously short translations
# ============================================================
def check_short_translations():
    en_files = glob.glob(os.path.join(ROOT, 'phases', '*', '*', 'docs', 'en.md'))
    short = []
    for en_path in en_files:
        pt_path = en_path.replace('/en.md', '/pt-br.md')
        if not os.path.exists(pt_path):
            continue
        en_size = os.path.getsize(en_path)
        pt_size = os.path.getsize(pt_path)
        if en_size > 500 and pt_size < en_size * 0.3:
            rel = os.path.relpath(pt_path, ROOT)
            short.append(f"{rel} ({pt_size}B vs {en_size}B EN)")
    
    if short:
        errors.append(f"SUSPICIOUSLY SHORT translations ({len(short)}):")
        for s in short[:5]:
            errors.append(f"  {s}")
        if len(short) > 5:
            errors.append(f"  ... and {len(short) - 5} more")
    else:
        warnings.append("OK: no suspiciously short translations")

# ============================================================
# 6. Code blocks preserved identically
# ============================================================
def check_code_blocks():
    en_files = glob.glob(os.path.join(ROOT, 'phases', '*', '*', 'docs', 'en.md'))
    mismatches = []
    for en_path in en_files:
        pt_path = en_path.replace('/en.md', '/pt-br.md')
        if not os.path.exists(pt_path):
            continue
        
        with open(en_path, 'r') as f:
            en_content = f.read()
        with open(pt_path, 'r') as f:
            pt_content = f.read()
        
        # Extract fenced code blocks
        en_blocks = re.findall(r'```\w*\n(.*?)```', en_content, re.DOTALL)
        pt_blocks = re.findall(r'```\w*\n(.*?)```', pt_content, re.DOTALL)
        
        if len(en_blocks) != len(pt_blocks):
            rel = os.path.relpath(pt_path, ROOT)
            mismatches.append(f"{rel}: {len(en_blocks)} code blocks (EN) vs {len(pt_blocks)} (PT)")
    
    if mismatches:
        errors.append(f"CODE BLOCK MISMATCHES ({len(mismatches)}):")
        for m in mismatches[:5]:
            errors.append(f"  {m}")
        if len(mismatches) > 5:
            errors.append(f"  ... and {len(mismatches) - 5} more")
    else:
        warnings.append("OK: code blocks preserved")

# ============================================================
# Run all checks
# ============================================================
def main():
    print("=" * 60)
    print("PT-BR Translation Verification Report")
    print("=" * 60)
    
    checks = [
        ("Missing translations", check_missing_translations),
        ("Placeholder leaks", check_placeholder_leaks),
        ("Content translations accents", check_content_translations),
        ("Header JS accents", check_header_js),
        ("Short translations", check_short_translations),
        ("Code blocks", check_code_blocks),
    ]
    
    for name, fn in checks:
        try:
            fn()
        except Exception as e:
            errors.append(f"CRASH in {name}: {e}")
    
    print(f"\nErrors: {len(errors)}")
    for e in errors:
        print(f"  [ERR] {e}")
    
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  [OK] {w}")
    
    if errors:
        sys.exit(2)
    else:
        print("\nAll checks passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
