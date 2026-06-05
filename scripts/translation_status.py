#!/usr/bin/env python3
"""Generate a status report of all PT-BR translations."""

import glob
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

report = {
    "summary": {"placeholder": 0, "partial_lt_40": 0, "partial_40_70": 0, "complete_gt_70": 0, "missing": 0},
    "lessons": []
}

en_files = sorted(glob.glob(os.path.join(ROOT, 'phases', '*', '*', 'docs', 'en.md')))
report['summary']['total_en_files'] = len(en_files)

for en_path in en_files:
    pt_path = en_path.replace('/en.md', '/pt-br.md')
    rel_en = os.path.relpath(en_path, ROOT)
    
    lesson = {
        'en_path': rel_en,
        'pt_exists': os.path.exists(pt_path),
        'status': 'missing',
        'en_size': os.path.getsize(en_path),
        'pt_size': 0,
        'ratio': 0,
    }
    
    if os.path.exists(pt_path):
        with open(pt_path, 'r') as f:
            first_chars = f.read()[:200].lower()
        
        lesson['pt_size'] = os.path.getsize(pt_path)
        lesson['ratio'] = lesson['pt_size'] / lesson['en_size'] if lesson['en_size'] > 0 else 0
        
        if 'tradução em andamento' in first_chars or 'traducao em andamento' in first_chars:
            lesson['status'] = 'placeholder'
        elif lesson['ratio'] < 0.40:
            lesson['status'] = 'partial_lt_40'
        elif lesson['ratio'] < 0.70:
            lesson['status'] = 'partial_40_70'
        else:
            lesson['status'] = 'complete_gt_70'
    
    report['lessons'].append(lesson)
    s = lesson['status']
    if s not in report['summary']:
        report['summary'][s] = 0
    report['summary'][s] += 1

# Sort by status for clarity
by_status = {'placeholder': [], 'partial_lt_40': [], 'partial_40_70': [], 'complete_gt_70': [], 'missing': []}
for l in report['lessons']:
    by_status[l['status']].append(l)

print("=" * 70)
print("PT-BR TRANSLATION STATUS REPORT")
print("=" * 70)
print(f"\nTotal EN files: {report['summary']['total_en_files']}")
total_pt = report['summary']['placeholder'] + report['summary']['partial_lt_40'] + report['summary']['partial_40_70'] + report['summary']['complete_gt_70']
print(f"Total PT-BR files: {total_pt}")
print(f"  Translation not started (placeholder): {report['summary']['placeholder']}")
print(f"  Partial < 40%:  {report['summary']['partial_lt_40']}")
print(f"  Partial 40-70%: {report['summary']['partial_40_70']}")
print(f"  Complete > 70%: {report['summary']['complete_gt_70']}")
print(f"  Missing entirely: {report['summary']['missing']}")

print("\n--- PLACEHOLDER (not started) ---")
for l in by_status['placeholder']:
    print(f"  {l['en_path']}")

print(f"\n--- PARTIAL < 40% (n={len(by_status['partial_lt_40'])}) ---")
for l in by_status['partial_lt_40'][:10]:
    phase = l['en_path'].split('/')[1]
    print(f"  [{phase}] {l['en_path']} ({l['pt_size']}B / {l['en_size']}B EN = {l['ratio']*100:.0f}%)")
if len(by_status['partial_lt_40']) > 10:
    print(f"  ... and {len(by_status['partial_lt_40'])-10} more")

print(f"\n--- PARTIAL 40-70% (n={len(by_status['partial_40_70'])}) ---")
for l in by_status['partial_40_70'][:10]:
    phase = l['en_path'].split('/')[1]
    print(f"  [{phase}] {l['en_path']} ({l['ratio']*100:.0f}%)")
if len(by_status['partial_40_70']) > 10:
    print(f"  ... and {len(by_status['partial_40_70'])-10} more")

print(f"\n--- COMPLETE > 70% (n={len(by_status['complete_gt_70'])}) ---")
print("  (all complete, ready for quality review)")

# Save as JSON for later use
out_path = os.path.join(ROOT, 'TRANSLATION_STATUS.json')
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nFull report saved to TRANSLATION_STATUS.json")
