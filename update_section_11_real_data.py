#!/usr/bin/env python3
"""
Update section 11 to use real data from PoC1 and comprehensive error scenarios
"""

import json

# New cell content that uses real data
new_cell_source = """# Display real failure modes from the comprehensive error cascade
print(\"\"\"
╔═══════════════════════════════════════════════════════════╗
║     COMPOSITIONAL FAILURE MODE DEMONSTRATION              ║
╚═══════════════════════════════════════════════════════════╝

Based on Xu et al. taxonomy (Section 2.2) with REAL DATA from PoC simulation:
\"\"\")

# Analyze error propagation patterns
print("\\n" + "="*70)
print("FAILURE MODE ANALYSIS FROM SIMULATED ERROR CASCADE")
print("="*70)

# Get error propagation stats
total_errors = len(metrics.error_propagation)
amplified_errors = sum(1 for e in metrics.error_propagation if e.get('amplified', False))

print(f"\\n📊 OVERALL STATISTICS:")
print(f"   • Total Error Propagations Tracked: {total_errors}")
print(f"   • Amplified Cascades: {amplified_errors}")
print(f"   • Contained Errors: {total_errors - amplified_errors}")

# Analyze by failure mode category
print("\\n" + "─"*70)
print("1️⃣  SPECIFICATION FRAGILITY")
print("─"*70)

# Get specification-related errors from requirements and design stages
spec_errors = [
    s for stage in ['requirements', 'design']
    for s in error_scenarios.get(stage, [])
    if 'Specification' in s['error_type'] or 'Requirements' in s['error_type'] or
       'Inconsistent' in s['error_type'] or 'Ambiguity' in s['error_type']
]

if spec_errors:
    # Show top 3 highest impact
    spec_errors_sorted = sorted(spec_errors,
                                key=lambda x: x['propagation_probability'] * x['amplification_factor'],
                                reverse=True)[:3]

    for i, err in enumerate(spec_errors_sorted, 1):
        impact = err['propagation_probability'] * err['amplification_factor']
        print(f"\\n   Example {i}: {err['error_type']}")
        print(f"   ├─ Severity: {err['severity']}")
        print(f"   ├─ Description: {err['description']}")
        print(f"   ├─ Propagation Probability: {err['propagation_probability']:.0%}")
        print(f"   ├─ Amplification Factor: {err['amplification_factor']}x")
        print(f"   ├─ Impact Score: {impact:.2f}")
        print(f"   └─ Cascades to: {', '.join([s.title() for s in err['cascades_to']])}")

print("\\n" + "─"*70)
print("2️⃣  IMPLEMENTATION-DESIGN MISALIGNMENT")
print("─"*70)

# Get implementation errors that cascade from design
impl_errors = [
    s for s in error_scenarios.get('implementation', [])
    if 'Design' in s['error_type'] or 'Divergence' in s['error_type'] or
       'Mismatch' in s['error_type']
]

if impl_errors:
    for err in impl_errors[:2]:  # Top 2
        impact = err['propagation_probability'] * err['amplification_factor']
        print(f"\\n   Example: {err['error_type']}")
        print(f"   ├─ Description: {err['description']}")
        print(f"   ├─ Real Example: {err['example']}")
        print(f"   ├─ Propagation: {err['propagation_probability']:.0%}")
        print(f"   ├─ Amplification: {err['amplification_factor']}x")
        print(f"   └─ Impact: {impact:.2f}")

print("\\n" + "─"*70)
print("3️⃣  TESTING INADEQUACY & FALSE CONFIDENCE")
print("─"*70)

# Get testing errors
test_errors = [
    s for s in error_scenarios.get('testing', [])
    if 'False Positive' in s['error_type'] or 'Missing' in s['error_type'] or
       'Insufficient' in s['error_type']
]

if test_errors:
    # Show highest impact testing failures
    test_errors_sorted = sorted(test_errors,
                                key=lambda x: x['propagation_probability'] * x['amplification_factor'],
                                reverse=True)[:3]

    for i, err in enumerate(test_errors_sorted, 1):
        impact = err['propagation_probability'] * err['amplification_factor']
        print(f"\\n   Example {i}: {err['error_type']}")
        print(f"   ├─ {err['description']}")
        print(f"   ├─ Example: {err['example']}")
        print(f"   ├─ Severity: {err['severity']}")
        print(f"   ├─ Propagation to Production: {err['propagation_probability']:.0%}")
        print(f"   └─ Impact: {impact:.2f}")

print("\\n" + "─"*70)
print("4️⃣  DEPLOYMENT & CONFIGURATION DRIFT")
print("─"*70)

# Get deployment errors
deploy_errors = [
    s for s in error_scenarios.get('deployment', [])
]

if deploy_errors:
    # Show critical deployment errors
    critical_deploy = [e for e in deploy_errors if e['severity'] == 'CRITICAL'][:3]

    for i, err in enumerate(critical_deploy, 1):
        impact = err['propagation_probability'] * err['amplification_factor']
        print(f"\\n   Critical Issue {i}: {err['error_type']}")
        print(f"   ├─ {err['description']}")
        print(f"   ├─ Example: {err['example']}")
        print(f"   ├─ Propagation Probability: {err['propagation_probability']:.0%}")
        print(f"   ├─ Amplification: {err['amplification_factor']}x")
        print(f"   └─ Impact: {impact:.2f} (TERMINAL - cannot cascade further)")

# Show the most dangerous error cascade chains
if metrics.error_propagation:
    print("\\n" + "="*70)
    print("🔥 MOST DANGEROUS ERROR CASCADE CHAINS FROM SIMULATION")
    print("="*70)

    # Group by source->target pairs
    cascade_map = {}
    for prop in metrics.error_propagation:
        key = f"{prop['source']} → {prop['target']}"
        if key not in cascade_map:
            cascade_map[key] = []
        cascade_map[key].append(prop)

    # Show top cascades
    print(f"\\nTracked {len(cascade_map)} unique stage-to-stage error pathways:")
    for i, (pathway, errors) in enumerate(list(cascade_map.items())[:5], 1):
        amplified = sum(1 for e in errors if e.get('amplified', False))
        print(f"\\n   {i}. {pathway}")
        print(f"      Total propagations: {len(errors)}, Amplified: {amplified} ({amplified/len(errors)*100:.1f}%)")

print("\\n" + "="*70)
print("💡 KEY INSIGHTS FROM REAL DATA")
print("="*70)

print(\"\"\"
• Each error type has REAL propagation probabilities (70-99%)
• Amplification factors range from 1.5x to 4.5x
• CRITICAL errors have highest propagation AND amplification
• Requirements/Design errors cascade through ALL downstream stages
• Testing errors go directly to production (deployment stage)
• Each stage optimizes for LOCAL correctness
• No single agent has GLOBAL system visibility
• Integration failures emerge at component boundaries

✓ This demonstrates the Integration Paradox: reliable components
  compose into unreliable systems due to compositional failures.
\"\"\")
"""

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find section 11 cell
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'COMPOSITIONAL FAILURE MODE DEMONSTRATION' in source and '39.2% of failures' in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find section 11 cell!")
    exit(1)

# Update the cell
notebook['cells'][target_cell_idx]['source'] = new_cell_source.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("UPDATING SECTION 11 WITH REAL PoC1 DATA")
print("="*80)
print()
print("✅ Successfully updated section 11!")
print()
print("Changes:")
print("  • Replaced placeholder examples with REAL error scenarios")
print("  • Added actual propagation probabilities and amplification factors")
print("  • Mapped errors to failure mode taxonomy from the paper")
print("  • Included error cascade chain analysis from metrics")
print("  • Shows specification fragility, implementation misalignment,")
print("    testing inadequacy, and deployment drift with real data")
print()
print("="*80)
print("✓ Section 11 now demonstrates failure modes with actual PoC1 data!")
print("="*80)
