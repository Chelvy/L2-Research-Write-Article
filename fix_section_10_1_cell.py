#!/usr/bin/env python3
"""
Fix Section 10.1 (Cell 28) - Detailed Error Propagation Analysis
Cell 28 has been corrupted with orphaned code fragments.
Replace with proper error propagation analysis.
"""

import json

# Read notebook
with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

# Create the correct Section 10.1 analysis cell
correct_cell_source = '''# Analyze error propagation in detail
print("\\n" + "="*80)
print("DETAILED ERROR PROPAGATION ANALYSIS")
print("="*80 + "\\n")

# Collect all error scenarios across stages
all_errors = []
for stage, scenarios in error_scenarios.items():
    for scenario in scenarios:
        error_info = {
            'stage': stage,
            'error_type': scenario['error_type'],
            'severity': scenario['severity'],
            'propagation_probability': scenario['propagation_probability'],
            'amplification_factor': scenario['amplification_factor'],
            'impact': scenario['propagation_probability'] * scenario['amplification_factor'],
            'cascades_to': scenario.get('cascades_to', [])
        }
        all_errors.append(error_info)

# Sort by impact (propagation_probability × amplification_factor)
all_errors_sorted = sorted(all_errors, key=lambda x: x['impact'], reverse=True)

# Display TOP 10 ERROR TYPES BY TOTAL IMPACT
print("📊 TOP 10 ERROR TYPES BY TOTAL IMPACT")
print("   (Impact = Propagation Probability × Amplification Factor)\\n")

for i, err in enumerate(all_errors_sorted[:10], 1):
    severity_icon = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }.get(err['severity'], '⚪')

    print(f"{i:2d}. {severity_icon} {err['error_type']}")
    print(f"    Stage: {err['stage'].upper()}")
    print(f"    Severity: {err['severity']}")
    print(f"    Propagation Probability: {err['propagation_probability']*100:.0f}%")
    print(f"    Amplification Factor: {err['amplification_factor']:.1f}x")
    print(f"    ⚡ IMPACT SCORE: {err['impact']:.2f}")
    if err['cascades_to']:
        print(f"    Cascades to: {', '.join(err['cascades_to'])}")
    print()

# Analyze propagation patterns
print("\\n" + "="*80)
print("ERROR PROPAGATION PATTERNS")
print("="*80 + "\\n")

# Group by severity
severity_stats = {}
for err in all_errors:
    sev = err['severity']
    if sev not in severity_stats:
        severity_stats[sev] = {
            'count': 0,
            'avg_propagation': 0,
            'avg_amplification': 0,
            'total_impact': 0
        }
    severity_stats[sev]['count'] += 1
    severity_stats[sev]['avg_propagation'] += err['propagation_probability']
    severity_stats[sev]['avg_amplification'] += err['amplification_factor']
    severity_stats[sev]['total_impact'] += err['impact']

# Calculate averages
for sev, stats in severity_stats.items():
    count = stats['count']
    stats['avg_propagation'] /= count
    stats['avg_amplification'] /= count

# Display severity analysis
print("📈 ANALYSIS BY SEVERITY LEVEL:\\n")
for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    if sev in severity_stats:
        stats = severity_stats[sev]
        print(f"{sev}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Propagation Probability: {stats['avg_propagation']*100:.1f}%")
        print(f"  Avg Amplification Factor: {stats['avg_amplification']:.2f}x")
        print(f"  Total Impact Score: {stats['total_impact']:.2f}")
        print()

# Analyze by stage
print("\\n" + "="*80)
print("ERROR PROPAGATION BY SDLC STAGE")
print("="*80 + "\\n")

stage_stats = {}
for err in all_errors:
    stage = err['stage']
    if stage not in stage_stats:
        stage_stats[stage] = {
            'count': 0,
            'high_propagation_count': 0,
            'high_amplification_count': 0,
            'total_impact': 0
        }
    stage_stats[stage]['count'] += 1
    if err['propagation_probability'] > 0.8:
        stage_stats[stage]['high_propagation_count'] += 1
    if err['amplification_factor'] > 2.0:
        stage_stats[stage]['high_amplification_count'] += 1
    stage_stats[stage]['total_impact'] += err['impact']

# Display stage analysis
stage_order = ['requirements', 'design', 'implementation', 'testing', 'deployment']
for stage in stage_order:
    if stage in stage_stats:
        stats = stage_stats[stage]
        print(f"🔹 {stage.upper()}:")
        print(f"   Total Errors: {stats['count']}")
        print(f"   High Propagation (>80%): {stats['high_propagation_count']}")
        print(f"   High Amplification (>2x): {stats['high_amplification_count']}")
        print(f"   Total Impact Score: {stats['total_impact']:.2f}")
        print()

# Key insights
print("\\n" + "="*80)
print("🎯 KEY INSIGHTS")
print("="*80 + "\\n")

total_errors = len(all_errors)
high_impact = sum(1 for e in all_errors if e['impact'] > 2.0)
cascade_errors = sum(1 for e in all_errors if len(e['cascades_to']) > 0)
multi_cascade = sum(1 for e in all_errors if len(e['cascades_to']) >= 3)

print(f"• Total Error Types Analyzed: {total_errors}")
print(f"• High Impact Errors (score > 2.0): {high_impact} ({high_impact/total_errors*100:.1f}%)")
print(f"• Errors That Cascade: {cascade_errors} ({cascade_errors/total_errors*100:.1f}%)")
print(f"• Multi-Stage Cascades (≥3 stages): {multi_cascade} ({multi_cascade/total_errors*100:.1f}%)")
print()
print(f"• Most Dangerous Error: {all_errors_sorted[0]['error_type']}")
print(f"  Impact Score: {all_errors_sorted[0]['impact']:.2f}")
print(f"  Stage: {all_errors_sorted[0]['stage'].upper()}")
'''

# Split into lines for proper notebook format
new_source = [line + '\n' for line in correct_cell_source.split('\n')]

# Replace cell 28
nb['cells'][28]['source'] = new_source

# Save notebook
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Fixed Section 10.1 (Cell 28) - Detailed Error Propagation Analysis")

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    # Try to compile the cell
    test_source = ''.join(test_nb['cells'][28]['source'])
    compile(test_source, '<cell28>', 'exec')
    print("✓ Cell 28 compiles successfully")
    print("✓ Notebook JSON is valid")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
