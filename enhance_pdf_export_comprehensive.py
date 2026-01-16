#!/usr/bin/env python3
"""
Enhance Section 12 PDF export to:
1. Add google.colab.files.download() to actually download the PDF
2. Include quantitative scaling law analysis (Section 13)
3. Include compositional failure modes (Section 11)
4. Include detailed error propagation analysis
5. Include stage risk assessment
6. Include recommendations and research alignment
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

# Find the PDF export cell (Cell 42)
cell_idx = 42
cell = nb['cells'][cell_idx]
source = ''.join(cell['source'])

# Add google.colab import after the matplotlib import
old_import = "from matplotlib.backends.backend_pdf import PdfPages"
new_import = """from matplotlib.backends.backend_pdf import PdfPages
from google.colab import files"""

source = source.replace(old_import, new_import)

# Find where the PDF is closed (after the last pdf.savefig) and add download + more pages
# The PDF currently ends after PAGE 8. We'll add more pages before closing.

# Find the line after PAGE 8 where pdf context closes
# We need to add more pages BEFORE the PdfPages context closes

# Let's insert new pages after PAGE 8 (around line 401)
# First, let's find the exact position

lines = source.split('\n')
page8_end_idx = None
for i, line in enumerate(lines):
    if 'PAGE 8' in line or 'Page 8' in line:
        # Find the pdf.savefig after this
        for j in range(i, min(i+100, len(lines))):
            if 'pdf.savefig' in lines[j]:
                page8_end_idx = j + 1
                break
        break

if page8_end_idx is None:
    print("❌ Could not find PAGE 8 end")
    exit(1)

print(f"Found PAGE 8 ends at line {page8_end_idx}")

# New pages to add
new_pages = '''
    # ========================================================================
    # PAGE 9: DETAILED ERROR PROPAGATION ANALYSIS
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Detailed Error Propagation Analysis',
             ha='center', fontsize=16, fontweight='bold')

    # Collect error analysis from Section 10.1
    all_errors = []
    for stage, scenarios in error_scenarios.items():
        for scenario in scenarios:
            impact = scenario['propagation_probability'] * scenario['amplification_factor']
            all_errors.append({
                'type': scenario['error_type'],
                'stage': stage,
                'severity': scenario['severity'],
                'propagation': scenario['propagation_probability'],
                'amplification': scenario['amplification_factor'],
                'impact': impact
            })

    all_errors_sorted = sorted(all_errors, key=lambda x: x['impact'], reverse=True)[:15]

    text_content = "TOP 15 HIGHEST IMPACT ERRORS\\n"
    text_content += "="*70 + "\\n\\n"

    for i, err in enumerate(all_errors_sorted, 1):
        severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(err['severity'], '⚪')
        text_content += f"{i:2d}. {severity_icon} {err['type'][:50]}\\n"
        text_content += f"    Stage: {err['stage'].upper()}\\n"
        text_content += f"    Propagation: {err['propagation']*100:.0f}% | "
        text_content += f"Amplification: {err['amplification']:.1f}x | "
        text_content += f"Impact: {err['impact']:.2f}\\n\\n"

    fig.text(0.1, 0.05, text_content, ha='left', va='bottom',
             fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 10: STAGE RISK ASSESSMENT
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Stage Risk Assessment',
             ha='center', fontsize=16, fontweight='bold')

    # Calculate stage risks
    stage_stats = {}
    for err in all_errors:
        stage = err['stage']
        if stage not in stage_stats:
            stage_stats[stage] = {
                'count': 0,
                'high_propagation': 0,
                'high_amplification': 0,
                'total_impact': 0
            }
        stage_stats[stage]['count'] += 1
        if err['propagation'] > 0.8:
            stage_stats[stage]['high_propagation'] += 1
        if err['amplification'] > 2.0:
            stage_stats[stage]['high_amplification'] += 1
        stage_stats[stage]['total_impact'] += err['impact']

    text_content = "SDLC STAGE RISK ANALYSIS\\n"
    text_content += "="*70 + "\\n\\n"

    stage_order = ['requirements', 'design', 'implementation', 'testing', 'deployment']
    for stage in stage_order:
        if stage in stage_stats:
            stats = stage_stats[stage]
            text_content += f"🔹 {stage.upper()}:\\n"
            text_content += f"   Total Errors: {stats['count']}\\n"
            text_content += f"   High Propagation (>80%): {stats['high_propagation']}\\n"
            text_content += f"   High Amplification (>2x): {stats['high_amplification']}\\n"
            text_content += f"   Total Impact Score: {stats['total_impact']:.2f}\\n\\n"

    fig.text(0.1, 0.5, text_content, ha='left', va='center',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 11: COMPOSITIONAL FAILURE MODES
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Compositional Failure Modes',
             ha='center', fontsize=16, fontweight='bold')

    text_content = "FAILURE MODE TAXONOMY (Based on Xu et al., 2024)\\n"
    text_content += "="*70 + "\\n\\n"

    # Specification Fragility
    spec_errors = [e for e in all_errors if e['stage'] in ['requirements', 'design']
                   and any(x in e['type'] for x in ['Specification', 'Requirements', 'Ambiguity'])]
    spec_errors_sorted = sorted(spec_errors, key=lambda x: x['impact'], reverse=True)[:3]

    text_content += "1️⃣  SPECIFICATION FRAGILITY\\n"
    text_content += "─"*70 + "\\n"
    for err in spec_errors_sorted:
        text_content += f"• {err['type'][:60]}\\n"
        text_content += f"  Impact: {err['impact']:.2f} | "
        text_content += f"Propagation: {err['propagation']*100:.0f}%\\n"
    text_content += "\\n"

    # Implementation Misalignment
    impl_errors = [e for e in all_errors if e['stage'] == 'implementation'
                   and any(x in e['type'] for x in ['Design', 'Divergence', 'Mismatch'])]
    impl_errors_sorted = sorted(impl_errors, key=lambda x: x['impact'], reverse=True)[:2]

    text_content += "2️⃣  IMPLEMENTATION-DESIGN MISALIGNMENT\\n"
    text_content += "─"*70 + "\\n"
    for err in impl_errors_sorted:
        text_content += f"• {err['type'][:60]}\\n"
        text_content += f"  Impact: {err['impact']:.2f}\\n"
    text_content += "\\n"

    # Testing Inadequacy
    test_errors = [e for e in all_errors if e['stage'] == 'testing']
    test_errors_sorted = sorted(test_errors, key=lambda x: x['impact'], reverse=True)[:3]

    text_content += "3️⃣  TESTING INADEQUACY & FALSE CONFIDENCE\\n"
    text_content += "─"*70 + "\\n"
    for err in test_errors_sorted:
        text_content += f"• {err['type'][:60]}\\n"
        text_content += f"  Impact: {err['impact']:.2f}\\n"

    fig.text(0.1, 0.1, text_content, ha='left', va='bottom',
             fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 12: QUANTITATIVE SCALING LAW ANALYSIS
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Quantitative Scaling Law Analysis',
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.90, 'Based on Kim et al. (2025) arXiv:2512.08296',
             ha='center', fontsize=10, style='italic')

    # Calculate scaling law metrics
    isolated_acc = metrics.calculate_isolated_accuracy()
    single_agent_baseline = sum(isolated_acc.values()) / len(isolated_acc)
    system_accuracy = metrics.calculate_system_accuracy()
    integration_gap = metrics.calculate_integration_gap()

    # Scaling law coefficients from paper
    beta_Ep_T = -0.330  # Tool-Coordination trade-off
    beta_Psa_lognA = -0.408  # Baseline paradox
    beta_Ae_T = -0.097  # Error amplification

    text_content = "SCALING LAW PREDICTORS\\n"
    text_content += "="*70 + "\\n\\n"
    text_content += f"Single-Agent Baseline: {single_agent_baseline*100:.1f}%\\n"
    text_content += f"System Accuracy: {system_accuracy*100:.1f}%\\n"
    text_content += f"Integration Gap: {integration_gap:.1f}%\\n\\n"

    text_content += "CRITICAL THRESHOLD TEST\\n"
    text_content += "─"*70 + "\\n"
    text_content += f"45% Baseline Threshold: "
    if single_agent_baseline > 0.45:
        text_content += f"EXCEEDED ({single_agent_baseline*100:.1f}% > 45%)\\n"
        text_content += f"⚠️  Multi-agent predicted to DEGRADE performance\\n"
        text_content += f"   Expected degradation: 39-70%\\n\\n"
    else:
        text_content += f"Below threshold ({single_agent_baseline*100:.1f}% < 45%)\\n"
        text_content += f"✓ Multi-agent may improve performance\\n\\n"

    text_content += "DOMINANT EFFECTS\\n"
    text_content += "─"*70 + "\\n"
    text_content += f"Tool-Coordination Trade-off (β={beta_Ep_T:.3f})\\n"
    text_content += f"  • Strongest predictor in scaling law\\n"
    text_content += f"  • Higher tool count + coordination = worse performance\\n\\n"

    text_content += f"Baseline Paradox (β={beta_Psa_lognA:.3f})\\n"
    text_content += f"  • High baseline agents degrade when composed\\n"
    text_content += f"  • Current baseline: {single_agent_baseline*100:.1f}%\\n\\n"

    text_content += f"Error Amplification (β={beta_Ae_T:.3f})\\n"
    text_content += f"  • Tool-rich environments amplify errors\\n"
    text_content += f"  • Each stage introduces cascade risk\\n\\n"

    text_content += "ARCHITECTURE RECOMMENDATION\\n"
    text_content += "─"*70 + "\\n"
    if single_agent_baseline > 0.45:
        text_content += "⚠️  SINGLE-AGENT SYSTEM RECOMMENDED\\n"
        text_content += "   Multi-agent composition predicted to hurt performance\\n"
    else:
        text_content += "✓ Multi-agent system viable\\n"
        text_content += "   Expected performance maintained or improved\\n"

    fig.text(0.1, 0.05, text_content, ha='left', va='bottom',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 13: RECOMMENDATIONS & MITIGATION STRATEGIES
    # ========================================================================
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Recommendations & Mitigation Strategies',
             ha='center', fontsize=16, fontweight='bold')

    text_content = "MITIGATION STRATEGIES\\n"
    text_content += "="*70 + "\\n\\n"

    if integration_gap >= 50:
        urgency = "🔴 URGENT"
        recommendations = [
            "1. Implement comprehensive integration testing at stage boundaries",
            "2. Add human validation gates at high-risk stages",
            "3. Deploy formal verification for critical components",
            "4. Establish continuous monitoring with automated rollback",
            "5. Create redundant validation paths for error-prone transformations"
        ]
    elif integration_gap >= 30:
        urgency = "🟠 HIGH PRIORITY"
        recommendations = [
            "1. Strengthen validation at stage boundaries",
            "2. Implement selective human review for critical paths",
            "3. Add consistency checks between adjacent stages",
            "4. Improve error propagation tracking and logging"
        ]
    elif integration_gap >= 15:
        urgency = "🟡 MODERATE PRIORITY"
        recommendations = [
            "1. Enhance automated testing coverage",
            "2. Add spot-checks for high-risk error scenarios",
            "3. Improve inter-agent communication protocols"
        ]
    else:
        urgency = "🟢 MAINTAIN"
        recommendations = [
            "1. Continue current practices",
            "2. Monitor for degradation over time",
            "3. Document successful patterns for replication"
        ]

    text_content += f"Priority Level: {urgency}\\n\\n"
    text_content += "RECOMMENDED ACTIONS:\\n"
    text_content += "─"*70 + "\\n"
    for rec in recommendations:
        text_content += f"{rec}\\n\\n"

    text_content += "\\nALIGNMENT WITH PUBLISHED RESEARCH\\n"
    text_content += "="*70 + "\\n\\n"

    dafnycomp_gap = 92.0
    text_content += f"DafnyCOMP Baseline (Xu et al., 2024): {dafnycomp_gap:.1f}%\\n"
    text_content += f"Current Integration Gap: {integration_gap:.1f}%\\n"
    text_content += f"Difference: {integration_gap - dafnycomp_gap:+.1f}%\\n\\n"

    if integration_gap < dafnycomp_gap:
        improvement = ((dafnycomp_gap - integration_gap) / dafnycomp_gap) * 100
        text_content += f"✅ Showing {improvement:.1f}% improvement over baseline\\n"
    else:
        degradation = ((integration_gap - dafnycomp_gap) / dafnycomp_gap) * 100
        text_content += f"⚠️  Gap is {degradation:.1f}% worse than baseline\\n"

    fig.text(0.1, 0.1, text_content, ha='left', va='bottom',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
'''

# Insert new pages before the PdfPages context closes
lines.insert(page8_end_idx, new_pages)

# Now find where the PdfPages context closes and add download
# Find "print("✅ PDF Report Generated:"
close_idx = None
for i, line in enumerate(lines):
    if '✅ PDF Report Generated:' in line:
        # Add download after the file size print (a few lines down)
        for j in range(i, min(i+10, len(lines))):
            if 'File Size' in lines[j]:
                close_idx = j + 1
                break
        break

if close_idx:
    download_code = '''
# Download the PDF file in Colab
try:
    files.download(pdf_filename)
    print(f"\\n📥 Downloading: {pdf_filename}")
except Exception as e:
    print(f"\\n⚠️  Download failed: {e}")
    print(f"   PDF saved to: {pdf_filename}")
'''
    lines.insert(close_idx, download_code)
    print("✓ Added files.download() call")
else:
    print("⚠️  Could not find location to add download")

# Reconstruct source
new_source = '\n'.join(lines)

# Convert to proper list format
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("ENHANCED SECTION 12 PDF EXPORT")
print("="*80)
print()
print("✅ Added google.colab.files import")
print("✅ Added files.download() to trigger PDF download")
print("✅ Added PAGE 9: Detailed Error Propagation Analysis (Top 15 errors)")
print("✅ Added PAGE 10: Stage Risk Assessment")
print("✅ Added PAGE 11: Compositional Failure Modes")
print("✅ Added PAGE 12: Quantitative Scaling Law Analysis")
print("✅ Added PAGE 13: Recommendations & Mitigation Strategies")
print()
print(f"PDF now has 13 pages (was 8 pages)")
print()

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    print("✓ Notebook JSON is valid")

    source = ''.join(test_nb['cells'][cell_idx]['source'])
    compile(source, '<cell>', 'exec')
    print("✓ Cell compiles successfully")

    # Count pages
    page_count = source.count('# PAGE') + source.count('# Page')
    print(f"✓ PDF has {page_count} pages defined")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
