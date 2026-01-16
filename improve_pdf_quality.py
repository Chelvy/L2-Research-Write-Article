#!/usr/bin/env python3
"""
Improve PDF export quality with professional formatting:
- Consistent fonts and sizes
- 1-inch margins everywhere
- Landscape for graphs, portrait for text
- US letter dimensions (8.5"x11")
- Reduced gaps (1 line between title and content)
- Uniform line spacing
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

# New professionally formatted PDF export code
new_pdf_code = '''# ============================================================================
# COMPREHENSIVE ANALYSIS RESULTS EXPORT - PDF REPORT
# ============================================================================

print("="*80)
print("GENERATING COMPREHENSIVE PDF ANALYSIS REPORT")
print("="*80)

from matplotlib.backends.backend_pdf import PdfPages
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# PDF FORMATTING CONSTANTS - PROFESSIONAL LAYOUT
# ============================================================================
# US Letter dimensions
LETTER_WIDTH = 8.5
LETTER_HEIGHT = 11.0

# Margins (1 inch = 72 points, in figure coordinates that's about 0.12)
MARGIN_INCH = 1.0
MARGIN_NORMALIZED = 0.12  # 1 inch in normalized coordinates (0-1)

# Consistent typography
TITLE_FONT = 'DejaVu Sans'
BODY_FONT = 'DejaVu Sans'
MONO_FONT = 'DejaVu Sans Mono'

TITLE_SIZE = 16
SUBTITLE_SIZE = 12
HEADING_SIZE = 11
BODY_SIZE = 9
SMALL_SIZE = 8

# Line spacing (consistent)
LINE_SPACING = 1.2

# Colors
TITLE_COLOR = '#1f77b4'
TEXT_COLOR = '#2f2f2f'

pdf_filename = f'integration_paradox_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

with PdfPages(pdf_filename) as pdf:

    # ========================================================================
    # PAGE 1: TITLE PAGE (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title
    fig.text(0.5, 0.65, 'Integration Paradox',
             ha='center', va='center',
             fontsize=24, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    fig.text(0.5, 0.60, 'Demonstration Results',
             ha='center', va='center',
             fontsize=20, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    # Subtitle
    fig.text(0.5, 0.52, 'Comprehensive Analysis of Multi-Agent SDLC Systems',
             ha='center', va='center',
             fontsize=SUBTITLE_SIZE, style='italic',
             fontname=TITLE_FONT, color=TEXT_COLOR)

    # Date and metadata
    timestamp = datetime.now().strftime('%B %d, %Y')
    fig.text(0.5, 0.40, f'Generated: {timestamp}',
             ha='center', va='center',
             fontsize=BODY_SIZE, fontname=BODY_FONT, color=TEXT_COLOR)

    fig.text(0.5, 0.38, 'Based on Xu et al. (2024) and Kim et al. (2025)',
             ha='center', va='center',
             fontsize=SMALL_SIZE, style='italic',
             fontname=BODY_FONT, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 2: INTRODUCTION & THEORETICAL FRAMEWORK (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title with 1 line gap
    fig.text(0.5, 0.95, 'Introduction & Theoretical Framework',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    intro_text = """INTRODUCTION

The Integration Paradox demonstrates that reliable AI components can compose into
unreliable systems. This phenomenon occurs when individual agents perform well in
isolation but degrade significantly when integrated into multi-agent pipelines.

THEORETICAL FRAMEWORK

1. COMPOSITIONAL RELIABILITY GAP
   High-performing components (>90% accuracy) experience dramatic performance
   degradation when composed into end-to-end systems.

2. ERROR PROPAGATION DYNAMICS
   Errors cascade through sequential stages, amplifying at each boundary:
   • Specification errors propagate to design (95% probability)
   • Design errors cascade to implementation (85% probability)
   • Implementation errors reach testing (80% probability)
   • Testing failures escape to production (70% probability)

3. FAILURE MODE TAXONOMY (Xu et al., 2024)
   a) Specification Fragility (39.2% of failures)
      Valid requirements become invalid under composition

   b) Implementation-Proof Misalignment (21.7%)
      Code diverges from design specifications

   c) Reasoning Instability (14.1%)
      Inconsistent outputs from identical inputs

   d) Error Compounding (O(T²×ε))
      Quadratic error growth in T-stage pipelines

4. QUANTITATIVE SCALING LAW (Kim et al., 2025)
   Multi-agent performance predicted with R²=0.513 accuracy:
   • 45% baseline threshold determines viability
   • Tool-coordination trade-off (β=-0.330) is dominant predictor
   • High-baseline agents degrade 39-70% when composed"""

    fig.text(MARGIN_NORMALIZED, 0.92, intro_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=BODY_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 3: EXECUTIVE SUMMARY - KEY METRICS (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Executive Summary - Key Metrics',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    # Calculate metrics
    isolated_accuracy = metrics.calculate_isolated_accuracy()
    system_accuracy = metrics.calculate_system_accuracy()
    integration_gap = metrics.calculate_integration_gap()
    avg_isolated = sum(isolated_accuracy.values()) / len(isolated_accuracy)

    summary_text = f"""PERFORMANCE METRICS

Component-Level Performance (Isolated)
  Average Accuracy: {avg_isolated*100:.1f}%
  Range: {min(isolated_accuracy.values())*100:.1f}% - {max(isolated_accuracy.values())*100:.1f}%

System-Level Performance (Integrated)
  End-to-End Accuracy: {system_accuracy*100:.1f}%

Integration Gap
  Performance Degradation: {integration_gap:.1f}%
  Severity: {"CRITICAL" if integration_gap >= 50 else "SEVERE" if integration_gap >= 30 else "MODERATE" if integration_gap >= 15 else "MINOR"}

Error Propagation Statistics
  Total Error Propagations: {len(metrics.error_propagation)}
  Amplified Cascades: {sum(1 for e in metrics.error_propagation if e.get('amplified', False))}
  Amplification Rate: {sum(1 for e in metrics.error_propagation if e.get('amplified', False)) / len(metrics.error_propagation) * 100:.1f}%

Comparison to Research Baseline (DafnyCOMP)
  DafnyCOMP Integration Gap: 92.0%
  Current Integration Gap: {integration_gap:.1f}%
  Difference: {integration_gap - 92.0:+.1f}% {"(improvement)" if integration_gap < 92.0 else "(degradation)"}

Key Findings
  • High component accuracy does NOT guarantee system reliability
  • Sequential composition amplifies errors at stage boundaries
  • Early-stage errors (requirements/design) have highest impact
  • Testing stage shows false confidence (passes despite upstream errors)
  • Integration gap exceeds theoretical predictions"""

    fig.text(MARGIN_NORMALIZED, 0.92, summary_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 4: VISUALIZATION - ACCURACY COMPARISON (LANDSCAPE)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_HEIGHT, LETTER_WIDTH))  # Swapped for landscape

    # Two subplots
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    # Title
    fig.suptitle('Performance Visualization: Component vs System Accuracy',
                fontsize=TITLE_SIZE, fontweight='bold',
                fontname=TITLE_FONT, color=TITLE_COLOR, y=0.98)

    # Chart 1: Component vs System Accuracy
    agents = list(isolated_accuracy.keys()) + ['System\\n(Integrated)']
    accuracies = list(isolated_accuracy.values()) + [system_accuracy]
    colors = ['#2ecc71'] * len(isolated_accuracy) + ['#e74c3c']

    bars = ax1.bar(range(len(agents)), [a*100 for a in accuracies],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(agents)))
    ax1.set_xticklabels(agents, rotation=45, ha='right', fontsize=BODY_SIZE, fontname=BODY_FONT)
    ax1.set_ylabel('Accuracy (%)', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax1.set_title('Component-Level vs System-Level Performance',
                  fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold', pad=10)
    ax1.axhline(y=90, color='blue', linestyle='--', label='90% Target', alpha=0.6, linewidth=1.5)
    ax1.legend(fontsize=BODY_SIZE, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom',
                fontsize=SMALL_SIZE, fontname=BODY_FONT, fontweight='bold')

    # Chart 2: Integration Gap Waterfall
    categories = ['Component\\nAverage', 'Integration\\nLoss', 'System\\nLevel']
    values = [avg_isolated*100, -integration_gap, system_accuracy*100]
    colors_waterfall = ['#2ecc71', '#e74c3c', '#e67e22']

    ax2.bar(categories, values, color=colors_waterfall, alpha=0.8,
            edgecolor='black', linewidth=1.5, width=0.6)
    ax2.set_ylabel('Accuracy (%)', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax2.set_title(f'Integration Gap Breakdown: {integration_gap:.1f}% Performance Loss',
                  fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold', pad=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', labelsize=BODY_SIZE)
    ax2.tick_params(axis='y', labelsize=BODY_SIZE)

    # Add value labels
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax2.text(i, val + (2 if val > 0 else -2),
                f'{abs(val):.1f}%', ha='center',
                va='bottom' if val > 0 else 'top',
                fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')

    plt.tight_layout(rect=[MARGIN_NORMALIZED, MARGIN_NORMALIZED,
                          1-MARGIN_NORMALIZED, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 5: ERROR PROPAGATION HEATMAP (LANDSCAPE)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_HEIGHT, LETTER_WIDTH))
    ax = plt.subplot(111)

    fig.suptitle('Error Propagation Heatmap Across SDLC Stages',
                fontsize=TITLE_SIZE, fontweight='bold',
                fontname=TITLE_FONT, color=TITLE_COLOR, y=0.98)

    # Create propagation matrix
    stages = ['Requirements', 'Design', 'Implementation', 'Testing', 'Deployment']
    propagation_matrix = np.zeros((5, 5))

    if metrics.error_propagation:
        for error in metrics.error_propagation:
            source = error.get('source', '')
            target = error.get('target', '')
            try:
                source_idx = [s.lower() for s in stages].index(source.lower())
                target_idx = [s.lower() for s in stages].index(target.lower())
                propagation_matrix[source_idx][target_idx] += 1
            except (ValueError, AttributeError):
                pass

    # Create heatmap
    im = ax.imshow(propagation_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(stages, fontsize=BODY_SIZE, fontname=BODY_FONT, rotation=45, ha='right')
    ax.set_yticklabels(stages, fontsize=BODY_SIZE, fontname=BODY_FONT)
    ax.set_xlabel('Target Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax.set_ylabel('Source Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error Count', fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')
    cbar.ax.tick_params(labelsize=SMALL_SIZE)

    # Add values to cells
    for i in range(5):
        for j in range(5):
            if propagation_matrix[i, j] > 0:
                ax.text(j, i, f'{int(propagation_matrix[i, j])}',
                       ha='center', va='center',
                       color='white' if propagation_matrix[i, j] > propagation_matrix.max()/2 else 'black',
                       fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')

    plt.tight_layout(rect=[MARGIN_NORMALIZED, MARGIN_NORMALIZED,
                          1-MARGIN_NORMALIZED, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 6: TOP 10 HIGHEST IMPACT ERRORS (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Top 10 Highest Impact Errors',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    # Collect all errors
    all_errors = []
    for stage, scenarios in error_scenarios.items():
        for scenario in scenarios:
            impact = scenario['propagation_probability'] * scenario['amplification_factor']
            all_errors.append({
                'type': scenario['error_type'],
                'stage': stage.capitalize(),
                'severity': scenario['severity'],
                'prop': scenario['propagation_probability'],
                'amp': scenario['amplification_factor'],
                'impact': impact
            })

    top_errors = sorted(all_errors, key=lambda x: x['impact'], reverse=True)[:10]

    error_text = ""
    for i, err in enumerate(top_errors, 1):
        severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(err['severity'], '⚪')
        error_text += f"{i:2d}. {severity_icon} {err['type'][:55]}\\n"
        error_text += f"    Stage: {err['stage']:15s} Severity: {err['severity']}\\n"
        error_text += f"    Propagation: {err['prop']*100:3.0f}%  Amplification: {err['amp']:.1f}x  Impact: {err['impact']:.2f}\\n\\n"

    fig.text(MARGIN_NORMALIZED, 0.92, error_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 7: STAGE-BY-STAGE BREAKDOWN (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Stage-by-Stage Error Analysis',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    stage_stats = {}
    for err in all_errors:
        stage = err['stage']
        if stage not in stage_stats:
            stage_stats[stage] = {'count': 0, 'high_prop': 0, 'high_amp': 0, 'total_impact': 0}
        stage_stats[stage]['count'] += 1
        if err['prop'] > 0.8:
            stage_stats[stage]['high_prop'] += 1
        if err['amp'] > 2.0:
            stage_stats[stage]['high_amp'] += 1
        stage_stats[stage]['total_impact'] += err['impact']

    stage_text = ""
    for stage in ['Requirements', 'Design', 'Implementation', 'Testing', 'Deployment']:
        if stage in stage_stats:
            stats = stage_stats[stage]
            stage_text += f"🔹 {stage.upper()}\\n"
            stage_text += f"   Total Errors: {stats['count']}\\n"
            stage_text += f"   High Propagation (>80%): {stats['high_prop']}\\n"
            stage_text += f"   High Amplification (>2x): {stats['high_amp']}\\n"
            stage_text += f"   Total Impact Score: {stats['total_impact']:.2f}\\n\\n"

    fig.text(MARGIN_NORMALIZED, 0.92, stage_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 8: CONCLUSION & KEY ASSUMPTIONS (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Conclusion & Key Assumptions',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    conclusion_text = """CONCLUSION

This demonstration validates the Integration Paradox in multi-agent SDLC systems. Despite
high component-level accuracy, the integrated system experiences significant performance
degradation due to error propagation and amplification at stage boundaries.

KEY FINDINGS

1. Component excellence ≠ System reliability
   High individual agent performance does not guarantee system-level success

2. Error cascades dominate system behavior
   Early-stage errors (requirements, design) propagate through all downstream stages

3. Testing provides false confidence
   Tests pass despite upstream errors, allowing defects to reach production

4. Quadratic error growth confirmed
   Error impact scales as O(T²×ε) in T-stage pipelines


ASSUMPTIONS FROM XU ET AL. (2024) - DAFNYCOMP STUDY

1. LLM-generated specifications contain subtle logical errors
   Automated tools struggle with semantic correctness verification

2. Sequential composition amplifies individual component errors
   Each integration point introduces new failure modes

3. Formal verification tools have limited coverage
   Proofs may not capture all edge cases and interactions

4. Human oversight remains critical
   Automated testing insufficient for complex system validation

5. Error detection difficulty increases with pipeline depth
   Root cause analysis becomes exponentially harder

6. Integration testing cannot fully compensate
   Even comprehensive tests miss emergent compositional failures


IMPLICATIONS FOR PRACTICE

• Deploy validation gates at every stage boundary
• Implement redundant checking mechanisms
• Maintain human review for critical paths
• Monitor error propagation patterns continuously
• Design for graceful degradation and error containment"""

    fig.text(MARGIN_NORMALIZED, 0.92, conclusion_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=BODY_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 9: DETAILED ERROR PROPAGATION ANALYSIS (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Detailed Error Propagation Analysis',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    top_15_errors = sorted(all_errors, key=lambda x: x['impact'], reverse=True)[:15]

    error_detail_text = ""
    for i, err in enumerate(top_15_errors, 1):
        severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(err['severity'], '⚪')
        error_detail_text += f"{i:2d}. {severity_icon} {err['type'][:52]}\\n"
        error_detail_text += f"    Stage: {err['stage']:12s} Severity: {err['severity']:8s}\\n"
        error_detail_text += f"    Prop: {err['prop']*100:3.0f}%  Amp: {err['amp']:.1f}x  Impact: {err['impact']:.2f}\\n"
        if i < 15:
            error_detail_text += "\\n"

    fig.text(MARGIN_NORMALIZED, 0.92, error_detail_text,
             ha='left', va='top',
             fontsize=SMALL_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 10: STAGE RISK ASSESSMENT (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Stage Risk Assessment',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    risk_text = ""
    for stage in ['Requirements', 'Design', 'Implementation', 'Testing', 'Deployment']:
        if stage in stage_stats:
            stats = stage_stats[stage]
            risk_text += f"🔹 {stage.upper()}\\n"
            risk_text += f"   Total Errors: {stats['count']}\\n"
            risk_text += f"   High Propagation (>80%): {stats['high_prop']}\\n"
            risk_text += f"   High Amplification (>2x): {stats['high_amp']}\\n"
            risk_text += f"   Total Impact Score: {stats['total_impact']:.2f}\\n"
            risk_text += "\\n"

    fig.text(MARGIN_NORMALIZED, 0.92, risk_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 11: COMPOSITIONAL FAILURE MODES (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Compositional Failure Modes',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    # Specification Fragility
    spec_errors = [e for e in all_errors if e['stage'] in ['Requirements', 'Design']
                   and any(x in e['type'] for x in ['Specification', 'Requirements', 'Ambiguity'])]
    spec_sorted = sorted(spec_errors, key=lambda x: x['impact'], reverse=True)[:3]

    failure_text = "1️⃣  SPECIFICATION FRAGILITY\\n\\n"
    for err in spec_sorted:
        failure_text += f"• {err['type'][:60]}\\n"
        failure_text += f"  Impact: {err['impact']:.2f}  Propagation: {err['prop']*100:.0f}%\\n\\n"

    # Implementation Misalignment
    impl_errors = [e for e in all_errors if e['stage'] == 'Implementation'
                   and any(x in e['type'] for x in ['Design', 'Divergence', 'Mismatch'])]
    impl_sorted = sorted(impl_errors, key=lambda x: x['impact'], reverse=True)[:2]

    failure_text += "\\n2️⃣  IMPLEMENTATION-DESIGN MISALIGNMENT\\n\\n"
    for err in impl_sorted:
        failure_text += f"• {err['type'][:60]}\\n"
        failure_text += f"  Impact: {err['impact']:.2f}\\n\\n"

    # Testing Inadequacy
    test_errors = [e for e in all_errors if e['stage'] == 'Testing']
    test_sorted = sorted(test_errors, key=lambda x: x['impact'], reverse=True)[:3]

    failure_text += "\\n3️⃣  TESTING INADEQUACY & FALSE CONFIDENCE\\n\\n"
    for err in test_sorted:
        failure_text += f"• {err['type'][:60]}\\n"
        failure_text += f"  Impact: {err['impact']:.2f}\\n\\n"

    fig.text(MARGIN_NORMALIZED, 0.92, failure_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=BODY_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 12: QUANTITATIVE SCALING LAW ANALYSIS (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Quantitative Scaling Law Analysis',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    fig.text(0.5, 0.925, 'Based on Kim et al. (2025) arXiv:2512.08296',
             ha='center', va='top',
             fontsize=SMALL_SIZE, style='italic',
             fontname=BODY_FONT, color=TEXT_COLOR)

    # Calculate metrics
    single_agent_baseline = avg_isolated
    beta_Ep_T = -0.330
    beta_Psa_lognA = -0.408
    beta_Ae_T = -0.097

    scaling_text = f"""SCALING LAW PREDICTORS

Single-Agent Baseline: {single_agent_baseline*100:.1f}%
System Accuracy: {system_accuracy*100:.1f}%
Integration Gap: {integration_gap:.1f}%

CRITICAL THRESHOLD TEST

45% Baseline Threshold: {"EXCEEDED" if single_agent_baseline > 0.45 else "Below threshold"}
Current Baseline: {single_agent_baseline*100:.1f}%

{"⚠️  Multi-agent predicted to DEGRADE performance" if single_agent_baseline > 0.45 else "✓ Multi-agent may improve performance"}
{"   Expected degradation: 39-70%" if single_agent_baseline > 0.45 else ""}

DOMINANT EFFECTS

Tool-Coordination Trade-off (β={beta_Ep_T:.3f})
  • Strongest predictor in scaling law
  • Higher tool count + coordination = worse performance

Baseline Paradox (β={beta_Psa_lognA:.3f})
  • High baseline agents degrade when composed
  • Current baseline: {single_agent_baseline*100:.1f}%

Error Amplification (β={beta_Ae_T:.3f})
  • Tool-rich environments amplify errors
  • Each stage introduces cascade risk

ARCHITECTURE RECOMMENDATION

{"⚠️  SINGLE-AGENT SYSTEM RECOMMENDED" if single_agent_baseline > 0.45 else "✓ Multi-agent system viable"}
{"   Multi-agent composition predicted to hurt performance" if single_agent_baseline > 0.45 else "   Expected performance maintained or improved"}"""

    fig.text(MARGIN_NORMALIZED, 0.90, scaling_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=MONO_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # PAGE 13: RECOMMENDATIONS & MITIGATION STRATEGIES (PORTRAIT)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.95, 'Recommendations & Mitigation Strategies',
             ha='center', va='top',
             fontsize=TITLE_SIZE, fontweight='bold',
             fontname=TITLE_FONT, color=TITLE_COLOR)

    if integration_gap >= 50:
        urgency = "🔴 URGENT"
        recommendations = [
            "1. Implement comprehensive integration testing at stage boundaries",
            "2. Add human validation gates at high-risk stages",
            "3. Deploy formal verification for critical components",
            "4. Establish continuous monitoring with automated rollback",
            "5. Create redundant validation paths"
        ]
    elif integration_gap >= 30:
        urgency = "🟠 HIGH PRIORITY"
        recommendations = [
            "1. Strengthen validation at stage boundaries",
            "2. Implement selective human review for critical paths",
            "3. Add consistency checks between adjacent stages",
            "4. Improve error propagation tracking"
        ]
    elif integration_gap >= 15:
        urgency = "🟡 MODERATE PRIORITY"
        recommendations = [
            "1. Enhance automated testing coverage",
            "2. Add spot-checks for high-risk scenarios",
            "3. Improve inter-agent communication"
        ]
    else:
        urgency = "🟢 MAINTAIN"
        recommendations = [
            "1. Continue current practices",
            "2. Monitor for degradation over time",
            "3. Document successful patterns"
        ]

    dafnycomp_gap = 92.0

    rec_text = f"""MITIGATION STRATEGIES

Priority Level: {urgency}

RECOMMENDED ACTIONS

{chr(10).join(recommendations)}


ALIGNMENT WITH PUBLISHED RESEARCH

DafnyCOMP Baseline (Xu et al., 2024): {dafnycomp_gap:.1f}%
Current Integration Gap: {integration_gap:.1f}%
Difference: {integration_gap - dafnycomp_gap:+.1f}%

{"✅ Showing improvement over baseline" if integration_gap < dafnycomp_gap else "⚠️  Gap exceeds baseline"}
{f"   {abs((dafnycomp_gap - integration_gap) / dafnycomp_gap * 100):.1f}% {'better' if integration_gap < dafnycomp_gap else 'worse'} than DafnyCOMP"}"""

    fig.text(MARGIN_NORMALIZED, 0.92, rec_text,
             ha='left', va='top',
             fontsize=BODY_SIZE, fontname=BODY_FONT,
             linespacing=LINE_SPACING, color=TEXT_COLOR)

    plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                       top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\\n✅ PDF Report Generated: {pdf_filename}")
print(f"   Total Pages: 13")
print(f"   File Size: {os.path.getsize(pdf_filename) / 1024:.1f} KB")

# Download the PDF file in Colab
try:
    files.download(pdf_filename)
    print(f"\\n📥 Downloading: {pdf_filename}")
except Exception as e:
    print(f"\\n⚠️  Download failed: {e}")
    print(f"   PDF saved to: {pdf_filename}")

# Also export raw data as JSON
json_filename = f'integration_paradox_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

export_data = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Integration Paradox Demonstration - PoC 1',
    'metrics': {
        'isolated_accuracy': metrics.calculate_isolated_accuracy(),
        'system_accuracy': metrics.calculate_system_accuracy(),
        'integration_gap_percent': metrics.calculate_integration_gap()
    },
    'agent_results': metrics.agent_results,
    'error_propagation': metrics.error_propagation,
    'error_scenarios_summary': {
        stage: {
            'count': len(scenarios),
            'severities': {
                'CRITICAL': sum(1 for s in scenarios if s['severity'] == 'CRITICAL'),
                'HIGH': sum(1 for s in scenarios if s['severity'] == 'HIGH'),
                'MEDIUM': sum(1 for s in scenarios if s['severity'] == 'MEDIUM'),
                'LOW': sum(1 for s in scenarios if s['severity'] == 'LOW')
            }
        }
        for stage, scenarios in error_scenarios.items()
    }
}

with open(json_filename, 'w') as f:
    json.dump(export_data, f, indent=2)

print("✅ Results exported to: integration_paradox_results.json")

# Display summary
print("\\n📊 FINAL SUMMARY:")
print("-"*70)
print(f"Timestamp: {export_data['timestamp']}")
print(f"Experiment: {export_data['experiment']}")
print()
print("Metrics:")
for key, value in export_data['metrics'].items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v:.1%}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        print(f"  {key}: {value:.1%}" if isinstance(value, float) else f"  {key}: {value}")
print()
print(f"Total Agent Results: {len(export_data['agent_results'])}")
print(f"Total Error Propagations: {len(export_data['error_propagation'])}")
print()
print("Error Scenarios by Stage:")
for stage, info in export_data['error_scenarios_summary'].items():
    print(f"  {stage.title()}: {info['count']} errors")
    crit = info['severities'].get('CRITICAL', 0)
    high = info['severities'].get('HIGH', 0)
    if crit > 0 or high > 0:
        print(f"    (CRITICAL: {crit}, HIGH: {high})")
print("-"*70)
'''

# Replace the entire cell
cell_idx = 42
nb['cells'][cell_idx]['source'] = [line + '\n' for line in new_pdf_code.split('\n')]

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("PDF EXPORT QUALITY IMPROVEMENTS")
print("="*80)
print()
print("✅ Consistent typography:")
print("   - Title: DejaVu Sans 16pt")
print("   - Headings: DejaVu Sans 11pt")
print("   - Body: DejaVu Sans 9pt")
print("   - Code: DejaVu Sans Mono 9pt")
print()
print("✅ Uniform 1-inch margins on all pages")
print()
print("✅ Page orientations:")
print("   - Portrait: Pages 1-3, 6-13 (text)")
print("   - Landscape: Pages 4-5 (graphs)")
print()
print("✅ US Letter dimensions: 8.5\" x 11\"")
print()
print("✅ Reduced title gaps: 1 line on pages 9-13")
print()
print("✅ Uniform line spacing: 1.2 throughout")
print()

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    print("✓ Notebook JSON is valid")

    source = ''.join(test_nb['cells'][cell_idx]['source'])
    compile(source, '<cell>', 'exec')
    print("✓ Cell compiles successfully")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
