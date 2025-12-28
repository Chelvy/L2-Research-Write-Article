#!/usr/bin/env python3
"""
Integration script to add enhanced reporting capabilities to the Integration Paradox notebook.
This replaces the basic report generation with comprehensive metrics and visualizations.
"""

import json
import sys
from typing import Dict, List, Any

def create_enhanced_reporting_cells() -> List[Dict[str, Any]]:
    """Create cells for enhanced Integration Paradox reporting."""

    cells = []

    # Cell 1: Update Error Propagation Analysis to use comprehensive scenarios
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Analyze Error Propagation (Enhanced)\n",
            "\n",
            "Using comprehensive error scenarios across all SDLC stages with realistic propagation patterns."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import comprehensive error scenarios\n",
            "from comprehensive_error_scenarios import (\n",
            "    get_comprehensive_error_scenarios,\n",
            "    simulate_comprehensive_error_cascade\n",
            ")\n",
            "\n",
            "# Get all error scenarios\n",
            "error_scenarios = get_comprehensive_error_scenarios()\n",
            "\n",
            "# Display summary of error catalog\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"COMPREHENSIVE ERROR SCENARIO CATALOG\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "total_scenarios = 0\n",
            "for stage, scenarios in error_scenarios.items():\n",
            "    count = len(scenarios)\n",
            "    total_scenarios += count\n",
            "    print(f\"\\n{stage.upper():.<30} {count:>3} error types\")\n",
            "    \n",
            "    # Show severity distribution\n",
            "    severity_counts = {}\n",
            "    for s in scenarios:\n",
            "        sev = s['severity']\n",
            "        severity_counts[sev] = severity_counts.get(sev, 0) + 1\n",
            "    \n",
            "    severity_str = \", \".join([f\"{k}:{v}\" for k, v in severity_counts.items()])\n",
            "    print(f\"  └─ Severity: {severity_str}\")\n",
            "\n",
            "print(f\"\\n{'TOTAL SCENARIOS':.<30} {total_scenarios:>3}\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Run comprehensive error cascade simulation\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"SIMULATING ERROR CASCADE WITH COMPREHENSIVE SCENARIOS\")\n",
            "print(\"=\"*70 + \"\\n\")\n",
            "\n",
            "cascade_results = simulate_comprehensive_error_cascade(metrics, verbose=True)"
        ]
    })

    # Cell 2: Enhanced Report Generation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Generate Enhanced Integration Paradox Report\n",
            "\n",
            "Comprehensive report with detailed metrics, visualizations, and research alignment."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import enhanced report generator\n",
            "from enhanced_report_generator import (\n",
            "    EnhancedIntegrationMetrics,\n",
            "    generate_enhanced_report_and_visualizations\n",
            ")\n",
            "\n",
            "# Generate comprehensive report and visualizations\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"GENERATING ENHANCED INTEGRATION PARADOX REPORT\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "enhanced = generate_enhanced_report_and_visualizations(metrics, error_scenarios)\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"✓ Enhanced report generation complete!\")\n",
            "print(\"  - Comprehensive 9-section report generated\")\n",
            "print(\"  - 3x3 visualization dashboard created\")\n",
            "print(\"  - Dynamic recommendations provided\")\n",
            "print(\"  - Research baseline comparison included\")\n",
            "print(\"=\"*80)"
        ]
    })

    # Cell 3: Detailed Error Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10.1 Detailed Error Propagation Analysis\n",
            "\n",
            "Deep dive into error propagation patterns and amplification effects."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze error propagation in detail\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"DETAILED ERROR PROPAGATION ANALYSIS\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "propagation_analysis = enhanced.analyze_error_propagation()\n",
            "\n",
            "# Show top error types by total impact\n",
            "print(\"\\n📊 TOP 10 ERROR TYPES BY TOTAL IMPACT:\")\n",
            "print(\"─\" * 80)\n",
            "\n",
            "error_impacts = []\n",
            "for stage, scenarios in error_scenarios.items():\n",
            "    for scenario in scenarios:\n",
            "        impact = scenario['propagation_prob'] * scenario['amplification']\n",
            "        error_impacts.append({\n",
            "            'type': scenario['error_type'],\n",
            "            'stage': stage.capitalize(),\n",
            "            'severity': scenario['severity'],\n",
            "            'impact': impact,\n",
            "            'prop_prob': scenario['propagation_prob'],\n",
            "            'amplification': scenario['amplification']\n",
            "        })\n",
            "\n",
            "# Sort by impact\n",
            "error_impacts.sort(key=lambda x: x['impact'], reverse=True)\n",
            "\n",
            "for i, err in enumerate(error_impacts[:10], 1):\n",
            "    severity_icon = {\n",
            "        'CRITICAL': '🔴',\n",
            "        'HIGH': '🟠',\n",
            "        'MEDIUM': '🟡',\n",
            "        'LOW': '🟢'\n",
            "    }.get(err['severity'], '⚪')\n",
            "    \n",
            "    print(f\"{i:2d}. {severity_icon} {err['type']:<40} [{err['stage']}]\")\n",
            "    print(f\"    Impact: {err['impact']:.2f} | Prob: {err['prop_prob']:.0%} | Amp: {err['amplification']:.1f}x\")\n",
            "\n",
            "# Show propagation matrix\n",
            "print(\"\\n\\n📈 ERROR PROPAGATION MATRIX:\")\n",
            "print(\"─\" * 80)\n",
            "print(f\"Average amplification rate: {propagation_analysis['average_amplification_rate']:.2f}x\")\n",
            "print(f\"Errors that amplify: {propagation_analysis['amplifying_errors']}\")\n",
            "print(f\"Errors that are contained: {propagation_analysis['contained_errors']}\")\n",
            "\n",
            "print(\"\\nPropagation patterns:\")\n",
            "for pattern, count in propagation_analysis['propagation_patterns'].items():\n",
            "    print(f\"  {pattern}: {count} errors\")"
        ]
    })

    # Cell 4: Stage Risk Assessment
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10.2 Stage Risk Assessment\n",
            "\n",
            "Risk scoring and bottleneck identification across SDLC stages."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate and display stage risk scores\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"STAGE RISK ASSESSMENT\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "stage_risks = enhanced.calculate_stage_risk_scores(error_scenarios)\n",
            "\n",
            "# Sort stages by risk\n",
            "sorted_stages = sorted(stage_risks.items(), key=lambda x: x[1], reverse=True)\n",
            "\n",
            "print(\"Risk Scores by SDLC Stage:\")\n",
            "print(\"─\" * 80)\n",
            "\n",
            "max_risk = max(stage_risks.values())\n",
            "for stage, risk in sorted_stages:\n",
            "    # Normalize risk for visual bar\n",
            "    bar_length = int((risk / max_risk) * 40)\n",
            "    bar = '█' * bar_length\n",
            "    \n",
            "    # Color code based on risk level\n",
            "    if risk > max_risk * 0.8:\n",
            "        risk_icon = '🔴'\n",
            "        risk_level = 'CRITICAL'\n",
            "    elif risk > max_risk * 0.6:\n",
            "        risk_icon = '🟠'\n",
            "        risk_level = 'HIGH'\n",
            "    elif risk > max_risk * 0.4:\n",
            "        risk_icon = '🟡'\n",
            "        risk_level = 'MEDIUM'\n",
            "    else:\n",
            "        risk_icon = '🟢'\n",
            "        risk_level = 'LOW'\n",
            "    \n",
            "    print(f\"{risk_icon} {stage.capitalize():<15} {bar:<40} {risk:>6.1f} [{risk_level}]\")\n",
            "\n",
            "print(\"\\n📌 Key Insights:\")\n",
            "print(f\"   Highest risk stage: {sorted_stages[0][0].capitalize()} ({sorted_stages[0][1]:.1f})\")\n",
            "print(f\"   Lowest risk stage: {sorted_stages[-1][0].capitalize()} ({sorted_stages[-1][1]:.1f})\")\n",
            "print(f\"   Risk range: {sorted_stages[0][1] - sorted_stages[-1][1]:.1f}\")"
        ]
    })

    # Cell 5: Recommendations and Mitigation Strategies
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10.3 Recommendations & Mitigation Strategies\n",
            "\n",
            "Actionable recommendations based on integration gap severity and error patterns."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display dynamic recommendations\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"RECOMMENDATIONS & MITIGATION STRATEGIES\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "# Calculate integration gap\n",
            "isolated_avg = sum(metrics.isolated_accuracy.values()) / len(metrics.isolated_accuracy)\n",
            "integrated_avg = sum(metrics.integrated_accuracy.values()) / len(metrics.integrated_accuracy)\n",
            "integration_gap = ((isolated_avg - integrated_avg) / isolated_avg) * 100\n",
            "\n",
            "print(f\"Integration Gap: {integration_gap:.1f}%\\n\")\n",
            "\n",
            "# Dynamic recommendations based on gap severity\n",
            "if integration_gap >= 50:\n",
            "    urgency = \"🔴 URGENT\"\n",
            "    recommendations = [\n",
            "        \"Implement comprehensive integration testing at every stage boundary\",\n",
            "        \"Add human validation gates at high-risk stages (see Stage Risk Assessment)\",\n",
            "        \"Deploy formal verification for critical components (requirements, design)\",\n",
            "        \"Establish continuous monitoring with automated rollback capabilities\",\n",
            "        \"Create redundant validation paths for error-prone transformations\"\n",
            "    ]\n",
            "elif integration_gap >= 30:\n",
            "    urgency = \"🟠 HIGH PRIORITY\"\n",
            "    recommendations = [\n",
            "        \"Strengthen validation at stage boundaries\",\n",
            "        \"Implement selective human review for critical paths\",\n",
            "        \"Add consistency checks between adjacent stages\",\n",
            "        \"Improve error propagation tracking and logging\"\n",
            "    ]\n",
            "elif integration_gap >= 15:\n",
            "    urgency = \"🟡 MODERATE PRIORITY\"\n",
            "    recommendations = [\n",
            "        \"Enhance automated testing coverage\",\n",
            "        \"Add spot-checks for high-risk error scenarios\",\n",
            "        \"Improve inter-agent communication protocols\"\n",
            "    ]\n",
            "else:\n",
            "    urgency = \"🟢 MAINTAIN\"\n",
            "    recommendations = [\n",
            "        \"Continue current practices\",\n",
            "        \"Monitor for degradation over time\",\n",
            "        \"Document successful patterns for replication\"\n",
            "    ]\n",
            "\n",
            "print(f\"{urgency}\\n\")\n",
            "print(\"Recommended Actions:\")\n",
            "for i, rec in enumerate(recommendations, 1):\n",
            "    print(f\"  {i}. {rec}\")\n",
            "\n",
            "# Additional targeted recommendations based on stage risks\n",
            "print(\"\\n\\nStage-Specific Recommendations:\")\n",
            "print(\"─\" * 80)\n",
            "\n",
            "for stage, risk in sorted_stages[:3]:  # Top 3 risky stages\n",
            "    print(f\"\\n{stage.capitalize()}:\")\n",
            "    \n",
            "    # Count errors by severity\n",
            "    critical = sum(1 for s in error_scenarios[stage] if s['severity'] == 'CRITICAL')\n",
            "    high = sum(1 for s in error_scenarios[stage] if s['severity'] == 'HIGH')\n",
            "    \n",
            "    if critical > 0:\n",
            "        print(f\"  ⚠️  Contains {critical} CRITICAL error scenarios - prioritize mitigation\")\n",
            "    if high > 2:\n",
            "        print(f\"  ⚠️  Contains {high} HIGH severity scenarios - increase validation\")\n",
            "    \n",
            "    # Check propagation\n",
            "    high_prop = sum(1 for s in error_scenarios[stage] if s['propagation_prob'] > 0.8)\n",
            "    if high_prop > 0:\n",
            "        print(f\"  🔄 {high_prop} errors have high propagation probability - add stage boundaries\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)"
        ]
    })

    # Cell 6: Research Alignment
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10.4 Alignment with Published Research\n",
            "\n",
            "Comparison to baseline from DafnyCOMP paper (Xu et al., 2024)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare to research baseline\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"ALIGNMENT WITH PUBLISHED RESEARCH\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "# DafnyCOMP baseline (from Xu et al., 2024)\n",
            "dafnycomp_gap = 92.0  # 99% isolated → 7% integrated\n",
            "\n",
            "print(\"Comparison to DafnyCOMP Baseline:\")\n",
            "print(\"─\" * 80)\n",
            "print(f\"DafnyCOMP Integration Gap:  {dafnycomp_gap:.1f}%\")\n",
            "print(f\"Current Integration Gap:    {integration_gap:.1f}%\")\n",
            "print(f\"Difference:                 {integration_gap - dafnycomp_gap:+.1f}%\\n\")\n",
            "\n",
            "if integration_gap < dafnycomp_gap:\n",
            "    improvement = ((dafnycomp_gap - integration_gap) / dafnycomp_gap) * 100\n",
            "    print(f\"✅ Your integration approach shows {improvement:.1f}% improvement over baseline\")\n",
            "    print(\"   This suggests effective mitigation strategies are in place.\")\n",
            "else:\n",
            "    degradation = ((integration_gap - dafnycomp_gap) / dafnycomp_gap) * 100\n",
            "    print(f\"⚠️  Integration gap is {degradation:.1f}% worse than baseline\")\n",
            "    print(\"   Consider adopting mitigation strategies from the recommendations above.\")\n",
            "\n",
            "# Show compositional failure mode alignment\n",
            "print(\"\\n\\nCompositional Failure Modes (from Xu et al., Section 2.2):\")\n",
            "print(\"─\" * 80)\n",
            "\n",
            "failure_modes = [\n",
            "    (\"Specification Fragility\", \"39.2%\", \"Errors in LLM-generated specifications\"),\n",
            "    (\"Implementation-Proof Misalignment\", \"21.7%\", \"Code doesn't match formal proofs\"),\n",
            "    (\"Reasoning Instability\", \"14.1%\", \"Inconsistent outputs from same input\"),\n",
            "    (\"Error Compounding\", \"O(T²×ε)\", \"Quadratic growth in multi-stage pipelines\")\n",
            "]\n",
            "\n",
            "for mode, rate, description in failure_modes:\n",
            "    print(f\"\\n{mode}:\")\n",
            "    print(f\"  Rate: {rate}\")\n",
            "    print(f\"  └─ {description}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"\\n✓ Enhanced Integration Paradox analysis complete!\")\n",
            "print(\"  All sections generated with comprehensive metrics and visualizations.\")\n",
            "print(\"=\"*80)"
        ]
    })

    return cells


def integrate_enhanced_reporting(notebook_path: str) -> bool:
    """Integrate enhanced reporting into the notebook."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find section 9 and 10
        section_9_idx = None
        section_10_idx = None

        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                source = ''.join(cell['source'])
                if '## 9. Analyze Error Propagation' in source:
                    section_9_idx = i
                elif '## 10. Generate Integration Paradox Report' in source:
                    section_10_idx = i

        if section_9_idx is None or section_10_idx is None:
            print("❌ Could not find sections 9 and 10 in notebook")
            return False

        print(f"Found Section 9 at cell {section_9_idx}")
        print(f"Found Section 10 at cell {section_10_idx}")

        # Remove old sections 9 and 10 (keep going until we hit section 11)
        section_11_idx = None
        for i in range(section_10_idx + 1, len(nb['cells'])):
            if nb['cells'][i]['cell_type'] == 'markdown':
                source = ''.join(nb['cells'][i]['source'])
                if '## 11.' in source:
                    section_11_idx = i
                    break

        if section_11_idx is None:
            # No section 11, remove until end
            cells_to_remove = len(nb['cells']) - section_9_idx
        else:
            cells_to_remove = section_11_idx - section_9_idx

        print(f"Removing {cells_to_remove} cells (old sections 9 and 10)")

        # Remove old cells
        for _ in range(cells_to_remove):
            nb['cells'].pop(section_9_idx)

        # Insert new enhanced reporting cells
        new_cells = create_enhanced_reporting_cells()
        print(f"Inserting {len(new_cells)} new enhanced reporting cells")

        for i, cell in enumerate(new_cells):
            nb['cells'].insert(section_9_idx + i, cell)

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully integrated enhanced reporting!")
        print(f"   - Replaced sections 9 and 10")
        print(f"   - Added {len(new_cells)} new cells")
        print(f"   - Total notebook cells: {len(nb['cells'])}")

        return True

    except Exception as e:
        print(f"❌ Error integrating enhanced reporting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("INTEGRATING ENHANCED REPORTING INTO NOTEBOOK")
    print("="*80 + "\n")

    success = integrate_enhanced_reporting(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Integration complete!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Integration failed")
        print("="*80)
        sys.exit(1)
