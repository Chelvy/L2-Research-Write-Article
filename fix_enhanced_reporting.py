#!/usr/bin/env python3
"""
Fix enhanced reporting integration by embedding code in notebook cells.
"""

import json
import sys

def get_enhanced_report_code():
    """Get the complete enhanced reporting code to embed."""

    code = '''# ============================================================================
# ENHANCED INTEGRATION PARADOX REPORTING
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class EnhancedIntegrationMetrics:
    """Enhanced metrics tracking with comprehensive analysis capabilities."""

    def __init__(self, base_metrics):
        """
        Initialize with existing IntegrationMetrics instance.

        Args:
            base_metrics: Existing IntegrationMetrics object from PoC 1
        """
        self.base_metrics = base_metrics

    def analyze_error_propagation(self):
        """Analyze error propagation patterns in detail."""
        if not self.base_metrics.error_propagation:
            return {
                'total_propagations': 0,
                'amplified_count': 0,
                'average_amplification_rate': 0.0,
                'amplifying_errors': 0,
                'contained_errors': 0,
                'propagation_patterns': {}
            }

        df = pd.DataFrame(self.base_metrics.error_propagation)

        amplified_count = df['amplified'].sum() if 'amplified' in df.columns else 0
        total_count = len(df)

        # Count propagation patterns
        propagation_patterns = {}
        if 'source' in df.columns and 'target' in df.columns:
            for _, row in df.iterrows():
                pattern = f"{row['source']} → {row['target']}"
                propagation_patterns[pattern] = propagation_patterns.get(pattern, 0) + 1

        analysis = {
            'total_propagations': total_count,
            'amplified_count': int(amplified_count),
            'average_amplification_rate': amplified_count / total_count if total_count > 0 else 0.0,
            'amplifying_errors': int(amplified_count),
            'contained_errors': total_count - int(amplified_count),
            'propagation_patterns': propagation_patterns
        }

        return analysis

    def calculate_error_severity_distribution(self, error_scenarios):
        """Calculate distribution of errors by severity."""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        if error_scenarios:
            for stage, errors in error_scenarios.items():
                for error in errors:
                    severity = error.get('severity', 'MEDIUM')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return severity_counts

    def calculate_stage_risk_scores(self, error_scenarios):
        """Calculate risk scores for each SDLC stage."""
        stage_risks = {}

        if error_scenarios:
            for stage, errors in error_scenarios.items():
                total_risk = 0
                for error in errors:
                    # Risk = severity weight × propagation probability × amplification
                    severity_weight = {
                        'CRITICAL': 4.0,
                        'HIGH': 3.0,
                        'MEDIUM': 2.0,
                        'LOW': 1.0
                    }.get(error.get('severity', 'MEDIUM'), 2.0)

                    prop_prob = error.get('propagation_prob', 0.5)
                    amplification = error.get('amplification', 1.0)

                    error_risk = severity_weight * prop_prob * amplification
                    total_risk += error_risk

                # Return normalized risk score
                stage_risks[stage] = total_risk / len(errors) if errors else 0

        return stage_risks

    def generate_comprehensive_report(self, error_scenarios=None) -> str:
        """Generate comprehensive Integration Paradox report."""
        report = []
        report.append("╔" + "═"*68 + "╗")
        report.append("║" + " "*15 + "INTEGRATION PARADOX ANALYSIS REPORT" + " "*18 + "║")
        report.append("║" + " "*20 + "Comprehensive Edition" + " "*26 + "║")
        report.append("╚" + "═"*68 + "╝")
        report.append("")

        # Timestamp
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # ================================================================
        # SECTION 1: COMPONENT-LEVEL ACCURACY
        # ================================================================
        report.append("═"*70)
        report.append("📊 SECTION 1: COMPONENT-LEVEL ACCURACY (Isolated Performance)")
        report.append("═"*70)
        report.append("")

        isolated = self.base_metrics.calculate_isolated_accuracy()

        if isolated:
            report.append("Individual Agent Performance:")
            report.append("─" * 70)

            for agent, accuracy in sorted(isolated.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(accuracy * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                percentage = accuracy * 100

                status = "✓" if accuracy >= 0.9 else "⚠" if accuracy >= 0.7 else "✗"
                report.append(f"  {status} {agent:30s} [{bar}] {percentage:5.1f}%")

            avg_isolated = sum(isolated.values()) / len(isolated)
            report.append("")
            report.append(f"📈 Average Isolated Accuracy: {avg_isolated*100:.1f}%")

            # Performance distribution
            excellent = sum(1 for acc in isolated.values() if acc >= 0.9)
            good = sum(1 for acc in isolated.values() if 0.7 <= acc < 0.9)
            poor = sum(1 for acc in isolated.values() if acc < 0.7)

            report.append(f"   • Excellent (≥90%): {excellent} agents")
            report.append(f"   • Good (70-89%):    {good} agents")
            report.append(f"   • Poor (<70%):      {poor} agents")
        else:
            report.append("⚠️  No isolated accuracy data available")

        report.append("")

        # ================================================================
        # SECTION 2: SYSTEM-LEVEL PERFORMANCE
        # ================================================================
        report.append("═"*70)
        report.append("🔗 SECTION 2: SYSTEM-LEVEL PERFORMANCE (Integrated Pipeline)")
        report.append("═"*70)
        report.append("")

        system_accuracy = self.base_metrics.calculate_system_accuracy()
        report.append(f"End-to-End System Success Rate: {system_accuracy*100:.1f}%")
        report.append("")

        if system_accuracy >= 0.9:
            report.append("✓ System performance: EXCELLENT")
        elif system_accuracy >= 0.7:
            report.append("⚠ System performance: ACCEPTABLE")
        elif system_accuracy >= 0.5:
            report.append("⚠ System performance: MARGINAL")
        else:
            report.append("✗ System performance: CRITICAL")

        report.append("")

        # ================================================================
        # SECTION 3: INTEGRATION PARADOX GAP
        # ================================================================
        report.append("═"*70)
        report.append("⚠️  SECTION 3: INTEGRATION PARADOX GAP")
        report.append("═"*70)
        report.append("")

        integration_gap = self.base_metrics.calculate_integration_gap()

        if isolated:
            avg_isolated = sum(isolated.values()) / len(isolated)

            report.append("Performance Degradation Analysis:")
            report.append("─" * 70)
            report.append(f"  Component-level (isolated):  {avg_isolated*100:5.1f}%")
            report.append(f"  System-level (integrated):   {system_accuracy*100:5.1f}%")
            report.append(f"  Integration Gap:             {integration_gap:5.1f}%")
            report.append("")

            # Severity classification
            if integration_gap >= 50:
                severity = "🔴 CRITICAL"
                assessment = "Severe integration issues detected"
            elif integration_gap >= 30:
                severity = "🟠 SEVERE"
                assessment = "Significant performance degradation"
            elif integration_gap >= 15:
                severity = "🟡 MODERATE"
                assessment = "Notable integration challenges"
            else:
                severity = "🟢 MINOR"
                assessment = "Integration impact within acceptable range"

            report.append(f"Gap Severity: {severity}")
            report.append(f"Assessment: {assessment}")

        report.append("")

        # ================================================================
        # SECTION 4: ERROR PROPAGATION ANALYSIS
        # ================================================================
        if error_scenarios:
            report.append("═"*70)
            report.append("🔄 SECTION 4: ERROR PROPAGATION ANALYSIS")
            report.append("═"*70)
            report.append("")

            propagation = self.analyze_error_propagation()

            report.append(f"Total Error Propagations: {propagation['total_propagations']}")
            report.append(f"Amplified Errors: {propagation['amplified_count']}")
            report.append(f"Contained Errors: {propagation['contained_errors']}")
            report.append(f"Amplification Rate: {propagation['average_amplification_rate']*100:.1f}%")
            report.append("")

        # ================================================================
        # SECTION 5: ERROR SEVERITY DISTRIBUTION
        # ================================================================
        if error_scenarios:
            report.append("═"*70)
            report.append("📊 SECTION 5: ERROR SEVERITY DISTRIBUTION")
            report.append("═"*70)
            report.append("")

            severity_dist = self.calculate_error_severity_distribution(error_scenarios)
            total_errors = sum(severity_dist.values())

            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = severity_dist[severity]
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
                report.append(f"  {icon} {severity:10s}: {count:3d} ({percentage:5.1f}%)")

            report.append("")

        # ================================================================
        # SECTION 6: STAGE RISK ASSESSMENT
        # ================================================================
        if error_scenarios:
            report.append("═"*70)
            report.append("🎯 SECTION 6: STAGE RISK ASSESSMENT")
            report.append("═"*70)
            report.append("")

            stage_risks = self.calculate_stage_risk_scores(error_scenarios)

            if stage_risks:
                sorted_stages = sorted(stage_risks.items(), key=lambda x: x[1], reverse=True)
                max_risk = max(stage_risks.values()) if stage_risks else 1.0

                for stage, risk in sorted_stages:
                    bar_length = int((risk / max_risk) * 40) if max_risk > 0 else 0
                    bar = "█" * bar_length
                    report.append(f"  {stage.capitalize():15s} [{bar:<40}] {risk:6.2f}")

                report.append("")
                report.append(f"Highest Risk: {sorted_stages[0][0].capitalize()} ({sorted_stages[0][1]:.2f})")
                report.append(f"Lowest Risk:  {sorted_stages[-1][0].capitalize()} ({sorted_stages[-1][1]:.2f})")

        report.append("")

        # ================================================================
        # SECTION 7: COMPOSITIONAL FAILURE MODES
        # ================================================================
        report.append("═"*70)
        report.append("⚙️  SECTION 7: COMPOSITIONAL FAILURE MODES")
        report.append("═"*70)
        report.append("")
        report.append("Based on Xu et al., 2024 (DafnyCOMP):")
        report.append("")
        report.append("  1. Specification Fragility (39.2%)")
        report.append("     └─ LLM-generated specs contain subtle errors")
        report.append("")
        report.append("  2. Implementation-Proof Misalignment (21.7%)")
        report.append("     └─ Code doesn't match formal verification proofs")
        report.append("")
        report.append("  3. Reasoning Instability (14.1%)")
        report.append("     └─ Inconsistent outputs from identical inputs")
        report.append("")
        report.append("  4. Error Compounding (O(T²×ε))")
        report.append("     └─ Quadratic error growth in multi-stage pipelines")
        report.append("")

        # ================================================================
        # SECTION 8: RECOMMENDATIONS
        # ================================================================
        report.append("═"*70)
        report.append("💡 SECTION 8: RECOMMENDATIONS & MITIGATION STRATEGIES")
        report.append("═"*70)
        report.append("")

        if integration_gap >= 50:
            urgency = "🔴 URGENT"
            recommendations = [
                "Implement comprehensive integration testing at every stage boundary",
                "Add human validation gates at high-risk stages",
                "Deploy formal verification for critical components",
                "Establish continuous monitoring with automated rollback",
                "Create redundant validation paths for error-prone transformations"
            ]
        elif integration_gap >= 30:
            urgency = "🟠 HIGH PRIORITY"
            recommendations = [
                "Strengthen validation at stage boundaries",
                "Implement selective human review for critical paths",
                "Add consistency checks between adjacent stages",
                "Improve error propagation tracking and logging"
            ]
        elif integration_gap >= 15:
            urgency = "🟡 MODERATE PRIORITY"
            recommendations = [
                "Enhance automated testing coverage",
                "Add spot-checks for high-risk error scenarios",
                "Improve inter-agent communication protocols"
            ]
        else:
            urgency = "🟢 MAINTAIN"
            recommendations = [
                "Continue current practices",
                "Monitor for degradation over time",
                "Document successful patterns for replication"
            ]

        report.append(f"Priority Level: {urgency}")
        report.append("")
        report.append("Recommended Actions:")
        for i, rec in enumerate(recommendations, 1):
            report.append(f"  {i}. {rec}")

        report.append("")

        # ================================================================
        # SECTION 9: ALIGNMENT WITH PUBLISHED RESEARCH
        # ================================================================
        report.append("═"*70)
        report.append("📚 SECTION 9: ALIGNMENT WITH PUBLISHED RESEARCH")
        report.append("═"*70)
        report.append("")

        dafnycomp_gap = 92.0  # 99% isolated → 7% integrated

        report.append("Comparison to DafnyCOMP Baseline (Xu et al., 2024):")
        report.append("─" * 70)
        report.append(f"  DafnyCOMP Integration Gap:  {dafnycomp_gap:5.1f}%")
        report.append(f"  Current Integration Gap:    {integration_gap:5.1f}%")
        report.append(f"  Difference:                 {integration_gap - dafnycomp_gap:+5.1f}%")
        report.append("")

        if integration_gap < dafnycomp_gap:
            improvement = ((dafnycomp_gap - integration_gap) / dafnycomp_gap) * 100
            report.append(f"✅ Integration approach shows {improvement:.1f}% improvement over baseline")
        else:
            degradation = ((integration_gap - dafnycomp_gap) / dafnycomp_gap) * 100
            report.append(f"⚠️  Integration gap is {degradation:.1f}% worse than baseline")

        report.append("")
        report.append("═"*70)
        report.append("END OF REPORT")
        report.append("═"*70)

        return "\\n".join(report)

    def create_enhanced_visualizations(self, error_scenarios=None, figsize=(20, 16)):
        """Create comprehensive visualization dashboard."""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Enhanced Integration Paradox Analysis Dashboard',
                     fontsize=16, fontweight='bold')

        # Get metrics
        isolated = self.base_metrics.calculate_isolated_accuracy()
        system = self.base_metrics.calculate_system_accuracy()
        gap = self.base_metrics.calculate_integration_gap()

        # Plot 1: Component vs System Accuracy (Enhanced)
        ax = axes[0, 0]
        if isolated:
            agents = list(isolated.keys()) + ['System\\n(Integrated)']
            accuracies = list(isolated.values()) + [system]
            colors = ['green'] * len(isolated) + ['red']

            bars = ax.bar(range(len(agents)), [a*100 for a in accuracies],
                         color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(agents)))
            ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Component vs System Accuracy')
            ax.axhline(y=90, color='blue', linestyle='--', label='90% Target', alpha=0.5)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # Plot 2: Integration Gap Waterfall
        ax = axes[0, 1]
        if isolated:
            avg_isolated = sum(isolated.values()) / len(isolated) * 100
            categories = ['Component\\nLevel', 'Integration\\nLoss', 'System\\nLevel']
            values = [avg_isolated, -gap, system*100]
            colors_waterfall = ['green', 'red', 'darkred']

            ax.bar(categories, values, color=colors_waterfall, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Integration Gap: {gap:.1f}%')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='y', alpha=0.3)

        # Plot 3: Error Generation by Stage
        ax = axes[0, 2]
        if self.base_metrics.error_propagation:
            df = pd.DataFrame(self.base_metrics.error_propagation)
            if 'source' in df.columns:
                error_counts = df.groupby('source').size().sort_values()
                ax.barh(range(len(error_counts)), error_counts.values,
                       color='orange', alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(error_counts)))
                ax.set_yticklabels(error_counts.index, fontsize=8)
                ax.set_xlabel('Errors Generated')
                ax.set_title('Error Generation by Stage')
                ax.grid(axis='x', alpha=0.3)

        # Plot 4: Error Severity Distribution
        ax = axes[1, 0]
        if error_scenarios:
            severity_dist = self.calculate_error_severity_distribution(error_scenarios)
            colors_severity = ['darkred', 'orange', 'yellow', 'lightgreen']
            ax.pie(severity_dist.values(), labels=severity_dist.keys(), autopct='%1.1f%%',
                  colors=colors_severity, startangle=90)
            ax.set_title('Error Severity Distribution')

        # Plot 5: Stage Risk Heatmap
        ax = axes[1, 1]
        if error_scenarios:
            stage_risks = self.calculate_stage_risk_scores(error_scenarios)
            if stage_risks:
                stages = list(stage_risks.keys())
                risks = list(stage_risks.values())

                # Normalize risks to 0-1 for color mapping
                max_risk = max(risks) if risks else 1.0
                normalized_risks = [r / max_risk for r in risks] if max_risk > 0 else risks

                # Create heatmap-style visualization
                im = ax.imshow([normalized_risks], cmap='RdYlGn_r', aspect='auto')
                ax.set_xticks(range(len(stages)))
                ax.set_xticklabels([s.capitalize() for s in stages], rotation=45, ha='right', fontsize=8)
                ax.set_yticks([])
                ax.set_title('Stage Risk Assessment')
                plt.colorbar(im, ax=ax, label='Normalized Risk')

        # Plot 6: Amplification Rate Analysis
        ax = axes[1, 2]
        propagation = self.analyze_error_propagation()
        categories = ['Amplified', 'Contained']
        values = [propagation['amplifying_errors'], propagation['contained_errors']]
        colors_amp = ['red', 'green']
        ax.bar(categories, values, color=colors_amp, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Error Count')
        ax.set_title('Error Amplification Analysis')
        ax.grid(axis='y', alpha=0.3)

        # Plot 7: Top Error Types
        ax = axes[2, 0]
        if error_scenarios:
            # Collect all error types
            all_errors = []
            for stage, errors in error_scenarios.items():
                for error in errors:
                    all_errors.append({
                        'type': error['error_type'],
                        'impact': error.get('propagation_prob', 0.5) * error.get('amplification', 1.0)
                    })

            if all_errors:
                df_errors = pd.DataFrame(all_errors)
                top_errors = df_errors.groupby('type')['impact'].sum().sort_values(ascending=True).tail(10)
                ax.barh(range(len(top_errors)), top_errors.values,
                       color='crimson', alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(top_errors)))
                ax.set_yticklabels(top_errors.index, fontsize=7)
                ax.set_xlabel('Total Impact Score')
                ax.set_title('Top 10 Error Types by Impact')
                ax.grid(axis='x', alpha=0.3)

        # Plot 8: Comparison to Research Baseline
        ax = axes[2, 1]
        dafnycomp_gap = 92.0
        categories = ['DafnyCOMP\\n(Baseline)', 'Current\\nSystem']
        values = [dafnycomp_gap, gap]
        colors_baseline = ['purple', 'red' if gap > dafnycomp_gap else 'green']
        ax.bar(categories, values, color=colors_baseline, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Integration Gap (%)')
        ax.set_title('Comparison to Research Baseline')
        ax.axhline(y=dafnycomp_gap, color='purple', linestyle='--',
                  label='DafnyCOMP: 92%', alpha=0.5)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 9: Summary Metrics Card
        ax = axes[2, 2]
        ax.axis('off')

        summary_text = f"""
        SUMMARY METRICS

        Component Accuracy: {sum(isolated.values())/len(isolated)*100:.1f}% (avg)
        System Accuracy: {system*100:.1f}%
        Integration Gap: {gap:.1f}%

        Error Propagations: {propagation['total_propagations']}
        Amplification Rate: {propagation['average_amplification_rate']*100:.1f}%

        vs. DafnyCOMP: {gap - dafnycomp_gap:+.1f}%
        """

        ax.text(0.5, 0.5, summary_text, ha='center', va='center',
               fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Key Metrics Summary')

        plt.tight_layout()
        plt.show()

        return fig


def generate_enhanced_report_and_visualizations(metrics, error_scenarios=None):
    """
    Convenience function to generate both report and visualizations.

    Args:
        metrics: IntegrationMetrics instance
        error_scenarios: Optional dict of error scenarios by stage

    Returns:
        EnhancedIntegrationMetrics instance
    """
    enhanced = EnhancedIntegrationMetrics(metrics)

    # Generate report
    report = enhanced.generate_comprehensive_report(error_scenarios)
    print(report)

    # Create visualizations
    print("\\n\\nGenerating enhanced visualizations...\\n")
    enhanced.create_enhanced_visualizations(error_scenarios)

    return enhanced


print("✅ Enhanced Integration Paradox reporting framework loaded!")
print("   - EnhancedIntegrationMetrics class available")
print("   - Use: generate_enhanced_report_and_visualizations(metrics, error_scenarios)")
'''

    return code


def fix_enhanced_reporting(notebook_path: str) -> bool:
    """Fix enhanced reporting by embedding code in notebook."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find the cell that tries to import enhanced_report_generator
        import_cell_idx = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'from enhanced_report_generator import' in source:
                    import_cell_idx = i
                    break

        if import_cell_idx is None:
            print("❌ Could not find enhanced_report_generator import cell")
            return False

        print(f"Found import cell at index {import_cell_idx}")

        # Create new code cell with embedded enhanced reporting code
        new_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": get_enhanced_report_code().split('\n')
        }

        # Replace the import cell with embedded code
        nb['cells'][import_cell_idx] = new_code_cell

        # Find and update the cell that calls generate_enhanced_report_and_visualizations
        for i in range(import_cell_idx + 1, min(import_cell_idx + 5, len(nb['cells']))):
            if nb['cells'][i]['cell_type'] == 'code':
                source = ''.join(nb['cells'][i]['source'])
                if 'generate_enhanced_report_and_visualizations' in source:
                    # Update this cell to just call the function
                    nb['cells'][i]['source'] = [
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
                    print(f"Updated report generation cell at index {i}")
                    break

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully fixed enhanced reporting!")
        print(f"   - Embedded EnhancedIntegrationMetrics class in cell {import_cell_idx}")
        print(f"   - Code is now self-contained in notebook")

        return True

    except Exception as e:
        print(f"❌ Error fixing enhanced reporting: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("FIXING ENHANCED REPORTING INTEGRATION")
    print("="*80 + "\n")

    success = fix_enhanced_reporting(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
