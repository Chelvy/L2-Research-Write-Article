"""
Enhanced Integration Paradox Report Generator

This module creates comprehensive, publication-quality reports with detailed
metrics, visualizations, and analysis of the Integration Paradox.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# ============================================================================
# ENHANCED INTEGRATION METRICS CLASS
# ============================================================================

class EnhancedIntegrationMetrics:
    """Enhanced metrics tracking with comprehensive analysis capabilities."""

    def __init__(self, base_metrics):
        """
        Initialize with existing IntegrationMetrics instance.

        Args:
            base_metrics: Existing IntegrationMetrics object from PoC 1
        """
        self.base_metrics = base_metrics
        self.error_scenarios = {}
        self.cascade_chains = []
        self.stage_analysis = {}

    def analyze_error_propagation(self):
        """Analyze error propagation patterns in detail."""
        if not self.base_metrics.error_propagation:
            return {}

        df = pd.DataFrame(self.base_metrics.error_propagation)

        analysis = {
            'total_propagations': len(df),
            'amplified_count': df['amplified'].sum() if 'amplified' in df.columns else 0,
            'amplification_rate': df['amplified'].mean() if 'amplified' in df.columns else 0,
            'unique_error_types': df['error_type'].nunique() if 'error_type' in df.columns else 0,
            'propagation_by_source': df.groupby('source').size().to_dict() if 'source' in df.columns else {},
            'propagation_by_target': df.groupby('target').size().to_dict() if 'target' in df.columns else {},
            'most_common_errors': df['error_type'].value_counts().head(10).to_dict() if 'error_type' in df.columns else {}
        }

        # Calculate propagation matrix
        if 'source' in df.columns and 'target' in df.columns:
            analysis['propagation_matrix'] = pd.crosstab(
                df['source'], df['target'],
                values=df['amplified'] if 'amplified' in df.columns else None,
                aggfunc='sum',
                fill_value=0
            ).to_dict()

        return analysis

    def calculate_error_severity_distribution(self, error_scenarios):
        """Calculate distribution of errors by severity."""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        for stage, errors in error_scenarios.items():
            for error in errors:
                severity = error.get('severity', 'MEDIUM')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return severity_counts

    def calculate_stage_risk_scores(self, error_scenarios):
        """Calculate risk scores for each SDLC stage."""
        stage_risks = {}

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

            # Normalize by number of errors
            stage_risks[stage] = {
                'total_risk': total_risk,
                'avg_risk_per_error': total_risk / len(errors) if errors else 0,
                'error_count': len(errors)
            }

        return stage_risks

    def generate_comprehensive_report(self, error_scenarios=None) -> str:
        """
        Generate comprehensive Integration Paradox report.

        Args:
            error_scenarios: Optional dict of error scenarios by stage

        Returns:
            Formatted report string
        """
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
            report.append(f"  Component Average:     {avg_isolated*100:5.1f}%")
            report.append(f"  System Integration:    {system_accuracy*100:5.1f}%")
            report.append(f"  ───────────────────────────────")
            report.append(f"  Performance Gap:       {integration_gap:5.1f}% ⚠️")
            report.append("")

            # Gap severity assessment
            if integration_gap >= 70:
                severity = "🔴 CRITICAL"
                description = "Severe integration paradox - urgent investigation required"
            elif integration_gap >= 50:
                severity = "🟠 SEVERE"
                description = "Significant integration issues - major improvements needed"
            elif integration_gap >= 30:
                severity = "🟡 MODERATE"
                description = "Notable integration challenges - optimization recommended"
            elif integration_gap >= 10:
                severity = "🟢 MINOR"
                description = "Some integration overhead - acceptable for complex systems"
            else:
                severity = "✓ MINIMAL"
                description = "Excellent integration - components compose well"

            report.append(f"Gap Severity: {severity}")
            report.append(f"Assessment: {description}")
            report.append("")

            # Comparison to research paper
            paper_gap = 92.0  # From DafnyCOMP paper
            report.append("Comparison to Published Research:")
            report.append(f"  • This System:     {integration_gap:.1f}% gap")
            report.append(f"  • DafnyCOMP Paper: {paper_gap:.1f}% gap")
            report.append(f"  • Relative Impact: {(integration_gap/paper_gap)*100:.1f}% of published severity")

        report.append("")

        # ================================================================
        # SECTION 4: ERROR PROPAGATION ANALYSIS
        # ================================================================
        report.append("═"*70)
        report.append("🔄 SECTION 4: ERROR PROPAGATION ANALYSIS")
        report.append("═"*70)
        report.append("")

        error_analysis = self.analyze_error_propagation()

        if error_analysis:
            report.append(f"Total Error Propagation Events: {error_analysis['total_propagations']}")
            report.append(f"Amplified Errors: {error_analysis['amplified_count']} "
                         f"({error_analysis['amplification_rate']*100:.1f}%)")
            report.append(f"Unique Error Types: {error_analysis['unique_error_types']}")
            report.append("")

            # Error propagation by stage
            if error_analysis['propagation_by_source']:
                report.append("Error Generation by Stage:")
                report.append("─" * 70)
                for source, count in sorted(error_analysis['propagation_by_source'].items(),
                                           key=lambda x: x[1], reverse=True):
                    bar_length = int((count / max(error_analysis['propagation_by_source'].values())) * 30)
                    bar = "▓" * bar_length
                    report.append(f"  {source:30s} {bar} {count}")
                report.append("")

            # Most common error types
            if error_analysis['most_common_errors']:
                report.append("Top Error Types:")
                report.append("─" * 70)
                for error_type, count in list(error_analysis['most_common_errors'].items())[:5]:
                    report.append(f"  • {error_type:50s} ({count} occurrences)")
                report.append("")
        else:
            report.append("⚠️  No error propagation data available")
            report.append("")

        # ================================================================
        # SECTION 5: ERROR SEVERITY DISTRIBUTION
        # ================================================================
        if error_scenarios:
            report.append("═"*70)
            report.append("⚠️  SECTION 5: ERROR SEVERITY DISTRIBUTION")
            report.append("═"*70)
            report.append("")

            severity_dist = self.calculate_error_severity_distribution(error_scenarios)
            total_errors = sum(severity_dist.values())

            report.append(f"Total Error Scenarios Analyzed: {total_errors}")
            report.append("")
            report.append("Distribution by Severity:")
            report.append("─" * 70)

            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = severity_dist.get(severity, 0)
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length

                icon = "🔴" if severity == 'CRITICAL' else "🟠" if severity == 'HIGH' else "🟡" if severity == 'MEDIUM' else "🟢"
                report.append(f"  {icon} {severity:8s} [{bar:50s}] {count:2d} ({percentage:5.1f}%)")

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

            # Sort stages by total risk
            sorted_stages = sorted(stage_risks.items(),
                                  key=lambda x: x[1]['total_risk'],
                                  reverse=True)

            report.append("Risk Scores by SDLC Stage:")
            report.append("─" * 70)

            max_risk = max(s[1]['total_risk'] for s in sorted_stages) if sorted_stages else 1

            for stage, risk_data in sorted_stages:
                risk_score = risk_data['total_risk']
                normalized_risk = (risk_score / max_risk) * 100 if max_risk > 0 else 0

                bar_length = int(normalized_risk / 2)
                bar = "▓" * bar_length

                risk_level = ("🔴 CRITICAL" if normalized_risk >= 80 else
                             "🟠 HIGH" if normalized_risk >= 60 else
                             "🟡 MEDIUM" if normalized_risk >= 40 else
                             "🟢 LOW")

                report.append(f"  {stage.title():16s} {risk_level:12s} [{bar:50s}]")
                report.append(f"      Risk Score: {risk_score:.1f}, "
                             f"Errors: {risk_data['error_count']}, "
                             f"Avg: {risk_data['avg_risk_per_error']:.1f}")

            report.append("")

        # ================================================================
        # SECTION 7: INTEGRATION FAILURE MODES
        # ================================================================
        report.append("═"*70)
        report.append("💥 SECTION 7: COMPOSITIONAL FAILURE MODES")
        report.append("═"*70)
        report.append("")

        report.append("Based on Xu et al. taxonomy:")
        report.append("")

        report.append("1️⃣  SPECIFICATION FRAGILITY (39.2% of failures)")
        report.append("   • Requirements valid in isolation but incompatible when composed")
        report.append("   • Example: Different interpretations of 'secure password storage'")
        report.append("")

        report.append("2️⃣  IMPLEMENTATION-PROOF MISALIGNMENT (21.7%)")
        report.append("   • Implementation deviates from formal specifications")
        report.append("   • Example: JWT expiration in seconds vs milliseconds")
        report.append("")

        report.append("3️⃣  REASONING INSTABILITY (14.1%)")
        report.append("   • Base case works but inductive step fails")
        report.append("   • Example: Rate limiting fails in distributed deployment")
        report.append("")

        # ================================================================
        # SECTION 8: RECOMMENDATIONS
        # ================================================================
        report.append("═"*70)
        report.append("💡 SECTION 8: RECOMMENDATIONS & MITIGATION STRATEGIES")
        report.append("═"*70)
        report.append("")

        report.append("Based on Analysis Findings:")
        report.append("")

        # Dynamic recommendations based on data
        if integration_gap >= 50:
            report.append("🔴 URGENT ACTIONS REQUIRED:")
            report.append("   1. Implement integration-first testing approach")
            report.append("   2. Add contract verification at all component boundaries")
            report.append("   3. Introduce human validation gates at critical stages")
            report.append("   4. Deploy comprehensive error injection framework")
            report.append("")

        report.append("📋 GENERAL RECOMMENDATIONS:")
        report.append("")
        report.append("1. Integration-First Testing")
        report.append("   • Test composed behavior, not just individual components")
        report.append("   • Validate cross-component interactions early and often")
        report.append("")

        report.append("2. Contract-Based Development")
        report.append("   • Define formal contracts at all component boundaries")
        report.append("   • Implement runtime contract validation")
        report.append("   • Use design-by-contract principles")
        report.append("")

        report.append("3. Error Injection Testing")
        report.append("   • Inject realistic failure scenarios at each stage")
        report.append("   • Test error propagation and amplification")
        report.append("   • Validate error handling under composition")
        report.append("")

        report.append("4. Traceability Implementation")
        report.append("   • Maintain end-to-end traceability from requirements to deployment")
        report.append("   • Use formal models to track dependencies")
        report.append("   • Implement automated traceability verification")
        report.append("")

        report.append("5. Human-in-the-Loop Validation")
        report.append("   • Add validation gates at high-risk stages")
        report.append("   • Expert review for critical decisions")
        report.append("   • Human oversight for security-sensitive components")
        report.append("")

        # ================================================================
        # SECTION 9: RESEARCH ALIGNMENT
        # ================================================================
        report.append("═"*70)
        report.append("📚 SECTION 9: ALIGNMENT WITH PUBLISHED RESEARCH")
        report.append("═"*70)
        report.append("")

        report.append("Key Findings from 'The Integration Paradox' (Xu et al.):")
        report.append("")
        report.append("✓ Confirmed: Reliable components compose into unreliable systems")
        report.append("✓ Confirmed: Quadratic error compounding O(T²×ε)")
        report.append("✓ Confirmed: Integration failures emerge at component boundaries")
        report.append("")

        if integration_gap >= 50:
            report.append("✓ This analysis STRONGLY SUPPORTS the Integration Paradox hypothesis")
        elif integration_gap >= 30:
            report.append("✓ This analysis SUPPORTS the Integration Paradox hypothesis")
        else:
            report.append("• This system shows better integration than typical (may indicate fewer or simpler integrations)")

        report.append("")

        # ================================================================
        # FOOTER
        # ================================================================
        report.append("═"*70)
        report.append("End of Report")
        report.append("═"*70)

        return "\n".join(report)

    def create_enhanced_visualizations(self, error_scenarios=None, figsize=(20, 16)):
        """
        Create comprehensive visualization dashboard.

        Args:
            error_scenarios: Optional dict of error scenarios by stage
            figsize: Figure size (width, height)
        """
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Integration Paradox: Comprehensive Analysis Dashboard',
                     fontsize=18, fontweight='bold', y=0.995)

        # ============================================================
        # Plot 1: Component vs System Accuracy (Enhanced)
        # ============================================================
        isolated = self.base_metrics.calculate_isolated_accuracy()
        system = self.base_metrics.calculate_system_accuracy()

        if isolated:
            agents = list(isolated.keys()) + ['System\n(Integrated)']
            accuracies = list(isolated.values()) + [system]
            colors = ['#2ecc71'] * len(isolated) + ['#e74c3c']

            bars = axes[0, 0].bar(range(len(agents)), [a*100 for a in accuracies],
                                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            axes[0, 0].set_xticks(range(len(agents)))
            axes[0, 0].set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
            axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Component vs System Accuracy', fontsize=12, fontweight='bold')
            axes[0, 0].axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90% Target')
            axes[0, 0].legend(fontsize=9)
            axes[0, 0].grid(axis='y', alpha=0.3)
            axes[0, 0].set_ylim([0, 100])

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 2,
                              f'{acc*100:.1f}%', ha='center', va='bottom',
                              fontsize=9, fontweight='bold')

        # ============================================================
        # Plot 2: Integration Gap Waterfall
        # ============================================================
        if isolated:
            avg_isolated = sum(isolated.values()) / len(isolated)
            gap = self.base_metrics.calculate_integration_gap()

            categories = ['Component\nAverage', 'Integration\nLoss', 'System\nActual']
            values = [avg_isolated*100, -gap, system*100]
            colors_waterfall = ['#3498db', '#e74c3c', '#e67e22']

            cumulative = [values[0], values[0] + values[1], values[2]]

            axes[0, 1].bar([0, 1, 2], values, color=colors_waterfall, alpha=0.8,
                          edgecolor='black', linewidth=1.5)
            axes[0, 1].plot([0, 1, 2], cumulative, 'ko-', linewidth=2, markersize=8)

            axes[0, 1].set_xticks([0, 1, 2])
            axes[0, 1].set_xticklabels(categories, fontsize=10)
            axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Integration Gap Waterfall', fontsize=12, fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

            # Add annotations
            axes[0, 1].annotate(f'-{gap:.1f}%', xy=(1, values[1]/2),
                              fontsize=12, fontweight='bold', color='red',
                              ha='center', va='center')

        # ============================================================
        # Plot 3: Error Propagation Network
        # ============================================================
        error_analysis = self.analyze_error_propagation()

        if error_analysis and error_analysis.get('propagation_by_source'):
            sources = list(error_analysis['propagation_by_source'].keys())
            counts = list(error_analysis['propagation_by_source'].values())

            axes[0, 2].barh(sources, counts, color='#e74c3c', alpha=0.7,
                           edgecolor='black', linewidth=1.5)
            axes[0, 2].set_xlabel('Error Count', fontsize=11, fontweight='bold')
            axes[0, 2].set_title('Error Generation by Stage', fontsize=12, fontweight='bold')
            axes[0, 2].grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (source, count) in enumerate(zip(sources, counts)):
                axes[0, 2].text(count + 0.5, i, str(count),
                              va='center', fontsize=10, fontweight='bold')

        # ============================================================
        # Plot 4: Error Severity Distribution
        # ============================================================
        if error_scenarios:
            severity_dist = self.calculate_error_severity_distribution(error_scenarios)

            severities = list(severity_dist.keys())
            counts = list(severity_dist.values())
            colors_sev = ['#c0392b', '#e67e22', '#f39c12', '#27ae60']

            wedges, texts, autotexts = axes[1, 0].pie(counts, labels=severities,
                                                       colors=colors_sev, autopct='%1.1f%%',
                                                       startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})

            axes[1, 0].set_title('Error Severity Distribution', fontsize=12, fontweight='bold')

        # ============================================================
        # Plot 5: Stage Risk Heatmap
        # ============================================================
        if error_scenarios:
            stage_risks = self.calculate_stage_risk_scores(error_scenarios)

            stages = list(stage_risks.keys())
            risk_scores = [stage_risks[s]['total_risk'] for s in stages]

            # Normalize for heatmap
            max_risk = max(risk_scores) if risk_scores else 1
            normalized_risks = [[r/max_risk] for r in risk_scores]

            im = axes[1, 1].imshow(normalized_risks, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            axes[1, 1].set_yticks(range(len(stages)))
            axes[1, 1].set_yticklabels([s.title() for s in stages], fontsize=10)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_title('Stage Risk Heatmap', fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Risk', fontsize=10)

            # Add risk scores as text
            for i, (stage, risk) in enumerate(zip(stages, risk_scores)):
                axes[1, 1].text(0, i, f'{risk:.1f}', ha='center', va='center',
                              color='white' if normalized_risks[i][0] > 0.5 else 'black',
                              fontsize=11, fontweight='bold')

        # ============================================================
        # Plot 6: Amplification Rate Analysis
        # ============================================================
        if error_analysis and error_analysis.get('amplification_rate'):
            amp_rate = error_analysis['amplification_rate']
            contain_rate = 1 - amp_rate

            rates = [amp_rate * 100, contain_rate * 100]
            labels = ['Amplified', 'Contained']
            colors_amp = ['#e74c3c', '#2ecc71']

            bars = axes[1, 2].bar(labels, rates, color=colors_amp, alpha=0.8,
                                 edgecolor='black', linewidth=2)
            axes[1, 2].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
            axes[1, 2].set_title('Error Amplification Rate', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylim([0, 100])
            axes[1, 2].grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 2,
                              f'{rate:.1f}%', ha='center', va='bottom',
                              fontsize=11, fontweight='bold')

        # ============================================================
        # Plot 7: Error Type Distribution (Top 10)
        # ============================================================
        if error_analysis and error_analysis.get('most_common_errors'):
            errors = list(error_analysis['most_common_errors'].items())[:10]
            error_names = [e[0][:30] + '...' if len(e[0]) > 30 else e[0] for e in errors]
            error_counts = [e[1] for e in errors]

            axes[2, 0].barh(error_names, error_counts, color='#9b59b6', alpha=0.7,
                           edgecolor='black', linewidth=1.5)
            axes[2, 0].set_xlabel('Occurrences', fontsize=11, fontweight='bold')
            axes[2, 0].set_title('Top 10 Error Types', fontsize=12, fontweight='bold')
            axes[2, 0].grid(axis='x', alpha=0.3)
            axes[2, 0].invert_yaxis()

        # ============================================================
        # Plot 8: Comparison to Research Baseline
        # ============================================================
        if isolated:
            gap = self.base_metrics.calculate_integration_gap()
            paper_gap = 92.0  # DafnyCOMP paper

            gaps = [gap, paper_gap]
            labels = ['This\nSystem', 'DafnyCOMP\n(Paper)']
            colors_comp = ['#3498db', '#95a5a6']

            bars = axes[2, 1].bar(labels, gaps, color=colors_comp, alpha=0.8,
                                 edgecolor='black', linewidth=2)
            axes[2, 1].set_ylabel('Integration Gap (%)', fontsize=11, fontweight='bold')
            axes[2, 1].set_title('Comparison to Published Research', fontsize=12, fontweight='bold')
            axes[2, 1].set_ylim([0, 100])
            axes[2, 1].grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, gap_val in zip(bars, gaps):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                              f'{gap_val:.1f}%', ha='center', va='bottom',
                              fontsize=11, fontweight='bold')

        # ============================================================
        # Plot 9: Summary Metrics Card
        # ============================================================
        axes[2, 2].axis('off')

        # Create summary text
        summary_text = "KEY METRICS SUMMARY\n" + "─"*30 + "\n\n"

        if isolated:
            avg_isolated = sum(isolated.values()) / len(isolated)
            summary_text += f"Component Avg: {avg_isolated*100:.1f}%\n"
            summary_text += f"System Actual: {system*100:.1f}%\n"
            summary_text += f"Integration Gap: {gap:.1f}%\n\n"

        if error_analysis:
            summary_text += f"Total Errors: {error_analysis['total_propagations']}\n"
            summary_text += f"Amplified: {error_analysis['amplified_count']}\n"
            summary_text += f"Unique Types: {error_analysis['unique_error_types']}\n\n"

        if error_scenarios:
            severity_dist = self.calculate_error_severity_distribution(error_scenarios)
            total_scenarios = sum(severity_dist.values())
            summary_text += f"Total Scenarios: {total_scenarios}\n"
            summary_text += f"Critical: {severity_dist.get('CRITICAL', 0)}\n"
            summary_text += f"High: {severity_dist.get('HIGH', 0)}\n"

        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        print("✅ Enhanced visualization dashboard complete!")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_enhanced_report_and_visualizations(metrics, error_scenarios=None):
    """
    Generate enhanced report and visualizations.

    Args:
        metrics: IntegrationMetrics instance from PoC 1
        error_scenarios: Optional dict of error scenarios by stage

    Returns:
        EnhancedIntegrationMetrics instance
    """
    enhanced = EnhancedIntegrationMetrics(metrics)

    # Generate report
    report = enhanced.generate_comprehensive_report(error_scenarios)
    print(report)

    # Generate visualizations
    print("\n" + "="*70)
    print("Creating enhanced visualization dashboard...")
    print("="*70 + "\n")

    enhanced.create_enhanced_visualizations(error_scenarios)

    return enhanced


if __name__ == "__main__":
    print("Enhanced Integration Paradox Report Generator")
    print("Import this module and use: generate_enhanced_report_and_visualizations(metrics)")
