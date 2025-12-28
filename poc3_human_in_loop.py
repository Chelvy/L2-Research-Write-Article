"""
PoC 3: Human-Centered AI for Software Engineering

This PoC demonstrates human-in-the-loop AI systems where:
- AI agents generate initial outputs
- Humans review and validate at critical decision points
- Human feedback improves subsequent stages
- Hybrid intelligence combines human judgment with AI capabilities

Research Questions:
1. How does human oversight affect the Integration Paradox?
2. At which stages is human intervention most valuable?
3. What is the optimal human-AI collaboration pattern?
4. How does human feedback quality impact system success?
"""

# ============================================================================
# HUMAN-IN-THE-LOOP FRAMEWORK
# ============================================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import json
import random

class HumanDecision(Enum):
    """Types of decisions a human can make."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    REQUEST_REVISION = "request_revision"
    ESCALATE = "escalate"


class InterventionLevel(Enum):
    """Levels of human intervention."""
    NONE = "none"  # No human involvement
    REVIEW_ONLY = "review_only"  # Human reviews but doesn't change
    APPROVE_REJECT = "approve_reject"  # Human approves or rejects
    COLLABORATIVE_EDIT = "collaborative_edit"  # Human modifies output
    HUMAN_DRIVEN = "human_driven"  # Human controls, AI assists


class ValidationGateType(Enum):
    """Types of validation gates."""
    REQUIREMENTS_REVIEW = "requirements_review"
    DESIGN_APPROVAL = "design_approval"
    CODE_REVIEW = "code_review"
    TEST_VALIDATION = "test_validation"
    DEPLOYMENT_SIGNOFF = "deployment_signoff"


@dataclass
class HumanFeedback:
    """Captures human feedback on AI output."""
    decision: HumanDecision
    confidence: float  # 0.0 to 1.0
    comments: str
    modifications: Optional[str] = None
    issues_identified: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    time_spent_seconds: float = 0.0
    reviewer_expertise: str = "medium"  # low, medium, high, expert
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationGate:
    """A checkpoint where human validation is required."""
    gate_type: ValidationGateType
    stage_name: str
    required: bool = True
    intervention_level: InterventionLevel = InterventionLevel.APPROVE_REJECT
    ai_output: str = ""
    human_feedback: Optional[HumanFeedback] = None
    final_output: str = ""
    passed: bool = False
    retry_count: int = 0
    max_retries: int = 3


# ============================================================================
# SIMULATED HUMAN REVIEWER
# ============================================================================

class SimulatedHumanReviewer:
    """
    Simulates human review behavior for testing.

    In production, this would be replaced with actual human input via
    interactive widgets or external review systems.
    """

    def __init__(self, expertise_level: str = "medium",
                 approval_threshold: float = 0.7,
                 thoroughness: float = 0.8):
        """
        Args:
            expertise_level: low, medium, high, expert
            approval_threshold: Quality threshold for approval (0-1)
            thoroughness: How carefully they review (0-1)
        """
        self.expertise_level = expertise_level
        self.approval_threshold = approval_threshold
        self.thoroughness = thoroughness

        # Expertise affects error detection rate
        self.error_detection_rates = {
            'low': 0.4,
            'medium': 0.7,
            'high': 0.85,
            'expert': 0.95
        }

    def review(self, ai_output: str, stage_name: str,
              gate_type: ValidationGateType) -> HumanFeedback:
        """
        Simulate human review of AI output.

        Args:
            ai_output: The output from AI agent(s)
            stage_name: Name of the SDLC stage
            gate_type: Type of validation gate

        Returns:
            HumanFeedback with decision and comments
        """
        # Simulate review time based on thoroughness
        review_time = self._calculate_review_time(ai_output)

        # Detect issues (based on expertise)
        issues = self._detect_issues(ai_output, stage_name)

        # Make decision
        decision = self._make_decision(ai_output, issues)

        # Generate comments
        comments = self._generate_comments(decision, issues, stage_name)

        # Calculate confidence
        confidence = self._calculate_confidence(issues, ai_output)

        # Generate improvements
        improvements = self._suggest_improvements(issues, stage_name)

        # Modifications if needed
        modifications = None
        if decision == HumanDecision.MODIFY:
            modifications = self._generate_modifications(ai_output, issues)

        return HumanFeedback(
            decision=decision,
            confidence=confidence,
            comments=comments,
            modifications=modifications,
            issues_identified=issues,
            improvement_suggestions=improvements,
            time_spent_seconds=review_time,
            reviewer_expertise=self.expertise_level
        )

    def _calculate_review_time(self, output: str) -> float:
        """Calculate simulated review time."""
        base_time = len(output) / 100.0  # ~1 second per 100 chars
        thoroughness_factor = 0.5 + (self.thoroughness * 1.5)
        return base_time * thoroughness_factor

    def _detect_issues(self, output: str, stage_name: str) -> List[str]:
        """Detect issues in AI output based on expertise."""
        issues = []

        detection_rate = self.error_detection_rates.get(
            self.expertise_level, 0.7
        )

        # Common issues by stage (simplified)
        potential_issues = {
            'Requirements': [
                'Ambiguous requirement specification',
                'Missing non-functional requirements',
                'Incomplete edge case coverage',
                'Unclear acceptance criteria'
            ],
            'Design': [
                'Security vulnerabilities in design',
                'Scalability concerns not addressed',
                'Missing error handling strategy',
                'Tight coupling between components'
            ],
            'Implementation': [
                'Code quality issues',
                'Missing input validation',
                'Inadequate error handling',
                'Security vulnerabilities'
            ],
            'Testing': [
                'Insufficient test coverage',
                'Missing security tests',
                'No performance tests',
                'Inadequate edge case testing'
            ],
            'Deployment': [
                'Missing rollback procedures',
                'Insufficient monitoring',
                'Security configuration issues',
                'No disaster recovery plan'
            ]
        }

        stage_issues = potential_issues.get(stage_name, [])

        # Detect issues based on expertise
        for issue in stage_issues:
            if random.random() < detection_rate:
                # Check if this issue actually exists in output
                # Simplified: random chance based on output quality
                if self._issue_exists_in_output(output, issue):
                    issues.append(issue)

        return issues

    def _issue_exists_in_output(self, output: str, issue: str) -> bool:
        """Check if an issue likely exists (simplified)."""
        # Simple heuristics
        output_lower = output.lower()

        if 'security' in issue.lower():
            return 'security' not in output_lower or len(output) < 200
        elif 'test' in issue.lower():
            return 'test' not in output_lower
        elif 'error' in issue.lower():
            return 'error' not in output_lower and 'exception' not in output_lower
        elif 'performance' in issue.lower():
            return 'performance' not in output_lower and 'scalability' not in output_lower

        # Default: 30% chance any issue exists
        return random.random() < 0.3

    def _make_decision(self, output: str, issues: List[str]) -> HumanDecision:
        """Make review decision based on issues found."""
        if not issues:
            # No issues - likely approve
            return HumanDecision.APPROVE if random.random() < 0.9 else HumanDecision.MODIFY

        # Calculate quality score
        quality_score = 1.0 - (len(issues) * 0.15)  # Each issue reduces quality

        if quality_score >= self.approval_threshold:
            # Good enough to approve, but might suggest modifications
            return HumanDecision.APPROVE if random.random() < 0.7 else HumanDecision.MODIFY
        elif quality_score >= self.approval_threshold - 0.2:
            # Needs minor modifications
            return HumanDecision.MODIFY
        else:
            # Significant issues - request revision
            return HumanDecision.REQUEST_REVISION

    def _generate_comments(self, decision: HumanDecision,
                          issues: List[str], stage_name: str) -> str:
        """Generate review comments."""
        if decision == HumanDecision.APPROVE:
            if not issues:
                return f"Excellent work on {stage_name}. Approved without modifications."
            else:
                return f"Approved with minor observations: {'; '.join(issues[:2])}. " \
                       f"Consider addressing in future iterations."

        elif decision == HumanDecision.MODIFY:
            return f"Good foundation, but needs modifications: {'; '.join(issues)}. " \
                   f"I've made corrections to address these issues."

        elif decision == HumanDecision.REQUEST_REVISION:
            return f"Significant issues identified in {stage_name}: {'; '.join(issues)}. " \
                   f"Please revise and resubmit."

        else:
            return f"Review complete for {stage_name}."

    def _calculate_confidence(self, issues: List[str], output: str) -> float:
        """Calculate reviewer confidence."""
        base_confidence = 0.6

        # Expertise increases confidence
        expertise_bonus = {
            'low': 0.0,
            'medium': 0.1,
            'high': 0.2,
            'expert': 0.3
        }.get(self.expertise_level, 0.1)

        # Thoroughness increases confidence
        thoroughness_bonus = self.thoroughness * 0.1

        # More issues = lower confidence (uncertainty)
        issue_penalty = len(issues) * 0.05

        confidence = base_confidence + expertise_bonus + thoroughness_bonus - issue_penalty
        return max(0.0, min(1.0, confidence))

    def _suggest_improvements(self, issues: List[str], stage_name: str) -> List[str]:
        """Suggest improvements based on identified issues."""
        improvements = []

        for issue in issues:
            if 'security' in issue.lower():
                improvements.append("Add security controls and threat modeling")
            elif 'test' in issue.lower():
                improvements.append("Expand test coverage to include edge cases")
            elif 'performance' in issue.lower():
                improvements.append("Add performance benchmarks and scalability analysis")
            elif 'error' in issue.lower():
                improvements.append("Implement comprehensive error handling")

        return improvements[:3]  # Return top 3

    def _generate_modifications(self, original: str, issues: List[str]) -> str:
        """Generate modified output addressing issues."""
        modifications = f"{original}\n\n--- HUMAN MODIFICATIONS ---\n"

        for issue in issues:
            modifications += f"\n✓ Addressed: {issue}\n"
            if 'security' in issue.lower():
                modifications += "  - Added security requirements/controls\n"
            elif 'test' in issue.lower():
                modifications += "  - Expanded test coverage\n"
            elif 'performance' in issue.lower():
                modifications += "  - Added performance requirements\n"

        return modifications


# ============================================================================
# INTERACTIVE HUMAN REVIEWER (for Colab)
# ============================================================================

class InteractiveHumanReviewer:
    """
    Interactive reviewer using Colab widgets.

    Allows real human review via interactive forms.
    """

    def __init__(self):
        self.feedback_history = []

    def review(self, ai_output: str, stage_name: str,
              gate_type: ValidationGateType) -> HumanFeedback:
        """
        Collect human feedback via interactive widgets.

        Note: This requires ipywidgets in Colab environment.
        """
        try:
            from IPython.display import display, HTML, clear_output
            import ipywidgets as widgets

            print("\n" + "="*70)
            print(f"🔍 HUMAN REVIEW REQUIRED: {stage_name}")
            print("="*70)
            print(f"\n📄 AI Output:\n{ai_output[:500]}...")
            print("\n" + "-"*70)

            # Create interactive form
            decision_dropdown = widgets.Dropdown(
                options=['approve', 'reject', 'modify', 'request_revision'],
                value='approve',
                description='Decision:',
            )

            confidence_slider = widgets.FloatSlider(
                value=0.7,
                min=0.0,
                max=1.0,
                step=0.1,
                description='Confidence:',
            )

            comments_text = widgets.Textarea(
                value='',
                placeholder='Enter your review comments...',
                description='Comments:',
                layout=widgets.Layout(width='100%', height='100px')
            )

            issues_text = widgets.Textarea(
                value='',
                placeholder='List any issues found (one per line)...',
                description='Issues:',
                layout=widgets.Layout(width='100%', height='80px')
            )

            submit_button = widgets.Button(
                description='Submit Review',
                button_style='success',
                tooltip='Submit your review'
            )

            output_area = widgets.Output()

            # Create feedback storage
            feedback_data = {}

            def on_submit(b):
                with output_area:
                    clear_output()
                    feedback_data['decision'] = decision_dropdown.value
                    feedback_data['confidence'] = confidence_slider.value
                    feedback_data['comments'] = comments_text.value
                    feedback_data['issues'] = [i.strip() for i in issues_text.value.split('\n') if i.strip()]
                    print("✅ Review submitted!")

            submit_button.on_click(on_submit)

            # Display form
            form = widgets.VBox([
                decision_dropdown,
                confidence_slider,
                comments_text,
                issues_text,
                submit_button,
                output_area
            ])

            display(form)

            # Wait for submission (in interactive mode)
            print("\n⏳ Waiting for human review...")
            print("(In simulation mode, using simulated reviewer)")

            # Fall back to simulated review if no interaction
            if not feedback_data:
                simulated = SimulatedHumanReviewer()
                return simulated.review(ai_output, stage_name, gate_type)

            # Create HumanFeedback from interactive input
            return HumanFeedback(
                decision=HumanDecision(feedback_data['decision']),
                confidence=feedback_data['confidence'],
                comments=feedback_data['comments'],
                issues_identified=feedback_data['issues'],
                improvement_suggestions=[],
                reviewer_expertise='human'
            )

        except ImportError:
            # Fallback to simulated if widgets not available
            print("⚠️  Interactive widgets not available, using simulated reviewer")
            simulated = SimulatedHumanReviewer()
            return simulated.review(ai_output, stage_name, gate_type)


# ============================================================================
# HUMAN-IN-THE-LOOP SDLC PIPELINE
# ============================================================================

class HumanInLoopSDLC:
    """
    SDLC Pipeline with human validation gates.

    Combines AI agent outputs with human review at critical checkpoints.
    """

    def __init__(self, use_interactive: bool = False,
                 reviewer_expertise: str = "medium"):
        """
        Args:
            use_interactive: If True, use interactive widgets (Colab only)
            reviewer_expertise: Level for simulated reviewer
        """
        self.use_interactive = use_interactive
        self.reviewer = (InteractiveHumanReviewer() if use_interactive
                        else SimulatedHumanReviewer(expertise_level=reviewer_expertise))

        self.validation_gates = []
        self.stage_results = []
        self.pipeline_metrics = {}

    def execute_stage_with_human_review(
        self,
        agent: Any,
        task_description: str,
        stage_name: str,
        gate_type: ValidationGateType,
        intervention_level: InterventionLevel = InterventionLevel.APPROVE_REJECT
    ) -> ValidationGate:
        """
        Execute a stage with human validation gate.

        Args:
            agent: AI agent to generate output
            task_description: Task for the agent
            stage_name: Name of the stage
            gate_type: Type of validation gate
            intervention_level: Level of human intervention

        Returns:
            ValidationGate with results
        """
        from crewai import Task, Crew

        print(f"\n{'='*70}")
        print(f"🤖 AI STAGE: {stage_name}")
        print(f"{'='*70}")

        # Create validation gate
        gate = ValidationGate(
            gate_type=gate_type,
            stage_name=stage_name,
            intervention_level=intervention_level
        )

        # AI generates initial output
        task = Task(
            description=task_description,
            agent=agent,
            expected_output=f"Output for {stage_name}"
        )

        try:
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            ai_output = str(crew.kickoff())
            gate.ai_output = ai_output

            print(f"✅ AI completed {stage_name}")
            print(f"📊 Output length: {len(ai_output)} characters")

        except Exception as e:
            print(f"❌ AI failed: {str(e)}")
            gate.ai_output = f"Error: {str(e)}"
            gate.passed = False
            self.validation_gates.append(gate)
            return gate

        # Human review
        if intervention_level != InterventionLevel.NONE:
            print(f"\n👤 HUMAN REVIEW: {stage_name}")
            feedback = self.reviewer.review(ai_output, stage_name, gate_type)
            gate.human_feedback = feedback

            print(f"   Decision: {feedback.decision.value}")
            print(f"   Confidence: {feedback.confidence:.1%}")
            print(f"   Issues found: {len(feedback.issues_identified)}")
            if feedback.comments:
                print(f"   Comments: {feedback.comments[:100]}...")

            # Process decision
            if feedback.decision == HumanDecision.APPROVE:
                gate.final_output = ai_output
                gate.passed = True

            elif feedback.decision == HumanDecision.MODIFY:
                # Human modifies output
                if feedback.modifications:
                    gate.final_output = feedback.modifications
                else:
                    gate.final_output = f"{ai_output}\n\n[Human approved with suggested improvements]"
                gate.passed = True

            elif feedback.decision == HumanDecision.REQUEST_REVISION:
                # Needs revision - could retry
                gate.retry_count += 1
                if gate.retry_count < gate.max_retries:
                    print(f"   🔄 Requesting revision (attempt {gate.retry_count + 1}/{gate.max_retries})")
                    # In production, would re-run with feedback
                    gate.final_output = ai_output + "\n[Revision requested but not implemented in this demo]"
                    gate.passed = False
                else:
                    print(f"   ⚠️  Max retries reached")
                    gate.final_output = ai_output
                    gate.passed = False

            else:  # REJECT or ESCALATE
                gate.passed = False
                gate.final_output = ai_output

        else:
            # No human review
            gate.final_output = ai_output
            gate.passed = True

        self.validation_gates.append(gate)
        return gate

    def execute_pipeline(self, agents: Dict[str, Any],
                        project_description: str) -> Dict[str, Any]:
        """
        Execute complete SDLC pipeline with human validation gates.

        Args:
            agents: Dictionary of agents by stage
            project_description: Project description

        Returns:
            Pipeline results with metrics
        """
        import time

        print("\n" + "="*70)
        print("   POC 3: HUMAN-IN-THE-LOOP AI SDLC PIPELINE")
        print("="*70)

        start_time = time.time()

        # Stage 1: Requirements with human review
        req_gate = self.execute_stage_with_human_review(
            agent=agents['requirements'],
            task_description=f"Analyze requirements for: {project_description}",
            stage_name="Requirements",
            gate_type=ValidationGateType.REQUIREMENTS_REVIEW,
            intervention_level=InterventionLevel.APPROVE_REJECT
        )

        # Stage 2: Design with human approval
        design_gate = self.execute_stage_with_human_review(
            agent=agents['design'],
            task_description=f"Create design based on: {req_gate.final_output[:300]}...",
            stage_name="Design",
            gate_type=ValidationGateType.DESIGN_APPROVAL,
            intervention_level=InterventionLevel.COLLABORATIVE_EDIT
        )

        # Stage 3: Implementation with code review
        impl_gate = self.execute_stage_with_human_review(
            agent=agents['implementation'],
            task_description=f"Implement based on design: {design_gate.final_output[:300]}...",
            stage_name="Implementation",
            gate_type=ValidationGateType.CODE_REVIEW,
            intervention_level=InterventionLevel.COLLABORATIVE_EDIT
        )

        # Stage 4: Testing with validation
        test_gate = self.execute_stage_with_human_review(
            agent=agents['testing'],
            task_description=f"Create tests for: {impl_gate.final_output[:300]}...",
            stage_name="Testing",
            gate_type=ValidationGateType.TEST_VALIDATION,
            intervention_level=InterventionLevel.APPROVE_REJECT
        )

        # Stage 5: Deployment with signoff
        deploy_gate = self.execute_stage_with_human_review(
            agent=agents['deployment'],
            task_description=f"Create deployment for: {test_gate.final_output[:300]}...",
            stage_name="Deployment",
            gate_type=ValidationGateType.DEPLOYMENT_SIGNOFF,
            intervention_level=InterventionLevel.APPROVE_REJECT
        )

        execution_time = time.time() - start_time

        print("\n" + "="*70)
        print("✅ HUMAN-IN-LOOP PIPELINE COMPLETE")
        print("="*70)

        # Calculate metrics
        self._calculate_metrics(execution_time)

        return {
            'validation_gates': self.validation_gates,
            'metrics': self.pipeline_metrics,
            'execution_time': execution_time
        }

    def _calculate_metrics(self, execution_time: float):
        """Calculate pipeline metrics."""
        total_gates = len(self.validation_gates)
        passed_gates = sum(1 for g in self.validation_gates if g.passed)

        total_issues = sum(len(g.human_feedback.issues_identified)
                          for g in self.validation_gates
                          if g.human_feedback)

        total_review_time = sum(g.human_feedback.time_spent_seconds
                               for g in self.validation_gates
                               if g.human_feedback)

        avg_confidence = (sum(g.human_feedback.confidence
                             for g in self.validation_gates
                             if g.human_feedback) / total_gates
                         if total_gates > 0 else 0)

        # Count decisions
        decisions = {}
        for gate in self.validation_gates:
            if gate.human_feedback:
                decision = gate.human_feedback.decision.value
                decisions[decision] = decisions.get(decision, 0) + 1

        self.pipeline_metrics = {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'gate_pass_rate': passed_gates / total_gates if total_gates > 0 else 0,
            'total_issues_found': total_issues,
            'avg_issues_per_stage': total_issues / total_gates if total_gates > 0 else 0,
            'total_human_review_time': total_review_time,
            'avg_review_time_per_stage': total_review_time / total_gates if total_gates > 0 else 0,
            'avg_human_confidence': avg_confidence,
            'decision_distribution': decisions,
            'execution_time': execution_time,
            'human_intervention_value': self._calculate_intervention_value()
        }

    def _calculate_intervention_value(self) -> float:
        """
        Calculate the value added by human intervention.

        Based on issues caught and corrections made.
        """
        total_value = 0.0

        for gate in self.validation_gates:
            if gate.human_feedback:
                # Each issue caught adds value
                issues_value = len(gate.human_feedback.issues_identified) * 0.1

                # Modifications add value
                if gate.human_feedback.decision == HumanDecision.MODIFY:
                    issues_value += 0.2

                # Rejections prevent bad outputs
                if gate.human_feedback.decision == HumanDecision.REQUEST_REVISION:
                    issues_value += 0.3

                total_value += issues_value

        return min(1.0, total_value)  # Cap at 1.0


# ============================================================================
# METRICS AND COMPARISON
# ============================================================================

class HumanAIMetrics:
    """Track metrics specific to human-AI collaboration."""

    def __init__(self):
        self.intervention_effectiveness = []
        self.human_ai_agreement = []
        self.error_prevention = []

    def compare_with_pure_ai(self, human_in_loop_results: Dict,
                            pure_ai_results: Dict) -> Dict:
        """Compare human-in-loop vs pure AI results."""

        comparison = {
            'quality_improvement': 0.0,
            'time_overhead': 0.0,
            'errors_prevented': 0,
            'human_value_added': 0.0
        }

        # Quality improvement from human review
        hil_success = human_in_loop_results['metrics']['gate_pass_rate']
        ai_success = pure_ai_results.get('system_accuracy', 0)
        comparison['quality_improvement'] = (hil_success - ai_success) * 100

        # Time overhead
        hil_time = human_in_loop_results['execution_time']
        ai_time = pure_ai_results.get('execution_time', hil_time / 2)
        comparison['time_overhead'] = ((hil_time - ai_time) / ai_time) * 100

        # Errors prevented
        comparison['errors_prevented'] = human_in_loop_results['metrics']['total_issues_found']

        # Human value
        comparison['human_value_added'] = human_in_loop_results['metrics']['human_intervention_value']

        return comparison


if __name__ == "__main__":
    print("PoC 3: Human-Centered AI for SE - Framework Ready")
    print("Import this module into your Colab notebook to use")
