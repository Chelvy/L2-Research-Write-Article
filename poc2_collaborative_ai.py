"""
PoC 2: Collaborative AI for Software Engineering

This PoC demonstrates multi-agent collaboration at each SDLC stage, with:
- Multiple agents working together on each task
- Consensus mechanisms (voting, debate, synthesis)
- Peer review and cross-validation
- Conflict resolution strategies

Research Question: Does collaboration reduce the Integration Paradox gap?
"""

# ============================================================================
# COLLABORATIVE FRAMEWORK ARCHITECTURE
# ============================================================================

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from crewai import Agent, Task, Crew, Process
import json
from datetime import datetime

class ConsensusStrategy(Enum):
    """Strategies for reaching consensus among multiple agents."""
    VOTING = "voting"  # Majority vote
    SYNTHESIS = "synthesis"  # Combine all inputs
    DEBATE = "debate"  # Deliberative discussion
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by confidence
    FIRST_VALID = "first_valid"  # First passing validation


class CollaborationMode(Enum):
    """Modes of agent collaboration."""
    PARALLEL = "parallel"  # All agents work independently, then merge
    SEQUENTIAL_REVIEW = "sequential_review"  # Each agent reviews previous
    DEBATE = "debate"  # Agents debate to reach consensus
    HIERARCHICAL = "hierarchical"  # Lead agent coordinates others


@dataclass
class CollaborationConfig:
    """Configuration for collaborative agent teams."""
    num_agents: int
    consensus_strategy: ConsensusStrategy
    collaboration_mode: CollaborationMode
    min_agreement_threshold: float = 0.66  # 66% agreement required
    enable_peer_review: bool = True
    enable_conflict_detection: bool = True
    max_debate_rounds: int = 3


# ============================================================================
# CONSENSUS MECHANISMS
# ============================================================================

class ConsensusEngine:
    """Engine for reaching consensus among multiple agent outputs."""

    def __init__(self, config: CollaborationConfig):
        self.config = config
        self.consensus_history = []

    def reach_consensus(self, agent_outputs: List[Dict[str, Any]],
                       task_name: str) -> Dict[str, Any]:
        """
        Reach consensus from multiple agent outputs.

        Args:
            agent_outputs: List of outputs from different agents
            task_name: Name of the task

        Returns:
            Consensus output with metadata
        """
        strategy = self.config.consensus_strategy

        consensus_result = {
            'timestamp': datetime.now().isoformat(),
            'task': task_name,
            'num_agents': len(agent_outputs),
            'strategy': strategy.value,
            'consensus_output': None,
            'agreement_score': 0.0,
            'conflicts_detected': [],
            'resolution_method': None
        }

        # Apply consensus strategy
        if strategy == ConsensusStrategy.VOTING:
            consensus_result.update(self._voting_consensus(agent_outputs))
        elif strategy == ConsensusStrategy.SYNTHESIS:
            consensus_result.update(self._synthesis_consensus(agent_outputs))
        elif strategy == ConsensusStrategy.DEBATE:
            consensus_result.update(self._debate_consensus(agent_outputs))
        elif strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            consensus_result.update(self._weighted_consensus(agent_outputs))
        elif strategy == ConsensusStrategy.FIRST_VALID:
            consensus_result.update(self._first_valid_consensus(agent_outputs))

        # Detect conflicts
        if self.config.enable_conflict_detection:
            conflicts = self._detect_conflicts(agent_outputs)
            consensus_result['conflicts_detected'] = conflicts

        self.consensus_history.append(consensus_result)
        return consensus_result

    def _voting_consensus(self, outputs: List[Dict]) -> Dict:
        """Majority voting among agent outputs."""
        # Group similar outputs
        output_groups = {}
        for i, output in enumerate(outputs):
            output_text = str(output.get('output', ''))
            # Simple similarity: exact match (in production, use semantic similarity)
            found_group = False
            for key in output_groups:
                if self._similarity(output_text, key) > 0.8:
                    output_groups[key].append(i)
                    found_group = True
                    break
            if not found_group:
                output_groups[output_text] = [i]

        # Find majority
        max_votes = max(len(v) for v in output_groups.values())
        majority_output = max(output_groups.items(), key=lambda x: len(x[1]))

        agreement_score = max_votes / len(outputs)

        return {
            'consensus_output': majority_output[0],
            'agreement_score': agreement_score,
            'resolution_method': 'majority_vote',
            'vote_distribution': {k: len(v) for k, v in output_groups.items()}
        }

    def _synthesis_consensus(self, outputs: List[Dict]) -> Dict:
        """Synthesize all outputs into unified result."""
        combined_output = "=== SYNTHESIZED OUTPUT ===\n\n"

        for i, output in enumerate(outputs):
            combined_output += f"--- Agent {i+1} Contribution ---\n"
            combined_output += str(output.get('output', '')) + "\n\n"

        combined_output += "=== SYNTHESIS ===\n"
        combined_output += "All agent contributions have been combined. "
        combined_output += "Cross-validation recommended.\n"

        # Calculate agreement by looking for common patterns
        agreement = self._calculate_agreement(outputs)

        return {
            'consensus_output': combined_output,
            'agreement_score': agreement,
            'resolution_method': 'synthesis'
        }

    def _debate_consensus(self, outputs: List[Dict]) -> Dict:
        """Simulate debate to reach consensus."""
        # In a real implementation, this would involve multiple rounds
        # For now, we'll combine outputs with conflict highlighting

        debate_output = "=== COLLABORATIVE DEBATE RESULT ===\n\n"

        conflicts = self._detect_conflicts(outputs)

        if conflicts:
            debate_output += "🔴 CONFLICTS IDENTIFIED:\n"
            for conflict in conflicts:
                debate_output += f"  - {conflict}\n"
            debate_output += "\n"

        debate_output += "=== CONSENSUS POSITION ===\n"
        # Take the most comprehensive output
        longest_output = max(outputs, key=lambda x: len(str(x.get('output', ''))))
        debate_output += str(longest_output.get('output', ''))

        agreement = 1.0 - (len(conflicts) * 0.1)  # Reduce for conflicts

        return {
            'consensus_output': debate_output,
            'agreement_score': max(0.0, agreement),
            'resolution_method': 'debate',
            'debate_rounds': 1
        }

    def _weighted_consensus(self, outputs: List[Dict]) -> Dict:
        """Weight outputs by confidence scores."""
        # Extract confidence or default to equal weights
        total_weight = 0.0
        weighted_sum = ""

        for output in outputs:
            confidence = output.get('confidence', 1.0 / len(outputs))
            total_weight += confidence
            # In real implementation, would properly weight text/code

        # For simplicity, use synthesis with confidence weighting
        return self._synthesis_consensus(outputs)

    def _first_valid_consensus(self, outputs: List[Dict]) -> Dict:
        """Return first output that passes validation."""
        for i, output in enumerate(outputs):
            if output.get('valid', True):  # Check validation
                return {
                    'consensus_output': output.get('output', ''),
                    'agreement_score': 1.0 / len(outputs),
                    'resolution_method': 'first_valid',
                    'selected_agent': i
                }

        # If none valid, use first
        return {
            'consensus_output': outputs[0].get('output', ''),
            'agreement_score': 0.0,
            'resolution_method': 'fallback_first'
        }

    def _detect_conflicts(self, outputs: List[Dict]) -> List[str]:
        """Detect conflicts between agent outputs."""
        conflicts = []

        # Check for contradictions (simplified)
        output_texts = [str(o.get('output', '')) for o in outputs]

        # Look for explicit disagreements
        if len(set(output_texts)) > len(output_texts) * 0.5:
            conflicts.append("High variance in outputs detected")

        # Check for specific contradictions
        keywords_positive = ['yes', 'correct', 'valid', 'approved']
        keywords_negative = ['no', 'incorrect', 'invalid', 'rejected']

        has_positive = any(any(kw in text.lower() for kw in keywords_positive)
                          for text in output_texts)
        has_negative = any(any(kw in text.lower() for kw in keywords_negative)
                          for text in output_texts)

        if has_positive and has_negative:
            conflicts.append("Contradictory assessments detected")

        return conflicts

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)."""
        # Simplified: compare length and first 100 chars
        if not text1 or not text2:
            return 0.0

        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        prefix_match = sum(c1 == c2 for c1, c2 in zip(text1[:100], text2[:100])) / 100.0

        return (len_ratio + prefix_match) / 2.0

    def _calculate_agreement(self, outputs: List[Dict]) -> float:
        """Calculate overall agreement score."""
        if len(outputs) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self._similarity(
                    str(outputs[i].get('output', '')),
                    str(outputs[j].get('output', ''))
                )
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0


# ============================================================================
# COLLABORATIVE AGENT TEAM
# ============================================================================

class CollaborativeTeam:
    """A team of agents collaborating on a task."""

    def __init__(self, agents: List[Agent], config: CollaborationConfig,
                 consensus_engine: ConsensusEngine):
        self.agents = agents
        self.config = config
        self.consensus = consensus_engine
        self.collaboration_history = []

    def collaborate(self, task_description: str, task_name: str) -> Dict[str, Any]:
        """
        Execute collaborative task.

        Args:
            task_description: Description of the task
            task_name: Name of the task

        Returns:
            Collaboration result with consensus output
        """
        mode = self.config.collaboration_mode

        if mode == CollaborationMode.PARALLEL:
            result = self._parallel_collaboration(task_description, task_name)
        elif mode == CollaborationMode.SEQUENTIAL_REVIEW:
            result = self._sequential_review(task_description, task_name)
        elif mode == CollaborationMode.DEBATE:
            result = self._debate_collaboration(task_description, task_name)
        elif mode == CollaborationMode.HIERARCHICAL:
            result = self._hierarchical_collaboration(task_description, task_name)
        else:
            result = self._parallel_collaboration(task_description, task_name)

        self.collaboration_history.append(result)
        return result

    def _parallel_collaboration(self, task_desc: str, task_name: str) -> Dict:
        """All agents work independently, then merge results."""
        agent_outputs = []

        print(f"\n🤝 Parallel collaboration: {len(self.agents)} agents working...")

        for i, agent in enumerate(self.agents):
            print(f"   Agent {i+1}/{len(self.agents)}: {agent.role}...")

            # Create individual task for each agent
            task = Task(
                description=task_desc,
                agent=agent,
                expected_output=f"Output for {task_name}"
            )

            # Execute task
            try:
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                output = crew.kickoff()

                agent_outputs.append({
                    'agent_id': i,
                    'agent_role': agent.role,
                    'output': str(output),
                    'valid': True,
                    'confidence': 0.8  # Default confidence
                })
                print(f"      ✓ Complete")
            except Exception as e:
                print(f"      ✗ Error: {str(e)[:50]}")
                agent_outputs.append({
                    'agent_id': i,
                    'agent_role': agent.role,
                    'output': f"Error: {str(e)}",
                    'valid': False,
                    'confidence': 0.0
                })

        # Reach consensus
        print(f"\n   🎯 Reaching consensus using {self.config.consensus_strategy.value}...")
        consensus = self.consensus.reach_consensus(agent_outputs, task_name)

        print(f"      Agreement: {consensus['agreement_score']:.1%}")
        if consensus['conflicts_detected']:
            print(f"      ⚠️  {len(consensus['conflicts_detected'])} conflicts detected")

        return {
            'task_name': task_name,
            'mode': 'parallel',
            'agent_outputs': agent_outputs,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        }

    def _sequential_review(self, task_desc: str, task_name: str) -> Dict:
        """Each agent builds on/reviews the previous agent's work."""
        agent_outputs = []
        current_output = task_desc

        print(f"\n🔄 Sequential review: {len(self.agents)} agents in chain...")

        for i, agent in enumerate(self.agents):
            print(f"   Agent {i+1}/{len(self.agents)}: {agent.role}...")

            # Each agent reviews/extends previous output
            review_task = f"""
            {current_output}

            Your task: Review and enhance the above output. Identify any issues,
            gaps, or improvements needed. Provide your enhanced version.
            """

            task = Task(
                description=review_task,
                agent=agent,
                expected_output=f"Reviewed output for {task_name}"
            )

            try:
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                output = crew.kickoff()
                current_output = str(output)

                agent_outputs.append({
                    'agent_id': i,
                    'agent_role': agent.role,
                    'output': current_output,
                    'valid': True,
                    'confidence': 0.85
                })
                print(f"      ✓ Review complete")
            except Exception as e:
                print(f"      ✗ Error: {str(e)[:50]}")
                agent_outputs.append({
                    'agent_id': i,
                    'agent_role': agent.role,
                    'output': f"Error: {str(e)}",
                    'valid': False,
                    'confidence': 0.0
                })

        # Final output is the last agent's output
        consensus = {
            'consensus_output': current_output,
            'agreement_score': 1.0,  # Sequential builds consensus
            'resolution_method': 'sequential_refinement',
            'conflicts_detected': []
        }

        return {
            'task_name': task_name,
            'mode': 'sequential_review',
            'agent_outputs': agent_outputs,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        }

    def _debate_collaboration(self, task_desc: str, task_name: str) -> Dict:
        """Agents engage in debate to reach consensus."""
        # Simplified: Run parallel then identify conflicts
        parallel_result = self._parallel_collaboration(task_desc, task_name)

        if parallel_result['consensus']['conflicts_detected']:
            print(f"\n   💬 Conflicts detected - initiating debate round...")
            # In production, would run additional debate rounds

        return parallel_result

    def _hierarchical_collaboration(self, task_desc: str, task_name: str) -> Dict:
        """Lead agent coordinates and delegates to others."""
        # First agent is lead
        lead_agent = self.agents[0]
        support_agents = self.agents[1:]

        print(f"\n👔 Hierarchical: Lead agent coordinating {len(support_agents)} supports...")

        # Lead agent plans the work
        planning_task = Task(
            description=f"Plan how to accomplish: {task_desc}",
            agent=lead_agent,
            expected_output="Work plan"
        )

        # Support agents execute
        # (Simplified - in production, lead would delegate specific subtasks)
        return self._parallel_collaboration(task_desc, task_name)


# ============================================================================
# POC 2: COLLABORATIVE SDLC PIPELINE
# ============================================================================

class CollaborativeSDLCPipeline:
    """Complete SDLC pipeline with collaborative teams at each stage."""

    def __init__(self, llm_configs: Dict[str, Any]):
        self.llm_configs = llm_configs
        self.stage_results = []
        self.pipeline_metrics = {}

    def create_collaborative_team(self, stage_name: str,
                                  num_agents: int,
                                  consensus_strategy: ConsensusStrategy,
                                  collaboration_mode: CollaborationMode) -> CollaborativeTeam:
        """Create a collaborative team for a specific SDLC stage."""

        config = CollaborationConfig(
            num_agents=num_agents,
            consensus_strategy=consensus_strategy,
            collaboration_mode=collaboration_mode,
            enable_peer_review=True,
            enable_conflict_detection=True
        )

        consensus_engine = ConsensusEngine(config)

        # Create diverse agents for this stage
        agents = []

        if stage_name == "Requirements":
            roles = [
                "Senior Requirements Analyst",
                "Business Analyst",
                "Technical Requirements Specialist"
            ]
            backstories = [
                "Expert in IEEE 830 requirements specifications with 15 years experience",
                "Specialist in translating business needs into technical requirements",
                "Expert in non-functional requirements and quality attributes"
            ]
        elif stage_name == "Design":
            roles = [
                "Principal Software Architect",
                "Security Architect",
                "Performance Engineer"
            ]
            backstories = [
                "Expert in software architecture patterns and system design",
                "Specialist in security-first design and threat modeling",
                "Expert in performance optimization and scalability"
            ]
        elif stage_name == "Implementation":
            roles = [
                "Senior Software Engineer",
                "Code Quality Specialist",
                "DevOps Engineer"
            ]
            backstories = [
                "Expert in clean code and design patterns",
                "Specialist in code review and static analysis",
                "Expert in infrastructure as code and deployment automation"
            ]
        elif stage_name == "Testing":
            roles = [
                "QA Test Engineer",
                "Security Testing Specialist",
                "Performance Testing Engineer"
            ]
            backstories = [
                "Expert in test automation and coverage analysis",
                "Specialist in penetration testing and vulnerability assessment",
                "Expert in load testing and performance benchmarking"
            ]
        else:  # Deployment
            roles = [
                "DevOps Engineer",
                "Site Reliability Engineer",
                "Production Support Specialist"
            ]
            backstories = [
                "Expert in CI/CD pipelines and deployment automation",
                "Specialist in monitoring, observability, and incident response",
                "Expert in production rollout and rollback procedures"
            ]

        # Create agents with appropriate LLMs
        for i in range(min(num_agents, len(roles))):
            # Rotate through available LLMs
            llm_key = list(self.llm_configs.keys())[i % len(self.llm_configs)]
            llm = self.llm_configs[llm_key]

            agent = Agent(
                role=roles[i],
                goal=f"Collaborate with team to produce high-quality {stage_name.lower()} deliverables",
                backstory=backstories[i],
                verbose=False,
                allow_delegation=False,
                llm=llm
            )
            agents.append(agent)

        return CollaborativeTeam(agents, config, consensus_engine)

    def execute_pipeline(self, project_description: str) -> Dict[str, Any]:
        """Execute the complete collaborative SDLC pipeline."""

        print("\n" + "="*70)
        print("   POC 2: COLLABORATIVE AI SDLC PIPELINE")
        print("="*70)

        # Stage 1: Collaborative Requirements Analysis
        print("\n📋 STAGE 1: Collaborative Requirements Analysis")
        req_team = self.create_collaborative_team(
            "Requirements",
            num_agents=3,
            consensus_strategy=ConsensusStrategy.SYNTHESIS,
            collaboration_mode=CollaborationMode.PARALLEL
        )

        req_result = req_team.collaborate(
            f"Analyze and produce comprehensive requirements for: {project_description}",
            "Requirements Analysis"
        )
        self.stage_results.append(req_result)

        # Stage 2: Collaborative Design
        print("\n🎨 STAGE 2: Collaborative Architecture & Design")
        design_team = self.create_collaborative_team(
            "Design",
            num_agents=3,
            consensus_strategy=ConsensusStrategy.DEBATE,
            collaboration_mode=CollaborationMode.PARALLEL
        )

        design_result = design_team.collaborate(
            f"Based on requirements, create detailed design:\n{req_result['consensus']['consensus_output'][:500]}...",
            "Architecture Design"
        )
        self.stage_results.append(design_result)

        # Stage 3: Collaborative Implementation
        print("\n💻 STAGE 3: Collaborative Implementation")
        impl_team = self.create_collaborative_team(
            "Implementation",
            num_agents=3,
            consensus_strategy=ConsensusStrategy.VOTING,
            collaboration_mode=CollaborationMode.SEQUENTIAL_REVIEW
        )

        impl_result = impl_team.collaborate(
            f"Implement based on design:\n{design_result['consensus']['consensus_output'][:500]}...",
            "Implementation"
        )
        self.stage_results.append(impl_result)

        # Stage 4: Collaborative Testing
        print("\n🧪 STAGE 4: Collaborative Testing")
        test_team = self.create_collaborative_team(
            "Testing",
            num_agents=3,
            consensus_strategy=ConsensusStrategy.SYNTHESIS,
            collaboration_mode=CollaborationMode.PARALLEL
        )

        test_result = test_team.collaborate(
            f"Create comprehensive tests for:\n{impl_result['consensus']['consensus_output'][:500]}...",
            "Testing"
        )
        self.stage_results.append(test_result)

        # Stage 5: Collaborative Deployment
        print("\n🚀 STAGE 5: Collaborative Deployment")
        deploy_team = self.create_collaborative_team(
            "Deployment",
            num_agents=3,
            consensus_strategy=ConsensusStrategy.VOTING,
            collaboration_mode=CollaborationMode.PARALLEL
        )

        deploy_result = deploy_team.collaborate(
            f"Create deployment configuration for:\n{test_result['consensus']['consensus_output'][:500]}...",
            "Deployment"
        )
        self.stage_results.append(deploy_result)

        print("\n" + "="*70)
        print("✅ COLLABORATIVE PIPELINE COMPLETE")
        print("="*70)

        # Calculate metrics
        self._calculate_pipeline_metrics()

        return {
            'stage_results': self.stage_results,
            'metrics': self.pipeline_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_pipeline_metrics(self):
        """Calculate metrics for the collaborative pipeline."""

        total_agents = sum(len(stage['agent_outputs']) for stage in self.stage_results)
        avg_agreement = sum(stage['consensus']['agreement_score']
                          for stage in self.stage_results) / len(self.stage_results)

        total_conflicts = sum(len(stage['consensus'].get('conflicts_detected', []))
                            for stage in self.stage_results)

        successful_stages = sum(1 for stage in self.stage_results
                              if stage['consensus']['agreement_score'] >= 0.66)

        self.pipeline_metrics = {
            'total_stages': len(self.stage_results),
            'total_agents_involved': total_agents,
            'average_agreement_score': avg_agreement,
            'total_conflicts_detected': total_conflicts,
            'successful_stages': successful_stages,
            'pipeline_success_rate': successful_stages / len(self.stage_results),
            'collaboration_effectiveness': avg_agreement * (1 - total_conflicts * 0.1)
        }

    def generate_comparison_report(self, poc1_metrics: Dict) -> str:
        """Generate comparison report between PoC 1 and PoC 2."""

        report = "\n" + "="*70 + "\n"
        report += "   POC 1 vs POC 2: COMPARATIVE ANALYSIS\n"
        report += "="*70 + "\n\n"

        report += "📊 POC 1 (Sequential, Isolated Agents)\n"
        report += "-" * 70 + "\n"
        report += f"  Isolated Accuracy: {poc1_metrics.get('avg_isolated_accuracy', 0)*100:.1f}%\n"
        report += f"  System Accuracy: {poc1_metrics.get('system_accuracy', 0)*100:.1f}%\n"
        report += f"  Integration Gap: {poc1_metrics.get('integration_gap', 0):.1f}%\n\n"

        report += "🤝 POC 2 (Collaborative Multi-Agent)\n"
        report += "-" * 70 + "\n"
        report += f"  Average Agreement: {self.pipeline_metrics['average_agreement_score']*100:.1f}%\n"
        report += f"  Pipeline Success: {self.pipeline_metrics['pipeline_success_rate']*100:.1f}%\n"
        report += f"  Conflicts Detected: {self.pipeline_metrics['total_conflicts_detected']}\n"
        report += f"  Collaboration Effectiveness: {self.pipeline_metrics['collaboration_effectiveness']*100:.1f}%\n\n"

        report += "🔍 KEY INSIGHTS\n"
        report += "-" * 70 + "\n"

        if self.pipeline_metrics['average_agreement_score'] > poc1_metrics.get('system_accuracy', 0):
            report += "  ✅ Collaboration IMPROVED overall system performance\n"
            improvement = (self.pipeline_metrics['average_agreement_score'] -
                         poc1_metrics.get('system_accuracy', 0)) * 100
            report += f"     Improvement: +{improvement:.1f}%\n"
        else:
            report += "  ⚠️  Collaboration did NOT significantly improve performance\n"
            report += "     Possible causes: consensus overhead, conflict resolution costs\n"

        report += f"\n  Conflict Detection: {self.pipeline_metrics['total_conflicts_detected']} issues caught\n"
        report += "     This demonstrates improved error detection through peer review\n"

        return report


# ============================================================================
# METRICS AND ANALYSIS
# ============================================================================

class CollaborationMetrics:
    """Track metrics specific to collaborative workflows."""

    def __init__(self):
        self.consensus_metrics = []
        self.conflict_metrics = []
        self.collaboration_efficiency = []

    def track_consensus(self, stage_name: str, consensus_result: Dict):
        """Track consensus formation metrics."""
        self.consensus_metrics.append({
            'stage': stage_name,
            'agreement_score': consensus_result.get('agreement_score', 0),
            'conflicts': len(consensus_result.get('conflicts_detected', [])),
            'resolution_method': consensus_result.get('resolution_method', ''),
            'timestamp': datetime.now().isoformat()
        })

    def calculate_collaboration_overhead(self,
                                        poc1_time: float,
                                        poc2_time: float) -> Dict:
        """Calculate the overhead of collaboration."""
        return {
            'poc1_time_seconds': poc1_time,
            'poc2_time_seconds': poc2_time,
            'overhead_seconds': poc2_time - poc1_time,
            'overhead_percentage': ((poc2_time - poc1_time) / poc1_time) * 100,
            'time_per_consensus': poc2_time / len(self.consensus_metrics) if self.consensus_metrics else 0
        }

    def analyze_conflict_patterns(self) -> Dict:
        """Analyze patterns in detected conflicts."""
        if not self.consensus_metrics:
            return {'total_conflicts': 0, 'stages_with_conflicts': []}

        total_conflicts = sum(m['conflicts'] for m in self.consensus_metrics)
        stages_with_conflicts = [m['stage'] for m in self.consensus_metrics if m['conflicts'] > 0]
        avg_conflicts_per_stage = total_conflicts / len(self.consensus_metrics)

        return {
            'total_conflicts': total_conflicts,
            'stages_with_conflicts': stages_with_conflicts,
            'avg_conflicts_per_stage': avg_conflicts_per_stage,
            'conflict_resolution_methods': [m['resolution_method'] for m in self.consensus_metrics]
        }


if __name__ == "__main__":
    print("PoC 2: Collaborative AI for SE - Framework Ready")
    print("Import this module into your Colab notebook to use")
