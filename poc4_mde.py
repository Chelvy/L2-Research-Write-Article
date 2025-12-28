"""
PoC 4: AI-Assisted Model-Driven Engineering (MDE)

This PoC demonstrates model-driven development where:
- Formal models are created at each SDLC stage
- AI performs model-to-model transformations
- Models are validated before transformation
- Traceability links models across stages
- Code is generated from formal models

Research Questions:
1. Does formalization reduce the Integration Paradox?
2. How do model transformations affect error propagation?
3. What is the value of model validation and traceability?
4. Can formal models prevent specification fragility?
"""

# ============================================================================
# MODEL-DRIVEN ENGINEERING FRAMEWORK
# ============================================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import json
import re

class ModelType(Enum):
    """Types of models in the MDE pipeline."""
    REQUIREMENTS_MODEL = "requirements_model"
    DESIGN_MODEL = "design_model"
    IMPLEMENTATION_MODEL = "implementation_model"
    TEST_MODEL = "test_model"
    DEPLOYMENT_MODEL = "deployment_model"


class ValidationLevel(Enum):
    """Levels of model validation."""
    SYNTAX = "syntax"  # Syntactic correctness
    SEMANTIC = "semantic"  # Semantic consistency
    COMPLETENESS = "completeness"  # All required elements present
    CONSISTENCY = "consistency"  # Internal consistency
    TRACEABILITY = "traceability"  # Links to previous models


@dataclass
class ModelElement:
    """A single element within a model."""
    element_id: str
    element_type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormalModel:
    """
    A formal model representing an SDLC artifact.

    Models are structured representations that can be:
    - Validated for correctness
    - Transformed into other models
    - Traced across SDLC stages
    """
    model_id: str
    model_type: ModelType
    stage_name: str
    elements: List[ModelElement] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    traceability_links: Dict[str, str] = field(default_factory=dict)
    validation_results: Dict[ValidationLevel, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_element(self, element: ModelElement):
        """Add an element to the model."""
        self.elements.append(element)

    def add_constraint(self, constraint: str):
        """Add a constraint to the model."""
        self.constraints.append(constraint)

    def add_traceability_link(self, target_element: str, source_element: str):
        """Add traceability link to previous model."""
        self.traceability_links[target_element] = source_element

    def get_element_by_id(self, element_id: str) -> Optional[ModelElement]:
        """Get element by ID."""
        for element in self.elements:
            if element.element_id == element_id:
                return element
        return None


# ============================================================================
# MODEL VALIDATION
# ============================================================================

class ModelValidator:
    """Validates formal models at various levels."""

    def __init__(self):
        self.validation_rules = {
            ModelType.REQUIREMENTS_MODEL: self._validate_requirements_model,
            ModelType.DESIGN_MODEL: self._validate_design_model,
            ModelType.IMPLEMENTATION_MODEL: self._validate_implementation_model,
            ModelType.TEST_MODEL: self._validate_test_model,
            ModelType.DEPLOYMENT_MODEL: self._validate_deployment_model
        }

    def validate_model(self, model: FormalModel,
                      levels: List[ValidationLevel] = None) -> Dict[ValidationLevel, Tuple[bool, List[str]]]:
        """
        Validate model at specified levels.

        Args:
            model: Model to validate
            levels: Validation levels to check (default: all)

        Returns:
            Dict mapping validation level to (pass/fail, issues)
        """
        if levels is None:
            levels = list(ValidationLevel)

        results = {}

        for level in levels:
            if level == ValidationLevel.SYNTAX:
                results[level] = self._validate_syntax(model)
            elif level == ValidationLevel.SEMANTIC:
                results[level] = self._validate_semantics(model)
            elif level == ValidationLevel.COMPLETENESS:
                results[level] = self._validate_completeness(model)
            elif level == ValidationLevel.CONSISTENCY:
                results[level] = self._validate_consistency(model)
            elif level == ValidationLevel.TRACEABILITY:
                results[level] = self._validate_traceability(model)

        return results

    def _validate_syntax(self, model: FormalModel) -> Tuple[bool, List[str]]:
        """Validate syntactic correctness."""
        issues = []

        # Check model has required fields
        if not model.model_id:
            issues.append("Missing model ID")
        if not model.stage_name:
            issues.append("Missing stage name")

        # Check elements are well-formed
        for element in model.elements:
            if not element.element_id:
                issues.append(f"Element missing ID: {element.name}")
            if not element.element_type:
                issues.append(f"Element missing type: {element.name}")
            if not element.name:
                issues.append(f"Element missing name")

        return (len(issues) == 0, issues)

    def _validate_semantics(self, model: FormalModel) -> Tuple[bool, List[str]]:
        """Validate semantic consistency."""
        issues = []

        # Check relationships point to valid elements
        element_ids = {e.element_id for e in model.elements}

        for element in model.elements:
            for rel_id in element.relationships:
                if rel_id not in element_ids:
                    issues.append(f"Invalid relationship in {element.name}: {rel_id}")

        # Model-specific semantic checks
        validator = self.validation_rules.get(model.model_type)
        if validator:
            model_issues = validator(model)
            issues.extend(model_issues)

        return (len(issues) == 0, issues)

    def _validate_completeness(self, model: FormalModel) -> Tuple[bool, List[str]]:
        """Validate model completeness."""
        issues = []

        # Check minimum elements
        if len(model.elements) == 0:
            issues.append("Model has no elements")

        # Model-specific completeness checks
        if model.model_type == ModelType.REQUIREMENTS_MODEL:
            # Needs functional and non-functional requirements
            types = {e.element_type for e in model.elements}
            if 'functional_requirement' not in types:
                issues.append("Missing functional requirements")
            if 'non_functional_requirement' not in types:
                issues.append("Missing non-functional requirements")

        elif model.model_type == ModelType.DESIGN_MODEL:
            # Needs components and interfaces
            types = {e.element_type for e in model.elements}
            if 'component' not in types:
                issues.append("Missing components")
            if 'interface' not in types:
                issues.append("Missing interfaces")

        elif model.model_type == ModelType.IMPLEMENTATION_MODEL:
            # Needs classes/functions
            types = {e.element_type for e in model.elements}
            if 'class' not in types and 'function' not in types:
                issues.append("Missing implementation elements")

        return (len(issues) == 0, issues)

    def _validate_consistency(self, model: FormalModel) -> Tuple[bool, List[str]]:
        """Validate internal consistency."""
        issues = []

        # Check constraints are satisfied
        for constraint in model.constraints:
            if not self._check_constraint(model, constraint):
                issues.append(f"Constraint violated: {constraint}")

        # Check no duplicate IDs
        element_ids = [e.element_id for e in model.elements]
        if len(element_ids) != len(set(element_ids)):
            issues.append("Duplicate element IDs found")

        return (len(issues) == 0, issues)

    def _validate_traceability(self, model: FormalModel) -> Tuple[bool, List[str]]:
        """Validate traceability links."""
        issues = []

        # Check all elements have traceability (except first model)
        if model.model_type != ModelType.REQUIREMENTS_MODEL:
            untraced = []
            for element in model.elements:
                if element.element_id not in model.traceability_links:
                    untraced.append(element.name)

            if untraced:
                issues.append(f"Elements without traceability: {', '.join(untraced[:3])}...")

        return (len(issues) == 0, issues)

    def _check_constraint(self, model: FormalModel, constraint: str) -> bool:
        """Check if a constraint is satisfied (simplified)."""
        # In production, would parse and evaluate constraint formally
        # For now, assume constraints are satisfied
        return True

    def _validate_requirements_model(self, model: FormalModel) -> List[str]:
        """Requirements-specific validation."""
        issues = []

        # Check requirements have priorities
        for element in model.elements:
            if element.element_type in ['functional_requirement', 'non_functional_requirement']:
                if 'priority' not in element.properties:
                    issues.append(f"Requirement missing priority: {element.name}")

        return issues

    def _validate_design_model(self, model: FormalModel) -> List[str]:
        """Design-specific validation."""
        issues = []

        # Check components have interfaces
        components = [e for e in model.elements if e.element_type == 'component']
        for comp in components:
            if not comp.relationships:
                issues.append(f"Component has no interfaces: {comp.name}")

        return issues

    def _validate_implementation_model(self, model: FormalModel) -> List[str]:
        """Implementation-specific validation."""
        issues = []

        # Check functions have return types
        for element in model.elements:
            if element.element_type == 'function':
                if 'return_type' not in element.properties:
                    issues.append(f"Function missing return type: {element.name}")

        return issues

    def _validate_test_model(self, model: FormalModel) -> List[str]:
        """Test-specific validation."""
        issues = []

        # Check tests have assertions
        for element in model.elements:
            if element.element_type == 'test_case':
                if 'assertions' not in element.properties:
                    issues.append(f"Test missing assertions: {element.name}")

        return issues

    def _validate_deployment_model(self, model: FormalModel) -> List[str]:
        """Deployment-specific validation."""
        issues = []

        # Check deployment has configuration
        configs = [e for e in model.elements if e.element_type == 'configuration']
        if not configs:
            issues.append("Missing deployment configuration")

        return issues


# ============================================================================
# MODEL TRANSFORMATION
# ============================================================================

class ModelTransformer:
    """Transforms models from one type to another using AI."""

    def __init__(self, validator: ModelValidator):
        self.validator = validator
        self.transformation_history = []

    def transform(self, source_model: FormalModel,
                 target_type: ModelType,
                 ai_agent: Any) -> FormalModel:
        """
        Transform source model to target model type using AI.

        Args:
            source_model: Source model
            target_type: Target model type
            ai_agent: AI agent to perform transformation

        Returns:
            Transformed model
        """
        from crewai import Task, Crew

        print(f"\n🔄 MODEL TRANSFORMATION: {source_model.model_type.value} → {target_type.value}")

        # Create transformation task
        transformation_prompt = self._create_transformation_prompt(
            source_model, target_type
        )

        task = Task(
            description=transformation_prompt,
            agent=ai_agent,
            expected_output=f"Formal model for {target_type.value}"
        )

        try:
            crew = Crew(agents=[ai_agent], tasks=[task], verbose=False)
            ai_output = str(crew.kickoff())

            # Parse AI output into formal model
            target_model = self._parse_ai_output_to_model(
                ai_output, target_type, source_model
            )

            # Validate transformed model
            print(f"   ✅ Transformation complete")
            print(f"   📊 Elements: {len(target_model.elements)}")

            validation_results = self.validator.validate_model(target_model)
            passed = sum(1 for passed, _ in validation_results.values() if passed)
            total = len(validation_results)

            print(f"   🔍 Validation: {passed}/{total} checks passed")

            # Store validation results
            for level, (result, issues) in validation_results.items():
                target_model.validation_results[level] = result
                if not result and issues:
                    print(f"      ⚠️  {level.value}: {issues[0]}")

            # Record transformation
            self.transformation_history.append({
                'source_type': source_model.model_type.value,
                'target_type': target_type.value,
                'source_elements': len(source_model.elements),
                'target_elements': len(target_model.elements),
                'validation_passed': passed,
                'validation_total': total,
                'timestamp': datetime.now().isoformat()
            })

            return target_model

        except Exception as e:
            print(f"   ❌ Transformation failed: {str(e)}")
            # Return minimal model on failure
            return FormalModel(
                model_id=f"failed_{target_type.value}",
                model_type=target_type,
                stage_name=target_type.value.replace('_model', '').title(),
                metadata={'error': str(e)}
            )

    def _create_transformation_prompt(self, source_model: FormalModel,
                                     target_type: ModelType) -> str:
        """Create prompt for AI transformation."""
        prompt = f"""
Transform the following {source_model.model_type.value} into a {target_type.value}.

SOURCE MODEL:
{self._serialize_model(source_model)}

Your task: Create a formal {target_type.value} that:
1. Traces back to all elements in the source model
2. Includes all necessary elements for {target_type.value}
3. Maintains consistency with source model
4. Follows best practices for {target_type.value}

Output format: Describe the model elements, their properties, and relationships.
"""
        return prompt

    def _serialize_model(self, model: FormalModel) -> str:
        """Serialize model for AI consumption."""
        serialized = f"Model Type: {model.model_type.value}\n"
        serialized += f"Stage: {model.stage_name}\n"
        serialized += f"Elements ({len(model.elements)}):\n"

        for i, element in enumerate(model.elements[:10], 1):  # Show first 10
            serialized += f"  {i}. {element.name} ({element.element_type})\n"
            if element.properties:
                for key, value in list(element.properties.items())[:3]:
                    serialized += f"     - {key}: {value}\n"

        if len(model.elements) > 10:
            serialized += f"  ... and {len(model.elements) - 10} more elements\n"

        if model.constraints:
            serialized += f"Constraints: {', '.join(model.constraints[:3])}\n"

        return serialized

    def _parse_ai_output_to_model(self, ai_output: str,
                                  target_type: ModelType,
                                  source_model: FormalModel) -> FormalModel:
        """
        Parse AI output into formal model.

        This is simplified - in production would use more sophisticated parsing.
        """
        target_model = FormalModel(
            model_id=f"{target_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            model_type=target_type,
            stage_name=target_type.value.replace('_model', '').title()
        )

        # Extract elements from AI output (simplified heuristic)
        lines = ai_output.split('\n')

        element_count = 0
        current_element = None

        for line in lines:
            line = line.strip()

            # Detect element definitions (simplified patterns)
            if self._looks_like_element(line, target_type):
                element_count += 1
                element_name = self._extract_element_name(line)

                current_element = ModelElement(
                    element_id=f"{target_type.value}_elem_{element_count}",
                    element_type=self._infer_element_type(line, target_type),
                    name=element_name,
                    properties={}
                )
                target_model.add_element(current_element)

                # Add traceability to source
                if source_model.elements:
                    # Link to corresponding source element (simplified)
                    source_idx = min(element_count - 1, len(source_model.elements) - 1)
                    source_elem = source_model.elements[source_idx]
                    target_model.add_traceability_link(
                        current_element.element_id,
                        source_elem.element_id
                    )

            # Extract properties for current element
            elif current_element and ':' in line:
                key, value = self._extract_property(line)
                if key and value:
                    current_element.properties[key] = value

        # If no elements detected, create minimal model
        if len(target_model.elements) == 0:
            target_model = self._create_minimal_model(target_type, source_model)

        return target_model

    def _looks_like_element(self, line: str, model_type: ModelType) -> bool:
        """Check if line looks like an element definition."""
        # Simplified heuristics
        if model_type == ModelType.REQUIREMENTS_MODEL:
            return any(kw in line.lower() for kw in ['requirement', 'req-', 'fr-', 'nfr-'])
        elif model_type == ModelType.DESIGN_MODEL:
            return any(kw in line.lower() for kw in ['component', 'class', 'interface', 'module'])
        elif model_type == ModelType.IMPLEMENTATION_MODEL:
            return any(kw in line.lower() for kw in ['class', 'function', 'method', 'def ', 'public'])
        elif model_type == ModelType.TEST_MODEL:
            return any(kw in line.lower() for kw in ['test', 'assert', 'verify'])
        elif model_type == ModelType.DEPLOYMENT_MODEL:
            return any(kw in line.lower() for kw in ['container', 'service', 'deployment', 'config'])

        return False

    def _extract_element_name(self, line: str) -> str:
        """Extract element name from line."""
        # Remove common prefixes and extract identifier
        line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
        line = re.sub(r'^[-*]\s*', '', line)  # Remove bullets

        # Extract first word-like sequence
        match = re.search(r'\b([A-Z][a-zA-Z0-9_]*)\b', line)
        if match:
            return match.group(1)

        # Fallback: first 50 chars
        return line[:50].strip()

    def _infer_element_type(self, line: str, model_type: ModelType) -> str:
        """Infer element type from line and model type."""
        line_lower = line.lower()

        if model_type == ModelType.REQUIREMENTS_MODEL:
            if 'functional' in line_lower or 'fr-' in line_lower:
                return 'functional_requirement'
            elif 'non-functional' in line_lower or 'nfr-' in line_lower:
                return 'non_functional_requirement'
            return 'requirement'

        elif model_type == ModelType.DESIGN_MODEL:
            if 'component' in line_lower:
                return 'component'
            elif 'interface' in line_lower:
                return 'interface'
            elif 'class' in line_lower:
                return 'class'
            return 'design_element'

        elif model_type == ModelType.IMPLEMENTATION_MODEL:
            if 'class' in line_lower:
                return 'class'
            elif 'function' in line_lower or 'method' in line_lower:
                return 'function'
            return 'code_element'

        elif model_type == ModelType.TEST_MODEL:
            return 'test_case'

        elif model_type == ModelType.DEPLOYMENT_MODEL:
            if 'container' in line_lower:
                return 'container'
            elif 'service' in line_lower:
                return 'service'
            return 'configuration'

        return 'element'

    def _extract_property(self, line: str) -> Tuple[str, str]:
        """Extract property key-value from line."""
        if ':' not in line:
            return None, None

        parts = line.split(':', 1)
        key = parts[0].strip().lower().replace(' ', '_')
        value = parts[1].strip()

        return key, value

    def _create_minimal_model(self, model_type: ModelType,
                             source_model: FormalModel) -> FormalModel:
        """Create minimal model when parsing fails."""
        model = FormalModel(
            model_id=f"{model_type.value}_minimal",
            model_type=model_type,
            stage_name=model_type.value.replace('_model', '').title()
        )

        # Create one element per source element
        for i, source_elem in enumerate(source_model.elements[:5], 1):
            element = ModelElement(
                element_id=f"{model_type.value}_elem_{i}",
                element_type=self._get_default_element_type(model_type),
                name=f"Derived from {source_elem.name}",
                properties={'source': source_elem.element_id}
            )
            model.add_element(element)
            model.add_traceability_link(element.element_id, source_elem.element_id)

        return model

    def _get_default_element_type(self, model_type: ModelType) -> str:
        """Get default element type for model type."""
        defaults = {
            ModelType.REQUIREMENTS_MODEL: 'requirement',
            ModelType.DESIGN_MODEL: 'component',
            ModelType.IMPLEMENTATION_MODEL: 'class',
            ModelType.TEST_MODEL: 'test_case',
            ModelType.DEPLOYMENT_MODEL: 'configuration'
        }
        return defaults.get(model_type, 'element')


# ============================================================================
# MDE SDLC PIPELINE
# ============================================================================

class MDEPipeline:
    """Model-Driven Engineering SDLC pipeline."""

    def __init__(self):
        self.validator = ModelValidator()
        self.transformer = ModelTransformer(self.validator)
        self.models = []
        self.pipeline_metrics = {}

    def execute_pipeline(self, agents: Dict[str, Any],
                        project_description: str) -> Dict[str, Any]:
        """
        Execute MDE pipeline with model transformations.

        Args:
            agents: Dictionary of AI agents by stage
            project_description: Project description

        Returns:
            Pipeline results with models and metrics
        """
        import time

        print("\n" + "="*70)
        print("   POC 4: AI-ASSISTED MODEL-DRIVEN ENGINEERING PIPELINE")
        print("="*70)

        start_time = time.time()

        # Stage 1: Requirements Model (AI generates from description)
        print("\n📋 STAGE 1: Requirements Model Generation")
        req_model = self._generate_initial_model(
            agents['requirements'],
            project_description,
            ModelType.REQUIREMENTS_MODEL
        )
        self.models.append(req_model)

        # Stage 2: Design Model (Transform from Requirements)
        print("\n🎨 STAGE 2: Design Model Transformation")
        design_model = self.transformer.transform(
            req_model,
            ModelType.DESIGN_MODEL,
            agents['design']
        )
        self.models.append(design_model)

        # Stage 3: Implementation Model (Transform from Design)
        print("\n💻 STAGE 3: Implementation Model Transformation")
        impl_model = self.transformer.transform(
            design_model,
            ModelType.IMPLEMENTATION_MODEL,
            agents['implementation']
        )
        self.models.append(impl_model)

        # Stage 4: Test Model (Transform from Implementation)
        print("\n🧪 STAGE 4: Test Model Transformation")
        test_model = self.transformer.transform(
            impl_model,
            ModelType.TEST_MODEL,
            agents['testing']
        )
        self.models.append(test_model)

        # Stage 5: Deployment Model (Transform from Test)
        print("\n🚀 STAGE 5: Deployment Model Transformation")
        deploy_model = self.transformer.transform(
            test_model,
            ModelType.DEPLOYMENT_MODEL,
            agents['deployment']
        )
        self.models.append(deploy_model)

        execution_time = time.time() - start_time

        print("\n" + "="*70)
        print("✅ MDE PIPELINE COMPLETE")
        print("="*70)

        # Calculate metrics
        self._calculate_metrics(execution_time)

        return {
            'models': self.models,
            'transformations': self.transformer.transformation_history,
            'metrics': self.pipeline_metrics,
            'execution_time': execution_time
        }

    def _generate_initial_model(self, agent: Any, description: str,
                                model_type: ModelType) -> FormalModel:
        """Generate initial model from project description."""
        from crewai import Task, Crew

        task = Task(
            description=f"Create formal requirements model for: {description}",
            agent=agent,
            expected_output="Formal requirements model"
        )

        try:
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            ai_output = str(crew.kickoff())

            # Parse into model
            model = FormalModel(
                model_id=f"{model_type.value}_initial",
                model_type=model_type,
                stage_name="Requirements"
            )

            # Simple parsing to extract requirements
            lines = [l.strip() for l in ai_output.split('\n') if l.strip()]
            elem_count = 0

            for line in lines:
                if any(kw in line.lower() for kw in ['requirement', 'req', 'must', 'should']):
                    elem_count += 1
                    element = ModelElement(
                        element_id=f"req_{elem_count}",
                        element_type='functional_requirement' if 'functional' in line.lower() else 'requirement',
                        name=line[:100],
                        properties={'priority': 'high' if 'must' in line.lower() else 'medium'}
                    )
                    model.add_element(element)

            # Add minimal elements if none found
            if len(model.elements) == 0:
                for i in range(3):
                    model.add_element(ModelElement(
                        element_id=f"req_{i+1}",
                        element_type='functional_requirement',
                        name=f"Requirement {i+1}",
                        properties={'priority': 'medium'}
                    ))

            print(f"   ✅ Generated {len(model.elements)} requirements")

            # Validate
            validation = self.validator.validate_model(model)
            passed = sum(1 for p, _ in validation.values() if p)
            print(f"   🔍 Validation: {passed}/{len(validation)} checks passed")

            return model

        except Exception as e:
            print(f"   ❌ Generation failed: {str(e)}")
            return FormalModel(
                model_id="req_fallback",
                model_type=model_type,
                stage_name="Requirements"
            )

    def _calculate_metrics(self, execution_time: float):
        """Calculate MDE pipeline metrics."""
        total_elements = sum(len(m.elements) for m in self.models)
        total_validations = sum(
            len(m.validation_results) for m in self.models
        )
        passed_validations = sum(
            sum(1 for v in m.validation_results.values() if v)
            for m in self.models
        )

        # Traceability completeness
        total_links = sum(len(m.traceability_links) for m in self.models[1:])  # Skip first
        expected_links = sum(len(m.elements) for m in self.models[1:])
        traceability_completeness = (total_links / expected_links
                                     if expected_links > 0 else 0)

        # Model quality
        avg_validation_rate = (passed_validations / total_validations
                              if total_validations > 0 else 0)

        self.pipeline_metrics = {
            'total_models': len(self.models),
            'total_elements': total_elements,
            'avg_elements_per_model': total_elements / len(self.models) if self.models else 0,
            'total_transformations': len(self.transformer.transformation_history),
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'validation_pass_rate': avg_validation_rate,
            'traceability_completeness': traceability_completeness,
            'execution_time': execution_time,
            'formalization_benefit': self._calculate_formalization_benefit()
        }

    def _calculate_formalization_benefit(self) -> float:
        """Calculate benefit of formalization."""
        # Based on validation pass rate and traceability
        if not self.models:
            return 0.0

        validation_benefit = self.pipeline_metrics.get('validation_pass_rate', 0) * 0.6
        traceability_benefit = self.pipeline_metrics.get('traceability_completeness', 0) * 0.4

        return validation_benefit + traceability_benefit


if __name__ == "__main__":
    print("PoC 4: AI-Assisted MDE - Framework Ready")
    print("Import this module into your Colab notebook to use")
