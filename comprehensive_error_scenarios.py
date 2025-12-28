"""
Comprehensive Error Scenario Catalog for SDLC Integration Paradox Analysis

This module provides an extensive catalog of realistic error scenarios that can occur
at each stage of the SDLC and how they propagate through the pipeline.
"""

# ============================================================================
# COMPREHENSIVE ERROR SCENARIO CATALOG
# ============================================================================

def get_comprehensive_error_scenarios():
    """
    Returns comprehensive error scenarios organized by stage and category.

    Categories:
    - Specification Errors: Ambiguity, incompleteness, inconsistency
    - Technical Errors: Architecture, implementation, configuration issues
    - Integration Errors: Interface mismatches, contract violations
    - Quality Errors: Performance, security, reliability issues
    - Process Errors: Communication, documentation, assumptions
    """

    scenarios = {
        # ====================================================================
        # REQUIREMENTS STAGE ERRORS
        # ====================================================================
        'requirements': [
            {
                'error_type': 'Specification Ambiguity',
                'severity': 'HIGH',
                'description': 'Vague or unclear requirement wording',
                'example': '"System should be secure" without defining security requirements',
                'propagation_probability': 0.95,
                'amplification_factor': 2.5,
                'cascades_to': ['design', 'implementation', 'testing', 'deployment']
            },
            {
                'error_type': 'Missing Non-Functional Requirements',
                'severity': 'CRITICAL',
                'description': 'Performance, scalability, or security requirements not specified',
                'example': 'No response time requirements for API endpoints',
                'propagation_probability': 0.90,
                'amplification_factor': 3.0,
                'cascades_to': ['design', 'implementation', 'testing']
            },
            {
                'error_type': 'Inconsistent Requirements',
                'severity': 'HIGH',
                'description': 'Conflicting requirements between different features',
                'example': 'REQ-001 requires real-time processing, REQ-010 requires batch processing of same data',
                'propagation_probability': 0.85,
                'amplification_factor': 2.0,
                'cascades_to': ['design', 'implementation']
            },
            {
                'error_type': 'Incomplete Edge Case Coverage',
                'severity': 'MEDIUM',
                'description': 'Missing specifications for boundary conditions and error cases',
                'example': 'No specification for handling null/empty inputs',
                'propagation_probability': 0.80,
                'amplification_factor': 1.8,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Incorrect Assumptions',
                'severity': 'HIGH',
                'description': 'Wrong assumptions about user behavior or system constraints',
                'example': 'Assuming all users have high-speed internet',
                'propagation_probability': 0.88,
                'amplification_factor': 2.2,
                'cascades_to': ['design', 'implementation', 'deployment']
            },
            {
                'error_type': 'Stakeholder Miscommunication',
                'severity': 'MEDIUM',
                'description': 'Requirements misunderstood due to poor stakeholder communication',
                'example': 'Business wants "immediate" (< 1 min), interpreted as "real-time" (< 100ms)',
                'propagation_probability': 0.75,
                'amplification_factor': 1.5,
                'cascades_to': ['design', 'implementation']
            },
            {
                'error_type': 'Missing Acceptance Criteria',
                'severity': 'MEDIUM',
                'description': 'No clear definition of done for requirements',
                'example': 'Requirement exists but no way to verify completion',
                'propagation_probability': 0.70,
                'amplification_factor': 1.6,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Regulatory Compliance Gap',
                'severity': 'CRITICAL',
                'description': 'Missing legal/regulatory requirements',
                'example': 'GDPR/HIPAA requirements not captured',
                'propagation_probability': 0.95,
                'amplification_factor': 3.5,
                'cascades_to': ['design', 'implementation', 'testing', 'deployment']
            }
        ],

        # ====================================================================
        # DESIGN STAGE ERRORS
        # ====================================================================
        'design': [
            {
                'error_type': 'Architecture-Requirements Mismatch',
                'severity': 'CRITICAL',
                'description': 'Design does not satisfy stated requirements',
                'example': 'Monolithic design for requirement specifying microservices',
                'propagation_probability': 0.92,
                'amplification_factor': 2.8,
                'cascades_to': ['implementation', 'testing', 'deployment']
            },
            {
                'error_type': 'Interface Contract Violation',
                'severity': 'HIGH',
                'description': 'API contracts inconsistent between components',
                'example': 'Component A expects JSON, Component B sends XML',
                'propagation_probability': 0.90,
                'amplification_factor': 2.5,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Security Design Flaw',
                'severity': 'CRITICAL',
                'description': 'Fundamental security vulnerability in architecture',
                'example': 'Storing passwords in plaintext, no authentication layer',
                'propagation_probability': 0.98,
                'amplification_factor': 4.0,
                'cascades_to': ['implementation', 'testing', 'deployment']
            },
            {
                'error_type': 'Scalability Bottleneck',
                'severity': 'HIGH',
                'description': 'Design includes components that won\'t scale',
                'example': 'Single database instance for distributed system',
                'propagation_probability': 0.85,
                'amplification_factor': 2.3,
                'cascades_to': ['implementation', 'deployment']
            },
            {
                'error_type': 'Tight Coupling',
                'severity': 'MEDIUM',
                'description': 'Components overly dependent on each other',
                'example': 'Direct database access from UI layer',
                'propagation_probability': 0.75,
                'amplification_factor': 1.7,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Missing Error Handling Strategy',
                'severity': 'HIGH',
                'description': 'No defined approach for error handling and recovery',
                'example': 'No specification for retry logic, circuit breakers',
                'propagation_probability': 0.88,
                'amplification_factor': 2.1,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Data Model Inconsistency',
                'severity': 'HIGH',
                'description': 'Database schema conflicts with domain model',
                'example': 'Entity relationships don\'t match business logic',
                'propagation_probability': 0.82,
                'amplification_factor': 2.0,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Performance Anti-Pattern',
                'severity': 'MEDIUM',
                'description': 'Design includes known performance anti-patterns',
                'example': 'N+1 queries, excessive synchronous calls',
                'propagation_probability': 0.78,
                'amplification_factor': 1.8,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Incomplete API Specification',
                'severity': 'MEDIUM',
                'description': 'API endpoints lack complete specification',
                'example': 'Missing error response codes, request/response schemas',
                'propagation_probability': 0.80,
                'amplification_factor': 1.9,
                'cascades_to': ['implementation', 'testing']
            },
            {
                'error_type': 'Technology Stack Mismatch',
                'severity': 'HIGH',
                'description': 'Chosen technologies incompatible with requirements',
                'example': 'Using synchronous framework for real-time requirements',
                'propagation_probability': 0.86,
                'amplification_factor': 2.4,
                'cascades_to': ['implementation', 'deployment']
            }
        ],

        # ====================================================================
        # IMPLEMENTATION STAGE ERRORS
        # ====================================================================
        'implementation': [
            {
                'error_type': 'Design-Code Divergence',
                'severity': 'HIGH',
                'description': 'Implementation deviates from design specifications',
                'example': 'Code structure doesn\'t match designed architecture',
                'propagation_probability': 0.87,
                'amplification_factor': 2.2,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Missing Input Validation',
                'severity': 'CRITICAL',
                'description': 'User inputs not validated or sanitized',
                'example': 'SQL injection vulnerability, XSS attacks possible',
                'propagation_probability': 0.95,
                'amplification_factor': 3.5,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Race Condition',
                'severity': 'HIGH',
                'description': 'Concurrent access to shared resources not properly synchronized',
                'example': 'Multiple threads modifying same data without locks',
                'propagation_probability': 0.82,
                'amplification_factor': 2.5,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Memory Leak',
                'severity': 'HIGH',
                'description': 'Resources not properly released',
                'example': 'Database connections, file handles not closed',
                'propagation_probability': 0.80,
                'amplification_factor': 2.3,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Inadequate Error Handling',
                'severity': 'MEDIUM',
                'description': 'Exceptions not caught or handled properly',
                'example': 'Swallowing exceptions, exposing stack traces to users',
                'propagation_probability': 0.75,
                'amplification_factor': 1.8,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Hardcoded Credentials',
                'severity': 'CRITICAL',
                'description': 'Secrets and credentials in source code',
                'example': 'API keys, passwords in code or config files',
                'propagation_probability': 0.99,
                'amplification_factor': 4.5,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Inefficient Algorithm',
                'severity': 'MEDIUM',
                'description': 'Using O(n²) when O(n log n) is possible',
                'example': 'Nested loops for operations that could be optimized',
                'propagation_probability': 0.72,
                'amplification_factor': 1.6,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Dependency Version Conflict',
                'severity': 'HIGH',
                'description': 'Incompatible library versions',
                'example': 'Package A requires LibX v1, Package B requires LibX v2',
                'propagation_probability': 0.85,
                'amplification_factor': 2.0,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Missing Logging',
                'severity': 'MEDIUM',
                'description': 'Insufficient logging for debugging',
                'example': 'No logs for critical operations or errors',
                'propagation_probability': 0.70,
                'amplification_factor': 1.5,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'API Rate Limit Violation',
                'severity': 'MEDIUM',
                'description': 'Code exceeds external API rate limits',
                'example': 'Making 1000 API calls/sec when limit is 100/sec',
                'propagation_probability': 0.78,
                'amplification_factor': 1.9,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Null Pointer Dereference',
                'severity': 'HIGH',
                'description': 'Accessing null/undefined values',
                'example': 'Not checking for null before accessing object properties',
                'propagation_probability': 0.83,
                'amplification_factor': 2.1,
                'cascades_to': ['testing', 'deployment']
            },
            {
                'error_type': 'Type Safety Violation',
                'severity': 'MEDIUM',
                'description': 'Type mismatches or unsafe type coercion',
                'example': 'Treating string as integer, implicit type conversions',
                'propagation_probability': 0.74,
                'amplification_factor': 1.7,
                'cascades_to': ['testing']
            }
        ],

        # ====================================================================
        # TESTING STAGE ERRORS
        # ====================================================================
        'testing': [
            {
                'error_type': 'Insufficient Test Coverage',
                'severity': 'HIGH',
                'description': 'Critical paths not tested',
                'example': 'Only 40% code coverage, missing edge cases',
                'propagation_probability': 0.88,
                'amplification_factor': 2.4,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'False Positive Tests',
                'severity': 'HIGH',
                'description': 'Tests pass but functionality is broken',
                'example': 'Mock objects hide real integration issues',
                'propagation_probability': 0.92,
                'amplification_factor': 3.0,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Missing Security Tests',
                'severity': 'CRITICAL',
                'description': 'No penetration testing or vulnerability scanning',
                'example': 'No tests for SQL injection, XSS, CSRF',
                'propagation_probability': 0.95,
                'amplification_factor': 3.8,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Environment-Specific Tests',
                'severity': 'HIGH',
                'description': 'Tests only work in specific environment',
                'example': 'Tests pass on developer machine, fail in CI/CD',
                'propagation_probability': 0.85,
                'amplification_factor': 2.3,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Missing Performance Tests',
                'severity': 'MEDIUM',
                'description': 'No load or stress testing',
                'example': 'No testing under concurrent users or high load',
                'propagation_probability': 0.80,
                'amplification_factor': 2.0,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Flaky Tests',
                'severity': 'MEDIUM',
                'description': 'Tests intermittently fail without code changes',
                'example': 'Race conditions in tests, time-dependent assertions',
                'propagation_probability': 0.70,
                'amplification_factor': 1.6,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Test Data Contamination',
                'severity': 'MEDIUM',
                'description': 'Tests affect each other through shared state',
                'example': 'Database not reset between tests',
                'propagation_probability': 0.75,
                'amplification_factor': 1.8,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Missing Integration Tests',
                'severity': 'HIGH',
                'description': 'No testing of component interactions',
                'example': 'Unit tests pass but components fail to integrate',
                'propagation_probability': 0.87,
                'amplification_factor': 2.5,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Inadequate Error Scenario Testing',
                'severity': 'MEDIUM',
                'description': 'Only happy path tested',
                'example': 'No tests for network failures, timeouts, invalid inputs',
                'propagation_probability': 0.82,
                'amplification_factor': 2.1,
                'cascades_to': ['deployment']
            },
            {
                'error_type': 'Missing Regression Tests',
                'severity': 'MEDIUM',
                'description': 'No tests to prevent reintroduction of bugs',
                'example': 'Fixed bugs reappear in later releases',
                'propagation_probability': 0.73,
                'amplification_factor': 1.7,
                'cascades_to': ['deployment']
            }
        ],

        # ====================================================================
        # DEPLOYMENT STAGE ERRORS
        # ====================================================================
        'deployment': [
            {
                'error_type': 'Configuration Drift',
                'severity': 'CRITICAL',
                'description': 'Production config differs from tested config',
                'example': 'Different database connection strings, API endpoints',
                'propagation_probability': 0.90,
                'amplification_factor': 3.5,
                'cascades_to': []  # Terminal stage
            },
            {
                'error_type': 'Missing Environment Variables',
                'severity': 'CRITICAL',
                'description': 'Required environment variables not set',
                'example': 'API keys, database URLs not configured',
                'propagation_probability': 0.88,
                'amplification_factor': 3.2,
                'cascades_to': []
            },
            {
                'error_type': 'Insufficient Resource Allocation',
                'severity': 'HIGH',
                'description': 'Not enough CPU/memory/disk for production load',
                'example': 'Container limited to 512MB when app needs 2GB',
                'propagation_probability': 0.85,
                'amplification_factor': 2.8,
                'cascades_to': []
            },
            {
                'error_type': 'Missing Monitoring',
                'severity': 'HIGH',
                'description': 'No observability into production system',
                'example': 'No metrics, logs, or alerts configured',
                'propagation_probability': 0.80,
                'amplification_factor': 2.5,
                'cascades_to': []
            },
            {
                'error_type': 'No Rollback Plan',
                'severity': 'CRITICAL',
                'description': 'Cannot revert to previous version if deployment fails',
                'example': 'No blue-green deployment, no version pinning',
                'propagation_probability': 0.92,
                'amplification_factor': 3.8,
                'cascades_to': []
            },
            {
                'error_type': 'Network Security Misconfiguration',
                'severity': 'CRITICAL',
                'description': 'Firewall rules, VPC settings incorrect',
                'example': 'Database exposed to public internet',
                'propagation_probability': 0.95,
                'amplification_factor': 4.2,
                'cascades_to': []
            },
            {
                'error_type': 'Certificate Expiration',
                'severity': 'HIGH',
                'description': 'SSL/TLS certificates expired or about to expire',
                'example': 'HTTPS certificate expired, breaking service',
                'propagation_probability': 0.78,
                'amplification_factor': 2.2,
                'cascades_to': []
            },
            {
                'error_type': 'Database Migration Failure',
                'severity': 'CRITICAL',
                'description': 'Schema migration fails in production',
                'example': 'Migration script works in dev, fails in production',
                'propagation_probability': 0.87,
                'amplification_factor': 3.3,
                'cascades_to': []
            },
            {
                'error_type': 'Dependency Service Unavailable',
                'severity': 'HIGH',
                'description': 'External services not accessible',
                'example': 'Third-party API unreachable from production network',
                'propagation_probability': 0.83,
                'amplification_factor': 2.6,
                'cascades_to': []
            },
            {
                'error_type': 'Insufficient Backup Strategy',
                'severity': 'HIGH',
                'description': 'No backup or disaster recovery plan',
                'example': 'No database backups, no point-in-time recovery',
                'propagation_probability': 0.75,
                'amplification_factor': 2.4,
                'cascades_to': []
            }
        ]
    }

    return scenarios


def simulate_comprehensive_error_cascade(metrics, verbose=True):
    """
    Simulate comprehensive error propagation through the SDLC pipeline.

    Args:
        metrics: IntegrationMetrics instance to record errors
        verbose: If True, print detailed output

    Returns:
        Dictionary with cascade analysis results
    """

    if verbose:
        print("\n" + "="*70)
        print("  COMPREHENSIVE ERROR PROPAGATION ANALYSIS")
        print("="*70)

    scenarios = get_comprehensive_error_scenarios()

    # Agent mapping
    stage_to_agent = {
        'requirements': 'Requirements Agent',
        'design': 'Design Agent',
        'implementation': 'Implementation Agent',
        'testing': 'Testing Agent',
        'deployment': 'Deployment Agent'
    }

    agent_order = ['requirements', 'design', 'implementation', 'testing', 'deployment']

    cascade_results = {
        'total_errors': 0,
        'total_amplified': 0,
        'total_contained': 0,
        'by_severity': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
        'by_stage': {},
        'cascade_chains': []
    }

    # Process each stage
    for stage_idx, stage in enumerate(agent_order):
        stage_errors = scenarios[stage]

        if verbose:
            print(f"\n{'─'*70}")
            print(f"📍 STAGE: {stage.upper()} ({len(stage_errors)} error scenarios)")
            print(f"{'─'*70}")

        cascade_results['by_stage'][stage] = {
            'total_errors': len(stage_errors),
            'amplified': 0,
            'contained': 0
        }

        for error in stage_errors:
            cascade_results['total_errors'] += 1
            cascade_results['by_severity'][error['severity']] += 1

            # Determine if error amplifies based on propagation probability
            import random
            amplified = random.random() < error['propagation_probability']

            if amplified:
                cascade_results['total_amplified'] += 1
                cascade_results['by_stage'][stage]['amplified'] += 1
                status = "🔴 AMPLIFIED"
            else:
                cascade_results['total_contained'] += 1
                cascade_results['by_stage'][stage]['contained'] += 1
                status = "🟢 CONTAINED"

            # Record error propagation for each target stage
            source_agent = stage_to_agent[stage]

            if error['cascades_to']:
                for target_stage in error['cascades_to']:
                    target_agent = stage_to_agent[target_stage]

                    metrics.record_error_propagation(
                        source_agent=source_agent,
                        target_agent=target_agent,
                        error_type=error['error_type'],
                        amplified=amplified
                    )

                    # Track cascade chain
                    cascade_results['cascade_chains'].append({
                        'source': stage,
                        'target': target_stage,
                        'error_type': error['error_type'],
                        'severity': error['severity'],
                        'amplified': amplified,
                        'amplification_factor': error['amplification_factor'] if amplified else 1.0
                    })

            if verbose:
                print(f"\n{status} [{error['severity']}] {error['error_type']}")
                print(f"  ├─ Description: {error['description']}")
                print(f"  ├─ Example: {error['example']}")
                print(f"  ├─ Propagation Probability: {error['propagation_probability']:.0%}")
                print(f"  ├─ Amplification Factor: {error['amplification_factor']}x")

                if error['cascades_to']:
                    cascade_targets = ' → '.join([s.title() for s in error['cascades_to']])
                    print(f"  └─ Cascades to: {cascade_targets}")
                else:
                    print(f"  └─ Terminal error (deployment stage)")

    # Summary statistics
    if verbose:
        print("\n" + "="*70)
        print("  CASCADE SUMMARY")
        print("="*70)
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  • Total Error Scenarios: {cascade_results['total_errors']}")
        print(f"  • Amplified Errors: {cascade_results['total_amplified']} ({cascade_results['total_amplified']/cascade_results['total_errors']*100:.1f}%)")
        print(f"  • Contained Errors: {cascade_results['total_contained']} ({cascade_results['total_contained']/cascade_results['total_errors']*100:.1f}%)")

        print(f"\n⚠️  BY SEVERITY:")
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = cascade_results['by_severity'].get(severity, 0)
            if count > 0:
                print(f"  • {severity:8s}: {count:2d} errors")

        print(f"\n📍 BY STAGE:")
        for stage in agent_order:
            stats = cascade_results['by_stage'][stage]
            print(f"  • {stage.title():16s}: {stats['total_errors']} total, "
                  f"{stats['amplified']} amplified, {stats['contained']} contained")

        # Most dangerous cascade chains
        print(f"\n🔥 TOP 10 DANGEROUS CASCADE CHAINS:")
        sorted_chains = sorted(
            cascade_results['cascade_chains'],
            key=lambda x: (
                {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['severity']],
                x['amplification_factor']
            ),
            reverse=True
        )

        for i, chain in enumerate(sorted_chains[:10], 1):
            print(f"\n  {i}. {chain['source'].title()} → {chain['target'].title()}")
            print(f"     Error: {chain['error_type']}")
            print(f"     Severity: {chain['severity']}, "
                  f"Amplification: {chain['amplification_factor']}x, "
                  f"Status: {'AMPLIFIED' if chain['amplified'] else 'CONTAINED'}")

        print("\n" + "="*70)
        print("✅ Comprehensive error propagation analysis complete!")
        print("="*70)

    return cascade_results


if __name__ == "__main__":
    # Example usage
    from datetime import datetime

    class IntegrationMetrics:
        def __init__(self):
            self.error_propagation = []

        def record_error_propagation(self, source_agent, target_agent, error_type, amplified):
            self.error_propagation.append({
                'timestamp': datetime.now().isoformat(),
                'source': source_agent,
                'target': target_agent,
                'error_type': error_type,
                'amplified': amplified
            })

    metrics = IntegrationMetrics()
    results = simulate_comprehensive_error_cascade(metrics, verbose=True)
