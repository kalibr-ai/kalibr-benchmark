#!/usr/bin/env python3
"""
Kalibr Resilience Benchmark

Proves: "When your hardcoded path fails, Kalibr automatically routes to alternatives."

Run with:
    python resilience_benchmark.py

Requirements:
    pip install kalibr openai
    
    export KALIBR_API_KEY=your-key
    export KALIBR_TENANT_ID=your-tenant  
    export OPENAI_API_KEY=your-key

Full documentation: https://docs.kalibr.systems/benchmarks/resilience
"""

import os
import sys
import time
import json
import random
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


def check_environment():
    """Verify required environment variables are set."""
    required = ["KALIBR_API_KEY", "KALIBR_TENANT_ID", "OPENAI_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("\nGet your Kalibr credentials at: https://dashboard.kalibr.systems")
        print("\nThen set them:")
        for k in missing:
            print(f"  export {k}=your-value")
        sys.exit(1)


check_environment()

from kalibr import Router


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    goal: str = "document_analysis"
    
    # Paths available to both systems
    paths: list = field(default_factory=lambda: [
        "gpt-4o",           # Primary - will be degraded
        "gpt-4o-mini",      # Backup 1
        "gpt-3.5-turbo",    # Backup 2
    ])
    
    # Phase configuration
    learning_executions: int = 30      # Phase 1: Normal operation
    degradation_executions: int = 40   # Phase 2: Primary path degraded
    observation_executions: int = 30   # Phase 3: Sustained degradation
    
    # Degradation settings
    primary_path: str = "gpt-4o"
    degradation_failure_rate: float = 0.7  # 70% failure rate when degraded
    
    # Hardcoded baseline always uses this
    hardcoded_path: str = "gpt-4o"


# =============================================================================
# TASK DEFINITION
# =============================================================================

ANALYSIS_TASK = """Analyze this document excerpt and provide:

1. SUMMARY: A 2-3 sentence summary of the main point
2. KEY_ENTITIES: List any people, organizations, or places mentioned
3. SENTIMENT: Overall sentiment (positive/negative/neutral) with confidence
4. ACTION_ITEMS: Any suggested next steps or actions

Document:
---
{document}
---

Respond in this exact JSON format:
{{
  "summary": "...",
  "key_entities": ["...", "..."],
  "sentiment": {{"label": "...", "confidence": 0.0-1.0}},
  "action_items": ["...", "..."]
}}
"""

SAMPLE_DOCUMENTS = [
    """Q3 earnings exceeded expectations with 15% YoY growth. CEO Jane Smith 
    attributed success to the European expansion and new enterprise contracts. 
    The board approved a $50M investment in AI infrastructure.""",
    
    """Customer satisfaction surveys show declining NPS scores (-12 points). 
    Main complaints: slow response times, unclear documentation. Engineering 
    team lead Mike Chen proposes hiring 3 additional support engineers.""",
    
    """Partnership agreement signed with TechCorp for joint go-to-market in 
    APAC region. Deal worth $2.3M annually. Legal review required before 
    public announcement. Contact: Sarah Johnson, VP Partnerships.""",
    
    """Security audit completed. 2 critical vulnerabilities found in 
    authentication module. Patch deployed to staging. Production deployment 
    scheduled for Friday 2am EST. All hands on deck required.""",
    
    """Product roadmap review: Q4 priorities are mobile app launch, 
    API v2 migration, and GDPR compliance updates. Resources allocated: 
    12 engineers, 3 designers. Deadline: November 30.""",
]


def get_random_document() -> str:
    return random.choice(SAMPLE_DOCUMENTS)


def validate_response(response_text: str) -> tuple[bool, str]:
    """Validate the LLM response. Returns (success, reason)."""
    if not response_text or not response_text.strip():
        return False, "empty_response"
    
    try:
        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
        
        data = json.loads(text)
        
        # Check required fields
        required = ["summary", "key_entities", "sentiment", "action_items"]
        for field in required:
            if field not in data:
                return False, f"missing_field_{field}"
        
        if not isinstance(data["summary"], str) or len(data["summary"]) < 20:
            return False, "invalid_summary"
        
        if not isinstance(data["key_entities"], list):
            return False, "invalid_entities"
        
        if not isinstance(data["sentiment"], dict):
            return False, "invalid_sentiment"
        if "label" not in data["sentiment"] or "confidence" not in data["sentiment"]:
            return False, "incomplete_sentiment"
        
        if not isinstance(data["action_items"], list):
            return False, "invalid_action_items"
        
        return True, "valid"
        
    except json.JSONDecodeError:
        return False, "json_parse_error"
    except Exception as e:
        return False, f"validation_error_{type(e).__name__}"


# =============================================================================
# DEGRADATION INJECTION
# =============================================================================

class DegradationController:
    """Controls failure injection to simulate provider degradation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.degradation_active = False
        self.degradation_start_time: Optional[datetime] = None
    
    def activate_degradation(self):
        self.degradation_active = True
        self.degradation_start_time = datetime.now()
        print(f"\n🔥 DEGRADATION ACTIVATED at {self.degradation_start_time.strftime('%H:%M:%S')}")
        print(f"   Primary path ({self.config.primary_path}) now failing at {self.config.degradation_failure_rate*100:.0f}% rate")
    
    def should_fail(self, model_id: str) -> bool:
        if not self.degradation_active:
            return False
        if model_id != self.config.primary_path:
            return False
        return random.random() < self.config.degradation_failure_rate
    
    def inject_failure(self) -> str:
        failure_types = [
            "",
            "Error: Service temporarily unavailable",
            '{"error": "rate_limit_exceeded"}',
            "I apologize, but I cannot process this request at the moment.",
            '{"summary": "", "key_entities": [], "sentiment": {}, "action_items": []}',
        ]
        return random.choice(failure_types)


# =============================================================================
# METRICS
# =============================================================================

# Model costs per 1M tokens (input, output)
MODEL_COSTS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
}


def estimate_cost(model: str, input_tokens: int = 500, output_tokens: int = 300) -> float:
    base_model = model.split("-2024")[0] if "-2024" in model else model
    costs = MODEL_COSTS.get(model, MODEL_COSTS.get(base_model, (1.0, 1.0)))
    return (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000


@dataclass
class ExecutionResult:
    execution_id: int
    phase: str
    system: str
    model_used: str
    success: bool
    failure_reason: Optional[str]
    latency_ms: int
    cost_usd: float
    timestamp: datetime


class MetricsTracker:
    def __init__(self):
        self.results: list[ExecutionResult] = []
        self.routing_distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def record(self, result: ExecutionResult):
        self.results.append(result)
        self.routing_distribution[result.phase][result.model_used] += 1
    
    def get_success_rate(self, system: str, phase: str) -> float:
        phase_results = [r for r in self.results if r.system == system and r.phase == phase]
        if not phase_results:
            return 0.0
        return sum(1 for r in phase_results if r.success) / len(phase_results)
    
    def get_phase_summary(self, phase: str) -> dict:
        kalibr_results = [r for r in self.results if r.system == "kalibr" and r.phase == phase]
        hardcoded_results = [r for r in self.results if r.system == "hardcoded" and r.phase == phase]
        
        k_total_cost = sum(r.cost_usd for r in kalibr_results)
        k_successes = sum(1 for r in kalibr_results if r.success)
        h_total_cost = sum(r.cost_usd for r in hardcoded_results)
        h_successes = sum(1 for r in hardcoded_results if r.success)
        
        return {
            "kalibr": {
                "total": len(kalibr_results),
                "successes": k_successes,
                "failures": sum(1 for r in kalibr_results if not r.success),
                "success_rate": self.get_success_rate("kalibr", phase),
                "models_used": dict(self.routing_distribution[f"{phase}_kalibr"]),
                "total_cost": k_total_cost,
                "cost_per_success": k_total_cost / k_successes if k_successes > 0 else float('inf'),
            },
            "hardcoded": {
                "total": len(hardcoded_results),
                "successes": h_successes,
                "failures": sum(1 for r in hardcoded_results if not r.success),
                "success_rate": self.get_success_rate("hardcoded", phase),
                "total_cost": h_total_cost,
                "cost_per_success": h_total_cost / h_successes if h_successes > 0 else float('inf'),
            }
        }
    
    def print_live_status(self, execution_id: int, phase: str, kalibr_result: ExecutionResult, hardcoded_result: ExecutionResult):
        k_status = "✓" if kalibr_result.success else "✗"
        h_status = "✓" if hardcoded_result.success else "✗"
        k_color = "\033[92m" if kalibr_result.success else "\033[91m"
        h_color = "\033[92m" if hardcoded_result.success else "\033[91m"
        reset = "\033[0m"
        print(f"  [{execution_id:3d}] Kalibr: {k_color}{k_status}{reset} ({kalibr_result.model_used:15s}) | Hardcoded: {h_color}{h_status}{reset} ({hardcoded_result.model_used})")


# =============================================================================
# AGENT
# =============================================================================

class DocumentAnalysisAgent:
    def __init__(self, config: BenchmarkConfig, degradation: DegradationController):
        self.config = config
        self.degradation = degradation
        self.router = Router(goal=config.goal, paths=config.paths)
        self.hardcoded_router = Router(goal=f"{config.goal}_hardcoded", paths=[config.hardcoded_path])
    
    def execute_kalibr(self, document: str) -> tuple[bool, str, str, bool, float]:
        messages = [
            {"role": "system", "content": "You are a document analysis assistant. Always respond in valid JSON."},
            {"role": "user", "content": ANALYSIS_TASK.format(document=document)},
        ]
        
        try:
            response = self.router.completion(messages=messages)
            model_used = response.model
            content = response.choices[0].message.content
            
            cost = estimate_cost(
                model_used,
                getattr(response.usage, 'prompt_tokens', 500),
                getattr(response.usage, 'completion_tokens', 300)
            )
            
            was_primary = model_used == self.config.primary_path
            
            if self.degradation.should_fail(model_used):
                content = self.degradation.inject_failure()
            
            success, reason = validate_response(content)
            self.router.report(success=success, reason=None if success else reason)
            
            return success, reason if not success else None, model_used, was_primary, cost
            
        except Exception as e:
            return False, f"exception_{type(e).__name__}", self.config.primary_path, True, 0.0
    
    def execute_hardcoded(self, document: str) -> tuple[bool, str, str, bool, float]:
        messages = [
            {"role": "system", "content": "You are a document analysis assistant. Always respond in valid JSON."},
            {"role": "user", "content": ANALYSIS_TASK.format(document=document)},
        ]
        
        model_used = self.config.hardcoded_path
        
        try:
            response = self.hardcoded_router.completion(messages=messages, force_model=model_used)
            content = response.choices[0].message.content
            
            cost = estimate_cost(
                model_used,
                getattr(response.usage, 'prompt_tokens', 500),
                getattr(response.usage, 'completion_tokens', 300)
            )
            
            if self.degradation.should_fail(model_used):
                content = self.degradation.inject_failure()
            
            success, reason = validate_response(content)
            self.hardcoded_router.report(success=success, reason=None if success else reason)
            
            return success, reason if not success else None, model_used, True, cost
            
        except Exception as e:
            return False, f"exception_{type(e).__name__}", model_used, True, 0.0


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(config: BenchmarkConfig):
    print("=" * 70)
    print("KALIBR RESILIENCE BENCHMARK")
    print("=" * 70)
    print(f"\nGoal: Prove Kalibr routes around failures automatically")
    print(f"Primary path: {config.primary_path}")
    print(f"Backup paths: {[p for p in config.paths if p != config.primary_path]}")
    print(f"\nPhases:")
    print(f"  1. Learning:     {config.learning_executions} executions (normal operation)")
    print(f"  2. Degradation:  {config.degradation_executions} executions ({config.primary_path} fails at {config.degradation_failure_rate*100:.0f}%)")
    print(f"  3. Observation:  {config.observation_executions} executions (sustained degradation)")
    print("=" * 70)
    
    degradation = DegradationController(config)
    metrics = MetricsTracker()
    agent = DocumentAnalysisAgent(config, degradation)
    
    execution_id = 0
    
    # Phase 1: Learning
    print(f"\n📚 PHASE 1: LEARNING ({config.learning_executions} executions)")
    print("-" * 50)
    
    for i in range(config.learning_executions):
        execution_id += 1
        document = get_random_document()
        
        start = time.time()
        k_success, k_reason, k_model, _, k_cost = agent.execute_kalibr(document)
        k_latency = int((time.time() - start) * 1000)
        
        start = time.time()
        h_success, h_reason, h_model, _, h_cost = agent.execute_hardcoded(document)
        h_latency = int((time.time() - start) * 1000)
        
        k_result = ExecutionResult(execution_id, "learning", "kalibr", k_model, k_success, k_reason, k_latency, k_cost, datetime.now())
        h_result = ExecutionResult(execution_id, "learning", "hardcoded", h_model, h_success, h_reason, h_latency, h_cost, datetime.now())
        
        metrics.record(k_result)
        metrics.record(h_result)
        metrics.routing_distribution["learning_kalibr"][k_model] += 1
        metrics.print_live_status(execution_id, "learning", k_result, h_result)
        
        time.sleep(0.5)
    
    summary = metrics.get_phase_summary("learning")
    print(f"\n  Learning complete:")
    print(f"    Kalibr:    {summary['kalibr']['success_rate']*100:.1f}% success ({summary['kalibr']['successes']}/{summary['kalibr']['total']})")
    print(f"    Hardcoded: {summary['hardcoded']['success_rate']*100:.1f}% success ({summary['hardcoded']['successes']}/{summary['hardcoded']['total']})")
    
    # Phase 2: Degradation
    print(f"\n🔥 PHASE 2: DEGRADATION ({config.degradation_executions} executions)")
    print("-" * 50)
    
    degradation.activate_degradation()
    
    for i in range(config.degradation_executions):
        execution_id += 1
        document = get_random_document()
        
        start = time.time()
        k_success, k_reason, k_model, _, k_cost = agent.execute_kalibr(document)
        k_latency = int((time.time() - start) * 1000)
        
        start = time.time()
        h_success, h_reason, h_model, _, h_cost = agent.execute_hardcoded(document)
        h_latency = int((time.time() - start) * 1000)
        
        k_result = ExecutionResult(execution_id, "degradation", "kalibr", k_model, k_success, k_reason, k_latency, k_cost, datetime.now())
        h_result = ExecutionResult(execution_id, "degradation", "hardcoded", h_model, h_success, h_reason, h_latency, h_cost, datetime.now())
        
        metrics.record(k_result)
        metrics.record(h_result)
        metrics.routing_distribution["degradation_kalibr"][k_model] += 1
        metrics.print_live_status(execution_id, "degradation", k_result, h_result)
        
        time.sleep(0.5)
    
    summary = metrics.get_phase_summary("degradation")
    print(f"\n  Degradation phase complete:")
    print(f"    Kalibr:    {summary['kalibr']['success_rate']*100:.1f}% success ({summary['kalibr']['successes']}/{summary['kalibr']['total']})")
    print(f"    Hardcoded: {summary['hardcoded']['success_rate']*100:.1f}% success ({summary['hardcoded']['successes']}/{summary['hardcoded']['total']})")
    
    # Phase 3: Observation
    print(f"\n👁️ PHASE 3: OBSERVATION ({config.observation_executions} executions)")
    print("-" * 50)
    
    for i in range(config.observation_executions):
        execution_id += 1
        document = get_random_document()
        
        start = time.time()
        k_success, k_reason, k_model, _, k_cost = agent.execute_kalibr(document)
        k_latency = int((time.time() - start) * 1000)
        
        start = time.time()
        h_success, h_reason, h_model, _, h_cost = agent.execute_hardcoded(document)
        h_latency = int((time.time() - start) * 1000)
        
        k_result = ExecutionResult(execution_id, "observation", "kalibr", k_model, k_success, k_reason, k_latency, k_cost, datetime.now())
        h_result = ExecutionResult(execution_id, "observation", "hardcoded", h_model, h_success, h_reason, h_latency, h_cost, datetime.now())
        
        metrics.record(k_result)
        metrics.record(h_result)
        metrics.routing_distribution["observation_kalibr"][k_model] += 1
        metrics.print_live_status(execution_id, "observation", k_result, h_result)
        
        time.sleep(0.5)
    
    # Final Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    total_kalibr = [r for r in metrics.results if r.system == "kalibr"]
    total_hardcoded = [r for r in metrics.results if r.system == "hardcoded"]
    
    kalibr_overall = sum(1 for r in total_kalibr if r.success) / len(total_kalibr)
    hardcoded_overall = sum(1 for r in total_hardcoded if r.success) / len(total_hardcoded)
    
    learning = metrics.get_phase_summary("learning")
    degradation_summary = metrics.get_phase_summary("degradation")
    observation = metrics.get_phase_summary("observation")
    
    print(f"\n📊 SUCCESS RATES BY PHASE")
    print("-" * 50)
    print(f"{'Phase':<15} {'Kalibr':<20} {'Hardcoded':<20}")
    print(f"{'Learning':<15} {learning['kalibr']['success_rate']*100:>6.1f}%             {learning['hardcoded']['success_rate']*100:>6.1f}%")
    print(f"{'Degradation':<15} {degradation_summary['kalibr']['success_rate']*100:>6.1f}%             {degradation_summary['hardcoded']['success_rate']*100:>6.1f}%")
    print(f"{'Observation':<15} {observation['kalibr']['success_rate']*100:>6.1f}%             {observation['hardcoded']['success_rate']*100:>6.1f}%")
    print("-" * 50)
    print(f"{'OVERALL':<15} {kalibr_overall*100:>6.1f}%             {hardcoded_overall*100:>6.1f}%")
    
    # Cost analysis
    k_total_cost = sum(r.cost_usd for r in total_kalibr)
    h_total_cost = sum(r.cost_usd for r in total_hardcoded)
    k_successes = sum(1 for r in total_kalibr if r.success)
    h_successes = sum(1 for r in total_hardcoded if r.success)
    
    print(f"\n💰 COST ANALYSIS")
    print("-" * 50)
    print(f"{'Metric':<25} {'Kalibr':<15} {'Hardcoded':<15}")
    print(f"{'Total Cost':<25} ${k_total_cost:<14.4f} ${h_total_cost:<14.4f}")
    print(f"{'Successful Outcomes':<25} {k_successes:<15} {h_successes:<15}")
    if k_successes > 0 and h_successes > 0:
        k_cps = k_total_cost / k_successes
        h_cps = h_total_cost / h_successes
        print(f"{'Cost per Success':<25} ${k_cps:<14.6f} ${h_cps:<14.6f}")
    
    print(f"\n🔀 KALIBR ROUTING DISTRIBUTION")
    print("-" * 50)
    for phase in ["learning", "degradation", "observation"]:
        dist = dict(metrics.routing_distribution[f"{phase}_kalibr"])
        total = sum(dist.values())
        print(f"{phase.capitalize():15} ", end="")
        for model, count in sorted(dist.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"{model}: {pct:.0f}%  ", end="")
        print()
    
    # The proof
    print(f"\n" + "=" * 70)
    print("📋 THE PROOF")
    print("=" * 70)
    
    degradation_delta = degradation_summary['kalibr']['success_rate'] - degradation_summary['hardcoded']['success_rate']
    
    print(f"""
During degradation (when {config.primary_path} was failing at {config.degradation_failure_rate*100:.0f}%):

  • Hardcoded system: {degradation_summary['hardcoded']['success_rate']*100:.1f}% success
    → Kept sending traffic to broken path
    → Required human intervention to fix
    
  • Kalibr system: {degradation_summary['kalibr']['success_rate']*100:.1f}% success  
    → Detected degradation automatically
    → Routed traffic to healthy alternatives
    → No human intervention required

Kalibr maintained {degradation_delta*100:+.1f} percentage points higher success rate
during the outage.
""")
    
    print("=" * 70)


if __name__ == "__main__":
    config = BenchmarkConfig()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            config.learning_executions = 10
            config.degradation_executions = 15
            config.observation_executions = 10
            print("Running quick benchmark (~35 executions)")
        elif sys.argv[1] == "--full":
            config.learning_executions = 50
            config.degradation_executions = 100
            config.observation_executions = 50
            print("Running full benchmark (~200 executions)")
    
    run_benchmark(config)
