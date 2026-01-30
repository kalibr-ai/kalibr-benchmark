#!/usr/bin/env python3
"""
Kalibr Resilience Benchmark

Proves: When your best execution path degrades, Kalibr routes around it automatically.
Hardcoded systems keep failing until a human intervenes.

This benchmark uses a multi-step research agent with real tool calls (Serper, Tavily)
to demonstrate execution path routing—not just model selection.
"""

import os
import sys
import json
import time
import random
import re
import httpx
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# =============================================================================
# ENVIRONMENT
# =============================================================================

REQUIRED_ENV = [
    "KALIBR_API_KEY",
    "KALIBR_TENANT_ID",
    "OPENAI_API_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY"
]

def check_environment():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("\nThis benchmark requires:")
        print("  - Kalibr API key and tenant ID (https://kalibr.dev)")
        print("  - OpenAI API key")
        print("  - Two search APIs (Serper + Tavily) to demonstrate path routing")
        print("\nSet them with:")
        for k in missing:
            print(f"  export {k}=your-value")
        sys.exit(1)

check_environment()

import logging
logging.getLogger("kalibr").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from kalibr import register_path, decide, report_outcome
from openai import OpenAI

client = OpenAI()
SERPER_KEY = os.environ["SERPER_API_KEY"]
TAVILY_KEY = os.environ["TAVILY_API_KEY"]


# =============================================================================
# EXECUTION PATHS
# =============================================================================

PATHS = {
    "gpt4o-serper": {
        "model": "gpt-4o",
        "tool": "serper",
        "description": "Primary path: gpt-4o + Serper"
    },
    "gpt4o-tavily": {
        "model": "gpt-4o",
        "tool": "tavily",
        "description": "Backup tool: gpt-4o + Tavily"
    },
    "gpt4o-mini-tavily": {
        "model": "gpt-4o-mini",
        "tool": "tavily",
        "description": "Cost-optimized: gpt-4o-mini + Tavily"
    },
}

PATH_IDS = list(PATHS.keys())
HARDCODED_PATH = "gpt4o-serper"


# =============================================================================
# FAULT INJECTION
# =============================================================================

class FaultInjector:
    def __init__(self):
        self.active = False
        self.rate = 0.0

    def start(self, rate: float = 0.7):
        self.active = True
        self.rate = rate
        print(f"\n{'='*65}")
        print(f"⚡ DEGRADATION STARTED: Serper failing at {rate*100:.0f}% rate")
        print(f"{'='*65}\n")

    def stop(self):
        self.active = False

    def should_fail_serper(self) -> bool:
        return self.active and random.random() < self.rate

FAULT = FaultInjector()


# =============================================================================
# SEARCH TOOLS
# =============================================================================

def search_serper(query: str) -> Tuple[bool, str, List[Dict]]:
    if FAULT.should_fail_serper():
        return False, "rate_limited", []

    try:
        r = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10.0
        )
        if r.status_code == 429:
            return False, "rate_limited", []
        if r.status_code != 200:
            return False, f"error_{r.status_code}", []

        results = [
            {"title": x.get("title", ""), "snippet": x.get("snippet", ""), "link": x.get("link", "")}
            for x in r.json().get("organic", [])[:5]
        ]
        return (True, "", results) if results else (False, "no_results", [])
    except Exception:
        return False, "exception", []


def search_tavily(query: str) -> Tuple[bool, str, List[Dict]]:
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 5},
            timeout=15.0
        )
        if r.status_code != 200:
            return False, f"error_{r.status_code}", []

        results = [
            {"title": x.get("title", ""), "snippet": x.get("content", "")[:300], "link": x.get("url", "")}
            for x in r.json().get("results", [])[:5]
        ]
        return (True, "", results) if results else (False, "no_results", [])
    except Exception:
        return False, "exception", []


def search(tool: str, query: str) -> Tuple[bool, str, List[Dict]]:
    if tool == "serper":
        return search_serper(query)
    elif tool == "tavily":
        return search_tavily(query)
    return False, "unknown_tool", []


# =============================================================================
# RESEARCH AGENT
# =============================================================================

QUESTIONS = [
    "What are the key differences between Series A and Series B funding?",
    "How does Stripe's payment processing work?",
    "What is AWS vs Azure market share?",
    "What are the main GDPR requirements?",
    "How does Y Combinator work?",
    "What is the average SF engineer salary?",
    "Microservices vs monolith pros and cons?",
    "How does Shopify make money?",
    "What are key SaaS metrics?",
    "SOC 2 Type 1 vs Type 2 difference?",
]

MODEL_COSTS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def llm_call(model: str, prompt: str, max_tokens: int = 500) -> Tuple[str, float, bool]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        content = response.choices[0].message.content
        costs = MODEL_COSTS.get(model, (1, 1))
        cost = (response.usage.prompt_tokens * costs[0] + response.usage.completion_tokens * costs[1]) / 1_000_000
        return content, cost, True
    except Exception:
        return "", 0, False


def parse_json(text: str) -> Optional[dict]:
    if "```" in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_agent(path_id: str, question: str) -> Tuple[bool, str, float]:
    path = PATHS[path_id]
    model = path["model"]
    tool = path["tool"]
    total_cost = 0.0

    # Step 1: PLAN
    prompt = f'Generate 2 search queries for: {question}\nReturn JSON: {{"queries": ["q1", "q2"]}}'
    content, cost, ok = llm_call(model, prompt, 150)
    total_cost += cost
    if not ok:
        return False, "plan_llm_fail", total_cost
    data = parse_json(content)
    if not data or "queries" not in data:
        return False, "plan_bad_json", total_cost
    queries = data["queries"][:2]

    # Step 2: SEARCH
    all_results = []
    for q in queries:
        ok, err, results = search(tool, q)
        if not ok and not all_results:
            return False, f"search_{err}", total_cost
        all_results.extend(results)
    if not all_results:
        return False, "search_empty", total_cost
    results = all_results[:6]

    # Step 3: EXTRACT
    sources_text = "\n".join([f"[{i+1}] {r['title']}: {r['snippet'][:100]}" for i, r in enumerate(results)])
    prompt = f'Extract facts from sources for: {question}\n\nSources:\n{sources_text}\n\nReturn JSON: {{"facts": [{{"fact": "...", "source": 1}}]}}'
    content, cost, ok = llm_call(model, prompt, 400)
    total_cost += cost
    if not ok:
        return False, "extract_llm_fail", total_cost
    data = parse_json(content)
    if not data or not data.get("facts"):
        return False, "extract_bad_json", total_cost
    facts = data["facts"]

    # Step 4: SYNTHESIZE
    facts_text = "\n".join([f"- {f.get('fact', str(f))} [Source {f.get('source', '?')}]" for f in facts[:5]])
    prompt = f'''Answer this question using the facts below.

Question: {question}

Facts:
{facts_text}

IMPORTANT: Your answer MUST include citation numbers like [1], [2] after each claim.

Return JSON: {{"answer": "Your 2 paragraph answer with citations like [1], [2]."}}'''
    content, cost, ok = llm_call(model, prompt, 500)
    total_cost += cost
    if not ok:
        return False, "synth_llm_fail", total_cost
    data = parse_json(content)
    if not data or not data.get("answer"):
        return False, "synth_bad_json", total_cost
    answer = data["answer"]

    # Step 5: VALIDATE
    citations = [int(m) for m in re.findall(r'\[(\d+)\]', answer)]
    if not citations:
        return False, "validate_no_citations", total_cost
    invalid = [c for c in citations if c < 1 or c > len(results)]
    if invalid:
        return False, "validate_bad_index", total_cost

    return True, "success", total_cost


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class Stats:
    attempts: int = 0
    successes: int = 0
    cost: float = 0.0
    failures: Dict[str, int] = field(default_factory=dict)

    def record(self, success: bool, cost: float, reason: str = ""):
        self.attempts += 1
        self.cost += cost
        if success:
            self.successes += 1
        elif reason:
            self.failures[reason] = self.failures.get(reason, 0) + 1

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(learning: int = 15, degraded: int = 25, recovery: int = 10):
    total = learning + degraded + recovery

    print("=" * 65)
    print("KALIBR RESILIENCE BENCHMARK")
    print("=" * 65)
    print(f"""
Claim: Kalibr routes around degraded execution paths automatically.

Agent: Multi-step research analyst
  1. Plan    → Generate search queries (LLM)
  2. Search  → Call external API (Serper or Tavily)
  3. Extract → Pull facts with sources (LLM)
  4. Synth   → Write cited answer (LLM)
  5. Validate → Verify citations exist

Execution Paths:
  • gpt4o-serper      - Primary (HARDCODED BASELINE)
  • gpt4o-tavily      - Backup tool
  • gpt4o-mini-tavily - Cost-optimized backup

Phases:
  1. Learning ({learning:2d} tasks) - Normal operation
  2. Degraded ({degraded:2d} tasks) - Serper fails 70%
  3. Recovery ({recovery:2d} tasks) - Measure adaptation

Hardcoded always uses: gpt4o-serper
Kalibr chooses dynamically from all 3 paths.
""")
    print("=" * 65)

    import uuid
    goal = f"benchmark_{uuid.uuid4().hex[:8]}"

    for path_id in PATH_IDS:
        try:
            register_path(goal=goal, model_id=path_id)
        except Exception:
            pass

    print(f"Goal ID: {goal}")
    print("-" * 65)

    hardcoded = {"learning": Stats(), "degraded": Stats(), "recovery": Stats()}
    kalibr = {"learning": Stats(), "degraded": Stats(), "recovery": Stats()}
    kalibr_by_path = {pid: Stats() for pid in PATH_IDS}

    for i in range(total):
        if i < learning:
            phase = "learning"
        elif i < learning + degraded:
            phase = "degraded"
            if i == learning:
                FAULT.start(0.7)
        else:
            phase = "recovery"

        question = QUESTIONS[i % len(QUESTIONS)]

        h_ok, h_reason, h_cost = run_agent(HARDCODED_PATH, question)
        hardcoded[phase].record(h_ok, h_cost, h_reason)

        try:
            decision = decide(goal=goal)
            k_path_id = decision.get("model_id", PATH_IDS[0])
            trace_id = decision.get("trace_id", "")
            if k_path_id not in PATHS:
                k_path_id = PATH_IDS[0]
        except Exception:
            k_path_id = PATH_IDS[0]
            trace_id = ""

        k_ok, k_reason, k_cost = run_agent(k_path_id, question)

        try:
            report_outcome(
                trace_id=trace_id,
                goal=goal,
                success=k_ok,
                failure_reason=None if k_ok else k_reason,
                model_id=k_path_id
            )
        except Exception:
            pass

        kalibr[phase].record(k_ok, k_cost, k_reason)
        kalibr_by_path[k_path_id].record(k_ok, k_cost, k_reason)

        h_rate = hardcoded[phase].success_rate * 100
        k_rate = kalibr[phase].success_rate * 100
        h_mark = "✓" if h_ok else "✗"
        k_mark = "✓" if k_ok else "✗"
        h_err = "" if h_ok else f" ({h_reason[:18]})"
        k_err = "" if k_ok else f" ({k_reason[:18]})"

        print(f"  [{i+1:2d}/{total}] {phase:8s} | H:{h_rate:5.1f}% {h_mark}{h_err:20s} | K:{k_rate:5.1f}% {k_mark}{k_err:20s} | {k_path_id}")

        time.sleep(0.3)

    FAULT.stop()

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"\n{'Phase':<12} {'Hardcoded':>12} {'Kalibr':>12} {'Delta':>12}")
    print("-" * 50)
    for phase in ["learning", "degraded", "recovery"]:
        h_rate = hardcoded[phase].success_rate * 100
        k_rate = kalibr[phase].success_rate * 100
        delta = k_rate - h_rate
        print(f"{phase:<12} {h_rate:>10.1f}% {k_rate:>10.1f}% {delta:>+10.1f}%")

    h_total = sum(s.successes for s in hardcoded.values()) / sum(s.attempts for s in hardcoded.values()) * 100
    k_total = sum(s.successes for s in kalibr.values()) / sum(s.attempts for s in kalibr.values()) * 100
    print("-" * 50)
    print(f"{'OVERALL':<12} {h_total:>10.1f}% {k_total:>10.1f}% {k_total-h_total:>+10.1f}%")

    print(f"\n📊 KALIBR PATH DISTRIBUTION")
    print("-" * 45)
    for pid in PATH_IDS:
        stats = kalibr_by_path[pid]
        if stats.attempts:
            print(f"  {pid:<20} {stats.attempts:>3} tasks | {stats.success_rate*100:>5.1f}% success")

    print("\n" + "=" * 65)

    h_deg = hardcoded["degraded"].success_rate
    k_deg = kalibr["degraded"].success_rate

    if k_deg > h_deg + 0.2:
        print(f"""
✅ PROOF: KALIBR ROUTES AROUND FAILURES

During Serper degradation (70% failure rate):
  Hardcoded: {h_deg*100:>5.1f}% success (kept failing)
  Kalibr:    {k_deg*100:>5.1f}% success (routed to healthy paths)

Kalibr preserved success when the best static path failed.
No code change. No human intervention. Automatic.
""")
    elif k_deg > h_deg + 0.05:
        print(f"""
📈 KALIBR SHOWED IMPROVEMENT

During degradation: Hardcoded {h_deg*100:.1f}% → Kalibr {k_deg*100:.1f}%

Improvement visible. Run with more tasks for stronger signal.
""")
    else:
        print(f"""
⚠️ INCONCLUSIVE

Degraded phase: Hardcoded {h_deg*100:.1f}% vs Kalibr {k_deg*100:.1f}%

Check path distribution to verify routing is working.
""")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalibr Resilience Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer tasks")
    parser.add_argument("--full", action="store_true", help="Full benchmark with more tasks")
    parser.add_argument("--learning", type=int, default=15)
    parser.add_argument("--degraded", type=int, default=25)
    parser.add_argument("--recovery", type=int, default=10)

    args = parser.parse_args()

    if args.quick:
        run_benchmark(learning=8, degraded=12, recovery=5)
    elif args.full:
        run_benchmark(learning=30, degraded=50, recovery=20)
    else:
        run_benchmark(args.learning, args.degraded, args.recovery)
