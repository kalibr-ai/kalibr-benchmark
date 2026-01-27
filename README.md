# Kalibr Resilience Benchmark

**Claim:** Kalibr routes execution traffic based on observed outcomes, allowing agents to continue operating when execution paths degrade—without code changes or human intervention.

This benchmark demonstrates that behavior.

---

## The Question

What happens when an agent's execution path starts failing in production?

- **Hardcoded systems** continue sending traffic to the same path until an engineer notices, diagnoses the issue, and deploys a fix.
- **Kalibr** continuously incorporates outcome feedback and shifts traffic toward paths that are succeeding.

This benchmark compares those two behaviors under identical conditions.

---

## Benchmark Overview

### Task

A document analysis agent that must return valid structured JSON containing:
- summary
- key entities
- sentiment (label + confidence)
- action items

A run is considered successful only if:
- the response parses as JSON
- all required fields are present
- basic structural validation passes

This task has real failure modes (formatting errors, partial responses, invalid JSON).

### Available Execution Paths

Both systems have access to the same three models:
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-3.5-turbo`

There is no difference in prompts, validation, or task structure. The only difference is how execution paths are chosen.

---

## Experimental Conditions

### Hardcoded Baseline
- Always uses `gpt-4o`
- No fallback logic
- No adaptive routing
- Represents a typical production agent configuration today

### Kalibr
- Chooses execution paths based on observed outcomes
- Continuously explores alternatives during normal operation
- Optimizes for success rate first, then cost
- No explicit failure signals or special-case logic

---

## Phases

| Phase | Executions | Description |
|-------|------------|-------------|
| Learning | 30 | Normal operation. No failures injected. |
| Degradation | 40 | `gpt-4o` responses fail probabilistically (70%). |
| Observation | 30 | Degradation continues. Measure steady state behavior. |

---

## Failure Injection

At execution 31, a 70% failure rate is injected on `gpt-4o`.

This simulates real-world degradation such as:
- provider instability
- rate limiting
- quality regressions
- partial or malformed responses

Failures are injected after model execution and before validation, so:
- both systems see identical responses
- no special handling exists for Kalibr
- the only difference is routing behavior

Neither system knows *why* failures occur. They only observe outcomes.

---

## Results

### Run 1

| Phase | Kalibr | Hardcoded |
|-------|--------|-----------|
| Learning | 100.0% | 100.0% |
| Degradation | 100.0% | 42.5% |
| Observation | 100.0% | 40.0% |
| **Overall** | **100.0%** | **59.0%** |

#### Routing Distribution (Run 1)

| Phase | gpt-4o | gpt-4o-mini | gpt-3.5-turbo |
|-------|--------|-------------|---------------|
| Learning | 3% | 60% | 37% |
| Degradation | 2% | 65% | 32% |
| Observation | 7% | 47% | 47% |

Kalibr reduced reliance on the brittle path early and maintained that behavior during degradation.

#### Cost per Successful Outcome (Run 1)

- Kalibr: **$0.000232**
- Hardcoded: **$0.002839**

Kalibr achieved the same outcomes at ~12× lower cost per success.

### Run 2 (Independent Re-run)

| Phase | Kalibr | Hardcoded |
|-------|--------|-----------|
| Learning | 93.3% | 100.0% |
| Degradation | 100.0% | 42.5% |
| Observation | 100.0% | 36.7% |
| **Overall** | **98.0%** | **58.0%** |

Results are consistent across runs.

---

## What This Demonstrates

### Outcome-driven routing works

During degradation:

**Hardcoded system:**
- continued routing 100% of traffic to a degraded path
- success rate dropped sharply
- required human intervention to recover

**Kalibr:**
- incorporated outcome feedback
- minimized traffic to degraded paths automatically
- maintained near-perfect success without intervention

This is not an optimization. It is a behavioral difference that hardcoded systems cannot exhibit.

---

## What This Does Not Claim

This benchmark does not demonstrate:
- superior reasoning or intelligence
- universal optimality across all tasks
- guaranteed reliability in all environments

Kalibr is a control system. It routes execution based on what is actually working.

This benchmark evaluates that control behavior under degradation.

---

## Limitations

- Single task type
- Small number of execution paths
- Synthetic failure injection

Results should not be extrapolated to all workloads.

The purpose is to validate adaptive execution control, not benchmark model quality.

---

## Run It Yourself

```bash
pip install kalibr openai

export KALIBR_API_KEY=your-key
export KALIBR_TENANT_ID=your-tenant
export OPENAI_API_KEY=your-key

python resilience_benchmark.py
```

**Options:**
```bash
python resilience_benchmark.py --quick  # ~35 executions, ~3 min
python resilience_benchmark.py          # ~100 executions, ~10 min
python resilience_benchmark.py --full   # ~200 executions, ~20 min
```

**Requirements:**
- ~$0.20 in API usage (standard run)
- Python 3.10+

---

## Summary

| Metric | Kalibr | Hardcoded |
|--------|--------|-----------|
| Success during degradation | ~100% | ~40% |
| Cost per success | $0.0002 | $0.003 |
| Human intervention required | No | Yes |

When execution paths degrade, hardcoded systems fail until humans intervene.

Kalibr adapts automatically.

---

## License

MIT
