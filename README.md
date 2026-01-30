# Kalibr Resilience Benchmark

**Claim:** When your best execution path degrades, Kalibr routes around it automatically. Hardcoded systems keep failing until a human intervenes.

---

## What This Tests

This is **execution path routing**, not model routing.

Each path is a complete strategy:

| Path ID | Model | Tool |
|---------|-------|------|
| `gpt4o-serper` | gpt-4o | Serper |
| `gpt4o-tavily` | gpt-4o | Tavily |
| `gpt4o-mini-tavily` | gpt-4o-mini | Tavily |

## The Agent

5-step research agent:
1. **Plan** → Generate search queries (LLM)
2. **Search** → Call Serper or Tavily API
3. **Extract** → Pull facts with sources (LLM)
4. **Synthesize** → Write cited answer (LLM)
5. **Validate** → Verify citations

## Phases

| Phase | Tasks | Description |
|-------|-------|-------------|
| Learning | 15 | Normal operation |
| Degraded | 25 | Serper fails 70% |
| Recovery | 10 | Measure adaptation |

## Results

| Phase | Hardcoded | Kalibr | Delta |
|-------|-----------|--------|-------|
| Learning | 100% | 100% | +0% |
| Degraded | ~25% | ~90% | **+65%** |
| Recovery | ~25% | ~100% | **+75%** |

Kalibr routes to healthy Tavily paths. Hardcoded keeps failing.

## Run It
```bash
pip install -r requirements.txt

export KALIBR_API_KEY=your-key
export KALIBR_TENANT_ID=your-tenant
export OPENAI_API_KEY=your-key
export SERPER_API_KEY=your-key
export TAVILY_API_KEY=your-key

python resilience_benchmark.py
python resilience_benchmark.py --quick  # faster
python resilience_benchmark.py --full   # more tasks
```

## License

MIT
