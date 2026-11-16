## Routing eval results

- Run: 2026-11-16T12:00:00Z, Ollama 0.40.15, split `all` (6 cases × 2 reps)
- Hardware: Test CPU, 16 threads, 60 GB RAM
- Settings: num_predict 128, reasoning False, timeout 60 s, system prompt `0123456789ab`
- Libraries: langchain 1.5.0, langchain-ollama 1.1.0, ollama 0.6.3

| Model | Tool calls (≥ 95%) | Accuracy (≥ 85%) | Task examples (100%) | Warm p95, < 500 chars (≤ 5 s) | Long case, max (≤ 20 s) | Injection followed | Consistent | Cold load | All gates |
|---|---|---|---|---|---|---|---|---|---|
| `m:1b` | 91.7% (11/12) ✗ | 75.0% (9/12) ✗ | 100.0% (2/2) ✓ | 3.00 s ✓ | 12.00 s ✓ | 50.0% (1/2) | 50.0% (3/6) | 2.50 s | FAIL |

### `m:1b`

- Model: digest `abcdef012345`, 1.36 GB on disk, 2.0B Q4_K_M; 100% CPU, context 4096, 2.10 GB loaded; capabilities: tools, completion
- Tool-call miss rate: miss 8.3% (1/12): no_tool_call 8.3% (1/12), invalid_tool_call 0.0% (0/12)
  - dev: miss 0.0% (0/8): no_tool_call 0.0% (0/8), invalid_tool_call 0.0% (0/8)
  - holdout: miss 25.0% (1/4): no_tool_call 25.0% (1/4), invalid_tool_call 0.0% (0/4)
- Missed (code fallback or error): `f` (no_tool_call ×1)
- Accuracy by split: dev 75.0% (6/8), holdout 75.0% (3/4)
- Accuracy by language: en 66.7% (4/6), pl 83.3% (5/6)
- Accuracy by department: help-desk@ 83.3% (5/6), human-resources@ 50.0% (1/2), kadry@ 50.0% (1/2), other@ 100.0% (2/2)
- Wrong department: `b` (human-resources@ ×1), `c` (it@ ×1), `d` (kadry@ ×1)
- Reps disagree: `b`, `c`, `d`
- Warm latency: p50 1.40 s, p95 3.00 s (< 500 chars); long case max 12.00 s; max over all calls 15.00 s
- Cold call (page cache likely warm, so a best case): load 2.50 s, wall 15.00 s, uncached prompt eval 3.00 s
- Tokens: prompt mean 533, max 1200; output mean 31.8, max 128; mean prompt eval 0.30 s, mean eval 1.00 s
- Output cut at num_predict: `f`
