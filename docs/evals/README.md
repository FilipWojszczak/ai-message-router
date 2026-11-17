# Model evaluation and routing quality

How the default Ollama model was chosen, what it gets right and wrong, and how to
measure it again. Raw results are in [`results/`](results/); the labeled set is
[`evals/cases.jsonl`](../../evals/cases.jsonl).

## Result

The default model is **`qwen3:4b-instruct-2507-q4_K_M`** (2.5 GB on disk, 3.2 GB in RAM,
CPU only). It is the only candidate that passed every gate of the decision rule below.

On the full labeled set (32 cases × 2 repetitions) with the shipped prompt:

- 100% of the emails were sent by the agent's own tool call (64/64); the code fallback
  never ran.
- 90.6% went to an accepted department (58/64); both task examples passed every time.
- None of the 3 prompt-injection cases changed the department (0/6).
- A typical request takes 1.7 s (p95 1.92 s); a message near the 2000-character limit
  takes about 8 s.

## Decision rule

Fixed before the full run: the winner is the **smallest** candidate that passes every
gate on the full set (dev + holdout) with 2 repetitions.

| Gate | Threshold |
|---|---|
| Tool-call rate (the agent's tool sent the email) | ≥ 95% |
| Accuracy (final department is in the accepted set) | ≥ 85% |
| Task examples | 100% (both, every repetition) |
| Warm p95 wall time, messages < 500 chars | ≤ 5 s |
| Warm wall time, the long case | ≤ 20 s |
| Injection-followed rate | reported; tie-breaker only |

## Results

Measured on 2026-11-16 at temperature 0 with `num_predict` 128, thinking off and a 60 s
deadline. Every model ran 100% on CPU with a 4096-token context.

| Model | Prompt | Tool calls | Accuracy | Task examples | Warm p95 | Long case | Injection followed | Consistent | Cold load | Gates |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3:0.6b` | `1db9c08be6c5` | 3.1% (2/64) | 15.6% (10/64) | 0/4 | 0.71 s | 1.77 s | 0/6 | 32/32 | 1.03 s | fail |
| `qwen3:1.7b` | `1db9c08be6c5` | 54.7% (35/64) | 45.3% (29/64) | 3/4 | 1.41 s | 3.71 s | 6/6 | 31/32 | 1.53 s | fail |
| `qwen3:4b-instruct-2507-q4_K_M` | `1db9c08be6c5` | 96.9% (62/64) | 90.6% (58/64) | 4/4 | 1.93 s | 8.00 s | 0/6 | 32/32 | 2.79 s | pass |
| `qwen3:4b-instruct-2507-q4_K_M` | `818965df7d34` (shipped) | 100% (64/64) | 90.6% (58/64) | 4/4 | 1.92 s | 8.24 s | 0/6 | 32/32 | 2.79 s | pass |

- "Prompt" is the first 12 hex digits of `sha256(SYSTEM_PROMPT)`. The rule was applied
  to the three models on the same prompt (`1db9c08be6c5`). The last row is the winner
  after the one prompt change described under [Prompt changes](#prompt-changes).
- "Consistent" counts cases where both repetitions chose the same department.
- `qwen3:0.6b` shows 0/6 injections followed only because it made no tool call on those
  cases: the fallback sent them to other@.

### Why this model

- **`qwen3:0.6b`** almost never emits a real tool call (2/64), so nearly every request
  falls back to other@.
- **`qwen3:1.7b`** calls the tool about half the time (35/64), reaches 45.3% accuracy,
  gets help-desk@ right in 1 of 20 runs, and followed every injected instruction (6/6).
- **`qwen3:4b-instruct-2507-q4_K_M`** passes all gates with room on latency: p95 is
  1.92 s against a 5 s limit.

The **full tag is pinned** in `docker-compose.yml` because the short `qwen3:4b` tag now
points at the thinking-only build (the same digest as `qwen3:4b-thinking-2507-q4_K_M`).
That build ignores `think: false`; in the 4-message probe done while writing the stage 4
spec it took 15–32 s per request and ran into the output cap without calling the tool.
It was not part of this eval.

Ollama lists `thinking` among the instruct tag's capabilities, but no reasoning was
observed: across all its calls the output never exceeded 24 tokens, the size of one
tool call.

### The shipped model in detail

`qwen3:4b-instruct-2507-q4_K_M`, prompt `818965df7d34`
([`results/qwen3-4b-instruct-2507-q4_K_M.json`](results/qwen3-4b-instruct-2507-q4_K_M.json)):

- Tool-call miss rate: 0.0% (0/64): `no_tool_call` 0.0% (0/64), `invalid_tool_call`
  0.0% (0/64). Dev 0/42, holdout 0/22.
- Accuracy by split: dev 95.2% (40/42), holdout 81.8% (18/22).
- Accuracy by language: Polish 94.4% (34/36), English 91.7% (22/24), mixed 50.0% (2/4).
- Accuracy by department: kadry@ 100% (16/16), it@ 100% (10/10), other@ 100% (8/8),
  help-desk@ 80.0% (16/20), human-resources@ 80.0% (8/10).
- Warm latency: p50 1.71 s, p95 1.92 s below 500 characters; 8.24 s at most for the
  1969-character case.
- Cold call: 8.44 s in total, of which 2.79 s is the model load and 4.13 s the first,
  uncached evaluation of the prompt. The model file was probably in the page cache, so
  a load from a cold disk takes longer.
- Tokens: 465 prompt tokens on average, 1064 for the long case; 22.8 output tokens on
  average, 24 at most.

### Known misroutes

All three are deterministic (both repetitions):

| Case | Split | Expected | Got |
|---|---|---|---|
| `long-laptop-complaint-pl` (1969 characters, a failing laptop) | dev | help-desk@ | it@ |
| `multisport-card-en` ("How do I sign up for the Multisport card?") | holdout | human-resources@ | other@ |
| `blue-screen-mixed` ("Mój laptop ma blue screen po update") | holdout | help-desk@ | it@ |

Two of the three confuse help-desk@ with it@ on laptop faults. The holdout failures are
reported, not tuned against. With 11 holdout cases, one case is worth 9 points of
holdout accuracy, so the set can show gross overfitting but not small regressions.

## Routing policy

The labels follow this policy. It must stay consistent with `DEPARTMENT_SCOPES` in
`src/message_router/departments.py`; a change to either is a prompt change.

- **kadry@**: administrative personnel matters. Leave requests (including "a day off"),
  sick leave (L4), contracts and their copies, payroll and missing salary, employment
  certificates.
- **human-resources@**: recruitment, training, development, benefits (for example
  Multisport), team conflicts.
- **help-desk@**: one user's device, account or application. A broken computer or
  peripherals, passwords (including the VPN password), printer, email client, crashes.
- **it@**: infrastructure and servers, access and permission requests to systems, buying
  or rolling out software for a team.
- **other@**: anything else (off-topic, garbled), chosen by the model through the tool.

Three cases are ambiguous and accept two departments: a salary raise conversation
(human-resources@ or kadry@), installing an app on one laptop (help-desk@ or it@), and a
laptop plus an account for a new hire (it@ or help-desk@).

## Labeled set

`evals/cases.jsonl` holds 32 cases, one JSON object per line: 21 in `dev` and 11 in
`holdout`; 18 Polish, 12 English and 2 mixed. It includes both task examples verbatim,
3 injection cases, 3 ambiguous cases, 3 off-topic and 2 garbled messages, and one
message near the 2000-character limit. `evals/dataset.py` validates the file on load.

- **Scoring:** a department inside the case's accepted set is correct; anything else is
  an error. other@ chosen through the tool is correct and counts as a tool call. other@
  reached through the code fallback is correct for accuracy but is not a tool call.
- **Holdout discipline:** prompt tuning looks only at `dev` results. The holdout is run
  once for each candidate's final report and once after the last prompt change.

## Prompt changes

Every change to `SYSTEM_PROMPT` or `DEPARTMENT_SCOPES` changes routing behavior and is
listed here.

### 1. The other@ rule names the tool (2026-11-16)

Prompt `1db9c08be6c5` → `818965df7d34`, in `src/message_router/prompts.py`.

- Before: `- If no department clearly fits, use other@example.com.`
- After: `- If no department clearly fits, call send_email with other@example.com.`

Why: on the dev case `canteen-menu-en` ("What's on the canteen menu today?") the model
answered with the plain text `other@example.com` instead of calling the tool, in both
repetitions. It followed the rule literally. The fallback still delivered the message to
other@, but not through the agent.

| | Before | After |
|---|---|---|
| Dev tool calls | 40/42 | 42/42 |
| Dev accuracy | 40/42 | 40/42 |
| Full-set tool calls | 62/64 | 64/64 |
| Full-set accuracy | 58/64 | 58/64 |

Dev cases fixed: `canteen-menu-en`. Dev cases broken: none. The three known misroutes
are unchanged. Evidence: the "before" full run is
[`results/qwen3-4b-instruct-2507-q4_K_M.prompt-1db9c08be6c5.json`](results/qwen3-4b-instruct-2507-q4_K_M.prompt-1db9c08be6c5.json),
the dev rerun is
[`results/qwen3-4b-instruct-2507-q4_K_M.dev.json`](results/qwen3-4b-instruct-2507-q4_K_M.dev.json).

## Settings confirmed by the eval

- **`ollama_num_predict` = 128.** The longest output of any candidate was 60 tokens
  (`qwen3:0.6b`); the shipped model never exceeded 24. No call was cut at the cap.
- **`ollama_timeout` = 60 s.** The slowest call across all runs was a cold call of the
  shipped model, 8.72 s. That leaves about 7× headroom for slower machines and for
  queueing.
- **Context window.** The 1969-character Polish message is 1064 prompt tokens against
  Ollama's default 4096-token context, and the Ollama log has no
  `truncating input prompt` line for the eval runs. The 2000-character API limit stays.
- **No retry after a missed tool call.** The fallback to other@ in code stays a safety
  net. Before the prompt change it ran in 3.1% of runs (2/64), all on one off-topic case
  whose correct department is other@ anyway; after the change it did not run (0/64).

## Readiness on the compose stack

`ollama-pull` pulls the model and then loads it into RAM; `api` starts only after it
succeeds. `OLLAMA_KEEP_ALIVE=-1` on the `ollama` service keeps the model loaded until
the stack stops. Measured on the same machine:

| Step | Result |
|---|---|
| `docker compose up -d --build` with an empty model volume | 2 min 23 s (image build, 2.5 GB download, warm-up); returns with the model loaded |
| `ollama-pull` log | `Model qwen3:4b-instruct-2507-q4_K_M loaded in 2.8 s.` |
| `ollama ps` | 3.2 GB, 100% CPU, context 4096, until "Forever" |
| First request after `up` | 5.6 s (before the API warm-up; now 1.6 s, see below) |
| Second request | 1.5 s |
| First request after `docker compose restart api` | 1.6 s (the model stays loaded) |
| First request after `docker compose restart ollama` | 7.5 s, no error |
| `docker compose up -d` after that | reruns `ollama-pull`: no pull, loaded in 2.5 s |
| `OLLAMA_MODEL=does-not-exist:1b docker compose up -d` | exit code 1, `api` not started, `ERROR: pull failed: pull model manifest: file does not exist` |

The download time depends on the connection.

`ollama-pull` loads the weights but not the prompt cache, so the first request used to
evaluate the system prompt and tool schema uncached (about 4 s of prefill). The API now
makes one model call at startup with the same prompt prefix (`warm_up_prompt_cache`,
`OLLAMA_WARMUP`, default on) and becomes healthy only after it. Measured after
`docker compose restart ollama` + `docker compose up -d --build --wait api`:

| Step | Before | After |
|---|---|---|
| Startup warm-up (`Prompt cache warmed in`) | none | 4.6 s, startup only |
| First request after `up` | 5.6 s | 1.6 s |
| Warm-up after `docker compose restart api` | none | 1.5 s (cache already warm) |

If the warm-up fails or hits `OLLAMA_TIMEOUT`, it logs a WARNING and the API starts
anyway; the first request is then slow, as before.

Known gaps, accepted:

- Restarting only Ollama does not rerun either warm-up. The next request pays the model
  load and the uncached prompt (7.5 s), well inside the 60 s deadline.
  `docker compose up -d` or restarting `api` warms it again.
- If the model cannot be loaded, `ollama-pull` exits with Ollama's own message (for
  example "model requires more system memory") and `docker compose up -d` fails before
  the API starts. This path, including the 300 s load timeout (shortened in the test),
  is covered by offline unit tests of `docker/ollama_pull.py` against a fake server,
  not by exhausting RAM.

## Notes for the project README

- **Requirements:** about 4 GB of free RAM (Ollama holds 3.2 GB for the model) and
  about 3 GB of disk for the model download.
- **Low-RAM override:** `OLLAMA_MODEL=qwen3:1.7b docker compose up -d` needs 1.9 GB of
  RAM and 1.4 GB of disk, but only about half of the requests are routed by the agent
  (see the table); the rest fall back to other@.
- **Concurrency:** Ollama serves one request at a time (`OLLAMA_NUM_PARALLEL` stays at
  its default of 1), so concurrent requests queue. From the measured latencies, the 60 s
  deadline covers roughly 30 queued short requests or 7 long ones; later ones get a 504
  with nothing sent, so a client retry is safe. This is an estimate, not a measurement.
  More slots would not help on CPU: they share the same cores and each needs its own
  prompt cache.

## Measurement setup

- **Hardware:** AMD Ryzen 7 PRO 8840U (16 threads), 60 GB RAM, NVMe.
- **Software:** Ollama 0.40.15 (`alpine/ollama`, CPU-only build), langchain 1.5.0,
  langchain-ollama 1.1.0, ollama (Python client) 0.6.3.
- **Code path:** the shipped `route_message`, with the real prompt, tool, fallback and
  deadline. Only `aiosmtplib.send` is replaced by a capturing fake, so no email leaves
  the process.
- **Protocol per model:** unload every loaded model, make one cold call (reported
  separately and excluded from every rate), then 2 sequential passes over the cases.
  The run fails if the model is not 100% on CPU at a 4096-token context.

## Running it again

The eval talks to a standalone Ollama on `localhost:11434`, not to the compose stack
(whose Ollama port is not published). Stop the compose stack and any native Ollama
first: they would compete for the port, RAM and CPU and skew the latency.

```bash
docker compose down
docker run -d --name ollama-eval -p 127.0.0.1:11434:11434 \
  -v ollama-eval-models:/root/.ollama alpine/ollama:0.40.15
docker exec ollama-eval ollama pull qwen3:4b-instruct-2507-q4_K_M

# Full set: writes docs/evals/results/<model>.json and prints the Markdown report
uv run python -m evals.run --model qwen3:4b-instruct-2507-q4_K_M

# While tuning the prompt: dev only, writes <model>.dev.json
uv run python -m evals.run --model qwen3:4b-instruct-2507-q4_K_M --split dev

docker stop ollama-eval
```

`--model` can be repeated to compare candidates. The script never pulls a model; it
stops with the `ollama pull` command to run if one is missing. Its own tests are
internal and not part of CI: `uv run pytest evals/tests -q`.
