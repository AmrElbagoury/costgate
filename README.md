# costgate

> Six PyPI packages tell you what your agent *spent*. costgate blocks the merge *before it spends it*.

**git diff for your token bill.** A static-analysis layer for AI-agent cost: it
predicts what an agent will spend, and what a prompt/tool change *adds*, at **PR
time** — so a cost regression fails the build like a test does, instead of showing
up on next month's invoice.

```
🛑 costgate — support-atlas · blocks merge

Projected spend ▲ $14,032/mo (+79%) · 1.79× at 4,000 tasks/day

|                | base    | this PR | Δ        |
|----------------|--------:|--------:|---------:|
| cost / month   | $17,745 | $31,777 | $14,032  |

Why:
• static context grew 401 tok → re-sent 11×/task (+4,411 tok/task)
• loop length 8→11 steps
• observed scaling: ~quadratic in steps (no caching detected)
```

## Quickstart (first number in ~2 minutes)

```bash
pip install costgate

costgate init                       # scaffold costgate.yaml
costgate run costgate.yaml         # instant cold-start estimate (no traces needed)
```

Then upgrade to provider-native accuracy by calibrating on your real runs:

```bash
costgate pull --source langfuse -o traces.jsonl    # uses LANGFUSE_* env vars
costgate run  costgate.yaml --traces traces.jsonl
```

Gate a pull request:

```bash
costgate diff base.yaml head.yaml --traces traces.jsonl --max-pct 20 --format pr
# exits non-zero when the verdict is "fail" → blocks CI
```

## Why it's different

Observability (Langfuse, Helicone, Datadog) tells you what you **already spent**.
Billing (Stripe, Metronome) charges your **customers**. Neither tells the
**developer**, at the moment they change a prompt, what the change costs at scale.
costgate is the only layer that runs **before execution, at PR time, with
code-level diff semantics** — closer to a compiler or `terraform plan` than a
dashboard. That position is also the hard-to-displace one: a model provider can add
"what-if pricing," but not into *your* CI against *your* git history across
*multiple* providers.

## The model (measured, not assumed)

Agents don't pay for a prompt once: each step re-sends the static context **and**
carries the growing conversation, so input cost can grow **super-linearly** in step
count. But real systems truncate, summarize, and cache — so growth is
**path-dependent**, not strictly N².

So costgate has two modes:

- **Cold start (analytic).** Closed-form compounding from your spec. Instant, no
  traces. Labelled as a directional upper-ish bound (assumes monotonic growth).
- **Calibrated (empirical).** Fits the **observed per-step cost curve** straight
  from provider-reported usage — **no tokenizer needed** — and *detects the regime*:

  | trace shape | reported scaling | cost in steps |
  |---|---|---|
  | steady context growth | `~quadratic (no caching detected)` | compounds hard |
  | growth flattens | `~linear (caching/summarization)` | tame |
  | context resets | `~linear (summarization/truncation)` | tame |

  Changes are projected by **perturbing that real curve** (a static-context delta is
  re-sent every step; added steps extend at the measured tail marginal). The
  confidence band reflects output variance only — the input side is measured.

## Commands

```
costgate init   [-o costgate.yaml] [--force]
costgate run    SPEC [--traces T.jsonl] [--format text|json]
costgate diff   BASE HEAD [--traces T.jsonl] [--max-pct N] [--max-month USD] [--format text|json|pr]
costgate pull   --source langfuse [-o traces.jsonl] [--host URL] [--limit N]
costgate pull   --source csv --file usage.csv --run-key run_id --in-key prompt_tokens --out-key completion_tokens
```

### Spec (`costgate.yaml`)

```yaml
name: support-atlas
model: claude-sonnet (example)
encoding: cl100k_base
static_files: [system_prompt.txt, tools.json]   # context re-sent every step
pricing: {input_per_m: 3.00, output_per_m: 15.00}   # edit to current rates
tasks_per_day: 4000
profile: {steps: 6, user_tokens: 220, output_tokens: 300, tool_tokens: 700}
```

### Traces (`traces.jsonl`)

One agent run per line; `costgate pull` produces this for you.

```json
{"task_id": "t1", "calls": [{"input_tokens": 1236, "output_tokens": 282}, {"input_tokens": 2266, "output_tokens": 343}]}
```

## CI integration

```yaml
# .github/workflows/cost.yml
on: pull_request
jobs:
  cost:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: your-org/costgate@v0
        with:
          base: examples/support_agent.yaml
          head: examples/support_agent_candidate.yaml
          traces: traces.jsonl     # optional; provider-native accuracy
          max-pct: "20"
```

The Action posts the cost diff as a sticky PR comment and fails the check on a
budget breach. Non-GitHub CI: use `costgate diff ... --format json` and read the
exit code.

## Roadmap to GA

In: the compounding model, empirical calibration with regime detection, diff +
gate, CLI, PR comment, GitHub Action, Langfuse + CSV importers, tests. Next:
publish to PyPI, GitHub App for hosted PR comments, more importers (Helicone,
OpenTelemetry, provider usage exports), and a hosted baseline store so `diff` knows
the target-branch spec automatically. Deliberately **not** building: an inline
gateway, a real-time dashboard, or billing — costgate is the pre-merge gate.

## Run the demo

```bash
cd examples
costgate run  support_agent.yaml --traces traces_steady.jsonl    # ~quadratic
costgate run  support_agent.yaml --traces traces_cached.jsonl    # ~linear (caching)
costgate diff support_agent.yaml support_agent_candidate.yaml \
               --traces traces_steady.jsonl --max-pct 20 --format pr
```
