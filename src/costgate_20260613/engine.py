"""
costgate engine — predict and diff AI-agent token cost before you ship.

Two prediction modes:

  ANALYTIC  (no traces, instant cold-start). Closed-form compounding from a spec.
            Clearly labelled as an upper-ish bound: it assumes context grows
            monotonically every step. Good for a directional first number.

  EMPIRICAL (calibrated from real traces — the accurate path). We do NOT assume a
            growth law. We fit the *observed* per-step input/output curve straight
            from the provider's reported usage, detect its shape (linear →
            ~quadratic cost in steps; flat/declining → caching or summarization →
            ~linear cost), and project changes by perturbing that real curve.
            No tokenizer needed; the numbers are provider-native truth.

This is why the cost scaling is reported as a *detected* property, not a claim:
real agents truncate, summarize, and cache, so growth is path-dependent and
super-linear — not strictly N². The engine measures which regime you're in.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from statistics import mean
import copy


# --------------------------------------------------------------------------- #
# Tokenizer (exact via tiktoken when reachable, else labelled approximation).
# Only used by the ANALYTIC path and to tokenize static files for the diff.
# --------------------------------------------------------------------------- #
class Tokenizer:
    def __init__(self, encoding: str = "cl100k_base"):
        self.encoding = encoding
        self.exact = False
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding)
            self.exact = True
        except Exception:
            self._enc = None

    def count(self, text: str, dense: bool = False) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        chars = len(text)
        words = len(text.split())
        punct = sum(c in "{}[]()<>:;,.\"'`/\\=+-*&|#@\n\t" for c in text)
        is_dense = dense or (chars and punct / chars > 0.16)
        by_char = chars / (3.3 if is_dense else 4.0)
        by_word = words * (1.75 if is_dense else 1.33)
        return max(0, round(by_char * 0.6 + by_word * 0.4))


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass
class Pricing:
    input_per_m: float = 3.00
    output_per_m: float = 15.00
    cached_input_per_m: float | None = None   # $/1M for cache-read tokens; defaults to 10% of input rate

    def cached_rate(self) -> float:
        return self.cached_input_per_m if self.cached_input_per_m is not None else self.input_per_m * 0.10


@dataclass
class StepProfile:
    steps: float = 6.0
    user_tokens: float = 200.0
    output_tokens: float = 350.0
    tool_tokens: float = 600.0


@dataclass
class AgentSpec:
    name: str = "agent"
    model: str = "unspecified"
    encoding: str = "cl100k_base"
    static_text: str = ""
    static_tokens: int | None = None
    profile: StepProfile = field(default_factory=StepProfile)
    pricing: Pricing = field(default_factory=Pricing)
    tasks_per_day: float = 500.0
    cache_fraction: float = 0.0     # share of input billed at cache-read rate (analytic path; empirical measures it)

    @staticmethod
    def from_dict(d: dict) -> "AgentSpec":
        return AgentSpec(
            name=d.get("name", "agent"),
            model=d.get("model", "unspecified"),
            encoding=d.get("encoding", "cl100k_base"),
            static_text=d.get("static_text", ""),
            static_tokens=d.get("static_tokens"),
            profile=StepProfile(**(d.get("profile") or {})),
            pricing=Pricing(**(d.get("pricing") or {})),
            tasks_per_day=float(d.get("tasks_per_day", 500.0)),
            cache_fraction=float(d.get("cache_fraction", 0.0)),
        )

    def static_count(self, tok: Tokenizer | None = None) -> tuple[int, bool]:
        if self.static_tokens is not None:
            return int(self.static_tokens), True
        tok = tok or Tokenizer(self.encoding)
        return int(tok.count(self.static_text, dense=True)), tok.exact


# --------------------------------------------------------------------------- #
# Projection result
# --------------------------------------------------------------------------- #
@dataclass
class Projection:
    method: str                 # "analytic" | "empirical"
    exact_basis: bool           # provider-native or exact tokenizer
    static_tokens: int
    steps: float
    input_per_task: float
    output_per_task: float
    tokens_per_task: float
    static_resend: float
    history_drag: float
    cost_per_task: float
    cost_per_day: float
    cost_per_month: float
    band_low_month: float
    band_high_month: float
    scaling: str                # detected/assumed: "~quadratic in steps" etc.
    tasks_per_day: float
    cache_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# ANALYTIC path — closed-form compounding (cold start)
# --------------------------------------------------------------------------- #
def project_analytic(spec: AgentSpec, tok: Tokenizer | None = None,
                     output_uncertainty: float = 0.35,
                     precounted: tuple[int, bool] | None = None) -> Projection:
    """`precounted` lets diff() pass an already-computed (static_tokens, exact)
    pair, avoiding double tokenization without mislabeling the basis."""
    tok = tok or Tokenizer(spec.encoding)
    S, exact = precounted if precounted is not None else spec.static_count(tok)
    p = spec.profile
    N, U, A, R = p.steps, p.user_tokens, p.output_tokens, p.tool_tokens

    static_resend = N * S
    history_drag = (A + R) * N * (N - 1) / 2.0
    input_per_task = static_resend + N * U + history_drag
    output_per_task = N * A
    return _finish(spec, "analytic", exact, S, N, input_per_task, output_per_task,
                   static_resend, history_drag, output_uncertainty,
                   scaling="~quadratic in steps (assumed: monotonic context growth)",
                   cache_fraction=spec.cache_fraction)


# --------------------------------------------------------------------------- #
# EMPIRICAL path — fit the real per-step curve, detect shape, project changes
# --------------------------------------------------------------------------- #
@dataclass
class Curve:
    input_by_step: list[float]   # mean input tokens at step k (k=0..L-1)
    output_by_step: list[float]
    n_mean: float
    n_min: int
    n_max: int
    marginal_tail: float         # avg input added per step near the tail (>=0)
    intercept: float             # input_by_step[0] ≈ fixed context + first user msg
    scaling: str                 # detected regime
    linearity_r2: float
    n_tasks: int
    n_calls: int
    cache_fraction: float = 0.0   # measured share of input tokens that were cache reads

    def cumulative_input(self, n: int, static_delta: float = 0.0) -> float:
        """Σ input over `n` steps, extending past observed length with the tail
        marginal, and shifting every step by `static_delta` (a re-sent change)."""
        total = 0.0
        L = len(self.input_by_step)
        last = self.input_by_step[-1] if L else self.intercept
        for k in range(max(0, n)):
            base = self.input_by_step[k] if k < L else last + self.marginal_tail * (k - (L - 1))
            total += base + static_delta
        return total

    def cumulative_output(self, n: int) -> float:
        total = 0.0
        L = len(self.output_by_step)
        tail = mean(self.output_by_step[-min(3, L):]) if L else 0.0
        for k in range(max(0, n)):
            total += self.output_by_step[k] if k < L else tail
        return total


def calibrate(traces: list[dict]) -> Curve:
    """traces: [{"calls": [{"input_tokens": int, "output_tokens": int}, ...]}, ...]"""
    lens, by_step_in, by_step_out = [], {}, {}
    n_calls = 0
    tot_in = tot_cached = 0.0
    for t in traces:
        calls = t.get("calls", [])
        lens.append(len(calls))
        for k, c in enumerate(calls):
            tin = float(c.get("input_tokens", 0))
            by_step_in.setdefault(k, []).append(tin)
            by_step_out.setdefault(k, []).append(float(c.get("output_tokens", 0)))
            tot_in += tin
            tot_cached += float(c.get("cache_read_tokens", 0))
            n_calls += 1
    if n_calls < 2:
        raise ValueError("need at least 2 calls across traces to calibrate")

    L = max(by_step_in)
    f = [mean(by_step_in[k]) for k in range(L + 1)]
    g = [mean(by_step_out.get(k, [0.0])) for k in range(L + 1)]

    diffs = [f[k + 1] - f[k] for k in range(len(f) - 1)]
    marginal_tail = max(0.0, mean(diffs[-min(3, len(diffs)):])) if diffs else 0.0

    # linearity: R^2 of a line through f vs step
    r2 = _linear_r2(list(range(len(f))), f)
    scaling = _classify(diffs, r2)

    return Curve(input_by_step=f, output_by_step=g, n_mean=mean(lens),
                 n_min=min(lens), n_max=max(lens), marginal_tail=marginal_tail,
                 intercept=f[0], scaling=scaling, linearity_r2=r2,
                 n_tasks=len(traces), n_calls=n_calls,
                 cache_fraction=min(1.0, tot_cached / tot_in) if tot_in else 0.0)


def _classify(diffs: list[float], r2: float) -> str:
    if not diffs:
        return "single-step (no compounding observed)"
    avg = mean(diffs)
    first = mean(diffs[:max(1, len(diffs) // 3)])
    last = mean(diffs[-max(1, len(diffs) // 3):])
    if any(d < -1 for d in diffs) and min(diffs) < -0.5 * abs(avg or 1):
        return "~linear in steps (context resets detected → summarization/truncation)"
    if first > 0 and last < 0.35 * first:
        return "~linear in steps (growth flattening → caching/summarization)"
    if r2 > 0.9 and avg > 0:
        return "~quadratic in steps (steady context growth — no caching detected)"
    return "super-linear in steps (path-dependent growth)"


def project_empirical(curve: Curve, *, n: int, static_delta: float = 0.0,
                      static_tokens: int, pricing: Pricing, tasks_per_day: float,
                      output_uncertainty: float = 0.35) -> Projection:
    input_per_task = curve.cumulative_input(n, static_delta)
    output_per_task = curve.cumulative_output(n)
    static_resend = static_tokens * n
    history_drag = max(0.0, input_per_task - static_resend)
    return _finish(_specless(pricing, tasks_per_day), "empirical", True,
                   static_tokens, n, input_per_task, output_per_task,
                   static_resend, history_drag, output_uncertainty,
                   scaling=curve.scaling, cache_fraction=curve.cache_fraction)


def _specless(pricing: Pricing, tasks_per_day: float) -> AgentSpec:
    return AgentSpec(pricing=pricing, tasks_per_day=tasks_per_day)


# --------------------------------------------------------------------------- #
def _finish(spec, method, exact, S, N, input_per_task, output_per_task,
            static_resend, history_drag, output_uncertainty, scaling,
            cache_fraction: float = 0.0) -> Projection:
    cf = max(0.0, min(1.0, cache_fraction))
    in_rate = spec.pricing.input_per_m * (1 - cf) + spec.pricing.cached_rate() * cf
    cost_per_task = (input_per_task / 1e6 * in_rate
                     + output_per_task / 1e6 * spec.pricing.output_per_m)
    cost_per_day = cost_per_task * spec.tasks_per_day
    cost_per_month = cost_per_day * 30.0
    out_cost = output_per_task / 1e6 * spec.pricing.output_per_m
    out_share = out_cost / cost_per_task if cost_per_task else 0.0
    spread = output_uncertainty * out_share
    return Projection(
        method=method, exact_basis=exact, static_tokens=int(round(S)), steps=N,
        input_per_task=input_per_task, output_per_task=output_per_task,
        tokens_per_task=input_per_task + output_per_task,
        static_resend=static_resend, history_drag=history_drag,
        cost_per_task=cost_per_task, cost_per_day=cost_per_day,
        cost_per_month=cost_per_month,
        band_low_month=cost_per_month * (1 - spread),
        band_high_month=cost_per_month * (1 + spread),
        scaling=scaling, tasks_per_day=spec.tasks_per_day, cache_fraction=cf,
    )


def _linear_r2(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys) or 1e-9
    ss_res = sum((ys[i] - (intercept + slope * xs[i])) ** 2 for i in range(n))
    return max(0.0, 1 - ss_res / ss_tot)


# --------------------------------------------------------------------------- #
# Diff — the wedge
# --------------------------------------------------------------------------- #
@dataclass
class CostDiff:
    base: Projection
    head: Projection
    d_tokens_per_task: float
    d_cost_per_task: float
    d_cost_per_month: float
    pct_month: float
    multiplier: float
    verdict: str
    reasons: list[str]


def diff(base: AgentSpec, head: AgentSpec, *, curve: Curve | None = None,
         max_pct: float = 20.0, max_month_usd: float | None = None,
         tok: Tokenizer | None = None) -> CostDiff:
    tok = tok or Tokenizer(head.encoding)
    base_S, base_exact = base.static_count(tok)
    head_S, head_exact = head.static_count(tok)

    if curve is not None:
        n_base = round(curve.n_mean)
        n_head = max(1, n_base + round(head.profile.steps - base.profile.steps))
        pb = project_empirical(curve, n=n_base, static_delta=0.0, static_tokens=base_S,
                               pricing=base.pricing, tasks_per_day=base.tasks_per_day)
        ph = project_empirical(curve, n=n_head, static_delta=(head_S - base_S),
                               static_tokens=head_S, pricing=head.pricing,
                               tasks_per_day=head.tasks_per_day)
    else:
        pb = project_analytic(base, tok, precounted=(base_S, base_exact))
        ph = project_analytic(head, tok, precounted=(head_S, head_exact))

    d_task_tok = ph.tokens_per_task - pb.tokens_per_task
    d_task_cost = ph.cost_per_task - pb.cost_per_task
    d_month = ph.cost_per_month - pb.cost_per_month
    if pb.cost_per_month:
        pct = d_month / pb.cost_per_month * 100.0
        mult = ph.cost_per_month / pb.cost_per_month
    elif d_month > 0:
        # Zero baseline + new spend must trip the gate, not sail through on 0%.
        pct, mult = 9999.0, float(ph.cost_per_month or 1.0)
    else:
        pct, mult = 0.0, 1.0

    reasons = []
    ds = head_S - base_S
    if abs(ds) >= 1:
        reasons.append(f"static context {('grew' if ds>0 else 'shrank')} {abs(ds):,} tok "
                       f"→ re-sent {ph.steps:.0f}×/task "
                       f"({'+' if ds>0 else '−'}{abs(ds)*ph.steps:,.0f} tok/task)")
    dn = ph.steps - pb.steps
    if abs(dn) >= 1:
        reasons.append(f"loop length {pb.steps:.0f}→{ph.steps:.0f} steps")
    if not pb.cost_per_month and d_month > 0:
        reasons.append(f"new spend introduced from a zero baseline "
                       f"(${ph.cost_per_month:,.0f}/mo)")
    reasons.append(f"observed scaling: {ph.scaling}")

    verdict = "pass"
    if pct > max_pct:
        verdict = "fail"; reasons.append(f"{pct:+.0f}% > {max_pct:.0f}% budget gate")
    elif pct > max_pct / 2:
        verdict = "warn"
    if max_month_usd is not None and ph.cost_per_month > max_month_usd:
        verdict = "fail"
        reasons.append(f"projected ${ph.cost_per_month:,.0f}/mo > ${max_month_usd:,.0f} ceiling")

    return CostDiff(base=pb, head=ph, d_tokens_per_task=d_task_tok,
                    d_cost_per_task=d_task_cost, d_cost_per_month=d_month,
                    pct_month=pct, multiplier=mult, verdict=verdict, reasons=reasons)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_projection(proj: Projection, name: str) -> None:
    exact = "exact" if proj.exact_basis else "estimated"
    print(f"[costgate] {name} — {proj.method} ({exact})")
    print(f"  tokens/task : {proj.tokens_per_task:,.0f}")
    print(f"  cost/task   : {_m(proj.cost_per_task)}")
    print(f"  cost/month  : {_m(proj.cost_per_month)}  [{_m(proj.band_low_month)} – {_m(proj.band_high_month)}]")
    print(f"  scaling     : {proj.scaling}")


def _print_diff(cd: CostDiff) -> None:
    label = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[cd.verdict]
    print(f"[costgate] {label}  {cd.pct_month:+.1f}%  {_m(cd.d_cost_per_month)}/mo")
    for r in cd.reasons:
        print(f"  · {r}")
    print(f"  base {_m(cd.base.cost_per_month)}/mo  →  head {_m(cd.head.cost_per_month)}/mo")


def main() -> None:
    import argparse, json, sys
    try:
        import yaml
    except ImportError:
        sys.exit("costgate requires pyyaml: pip install pyyaml")

    parser = argparse.ArgumentParser(prog="costgate",
                                     description="AI agent cost gate for CI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_proj = sub.add_parser("project", help="project cost for a single agent spec")
    p_proj.add_argument("spec", help="YAML spec file")
    p_proj.add_argument("--traces", help="JSON traces file (enables empirical mode)")
    p_proj.add_argument("--json", dest="as_json", action="store_true")

    p_diff = sub.add_parser("diff", help="diff two specs; exits 1 if cost gate fails")
    p_diff.add_argument("base", help="base YAML spec")
    p_diff.add_argument("head", help="head YAML spec")
    p_diff.add_argument("--traces", help="JSON traces file (enables empirical mode)")
    p_diff.add_argument("--max-pct", type=float, default=20.0,
                        help="fail if monthly cost grows by more than this %% (default 20)")
    p_diff.add_argument("--max-month", type=float, default=None,
                        help="fail if projected monthly cost exceeds this USD ceiling")
    p_diff.add_argument("--pr-comment", action="store_true",
                        help="output GitHub PR comment markdown")
    p_diff.add_argument("--json", dest="as_json", action="store_true")

    args = parser.parse_args()

    def _load_spec(path: str) -> AgentSpec:
        with open(path) as f:
            return AgentSpec.from_dict(yaml.safe_load(f))

    def _load_curve(path: str | None) -> Curve | None:
        if not path:
            return None
        with open(path) as f:
            return calibrate(json.load(f))

    if args.cmd == "project":
        spec = _load_spec(args.spec)
        curve = _load_curve(args.traces)
        if curve is not None:
            s_tok, _ = spec.static_count()
            proj = project_empirical(curve, n=round(curve.n_mean),
                                     static_tokens=s_tok,
                                     pricing=spec.pricing,
                                     tasks_per_day=spec.tasks_per_day)
        else:
            proj = project_analytic(spec)
        if args.as_json:
            print(json.dumps(proj.to_dict(), indent=2))
        else:
            _print_projection(proj, spec.name)

    elif args.cmd == "diff":
        base_spec = _load_spec(args.base)
        head_spec = _load_spec(args.head)
        curve = _load_curve(args.traces)
        cd = diff(base_spec, head_spec, curve=curve,
                  max_pct=args.max_pct, max_month_usd=args.max_month)
        if args.pr_comment:
            print(render_pr_comment(cd, head_spec.name))
        elif args.as_json:
            print(json.dumps({
                "verdict": cd.verdict,
                "pct_month": cd.pct_month,
                "d_cost_per_month": cd.d_cost_per_month,
                "reasons": cd.reasons,
            }, indent=2))
        else:
            _print_diff(cd)
        sys.exit(1 if cd.verdict == "fail" else 0)


# --------------------------------------------------------------------------- #
def _m(x: float) -> str:
    return f"${x:,.2f}" if abs(x) < 1000 else f"${x:,.0f}"


def render_pr_comment(cd: CostDiff, agent_name: str) -> str:
    icon = {"pass": "✅", "warn": "⚠️", "fail": "🛑"}[cd.verdict]
    head_word = {"pass": "within budget", "warn": "watch this", "fail": "blocks merge"}[cd.verdict]
    arrow = "▲" if cd.d_cost_per_month > 0 else ("▼" if cd.d_cost_per_month < 0 else "—")
    basis = "provider-native (calibrated)" if cd.head.method == "empirical" \
        else ("exact tokenizer" if cd.head.exact_basis else "estimated tokenizer*")
    mult_txt = f" · **{cd.multiplier:.2f}×**" if cd.multiplier >= 1.10 or cd.multiplier <= 0.9 else ""
    lines = [
        f"### {icon} costgate — `{agent_name}` · {head_word}",
        "",
        f"**Projected spend {arrow} {_m(cd.d_cost_per_month)}/mo ({cd.pct_month:+.0f}%)**{mult_txt} "
        f"at {cd.head.tasks_per_day:,.0f} tasks/day",
        "",
        "| | base | this PR | Δ |",
        "|---|--:|--:|--:|",
        f"| tokens / task | {cd.base.tokens_per_task:,.0f} | {cd.head.tokens_per_task:,.0f} | {cd.d_tokens_per_task:+,.0f} |",
        f"| cost / task | {_m(cd.base.cost_per_task)} | {_m(cd.head.cost_per_task)} | {_m(cd.d_cost_per_task)} |",
        f"| **cost / month** | {_m(cd.base.cost_per_month)} | {_m(cd.head.cost_per_month)} | **{_m(cd.d_cost_per_month)}** |",
        f"| projection range | | {_m(cd.head.band_low_month)} – {_m(cd.head.band_high_month)} | |",
        "",
        "**Why:**",
    ]
    lines += [f"- {r}" for r in cd.reasons]
    cache_note = (f" · {cd.head.cache_fraction*100:.0f}% of input billed at cache-read rate"
                  if cd.head.cache_fraction > 0.01 else "")
    lines += ["", f"<sub>basis: {basis} · static {cd.head.static_tokens:,} tok re-sent "
                  f"{cd.head.steps:.0f}×/task{cache_note} · band reflects output variance only "
                  f"(measured cost curve, not assumed)</sub>"]
    if cd.head.method == "analytic" and not cd.head.exact_basis:
        lines.append("<sub>* cold-start estimate; run `costgate pull` to calibrate "
                     "on real traces for provider-native numbers.</sub>")
    return "\n".join(lines)
