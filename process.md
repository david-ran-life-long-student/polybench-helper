# Process: Human–AI Teaming on the `correlation` Kernel Optimization

This document records how the work in this project was actually carried out as a collaboration between the human author (the "user") and a coding-focused AI assistant (Claude). The intent is to give the advisor an honest picture of the division of labor, the decision-making pattern, and the moments where each side's contribution was load-bearing.

## Overview

The task — optimize PolyBench's `correlation` kernel within the function-body constraint — broke naturally into three phases: **understand**, **measure**, **transform**. The user shaped the goals and judgments; the AI did the mechanical analysis, drafted artifacts, and implemented. At several specific decision points, the user's judgment overrode AI proposals or redirected the work in ways the AI would not have chosen alone. Conversely, on bookkeeping, code writing, and mathematical/cache-behavior derivation, the AI moved faster than the user would have alone. The final result reflects neither side individually.

## The user's role: ideation, judgment, and direction

The user acted as the **architect and decision-maker**, not as a passive consumer of AI output. Specific load-bearing contributions:

### Domain framing and constraints
- Established the working constraint that all edits must live inside `kernel_correlation` — without this, the AI would have proposed (and likely been allowed to take) a much simpler route by changing the storage layout of `data` in the call site. The constraint forced the more interesting in-kernel-transpose approach.
- Decided the structure of the measurement infrastructure: separate baseline vs opt source files, `#ifdef`-gated timing rather than function refactoring, sizes to sweep, and the file-naming convention (`opt` over `phaseN` once optimizations were combined).

### Critical judgment and rejection of risky proposals
- The clearest example: the AI proposed Phase 2, a single-pass variance computation via the `Var(X) = E[X²] − E[X]²` identity, complete with derivation and a floating-point caveat. The user examined the same information and explicitly ruled against it: *"I might have to rule against this optimization, it doesn't guarantee correctness."* This was a judgment call about acceptable risk — the AI had presented the trade-off but not made the decision. Without the user's intervention the kernel would have shipped with a numerical-stability landmine.
- Similarly directed the AI to keep numerically-safe versions of computations even when faster algorithmic alternatives existed.

### Original ideas the AI did not propose
- **Transpose the `data` matrix** — the most consequential idea in the whole project. The user proposed it during the very first optimization discussion: *"can we transpose the data matrix to convert the row iterations to column for cache friendliness?"* The AI had been analyzing surface-level loop fusion at the time; the transpose was a strictly broader move because it fixed three of the four regions at once (1, 2, and 4 all had stride-M access patterns that became stride-1 after the transpose). When the in-function-only constraint was later clarified — which would normally rule out a storage-layout change — the user insisted: *"I'm still into the transpose idea. it can even help region 4, right?"* That insistence under a constraint that should have killed the idea was the key move; it forced the in-kernel-allocated `dataT` scratch buffer that became the foundation of every subsequent optimization. The single largest measured win in the project (region 4 going from O(M²·N) cache-thrashing to vectorized unit-stride FMA) traces directly to this proposal.
- **Backward-`j` iteration** in region 4: the user suggested this as a way to preserve the next anchor row in L1 across outer-loop boundaries. The AI confirmed the mechanism (boustrophedon / snake iteration) but did not surface it independently.
- **FLOPs per cycle** instead of GFLOPs per second as the headline throughput metric: the user pushed back on the AI's initial proposal because the latter required guessing the clock rate under turbo. The user's framing was strictly better and the AI adopted it.
- **`DP_per_vec` (lane utilization)** was added at the user's prompt, fitting naturally into the existing metric set the AI had drafted.

### Detecting AI errors
- Caught a non-compiling C file with *"the localized code doesn't compile at all, please check."* The AI had verified compilation under one invocation pattern (`-DM=128 -DN=128` as separate shell arguments) but missed that the experiment-helper framework passes each `Mutable` value as a single `argv` entry, which gcc parses as one malformed `-D` flag. The user's prompt led the AI to reproduce the failure mode and ship a clean fix (`-DSIZE=...` with a derived `M`/`N` in the source).

### Strategic shaping of artifacts
- Asked for a consolidated `correlation.opt.c` rather than per-phase files once it became clear all optimizations would ship together.
- Requested the `compare_runtime.py` script merge experiment setup and analysis into one entry point — a structural choice the AI would have left as two separate scripts.
- Requested the profitability-table format and the analysis plots that go in the final report.

## The AI's role: implementation, mechanical analysis, and surfacing options

The AI acted as a fast, broad **implementation and explanation partner**. Specific contributions:

### Mechanical analysis and explanation
- Walked through the kernel's four loop nests in detail, building the access-pattern table (stride-1 vs stride-M, row vs column) that anchored every later optimization decision.
- Derived the `Var(X) = E[X²] − E[X]²` identity from first principles when the user asked about the invariance, including the floating-point cancellation analysis that motivated rejection.
- Computed cache-fit budgets (L1=32 KB, one column at N=2048 = 16 KB → fits, etc.) and predicted which optimizations would help at which sizes.
- Quantified expected wins (e.g., "stats+center DRAM traffic goes from ~4·M·N·8 to ~2·M·N·8 bytes") before measurement, giving the user a way to sanity-check measured results.

### Identifying optimization opportunities (when prompted)
- After the user asked about tiling region 4, the AI proposed and implemented **register-blocking with CORR_BI=4** plus a triangular-fringe cleanup and a tail loop — a non-trivial code structure that delivered the ~3× win on top of the transpose at N=1024.
- Recognized the boundary between *tileable* (region 4, with per-(i,j) accumulators) and *non-tileable due to reduction dependency* (regions 1–3, where the column-mean scalar blocks stripmining) and explained why the user's stripmining instinct on the stats regions could not work.

### Drafting and writing
- Authored the C source for `correlation.localized.c` and `correlation.opt.c`, including the `#ifdef TIME_REGION` instrumentation harness, the blocked transpose, the column-fusion loop, and the register-blocked region 4.
- Wrote all Python infrastructure: `check_correctness.py`, `compare_runtime.py`, the three analysis scripts (`analyze_speedup.py`, `analyze_breakdown.py`, `analyze_hwmetrics.py`), and `run_experiments.sh`.
- Wrote the project documentation (`profiling.md`, `optimization.md`, this `process.md`) at the user's request, condensing the conversation into reference material.

### Surfacing options for user judgment
- Used structured prompts (the `AskUserQuestion` flow) at decision points to give the user concrete alternatives with trade-offs rather than asking open-ended questions — e.g., "separate file, in-place `#ifdef`, or duplicate directory?" with the pros and cons of each.

## How decisions actually got made

The recurring pattern was:

1. The user posed a goal or asked a question.
2. The AI produced a structured response: typically several alternatives with trade-offs, or a single proposal with explicit caveats.
3. The user picked, modified, rejected, or counter-proposed.
4. The AI implemented the chosen direction and reported back results.

Two examples make the pattern concrete:

**Example 1 — Single-pass variance (rejected).** AI proposed Phase 2 with derivation and a numerical-safety caveat. User rejected on the caveat. AI then asked whether Phase 3 (column-at-a-time cache fusion) could stand alone without Phase 2. AI confirmed it could and showed the DRAM-traffic math. Phase 3 became the standalone change. Final code retains the numerically-safe two-pass variance.

**Example 2 — Region 4 optimization.** User asked about loop tiling and proposed backward-`j` iteration as a complementary idea. AI confirmed both mechanisms, then proposed combining tiling with **register-blocking** as the cleanest realization of "tiling for region 4" given AVX2's FMA throughput characteristics. User approved. AI implemented, hit a stale-variable compile error, fixed it, verified correctness, and measured the spot-check speedup (3× on top of the transpose at N=1024).

**Example 3 — The transpose (the foundation).** User asked early on whether the `data` matrix could be transposed for cache friendliness, initially framed as a question about regions 1 and 2. AI confirmed it would help those and connected it to region 4 as well. The in-function-only constraint then emerged, which would normally have ruled out a layout change. The user explicitly chose to keep the transpose alive *inside the kernel* as a one-time scratch copy, and asked the AI to confirm it would still help region 4 in that form. AI confirmed and implemented the blocked transpose plus the rewritten regions on top of `dataT`. This unblocked everything that followed: unit-stride inner loops in all four regions, which the column-at-a-time fusion and register-blocking then exploited.

The user did not write code in this project. The AI did not make the calls that mattered. Neither would have produced the same result alone — and the result is better than either would have produced alone in the same time budget.

## What worked, and what to take from it

A few patterns from this collaboration are worth naming:

- **Concrete trade-offs beat open-ended discussion.** When the AI presented forks as labeled alternatives with one-line consequences, the user could pick quickly. When the AI presented a long analysis without a clear choice, the conversation slowed.
- **The user's "yes" was as important as the AI's drafts.** The transpose was a user proposal that the AI confirmed but did not initiate, and the user defended it under a constraint that should have ruled it out. It is the foundation of every other optimization in `correlation.opt.c`. Without it, neither the column-at-a-time fusion nor the register-blocked region 4 would have been possible — both rely on the unit-stride access pattern that the transpose creates.
- **The user's "no" was as important as the AI's drafts.** The single-pass variance hack would have shipped if the user had not rejected it. The numerically-safe version exists because of a human judgment call about acceptable risk.
- **The AI surfaced opportunities the user knew to look for.** Register-blocking with four accumulators is a well-known technique; the AI proposed it specifically because the user had primed the discussion by asking about tiling. The pattern was "user names a goal in their vocabulary → AI maps it to a concrete technique → user decides whether to spend the complexity."
- **Verification was not optional.** Every optimization passed `check_correctness.py` against the upstream kernel before being measured. The AI ran the gate after each edit; the user requested it as a hard rule in the verification protocol. Several rounds of cleanup happened because of compile errors caught immediately — they never made it into the measurement pipeline.

The final artifact — a ~20× speedup at N=1024 and ~27× at N=2048, bit-identical to upstream output, with full measurement infrastructure and documented optimization rationale — is the product of this human/AI teaming.The work itself was shared in the way this document describes.
