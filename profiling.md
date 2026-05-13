# Profiling Plan: PolyBench `correlation` Kernel

This document captures the profiling methodology before any code is written. It covers what the kernel does, how we will instrument it, what hardware counters we will collect, and how we will interpret the results to drive optimization.

## 1. The Kernel

`polybench-c-4.2.1-beta/datamining/correlation/correlation.c` computes a Pearson correlation matrix.

- **Input**: `data[N][M]` — N observations of M variables (each column is a variable, each row is one joint observation).
- **Output**: `corr[M][M]` — symmetric matrix where `corr[i][j]` is the Pearson correlation between variable `i` and variable `j`.

Inside the `#pragma scop` block there are **four distinct loop nests**:

| Region | Work | Complexity | Notes |
|--------|------|-----------|-------|
| 1. Mean | `mean[j] = (1/N) Σ data[i][j]` | `O(N·M)` | Stride-M reads down columns |
| 2. Stddev | `stddev[j] = sqrt((1/N) Σ (data[i][j] − mean[j])²)` | `O(N·M)` | Includes per-column `sqrt` and an eps guard |
| 3. Center & reduce | `data[i][j] = (data[i][j] − mean[j]) / (sqrt(N)·stddev[j])` | `O(N·M)` | One `div` per element |
| 4. Correlation matrix | `corr[i][j] = Σ_k data[k][i] · data[k][j]` (upper triangle, mirrored) | `O(M²·N)` | Stride-N column reads — the hot loop |

Region 4 is almost certainly the dominant cost. It also has the worst memory access pattern: the innermost `k` loop reads `data[k][i]` and `data[k][j]` — two strided columns of a row-major array — so reuse depends entirely on the cache catching the columns between iterations of `j`.

## 2. Instrumentation Approach

### Goal

Time each of the four regions independently so we can identify which one(s) to optimize, while keeping the kernel's structure unchanged (realistic surrounding code, same compiler decisions, same cache state going into each region).

### Method

Create `correlation.localized.c` as a copy of upstream `correlation.c` with `#if`-guarded `polybench_start_instruments` / `polybench_stop_instruments` calls inserted **inside** the kernel around each region. A compile-time macro `TIME_REGION` selects which region's timing is active:

```c
#ifndef TIME_REGION
#  define TIME_REGION 0   // 0=whole scop, 1=mean, 2=stddev, 3=center, 4=corr
#endif
```

Skeleton:

```c
// in kernel_correlation, after #pragma scop:
#if TIME_REGION == 1
  polybench_start_instruments;
#endif
  for (j = 0; j < _PB_M; j++) { /* mean loop */ }
#if TIME_REGION == 1
  polybench_stop_instruments;
  polybench_print_instruments;
#endif

#if TIME_REGION == 2
  polybench_start_instruments;
#endif
  for (j = 0; j < _PB_M; j++) { /* stddev loop */ }
#if TIME_REGION == 2
  polybench_stop_instruments;
  polybench_print_instruments;
#endif

// regions 3 and 4 follow the same pattern
```

And in `main`, gate the whole-kernel timing so it fires only at `TIME_REGION=0`:

```c
#if TIME_REGION == 0
  polybench_start_instruments;
#endif
  kernel_correlation(...);
#if TIME_REGION == 0
  polybench_stop_instruments;
  polybench_print_instruments;
#endif
```

### Why not refactor the regions into separate functions?

A function-based refactor was considered and rejected. Keeping the original loop structure means the compiler sees the same surrounding code and makes the same inlining / aliasing / register-allocation decisions. Function calls would also change the timed region's cache state in subtle ways (spilled registers, function prologue/epilogue). The `#if`-guard approach is a strictly smaller delta from upstream.

### Why is this safe to measure?

- `polybench_start_instruments` / `polybench_stop_instruments` / `polybench_print_instruments` each fire exactly once per run for any value of `TIME_REGION`. The result parser sees exactly one line of output, same as upstream.
- All four regions run on every execution regardless of `TIME_REGION`. The data flow is identical to upstream, so the final `corr` matrix is identical.
- Region 4's trailing `corr[_PB_M-1][_PB_M-1] = 1.0` lives outside the outer `for` but inside the timed window — `stop_instruments` goes *after* that line.

## 3. Correctness Checking

The localized file is also the correctness oracle. Build it with `-DTIME_REGION=0 -DPOLYBENCH_DUMP_ARRAYS` and capture the dumped `corr[M][M]` matrix from stderr. After any optimization, rebuild and diff the new dump against the saved golden output.

- Floating-point tolerance: `~1e-6` per element (the kernel does many summations; reordering them produces small numerical differences even when the algorithm is correct).
- Standalone script `check_correctness.py`, invoked manually before re-running studies.
- Run at small `M=N` (e.g. 128 or 256) so the dump is fast and the diff is cheap.

## 4. Two Studies

We will run two parameter sweeps. Both use the existing `experiment_helper.py` framework (round-robin sweep, parallel builds, CSV checkpointing).

### Study A — Runtime (`hw4_correlation_runtime.py`)

`Study` (not `HWCounterStudy`). Measures wall-clock time per region.

Parameter sweep:
- `TIME_REGION` ∈ {0, 1, 2, 3, 4}
- `Opt_Level` ∈ {`-O0`, `-O2`, `-O3`}
- `-DM=...` `-DN=...` ∈ {128, 256, 512, 1024, 2048} (square: M == N to start; can split later)

This study answers:
- Which region dominates? (Confirmation that region 4 is the target.)
- How does each region scale with problem size?
- What does `-O3` already do for us?

### Study B — Hardware counters (`hw4_correlation_counters.py`)

`HWCounterStudy`. Same parameter sweep, focused on `TIME_REGION ∈ {1, 2, 3, 4}` initially (whole-kernel `TIME_REGION=0` is informative but is just the sum of the regions).

This study answers:
- Is region 4 compute-bound or memory-bound? Cache-bound or DRAM-bound?
- How well is it vectorizing?
- Where are the stall cycles going?

## 5. Hardware Counter Metrics

### Target machine

- CPU: Intel Xeon E-2278G (Coffee Lake, 8 cores, AVX2, no AVX-512)
- 10 hardware counters available; PAPI multiplexing supported but avoided
- Caches: L1d 32 KB/core, L2 256 KB/core, L3 16 MB shared
- Memory: dual-channel DDR4-2666, ~40 GB/s peak

### Theoretical peaks

- **Compute peak (DP, per core)**: 2 FMA ports × 4 DP lanes × 2 ops/FMA = **16 FLOPs/cycle/core**
- **DRAM bandwidth peak**: ~40 GB/s = ~5 GB/s/core when 8 cores share it
- **Machine balance** (single-core, base clock 3.4 GHz):
  - Compute: 16 × 3.4 = 54.4 GFLOP/s
  - Memory: 40 GB/s (shared, but we're single-threaded → effective ~10-20 GB/s)
  - Balance: ~3–5 FLOPs/byte (single core); ~6 FLOPs/byte (all-cores)
  - **Use ~5 FLOPs/byte as the rough threshold for compute- vs memory-bound.**

### Counter set (exactly 10, no multiplexing required)

| # | Counter | Used in |
|---|---------|---------|
| 1 | `PAPI_DP_OPS` | FLOPs/cycle, arithmetic intensity, vector quality |
| 2 | `PAPI_L3_TCM` | arithmetic intensity (DRAM bytes proxy), L3 miss rate |
| 3 | `PAPI_TOT_CYC` | IPC, FLOPs/cycle, %Stall denominators |
| 4 | `PAPI_TOT_INS` | IPC |
| 5 | `PAPI_LST_INS` | L1 miss rate denominator |
| 6 | `PAPI_L1_DCM` | L1 pressure, L2 miss rate denominator |
| 7 | `PAPI_L2_TCM` | L2 pressure |
| 8 | `PAPI_L3_TCA` | L3 miss rate denominator |
| 9 | `PAPI_RES_STL` | %Stall |
| 10 | `PAPI_VEC_DP` | vIPC, vector quality |

### Derived metrics and formulas

```python
def flops_per_cycle(PAPI_DP_OPS, PAPI_TOT_CYC):
    """DP FLOPs per cycle. Architectural peak on AVX2 = 16. """
    return PAPI_DP_OPS / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0

def arithmetic_intensity(PAPI_DP_OPS, PAPI_L3_TCM):
    """DRAM-level AI: FLOPs per byte read from DRAM (L3 miss × 64B line)."""
    return PAPI_DP_OPS / (PAPI_L3_TCM * 64) if PAPI_L3_TCM > 0 else float('inf')

def ipc(PAPI_TOT_INS, PAPI_TOT_CYC):
    """All instructions per cycle; integer + FP + loads + branches."""
    return PAPI_TOT_INS / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0

def vipc(PAPI_VEC_DP, PAPI_TOT_CYC):
    """DP vector instructions per cycle."""
    return PAPI_VEC_DP / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0

def dp_ops_per_vec_instr(PAPI_DP_OPS, PAPI_VEC_DP):
    """Effective lane utilization. AVX2 DP peak = 8 (FMA × 4 lanes).
       ~4 means mul or add only (no FMA fusion). ~1 means scalar."""
    return PAPI_DP_OPS / PAPI_VEC_DP if PAPI_VEC_DP > 0 else 0

def l1_miss_rate(PAPI_L1_DCM, PAPI_LST_INS):
    """Fraction of load/stores that miss in L1. PAPI_L1_DCA isn't
       available on this CPU so we use LST_INS as a request proxy."""
    return (PAPI_L1_DCM / PAPI_LST_INS) * 100 if PAPI_LST_INS > 0 else 0

def l2_miss_rate(PAPI_L2_TCM, PAPI_L1_DCM):
    """L2 only sees L1 misses, so L1_DCM is the L2 access count."""
    return (PAPI_L2_TCM / PAPI_L1_DCM) * 100 if PAPI_L1_DCM > 0 else 0

def l3_miss_rate(PAPI_L3_TCM, PAPI_L3_TCA):
    """L3 miss rate — anything missing here goes to DRAM."""
    return (PAPI_L3_TCM / PAPI_L3_TCA) * 100 if PAPI_L3_TCA > 0 else 0

def stall_rate(PAPI_RES_STL, PAPI_TOT_CYC):
    """Fraction of cycles where some resource stalled issue."""
    return (PAPI_RES_STL / PAPI_TOT_CYC) * 100 if PAPI_TOT_CYC > 0 else 0
```

### Metric interpretation rubric

#### Roofline lens (arithmetic intensity)

- **AI > 5 FLOPs/byte**: workload is theoretically compute-bound. If `flops_per_cycle` is still well below 16, the bottleneck is not memory — look at vectorization quality, dependency chains, or front-end issue width.
- **AI < 5 FLOPs/byte**: workload is theoretically memory-bound. Optimization should focus on reducing DRAM traffic (reuse, blocking, layout) rather than squeezing more FLOPs out per cycle.

#### Stall decomposition lens

| %Stall | L3 miss rate | L1/L2 miss rate | Likely bottleneck |
|--------|--------------|-----------------|-------------------|
| High | High | — | **DRAM-bound** — fix data layout / reduce passes |
| High | Low | High | **Cache-bound** — tileable / blockable |
| High | Low | Low | Dependency chain, divider/sqrt latency, branch mispredict |
| Low | — | — | Front-end or issue bottleneck — check IPC + vIPC |

#### IPC vs FLOPs/cycle cross-check

| IPC | FLOPs/cycle | Reading |
|-----|-------------|---------|
| High (>2) | High (>4) | Near-peak — well-utilized |
| High (>2) | Low (<1) | Lots of non-FP instructions — addressing overhead, or unvectorized scalar FP |
| Low (<1) | Low (<1) | Stalled — go to the stall lens above |
| Low | High | Unusual; dense vector FMAs (one instr = many FLOPs) |

#### Vectorization quality (`dp_ops_per_vec_instr`)

| Value | Meaning |
|-------|---------|
| ~8 | AVX2 FMA on full 4-lane DP vectors — ideal |
| ~4 | Vectorized but no FMA fusion (mul or add only) |
| ~1 | Scalar FP — vectorizer failed |
| 0 | No vector DP instructions at all |

## 6. Predicted Results

Useful as a sanity check — if the measurements don't roughly match, the instrumentation is probably wrong.

| Region | Predicted AI | Predicted classification | Reason |
|--------|-------------|--------------------------|--------|
| 1 (mean) | ~0.1 FLOP/B | DRAM-bound | One add per 8-byte read, single pass over data |
| 2 (stddev) | ~0.4 FLOP/B | DRAM-bound | ~3 FLOPs per element, single pass |
| 3 (center) | ~0.1 FLOP/B | DRAM-bound | Read + write, 2 FLOPs per element |
| 4 (corr, in theory) | high (~N/2 FLOP/B with full reuse) | compute-bound | M²N FLOPs over M×N + M² bytes |
| 4 (corr, in practice) | likely much lower | likely cache- or DRAM-bound | Stride-N column reads break reuse |

The gap between region 4's *theoretical* and *measured* AI is the optimization opportunity. A successful transformation (transpose `data`, tile the outer loops, or both) should pull measured AI up toward the theoretical curve.

## 7. Optimization Loop

1. **Baseline.** Run Study A and Study B on `correlation.localized.c` at `TIME_REGION ∈ {1..4}` across all `Opt_Level` × `M=N` combinations. Save CSVs.
2. **Identify target.** Confirm region 4 dominates (Study A). Read its metrics (Study B) to classify the bottleneck.
3. **Pick a transformation.** Match it to the bottleneck:
   - DRAM-bound → reduce passes, transpose data so inner loop is unit-stride
   - Cache-bound → tile the `i`/`j` loops on cache-line / L1 / L2 boundaries
   - Compute-bound but low FLOPs/cycle → check vectorization, unroll, register-block
4. **Apply.** Edit `correlation.localized.c` (the only source we own).
5. **Correctness gate.** Run `check_correctness.py`. If it fails, fix before re-measuring.
6. **Re-measure.** Re-run Study A and Study B. Diff against baseline CSVs.
7. **Iterate.** Repeat from step 2 until returns flatten or the roofline is reached.

## 8. Known Caveats

- **`PAPI_DP_OPS` counts `divsd` and `sqrtsd` as 1 FLOP each** despite their high latency. Region 2 has M sqrts, region 3 has N×M divs. AI numbers are slightly optimistic for those regions; region 4 (pure FMA) is unaffected.
- **`PAPI_L1_DCA` is not available** on this CPU. We approximate L1 access count with `PAPI_LST_INS` (load/store instructions retired). Close, but each vector instruction is one load/store *instruction* covering multiple lanes — so this slightly undercounts at the byte level. Use as a relative metric.
- **`PAPI_L3_TCM × 64` is a proxy for DRAM bytes**, not a direct measurement. It misses prefetch traffic that bypasses L3, but on Coffee Lake most prefetches go through L2 → L3 anyway. Good enough for relative comparisons across runs.
- **Wall-clock measurements include all four regions running.** Even when `TIME_REGION=4`, the untimed regions still execute beforehand, warming caches. This is intentional (realistic timing context) but means region 4's measured time is what it takes *after* the data has just been written by region 3 — best-case cache state.
- **First sweep iteration is usually noisy** (cold caches, dynamic CPU frequency ramp). The framework's trimmed mean (`compute_trimmed_mean`) drops the min and max across runs; with `RUNS_PER_EXPERIMENT=10` the middle 8 give a stable estimate.
