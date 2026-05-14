# Optimization Strategy: PolyBench `correlation` Kernel

This document captures the optimizations applied to `correlation.opt.c` and their measured impact. Companion `profiling.md` describes the measurement methodology.

## Constraint

All edits live inside `kernel_correlation`. We cannot change the storage layout of `data`, the kernel signature, the call site in `main`, or any surrounding scaffolding. Local scratch allocations inside the kernel are allowed.

## Bottleneck recap

Inside `data[N][M]` (row-major), the upstream kernel's four regions access memory like this:

| Region | Inner read | Stride | Pattern |
|--------|-----------|--------|---------|
| 1 mean | `data[i][j]` | M | column-walk |
| 2 stddev | `data[i][j]` | M | column-walk |
| 3 center | `data[i][j]` | 1 | row-walk ✓ |
| 4 corr | `data[k][i]`, `data[k][j]` | M × 2 | two column-walks |

Three of four regions walk columns of a row-major array. Region 4 does this `O(M²)` times — by far the dominant cost.

## Applied optimizations

All three transformations live in `correlation.opt.c`. The math is unchanged from upstream; everything below is a pure memory-locality / vectorization optimization.

### 1. In-kernel blocked transpose

At the top of `kernel_correlation`, allocate a scratch `dataT[M][N]` and copy `data` into it transposed using a 16×16 blocked transpose (so the transpose itself is cache-friendly). All four regions then read/write `dataT` instead of `data`, and every inner-loop access becomes unit-stride.

- Regions 1, 2 (mean, stddev): inner reads of `dataT[j][i]` are stride-1 → vectorizable.
- Region 3 (center): inner reads of `dataT[j][i]` are stride-1 after swapping outer/inner loop order.
- Region 4 (corr): inner reads of `dataT[i][k]` and `dataT[j][k]` are both stride-1 → compiler emits AVX2 FMA on the inner reduction; prefetcher hides DRAM latency.

Cost: one extra `O(M·N)` pass for the transpose plus `M·N·8` bytes of scratch heap. For M ≥ ~64 this is negligible against the region 4 savings. Introduces a new measurable region (`TIME_REGION=5`).

### 2. Column-at-a-time fusion of stats + center

The upstream kernel sweeps the full `dataT` three separate times (mean, stddev, center), each from DRAM since `dataT` is ~32 MB at 2048² — bigger than the 16 MB L3. Fuse the three outer-`j` loops into one, with three inner passes per column:

```c
for (j = 0; j < M; j++) {
    /* pass 1: mean of column j */
    /* pass 2: stddev of column j (two-pass variance — numerically safe) */
    /* pass 3: center column j */
}
```

Each column (~16 KB at N=2048, half of L1) is loaded once and stays L1-resident for passes 2 and 3. DRAM traffic for the stats+center bundle goes from ~4·M·N·8 bytes (three cold reads + one write) to ~2·M·N·8 bytes (one cold read + one write-back). The original two-pass variance is preserved — we considered the `E[X²] − E[X]²` single-pass identity but rejected it due to catastrophic-cancellation risk.

The three original regions can no longer be timed individually (they're interleaved per column), so a single `TIME_REGION=6` measures the whole fused block. Compare against the sum of baseline regions 1+2+3.

### 3. Region 4: register-block on `i` + backward `j`

After the transpose, region 4's inner `k` loop is already unit-stride. Two further changes squeeze it further:

**Register block, `CORR_BI=4`**: process 4 consecutive anchor rows simultaneously. Each iteration of the inner `k` loop does one load of `dataT[j][k]` feeding four FMA accumulators (`acc0..acc3`), one for each `dataT[ii..ii+3][k]`. AVX2 vectorizes each accumulator on 4-wide DP lanes, and the four independent FMA chains keep both FMA ports busy. The four anchor rows stay live in registers across the inner loop.

```c
for (k = 0; k < N; k++) {
    DATA_TYPE djk = dataT[j][k];
    acc0 += dataT[ii  ][k] * djk;
    acc1 += dataT[ii+1][k] * djk;
    acc2 += dataT[ii+2][k] * djk;
    acc3 += dataT[ii+3][k] * djk;
}
```

**Backward `j`** within each tile. At the end of one `ii` block, the last `j` used is the smallest in range, leaving its row hot for the next iteration boundary. Zero-cost change: just a loop direction flip.

Two scalar cleanup paths: (a) the triangular fringe `j ∈ (ii, ii+4)` inside each tile (≤6 (i,j) pairs), and (b) a tail loop for any `i` values that don't fill a full CORR_BI tile when `M mod 4 ≠ 0`.

## Profitability — spot checks

The full sweep is produced by `compare_runtime.py` (see verification protocol below). The numbers here are point measurements taken locally at `-O3` after each optimization landed; the full grid will report the same shape over (size, opt-level).

### Region-level comparison at N=1024, -O3

| Region | Baseline (ms) | Opt (ms) | Speedup |
|---|---|---|---|
| stats+center (base R1+R2+R3 vs opt R6) | 5.18 | 1.38 | **3.75×** |
| corr matrix (base R4 vs opt R4) | 1994.3 | 95.9 | **20.8×** |
| transpose (new — opt R5 only) | — | ~5 | — |
| whole kernel (base R0 vs opt R0) | ~1995 | 103.0 | **~19×** |

### Region 4 scaling

| Size | Baseline R4 (ms) | Opt R4 (ms) | Speedup |
|---|---|---|---|
| 512  | 233.2  | (transpose-only spot: 38.7) | ~6× transpose-only |
| 1024 | 1994.3 | 95.9  | **20.8×** |
| 2048 | 21500.9 | 796.3 | **27.0×** |

The speedup grows with N because region 4 is `O(M²·N)` and the baseline's stride-M memory pattern degrades faster than the work increases — the original kernel becomes more DRAM-bound at larger sizes, where the transpose + register-block extracts proportionally more.

### Contribution breakdown for region 4 at N=1024 -O3

Approximate, from spot measurements as optimizations landed:

| Variant | R4 (ms) | Cumulative speedup vs baseline |
|---|---|---|
| Baseline | 1994 | 1× |
| + transpose only | ~290 | ~6.9× |
| + register-block + backward-j | 95.9 | **20.8×** |

The register-block + backward-j step adds another ~3× on top of the transpose, indicating the post-transpose version was still under-utilizing FMA throughput (likely load-bound on a single anchor row).

## Considered and rejected

- **Single-pass variance (`Var = E[X²] − E[X]²`)**: would halve stats memory traffic. Rejected — catastrophic cancellation when mean is large relative to variance. Safe for PolyBench's synthetic init but unsafe as a general pattern; the safe alternative (Welford) is more complex and harder to vectorize. The two-pass column fusion (optimization 2 above) recovers most of the cache benefit without touching the math.
- **Storing `data` column-major from the start**: would obviate the transpose, but blocked by the constraint that we can only edit inside the kernel.
- **`k`-axis tiling of region 4**: would help if the register-blocked version were still memory-bound at the inner loop. Spot checks suggest it isn't — adding k-tiling would also force `corr[i][j] = 0` initialization to be split out, complicating the loop without obvious benefit. Reconsider only if measured FLOPs/cycle stays well below AVX2 peak (16) at the largest sizes.
- **Software prefetch in region 4**: hardware prefetcher handles unit-stride streams well; manual prefetch would mostly add overhead.

## Verification protocol

1. **Correctness gate**: `python3 check_correctness.py opt`. Must pass at 1e-6 tolerance. Currently bit-identical (max diff 0.000e+00).
2. **Runtime study + table**: `python3 compare_runtime.py`. Builds and runs both baseline (`correlation.localized.c`) and opt (`correlation.opt.c`) studies via `experiment_helper`, then prints the speedup table.
3. **HW counter study**: `python3 hw4_correlation_counters.py` (baseline) and `python3 hw4_correlation_counters_opt.py` (opt). Compare metrics like `FLOPs_per_cycle`, `AI`, `%L3m` between the two CSVs.

CSV outputs live in `build/runtime/`, `build/runtime_opt/`, `build/counters/`, `build/counters_opt/` — preserved across reruns thanks to the framework's caching of completed (config, run-iteration) pairs.
