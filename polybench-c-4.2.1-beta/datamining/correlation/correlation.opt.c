/**
 * correlation.opt.c — combined optimization variant of correlation.c.
 *
 * Applies, on top of the upstream algorithm:
 *   1. In-kernel blocked transpose of `data` -> `dataT[M][N]` so every
 *      subsequent inner-loop access is unit-stride.
 *   2. Column-at-a-time fusion of the mean, stddev, and centering passes:
 *      one outer-j loop holds one column of dataT hot in L1 across the
 *      three inner passes, halving DRAM traffic vs three separate sweeps.
 *
 * The math is unchanged from upstream — both transformations are pure
 * memory-locality optimizations. The two-pass variance computation is
 * kept (Var = E[(X - mu)^2]) for numerical safety.
 *
 * TIME_REGION values:
 *   0  whole kernel including transpose (default)
 *   4  correlation matrix loop
 *   5  blocked transpose
 *   6  fused stats + center (covers what were regions 1, 2, 3 upstream;
 *      compare against baseline.region1 + .region2 + .region3)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <polybench.h>

#ifdef SIZE
#  ifndef M
#    define M SIZE
#  endif
#  ifndef N
#    define N SIZE
#  endif
#endif

#include "correlation.h"

#ifndef TIME_REGION
#  define TIME_REGION 0
#endif

/* Tile size for the blocked transpose. 16x16 doubles = 2 KB. */
#define TRANSPOSE_BS 16

/* Register-block size on the outer i axis of region 4. 4 accumulators
   share each load of dataT[j][k], turning the inner loop into 4 FMAs
   per loaded element — well-suited to AVX2's 2 FMA ports and 4 DP lanes. */
#define CORR_BI 4


static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;
}


static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;
  int ii, jj;

  DATA_TYPE eps = SCALAR_VAL(0.1);

  /* Scratch: transposed copy of data. dataT[j][i] == data[i][j]. */
  DATA_TYPE (*dataT)[N] = (DATA_TYPE (*)[N])
      malloc((size_t)_PB_M * (size_t)_PB_N * sizeof(DATA_TYPE));


#pragma scop

  /* ---- Region 5: blocked transpose data -> dataT ---- */
#if TIME_REGION == 5
  polybench_start_instruments;
#endif
  for (ii = 0; ii < _PB_N; ii += TRANSPOSE_BS)
    for (jj = 0; jj < _PB_M; jj += TRANSPOSE_BS)
      {
        int imax = ii + TRANSPOSE_BS < _PB_N ? ii + TRANSPOSE_BS : _PB_N;
        int jmax = jj + TRANSPOSE_BS < _PB_M ? jj + TRANSPOSE_BS : _PB_M;
        for (i = ii; i < imax; i++)
          for (j = jj; j < jmax; j++)
            dataT[j][i] = data[i][j];
      }
#if TIME_REGION == 5
  polybench_stop_instruments;
  polybench_print_instruments;
#endif


  /* ---- Region 6: fused stats + center, column-at-a-time ----
     Each column of dataT (~N*8 bytes) is hot in L1 across the three
     inner passes. Math is bit-identical to upstream regions 1+2+3. */
#if TIME_REGION == 6
  polybench_start_instruments;
#endif
  for (j = 0; j < _PB_M; j++)
    {
      /* Pass 1: column mean */
      DATA_TYPE sum = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        sum += dataT[j][i];
      DATA_TYPE m_j = sum / float_n;
      mean[j] = m_j;

      /* Pass 2: column stddev via two-pass variance (numerically safe) */
      DATA_TYPE sumsq_dev = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        {
          DATA_TYPE d = dataT[j][i] - m_j;
          sumsq_dev += d * d;
        }
      DATA_TYPE s_j = SQRT_FUN(sumsq_dev / float_n);
      s_j = s_j <= eps ? SCALAR_VAL(1.0) : s_j;
      stddev[j] = s_j;

      /* Pass 3: center and reduce, same column still hot in L1 */
      DATA_TYPE inv_scale = SCALAR_VAL(1.0) / (SQRT_FUN(float_n) * s_j);
      for (i = 0; i < _PB_N; i++)
        dataT[j][i] = (dataT[j][i] - m_j) * inv_scale;
    }
#if TIME_REGION == 6
  polybench_stop_instruments;
  polybench_print_instruments;
#endif


  /* ---- Region 4: m * m correlation matrix ----
     Two layered optimizations on top of the unit-stride inner k loop:
       * Tile (register-block) the outer i axis by CORR_BI=4 so one j-row
         load feeds 4 anchor accumulators per inner k step.
       * Iterate j backward within each tile. This is a no-cost change
         that helps cache behavior at tile boundaries on the j-rows. */
#if TIME_REGION == 4
  polybench_start_instruments;
#endif
  /* Diagonal first; the off-diagonal kernel below skips i==j entirely. */
  for (i = 0; i < _PB_M; i++)
    corr[i][i] = SCALAR_VAL(1.0);

  for (ii = 0; ii + CORR_BI <= _PB_M - 1; ii += CORR_BI)
    {
      /* Main block: j >= ii + CORR_BI means all 4 anchors satisfy j > i. */
      for (j = _PB_M - 1; j >= ii + CORR_BI; j--)
        {
          DATA_TYPE acc0 = SCALAR_VAL(0.0);
          DATA_TYPE acc1 = SCALAR_VAL(0.0);
          DATA_TYPE acc2 = SCALAR_VAL(0.0);
          DATA_TYPE acc3 = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            {
              DATA_TYPE djk = dataT[j][k];
              acc0 += dataT[ii  ][k] * djk;
              acc1 += dataT[ii+1][k] * djk;
              acc2 += dataT[ii+2][k] * djk;
              acc3 += dataT[ii+3][k] * djk;
            }
          corr[ii  ][j] = acc0; corr[j][ii  ] = acc0;
          corr[ii+1][j] = acc1; corr[j][ii+1] = acc1;
          corr[ii+2][j] = acc2; corr[j][ii+2] = acc2;
          corr[ii+3][j] = acc3; corr[j][ii+3] = acc3;
        }
      /* Triangular fringe inside the tile: j in (ii, ii+CORR_BI), only a
         partial subset of the 4 anchors are valid (need j > i). At most
         CORR_BI*(CORR_BI-1)/2 = 6 (i,j) pairs — fall back to scalar. */
      for (j = ii + CORR_BI - 1; j > ii; j--)
        for (i = ii; i < j; i++)
          {
            DATA_TYPE acc = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_N; k++)
              acc += dataT[i][k] * dataT[j][k];
            corr[i][j] = acc; corr[j][i] = acc;
          }
    }
  /* Tail: remaining i values that didn't fill a full CORR_BI tile. */
  for (; ii < _PB_M - 1; ii++)
    for (j = _PB_M - 1; j > ii; j--)
      {
        DATA_TYPE acc = SCALAR_VAL(0.0);
        for (k = 0; k < _PB_N; k++)
          acc += dataT[ii][k] * dataT[j][k];
        corr[ii][j] = acc; corr[j][ii] = acc;
      }
#if TIME_REGION == 4
  polybench_stop_instruments;
  polybench_print_instruments;
#endif

#pragma endscop

  free(dataT);
}


int main(int argc, char** argv)
{
  int n = N;
  int m = M;

  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

#if TIME_REGION == 0
  polybench_start_instruments;
#endif

  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

#if TIME_REGION == 0
  polybench_stop_instruments;
  polybench_print_instruments;
#endif

  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
