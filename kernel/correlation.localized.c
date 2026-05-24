/**
 * correlation.localized.c — instrumented copy of correlation.c
 *
 * Adds compile-time selection of which loop region inside the kernel gets
 * timed/profiled. The whole kernel still runs on every execution so the
 * data flow (and therefore the final corr matrix) is identical to upstream.
 *
 *   -DTIME_REGION=0   whole kernel (default, identical to upstream)
 *   -DTIME_REGION=1   mean loop only
 *   -DTIME_REGION=2   stddev loop only
 *   -DTIME_REGION=3   center/reduce loop only
 *   -DTIME_REGION=4   correlation matrix loop only
 *
 * Exactly one start/stop/print pair fires per execution regardless of
 * TIME_REGION, so the polybench result parser sees one line of output.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <polybench.h>

/* Allow runners to set both M and N from a single -DSIZE=... flag.
   correlation.h skips its preset table if M and N are already defined. */
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

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop

  /* ---- Region 1: column means ---- */
#if TIME_REGION == 1
  polybench_start_instruments;
#endif
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }
#if TIME_REGION == 1
  polybench_stop_instruments;
  polybench_print_instruments;
#endif


  /* ---- Region 2: column standard deviations ---- */
#if TIME_REGION == 2
  polybench_start_instruments;
#endif
  for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }
#if TIME_REGION == 2
  polybench_stop_instruments;
  polybench_print_instruments;
#endif


  /* ---- Region 3: center and reduce ---- */
#if TIME_REGION == 3
  polybench_start_instruments;
#endif
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }
#if TIME_REGION == 3
  polybench_stop_instruments;
  polybench_print_instruments;
#endif


  /* ---- Region 4: m * m correlation matrix ---- */
#if TIME_REGION == 4
  polybench_start_instruments;
#endif
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#if TIME_REGION == 4
  polybench_stop_instruments;
  polybench_print_instruments;
#endif

#pragma endscop

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
