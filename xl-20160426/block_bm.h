#ifndef _BLOCK_BM_
#define _BLOCK_BM_

#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <stdlib.h>

#include <time.h>

#include "matrix_polynomial.h"
#include "matrix.h"

class block_BM
{
   public:
   
   template <unsigned m, unsigned n, unsigned deg_pol, unsigned deg_ai>
   static void block_bm(matrix_polynomial<n, m+n, deg_pol> &min_poly,
                        const matrix_array<m, n, deg_ai> &ai)
   {
       ECHO("running Coppersmith's BM.\n");
   
       double start_time = get_ms_time();
       double t_last = start_time;
   
       uint64_t sum_t = 0;
       uint64_t num_mults = 0;
   
       uint64_t exp = (ai.deg * ai.deg) >> 1;
   
   
       for (unsigned t = min_poly.max_nom_deg + 1; t < ai.deg; t++)
       {
          min_poly.update();
          min_poly.countCoffMat(ai, t);
   
          // ------- print some stats  -------------------------
          num_mults += (uint64_t)(2*n*n*n) * (uint64_t)(min_poly.max_nom_deg + 1);
          sum_t += t;
   
          double time = get_ms_time();
   
          if (time - t_last > 5000)
          {
             double t_diff = time - start_time;
   
             t_last = time;
   
             ECHO("(%i/%i), remaining: %.2f s, mults/cycle: %.2f\n",
                   t, ai.deg,
                   (t_diff / (double)sum_t * (double)exp - t_diff) / 1000,
                   num_mults / (t_diff * (double)CPU_FREQ * 1000.0));
          }
          // ---------------------------------------------------
       }
   
       min_poly.allgather();
   
       double total_time = get_ms_time() - start_time;
   
       ECHO("BM time: %.3f\n", total_time);
       ECHO_NL();
   }
};

#endif // ifndef _BLOCK_BM_

