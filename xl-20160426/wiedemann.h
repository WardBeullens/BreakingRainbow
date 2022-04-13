#ifndef _WIEDEMANN_
#define _WIEDEMANN_

#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <stdlib.h>

#include "matrix_array.h"
#include "matrix_polynomial.h"

#include "options.h"

#include "matrix.h"
#include "sp_matrix.h"

#include "bw1.h"
#include "block_bm.h"
#include "bw3.h"


class BW_Exception 
{
    const char *msg;

  public:
    BW_Exception(const char *msg) : msg(msg) {}

    const char* message() { return this->msg; }
};


template <class BW1, class BW3>
class BW
{

   Options *options;

   public:

   BW(Options *options) : options(options) {}

   template <unsigned m, unsigned n, unsigned N, unsigned deg>
   void bw1_read(sparse_matrix<n, N, Z_ROW_WEIGHT> &z_sp, 
         matrix_array<m, n, deg> &ai)
   {
      FILE *fd = NULL;

      if (mpi_rank == 0)
      {
         ECHO("rank %i reading result of bw1 from file '%s'\n", 
               mpi_rank, options->bw1_file);

         fd = fopen(options->bw1_file, "r");

         if (fd == NULL)
         {
            ECHO("Could not open file %s!\n", options->bm_file);
            ABORT;
         }

         fread(&z_sp, 1, sizeof z_sp, fd);
         fread(&ai, 1, sizeof ai, fd);

         fclose(fd);
      }

      ECHO("reading done\n");

#ifdef OPEN_MPI
      ECHO("broadcasting z\n");

      MPI_Bcast(&z_sp, sizeof z_sp, MPI_BYTE, 0, MPI_COMM_WORLD);

      ECHO("broadcasting ai\n");

      MPI_Bcast(&ai, sizeof ai, MPI_BYTE, 0, MPI_COMM_WORLD);

      ECHO("broadcasting done\n");
#endif // #ifdef OPEN_MPI
   }

   template <unsigned m, unsigned n, unsigned N, unsigned deg>
   void bw1_write(sparse_matrix<n, N, Z_ROW_WEIGHT> &z_sp, 
         matrix_array<m, n, deg> &ai)
   {
      FILE *fd = NULL;

      if (mpi_rank == 0)
      {
         ECHO("writing result of bw1 to file '%s'\n", options->bw1_file);

         fd = fopen(options->bw1_file, "w");

         if (fd == NULL)
         {
            ECHO("Could not open file %s!\n", options->bm_file);
            ABORT;
         }

         fwrite(&z_sp, 1, sizeof z_sp, fd);
         fwrite(&ai, 1, sizeof ai, fd);

         fclose(fd);
      }
   }

   template <unsigned n, unsigned N, unsigned m, unsigned deg_poly, unsigned deg_ai>
   void bm_read(sparse_matrix<n, N, Z_ROW_WEIGHT> &z_sp, 
         matrix_polynomial<n, m+n, deg_poly> &min_poly,
         matrix_array<m, n, deg_ai> &ai)
   {
      FILE *fd = NULL;

      if (mpi_rank == 0)
      {
         ECHO("reading result of bm from file '%s'\n", options->bm_file);

         fd = fopen(options->bm_file, "r");

         if (fd == NULL)
         {
            ECHO("Could not open file %s!\n", options->bm_file);
            ABORT;
         }

         fread(&z_sp, 1, sizeof z_sp, fd);
         fread(&min_poly, 1, sizeof min_poly, fd);
         fread(&ai, 1, sizeof ai, fd);

         fclose(fd);
      }
   }

   template <unsigned n, unsigned N, unsigned m, unsigned deg_poly, unsigned deg_ai>
   void bm_write(sparse_matrix<n, N, Z_ROW_WEIGHT> &z_sp, 
         matrix_polynomial<n, m+n, deg_poly> &min_poly,
         matrix_array<m, n, deg_ai> &ai)
   {
      FILE *fd = NULL;

      if (mpi_rank == 0)
      {
         ECHO("writing result of bm to file '%s'\n", options->bm_file);

         fd = fopen(options->bm_file, "w");

         if (fd == NULL)
         {
            ECHO("Could not open file %s!\n", options->bm_file);
            ABORT;
         }

         fwrite(&z_sp, 1, sizeof z_sp, fd);

         fwrite(&min_poly, 1, sizeof min_poly, fd);

         fwrite(&ai, 1, sizeof ai, fd);

         fclose(fd);
      }
   }

   template <unsigned n , unsigned m, class t_sol, class orig_sys, class Mac>
   void block_wiedemann(t_sol &sol, Mac &M, orig_sys &sys)
   {
      double start_bw = get_ms_time();

      ECHO("m = %u, n = %u\n", m, n);
      ECHO("dimension: %u\n", Mac::width);
      ECHO("weight/row: %.3lf\n", (double)M.num_entries()/Mac::width);
      ECHO("Z_ROW_WEIGHT: %i\n", Z_ROW_WEIGHT);

      static const unsigned num_iter = Mac::width/m + Mac::width/n + 8;

      typedef sparse_matrix<n, Mac::width, Z_ROW_WEIGHT> z_sp_t;

      boost::scoped_ptr<z_sp_t> z_sp(new z_sp_t);
      z_sp->rand();

      typedef matrix_array<m, n, num_iter> ai_t;

      boost::scoped_ptr<ai_t> ai(new ai_t);

//#ifdef _OPENMP
//      // make z NUMA aware
//#pragma omp parallel for schedule(static) 
//      for(unsigned r = 0; r < N; r++)
//         z_sp->L[r].set_zero();
//#endif

      //////////////////////////////////////////////////////////////////


      if (options->bw1_run)
			BW1::template bw_1(*ai, num_iter, M, *z_sp);


      if (options->bw1 == OP_BW1_WRITE)
         bw1_write(*z_sp, *ai);

      if (options->bw1 == OP_BW1_READ)
         bw1_read(*z_sp, *ai);

      /////////////////////////////////////////////////////////////////

      typedef matrix_polynomial<n, m+n, max((num_iter + 1) * 5/9, 100)> min_poly_t;

      boost::scoped_ptr<min_poly_t> min_poly(new min_poly_t(*ai));

      if (options->bm_run)
         block_BM::block_bm(*min_poly, *ai);


      if ((options->bm == OP_BM_WRITE) && (mpi_rank == 0))
         bm_write(*z_sp, *min_poly, *ai);

      if (options->bm == OP_BM_READ)
         bm_read(*z_sp, *min_poly, *ai);

      ////////////////////////////////////////////////////////////////

      if (options->bw3_run)
         BW3::template bw_3<Mac::width, m, n>(sol, *min_poly, *z_sp, *ai, sys);

      ECHO("BW total time: %.3f ms\n\n", get_ms_time() - start_bw);

      if (!options->bw3_run)
         throw BW_Exception("did not finish all steps for BW");
   }    

};

#endif // ifndef _WIEDEMANN_

