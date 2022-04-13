#ifndef _XL_H_
#define _XL_H_

#include "matrix.h"
#include "macaulay_matrix.h"

#include "util.h"

#include "options.h"

#include "wiedemann.h"

#include "bw1.h"
#include "bw3.h"

#include "binomial.h"

template <unsigned m, unsigned n>
class XL
{
  public:
#if QQ == 2
    typedef matrix<Sum_Binomial<NN, 2>::value, MM> orig_sys;
#else // #if QQ == 2
    typedef matrix<Binomial<NN + 2, 2>::value, MM> orig_sys;
#endif // #if QQ == 2


    typedef matrix<orig_sys::n, orig_sys::m> orig_sys_transp;

    typedef matrix<NN, NSOL> sol_t;


    Options *options;

    XL(Options *options) : options(options) {}

    void rand_sys(orig_sys &sys)
    {
       orig_sys_transp sys_tr;
       sys_tr.rand(); 

       gfv<NN> sol;
       sol.rand(); 

       if (mpi_rank == 0)
          sol.dump(stdout);

       monomial<NN> mon;

       gfv<MM> sum_v;
       sum_v.set_zero();


       mon.step();

       for(unsigned i = 1; i < sys_tr.n; i++)
       {
          gfv<MM> tmp_v = sys_tr.L[i];

          for(unsigned j = 0; j < NN; j++)
             for(unsigned d = 0; d < mon[j]; d++)
                tmp_v *= sol[j];

          mon.step();

          sum_v += tmp_v;
       }

       sys_tr.L[0] = -sum_v;

       sys_tr.transpose(sys);
    }

    void read_challenge(orig_sys &sys)
    {
       ECHO("reading challenge from file '%s'\n", options->challenge_file);

       FILE *fd = fopen(options->challenge_file, "r");

       if (fd == NULL)
       {
          ECHO("Could not open file %s!\n", options->challenge_file);
          ABORT;
       }

       sys.read_challenge(fd);

       fclose(fd);
    }

    void read_sys(orig_sys &sys)
    {
       ECHO("reading system from file '%s'\n", options->system_file);

       if (mpi_rank == 0)
       {
          FILE *fd = fopen(options->system_file, "r");

          if (fd == NULL)
          {
             ECHO("Could not open file %s!\n", options->system_file);
             ABORT;
          }

          fread(&sys, 1, sizeof sys, fd);

          fclose(fd);
       }

#ifdef OPEN_MPI
      ECHO("broadcasting sys\n");

      MPI_Bcast(&sys, sizeof sys, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif // #ifdef OPEN_MPI
    }

    void write_sys(orig_sys &sys)
    {
       if (mpi_rank != 0)
          return;

       ECHO("writing system to file '%s'\n", options->system_file);

       FILE *fd = fopen(options->system_file, "w");

       if (fd == NULL)
       {
          ECHO("Could not open file %s!\n", options->bm_file);
          ABORT;
       }

       fwrite(&sys, 1, sizeof sys, fd);

       fclose(fd);
    }


    void run()
    {
        typedef macaulay_matrix<orig_sys, NN, DD > Mac;

        ECHO_NL();
        ECHO("########## running XL ############");
        ECHO_NL();
        ECHO_NL();

        DUMP(QQ);
        DUMP(MM);
        DUMP(NN);
        DUMP(DD);
        DUMP(orig_sys::m);
        DUMP(Mac::width);
        DUMP(Mac::num_rb);
        DUMP(BW_M);
        DUMP(BW_N);

        fflush(stdout);

        double num_it = (double)Mac::width/(double)BW_M + (double)Mac::width/(double)BW_N + 8.0;

        DUMP(Mac::width/BW_M);
        DUMP(num_it);

        double exp_bm = ((num_it*num_it) / 4.0) * (double)(2*BW_N*BW_N*BW_N);

        ECHO("\n");
        ECHO("expected cost BW1: %.0f\n", 
              (double)((double)Mac::width * (double)orig_sys::m + (double)(BW_N*BW_N))
            * (double)BW_N * num_it);
        ECHO("expected cost BM:  %.0f\n", exp_bm);
        ECHO("\n");


        orig_sys sys;

        if (options->challenge == OP_CHALLENGE)
           read_challenge(sys);
        else if (options->system == OP_SYS_READ)
           read_sys(sys);
        else
           rand_sys(sys); 

        if (options->system == OP_SYS_WRITE)
           write_sys(sys);


        boost::scoped_ptr<Mac> M(new Mac(sys));
        sol_t sol;

#ifndef MPI
        BW<BW1, BW3_opt> bw(options);

        bw.template block_wiedemann<BW_N, BW_M>(sol, *M, sys);
#else
		  BW<BW1_ALGO, BW3_opt> bw(options);

	     bw.template block_wiedemann<BW_N, BW_M>(sol, *M, sys);
#endif
    }

};

#endif

