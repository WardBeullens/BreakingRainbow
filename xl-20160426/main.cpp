double CPU_FREQ;

#include "boost/shared_array.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/scoped_array.hpp"

#ifdef IBV
#include <vector>
#include <list>
#endif

#ifdef ISEND
#include <queue>
#endif


#if QQ == 2
   #include "gf/gf2.h"
#endif

#if QQ == 16
   #include "gf/gf16.h"
#endif

#if QQ == 31
   #include "gf/gf31.h"
#endif

#ifdef MPI
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

double GLOBAL_start_time;

#ifdef OPEN_MPI
unsigned mpi_rank, mpi_size; 
#else // #ifdef OPEN_MPI
const unsigned mpi_rank = 0;
const unsigned mpi_size = 1;
#endif // #ifdef OPEN_MPI


#include "params.h"

#include "xl.h"

#include <numa.h>

void dump_proc_status()
{
  int pid = getpid();

  char cmd[100];


  sprintf(cmd, "cat /proc/%d/status | grep Vm\n", pid);

#ifdef OPEN_MPI
  if(mpi_rank == 0)
#endif
    pid = system(cmd);
}

float get_cpu_freq()
{
   char buf[1024];

   FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "r");

   if (f != NULL)
   {
      if (fgets(buf, 1024, f) == NULL)
      {
         ECHO("error reading frequency!\n");
         fclose(f);

         return -1;
      }

      double freq = atof(buf) / 1000.0;

      fclose(f);

      return freq; 
   }
   else
   {
      f = fopen("/proc/cpuinfo", "r");
   
      do
      {
         if (strncmp("cpu MHz", buf, 7) == 0)
         {
            uint32_t pos = 0;
   
            while ((pos <= (strlen(buf)-1)) && (buf[pos] != ':'))
               pos ++;
   
            pos ++;
   
            double freq = atof(buf + pos);
   
            fclose(f);
   
            return freq;
         }
      }
      while (fgets(buf, 1024, f) != NULL);
   
      fclose(f);
   }

   return -1;
}

void setup(int argc, char * argv[], Options *options)
{
   GLOBAL_start_time = - get_ms_time();

   CPU_FREQ = get_cpu_freq();

   ECHO("running svn revision $Revision: 4129 $\n");

   ECHO("got cpu frequency from \\proc\\cpuinf: %f\n", CPU_FREQ);


#ifdef OPEN_MPI
    int size;
    int rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    mpi_size = size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mpi_rank = rank;
#endif

    dump_proc_status();

#ifdef OPEN_MPI
    DUMP(mpi_size);

#ifdef BLOCKING
    ECHO("MPI is in blocking mode...\n")
#endif

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(processor_name, &namelen);

    for (unsigned rank = 0; rank < mpi_size; rank++)
    {
       MPI_Status status;

       int pid = getpid();

       if (rank > 0)
       {

          if (mpi_rank == 0)
          {
             MPI_Recv(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   rank, 0, MPI_COMM_WORLD, &status);

             MPI_Recv(&pid, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
          }
          if (mpi_rank == rank)
          {
             MPI_Send(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   0, 0, MPI_COMM_WORLD);

             MPI_Send(&pid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          }
       }

       ECHO("rank %i is %s (%i)\n", rank, processor_name, pid);
    }

    ECHO_NL();

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef MPI_SIZE
    if (mpi_size != MPI_SIZE)
    {
        if (mpi_rank == 0)
            fprintf(stderr, 
                "%s was compiled for %i nodes!\n"
                "Please run 'mpirun ... -n %i ... %s'\n",
                argv[0], MPI_SIZE, MPI_SIZE, argv[0]);

        ABORT;
    }
#endif

#endif // #ifdef OPEN_MPI

    options->parse(argc, argv);

    if (options->bind != NULL)
       numa_run_on_node(options->bind[mpi_rank]);

#ifdef OPEN_MPI
    MPI_Bcast(&options->seed, sizeof(options->seed), MPI_BYTE, 0, 
          MPI_COMM_WORLD);
#endif // #ifdef OPEN_MPI

    srand(options->seed);

    ECHO("random seed = 0x%x\n", options->seed);

#ifdef _OPENMP
    ECHO("max threads: %d\n", omp_get_max_threads());
#endif

}

void teardown(void)
{
#ifdef OPEN_MPI
    MPI_Finalize();
#endif // #ifdef OPEN_MPI

    dump_proc_status();
}

int main(int argc, char *argv[])
{
    Options options;

    setup(argc, argv, &options);

    XL<MM,NN> xl(&options);

    try
    {
       double comp_time = 0;

       comp_time -= get_ms_time();
       xl.run();
       comp_time += get_ms_time();

       ECHO_NL();
       ECHO("total time of computation: %.3f ms\n", comp_time);
    }
    catch (BW_Exception &e) 
    { 
        ECHO_NL();
        ECHO("BW did not return solution; run all steps up to bw3.\nExiting...\n\n");
    }

    fflush(stdout);

    teardown();

    return 0;
}

