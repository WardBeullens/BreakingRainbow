#ifdef MPI
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#endif

#ifdef MPI
extern unsigned mpi_rank, mpi_size; 
#else
const unsigned mpi_rank = 0;
const unsigned mpi_size = 1;
#endif

extern double GLOBAL_start_time;

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <iostream>
#include <string.h>

#include "options.h"
#include "util.h"

void Options::help(char * argv[])
{
   ECHO("\n");
   ECHO("Usage: %s [OPTIONS]\n", argv[0]);
   ECHO("\n");
   ECHO("\n");
   ECHO("General options:\n");
   ECHO("                            \n");
   ECHO("      --seed=SEED           Use the seed SEED; 'random', 'rand' and 'r' give\n");
   ECHO("                            random seed. Default is 0.\n");
   ECHO("                            \n");
   ECHO("      --bind=...            \n");
   ECHO("                            \n");
   ECHO("      --challenge FILE      Read challenge from FILE.\n");
   ECHO("                            \n");
   ECHO("                            \n");
   ECHO("Options for Block Widemann (BW):\n");
   ECHO("                            \n");
   ECHO("      --write-bw1 FILE      Store the result of the first BW step (iterative\n");
   ECHO("                            matrix multiplications) to file FILE.\n");
   ECHO("                            \n");
   ECHO("      --read-bw1 FILE       Read previously stored result of BW1 from FILE.\n");
   ECHO("                            \n");
   ECHO("      --write-bm FILE       Write result of Berlekamp–Massey to FILE.\n");
   ECHO("                            \n");
   ECHO("      --read-bm FILE        Read previously stored result of BM from FILE.\n");
   ECHO("                            \n");
   ECHO("      --bw1                 Execute first step of Block Wiedemann.\n");
   ECHO("                            \n");
   ECHO("      --bm                  Execute Berlekamp–Massey step of Block Wiedemann.\n");
   ECHO("                            \n");
   ECHO("      --bw1                 Execute third step of Block Wiedemann.\n");
   ECHO("                            \n");
   ECHO("      --all                 Execute all three Block Wiedemann steps.");
   ECHO("                            \n");
   ECHO("                            \n");
   ECHO("Options for XL:\n");
   ECHO("                            \n");
   ECHO("      --write-system FILE   Write the Macaulay matrix of the system to FILE.\n");
   ECHO("                            \n");
   ECHO("      --read-system FILE    Read the Macaulay matrix of the system from FILE.\n");
   ECHO("                            \n");
   ECHO("                            \n");
   ECHO("                            \n");
   ECHO("                            \n");
   ECHO("Examples:                   \n");
   ECHO("                            \n");
   ECHO("  %s --read-bw1 bw1.out --write-bm bm.out --bm XL\n",
         argv[0]);
   ECHO("                            \n");
   ECHO("  This reads a previously stored result of bw1 from file bw1.out, runs\n");
   ECHO("  the Berlekamp–Massey step, stores the results to bm.out and exits.\n");
   ECHO("  \n");

   ABORT;
}

void Options::parse(int argc, char * argv[])
{
    int c;

    this->system_file = NULL;
    this->system = 0;

    this->bw1_run = false;
    this->bw1_file = NULL;
    this->bw1 = 0;

    this->bm_run = false;
    this->bm_file = NULL;
    this->bm = 0;

    this->bw3_run = false;

    this->seed = 0;

    this->iteration = -1;
    this->all_it = false;
    this->final_it = false;

    this->it_read = false;
    this->it_read_file = NULL;

    this->it_write = false;
    this->it_write_file = NULL;

    this->bind = NULL;

    this->challenge_file = NULL;
    this->challenge = 0;

    while (1)
    {
        static struct option long_options[] =
        {
            {"read-system", 1, 0, OP_SYS_READ},
            {"write-system", 1, 0, OP_SYS_WRITE},
            {"read-bw1", 1, 0, OP_BW1_READ},
            {"write-bw1", 1, 0, OP_BW1_WRITE},
            {"read-bm", 1, 0, OP_BM_READ},
            {"write-bm", 1, 0, OP_BM_WRITE},

            {"challenge", 1, 0, OP_CHALLENGE},

            {"bw1", 0, 0, OP_BW1_RUN},
            {"bm", 0, 0, OP_BM_RUN},
            {"bw3", 0, 0, OP_BW3_RUN},
            {"all", 0, 0, OP_ALL_RUN},

            {"seed", 1, 0, OP_SEED},
            {"el", 1, 0, OP_ITERATION},
            {"final", 0, 0, OP_FINAL_BW},

            {"read-el", 1, 0, OP_IT_READ},
            {"write-el", 1, 0, OP_IT_WRITE},

            {"bind", 1, 0, OP_BIND},

            {"help", 0, 0, 'h'},
            {0, 0, 0, 0}
        };
        /* getopt_long stores the option index here.   */
        int option_index = 0;

        c = getopt_long (argc, argv, "h",
                long_options, &option_index);

        /* Detect the end of the options.   */
        if (c == -1)
            break;

        switch (c)
        {
            case 0:
                /* If this option set a flag, do nothing else now.   */
                if (long_options[option_index].flag != 0)
                    break;
                ECHO ("option %s", long_options[option_index].name);
                if (optarg)
                    ECHO (" with arg %s", optarg);
                ECHO ("\n");
                break;

            case 'h':
                help(argv);
                break;

            case OP_SYS_READ:
                ECHO("option read-system\n");
                this->system_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->system_file, optarg, strlen(optarg) + 1);

                this->system = OP_SYS_READ;
                break;

            case OP_SYS_WRITE:
                ECHO("option write-system\n");
                this->system_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->system_file, optarg, strlen(optarg) + 1);

                this->system = OP_SYS_WRITE;
                break;

            case OP_BW1_READ:
                ECHO("option read-bw1\n");
                this->bw1_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->bw1_file, optarg, strlen(optarg) + 1);

                this->bw1 = OP_BW1_READ;
                break;

            case OP_BW1_WRITE:
                ECHO("option write-bw1\n");
                this->bw1_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->bw1_file, optarg, strlen(optarg) + 1);

                this->bw1 = OP_BW1_WRITE;
                break;

            case OP_BM_READ:
                ECHO("option read-bm\n");
                this->bm_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->bm_file, optarg, strlen(optarg) + 1);

                this->bm = OP_BM_READ;
                break;

            case OP_BM_WRITE:
                ECHO("option write-bm\n");
                this->bm_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->bm_file, optarg, strlen(optarg) + 1);

                this->bm = OP_BM_WRITE;
                break;

            case OP_IT_READ:
                ECHO("option read-it\n");
                this->it_read_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->it_read_file, optarg, strlen(optarg) + 1);

                this->it_read = true;
                break;

            case OP_IT_WRITE:
                ECHO("option write-it\n");
                this->it_write_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->it_write_file, optarg, strlen(optarg) + 1);

                this->it_write = true;
                break;

            case OP_BW1_RUN:
                ECHO("option bw1-run\n");

                this->bw1_run = true;
                break;

            case OP_BM_RUN:
                ECHO("option bm-run\n");

                this->bm_run = true;
                break;

            case OP_BW3_RUN:
                ECHO("option bw3-run\n");

                this->bw3_run = true;
                break;

            case OP_ALL_RUN:
                ECHO("option all-run\n");

                this->bw1_run = true;
                this->bm_run = true;
                this->bw3_run = true;
                break;

            case OP_SEED:
                ECHO("option seed\n");
                {
                    char *endp;

                    if ((strcmp(optarg, "random") == 0) || 
                            (strcmp(optarg, "rand") == 0)  ||
                            (strcmp(optarg, "r") == 0))
                        this->seed = time(NULL);
                    else
                        this->seed = strtol(optarg, &endp, 0);
                }
                break;

            case OP_FINAL_BW:
                ECHO("option final bw\n");
                this->final_it = true;
                break;

            case OP_ITERATION:
                ECHO("option iteration\n");
                {
                    char *endp;

                    if (strcmp(optarg, "all") == 0)
                        this->all_it = true;
                    if (strcmp(optarg, "final") == 0)
                        this->final_it = true;
                    else
                        this->iteration = strtol(optarg, &endp, 0);
                }
                break;

            case OP_BIND:
                ECHO("option bind\n");
                {
                   unsigned start = 0;
                   unsigned node = 0;

                   char *endp;

                   this->bind = (int*)malloc(4*mpi_size);

                   for (unsigned i = 0; (i <= strlen(optarg)) && (node < mpi_size); i++)
                      if ((optarg[i] == ',') || (optarg[i] == 0))
                      {
                         char tmp = optarg[i];

                         optarg[i] = 0;

                         this->bind[node] = strtol(&optarg[start], &endp, 0);
                         node++;

                         optarg[i] = tmp;

                         start = i+1;
                      }                      
                }
                break;

            case OP_CHALLENGE:
                ECHO("option challenge\n");
                this->challenge_file = (char*)malloc(strlen(optarg) + 1);
                strncpy(this->challenge_file, optarg, strlen(optarg) + 1);

                this->challenge = OP_CHALLENGE;
                break;

            case '?':
                /* getopt_long already printed an error message.   */
                break;

            default:
                abort ();
        }
    }

    if ((this->challenge == OP_CHALLENGE) & (this->system == OP_SYS_READ))
    {
       ECHO("You must use either --challenge or --read-system but not both of them!\n");
       abort();
    }

    if (optind < argc)
    {
       ECHO ("unrecognized parameter: %s\n", argv[optind]);
       ABORT;
    }
}

