#ifndef __UTIL_H
#define __UTIL_H

#include <iostream>
#include <stdint.h>
#include <sys/time.h>

#undef max
#undef min


#ifdef OPEN_MPI

inline unsigned mpi_start(unsigned n, unsigned r = mpi_rank)
{
    return (r * n) / mpi_size;
}

inline unsigned mpi_block(unsigned n, unsigned r = mpi_rank)
{
    return ((r+1) * n) / mpi_size - mpi_start(n, r);
}

#else

inline unsigned mpi_block(unsigned n)
{
    return n;
}

inline unsigned mpi_start(unsigned n)
{
    return 0;
}

#endif

template <class c>
inline void swap(c &a, c &b)
{
    c tmp;

    tmp = a;
    a = b;
    b = tmp;
}

template <typename A, typename B>
unsigned ceil_log(A base, B n)
{
    unsigned ret = 0;
    unsigned num = 1;

    while(num < n)
    {
        ret++;

        num *= base;
    }

    return ret;
}

inline double get_ms_time(void) 
{
	struct timeval timev;

	gettimeofday(&timev, NULL);
	return (double) timev.tv_sec * 1000 + (double) timev.tv_usec / 1000;
}


// general macro functions
#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

#define ceil2(a, b) ((a + b - 1)/b)

#ifdef OPEN_MPI
#define TRACE printf("%i - %s:%i ~ %s\n", mpi_rank, __FILE__, __LINE__, __func__)
#else
#define TRACE if (mpi_rank == 0) printf("%s:%i ~ %s\n", __FILE__, __LINE__, __func__)
#endif

#ifdef OPEN_MPI
#define DUMP(i) if (mpi_rank == 0) std::cout << #i << ": " << i << "\n"
#define DUMP_ALL(i) std::cout << mpi_rank << " - " << #i << ": " << i << "\n"
#else
#define DUMP(i) std::cout << #i << ": " << i << "\n"
#define DUMP_ALL(i) std::cout << #i << ": " << i << "\n"
#endif

#ifdef OPEN_MPI
#define ECHO_NL() if (mpi_rank == 0) \
   { printf("\n"); fflush(stdout); }
#define ECHO(...) if (mpi_rank == 0) \
   { printf("%6.2f - ", (GLOBAL_start_time + get_ms_time())/1000); printf(__VA_ARGS__); fflush(stdout); }
#define ECHO_R(...) if (mpi_rank == 0) \
   { printf("\r%6.2f - ", (GLOBAL_start_time + get_ms_time())/1000); printf(__VA_ARGS__); fflush(stdout); }
#else
#define ECHO_NL() \
   { printf("\n"); fflush(stdout); }
#define ECHO(...) \
   { printf("%6.2f - ", (GLOBAL_start_time + get_ms_time())/1000); printf(__VA_ARGS__); fflush(stdout); }
#define ECHO_R(...) \
   { printf("\r%6.2f - ", (GLOBAL_start_time + get_ms_time())/1000); printf(__VA_ARGS__); fflush(stdout); }
#endif

#ifdef OPEN_MPI
#define ABORT ECHO("Exiting...\n"); exit(-1);
#else
#define ABORT exit(-1);
#endif

#endif // #ifdef __UTIL_H

