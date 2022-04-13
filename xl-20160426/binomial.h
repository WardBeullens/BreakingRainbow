#ifndef __BINOMIAL_H
#define __BINOMIAL_H

#include <stdint.h>

template<uint64_t n, uint64_t k>
struct Binomial
{
     const static uint64_t value =  (Binomial<n-1,k-1>::value + Binomial<n-1,k>::value);
};

template<>
struct Binomial<0,0>
{
     const static uint64_t value = 1;
};

template<uint64_t n>
struct Binomial<n,0>
{
     const static uint64_t value = 1;
};

template<uint64_t n>
struct Binomial<n,n>
{
     const static uint64_t value = 1;
};

template<uint64_t n, uint64_t k>
struct Sum_Binomial
{
     const static uint64_t value = Binomial<n,k>::value + Sum_Binomial<n,k-1>::value;
};

template<uint64_t n>
struct Sum_Binomial<n,0>
{
     const static uint64_t value = Binomial<n,0>::value;
};


#endif

