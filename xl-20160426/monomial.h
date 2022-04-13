#ifndef TMATRIX_H
#define TMATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

template <unsigned N> //, unsigned MAX_D>
class monomial
{
   private:
      unsigned mon[N];

      // store precomputed binomials
      static unsigned bin[MAX_D + 1 + N][MAX_D + 1];

      // true if precomputed binomials have been compued
      static bool bin_initialized;

   public:

      monomial ()
      {
         memset(this->mon, 0, sizeof this->mon);

         // generate binomials
         if (!bin_initialized)
         {
            ECHO("generate binomials\n");

            for(unsigned i = 0; i < MAX_D + 1 + N; i++)
               for(unsigned j = 0; j < MAX_D + 1; j++)
                  bin[i][j] = binomial(i, j);

            bin_initialized = true;
         }

      }

      unsigned& operator[] (int i)
      {
         return this->mon[i];
      }

/* GF(2) is a special case because x^2 = x */
#if QQ == 2

      // count index in grevlex mode
      unsigned int monomial_to_index()
      {
         unsigned int i;
         int d = 0;
         unsigned int index = 0;
         unsigned int index_d = 0;
         for (i = 0; i < N; i++)
            if (this->mon[i] >= 1)
            {
               d++;

               //index += bin[i][d];
               //index_d += bin[N][d];	
               index += binomial(i, d);
               index_d += binomial(N, d);	
            }
         index = index_d - index;

         return index;
      }

      // compute next monomial in grvlex order
      void step()
      {
         unsigned int i,j;

         if (this->mon[0] == 0)
         {
            for (i = 1; this->mon[i] == 0; i++)
               if (i == N-1)
               {
                  this->mon[i] = 1;
                  return;
               }

            this->mon[i] = 0;
            this->mon[i-1] = 1;  

            return;
         }

         for (i = 1; this->mon[i] == 1; i++)
            if (i == N-1)	
               return;

         for (;i < N && this->mon[i] == 0; i++);
         if (i < N)
            this->mon[i] = 0;
         this->mon[i-1] = 1;	

         j = i-2;
         i = 0;
         for (; (i < j) & (j < N); i++, j--)
         {
            unsigned int temp = this->mon[i];
            this->mon[i] = this->mon[j];
            this->mon[j] = temp;
         } 	

         return;	
      }

#else // #if QQ == 2

      // count index in grvlex order
      unsigned int monomial_to_index()
      {
         unsigned int index = 0;

         int deg = 0;

         for (unsigned int i = 0; i < N; i++)
         {
            deg += mon[i];

            if(deg >= 1)
               //index += bin[deg+i][deg-1];
               index += binomial(deg+i, deg-1);
         }

         return index;
      }

      // somoute next monomial in grvlex order
      void step()
      {
         int i;

         if (mon[0] == 0)
         {
            for (i = 1; mon[i] == 0; i++)
               if (i == N-1)
               {
                  mon[i] = 1;
                  return;
               }

            mon[i] = mon[i] - 1;
            mon[i-1] = mon[i-1] + 1;

            return;
         }

         for (i = 1; mon[i] == 0; i++)
            if (i == N-1)
            {
               mon[N-1] = mon[0] + 1;
               mon[0] = 0;
               return;
            }

         mon[i] = mon[i] - 1;
         mon[i-1] = mon[0] + 1;

         if (i != 1)
            mon[0] = 0;

         return;	
      }

#endif // #if QQ == 2

      monomial<N> operator* (monomial<N> &x)
      {
         monomial<N> ret;

         for (unsigned k = 0; k < N; k++)
            ret[k] = this->mon[k] + x.mon[k];

         return ret;
      }

   private:

      unsigned int binomial(int m, int n)
      {
         if (m < n)
            return 0;
         unsigned int ret = 1;
         int i = 1;
         for(; i <= n; i++)
            ret = ret*(i + m - n)/i;
         return ret;
      }

};

template <unsigned N> //, unsigned MAX_D>
bool monomial<N>::bin_initialized = false;

template <unsigned N> //, unsigned MAX_D>
unsigned monomial<N>::bin[MAX_D + 1 + N][MAX_D + 1];

#endif

