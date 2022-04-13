#ifndef NAIVE_ROW_MASK_H
#define NAIVE_ROW_MASK_H

#include <assert.h>

#include "util.h"

template<unsigned height, unsigned num_rb, unsigned Sys_n>
class row_mask
{
public:

    unsigned L[height];

    unsigned block_cnt[num_rb];
    unsigned block_start[num_rb];

    row_mask()
    {
       if (height == num_rb * Sys_n) // keep all the rows
       {
           for(unsigned i = 0; i < height; i++)
               L[i] = i;

           for (unsigned i = 0; i < num_rb; i++)
           {
              block_start[i] = i * Sys_n;
              block_cnt[i] = Sys_n;
           }
       }
       else
       {
           uint8_t mask[Sys_n];
           unsigned t = 0;

           for(unsigned i = 0 ; i < Sys_n; i++)
               mask[i] = 0;

           for(unsigned rb = 0; rb < num_rb; rb++)
           {
               if(rb % 10000 == 0) 
               {
                   ECHO_R("generating mask: (%u/%u)", rb, num_rb);
                   fflush(stdout);
               }

               unsigned idx;

               for(uint64_t row = (uint64_t)rb * (uint64_t)height / (uint64_t)num_rb; 
                       row < (uint64_t)(rb + 1) * (uint64_t)height / (uint64_t)num_rb;
                       row++)
               {
                   do 
                   {
                       idx = rand() % Sys_n;
                   }
                   while (mask[ idx ]);

                   mask[ idx ] = 1;
               }


               block_start[rb] = t;

               unsigned cnt = 0;

               for(unsigned i = 0; i < Sys_n; i++)
               {
                   if(mask[i])
                   {
                       L[ t++ ] = i + rb * Sys_n;              
                       mask[i] = 0;
                       cnt++;
                   }
               }

               block_cnt[rb] = cnt;
           }

           ECHO_NL();

           assert(t == height);
       }
    }

    void dump()
    {
        for(unsigned i = 0; i < height; i++)
            printf("row %u: %u\n", i, L[i]);
    }

   inline unsigned block_idx(unsigned r) const
   {
      return L[r] / Sys_n;
   }

   inline unsigned blockrow_idx(unsigned r) const
   {
      return L[r] % Sys_n;
   }

   inline unsigned blockrow_order(unsigned r) const
   {
        unsigned rb = block_idx(r);

      return r - (uint64_t)rb * (uint64_t)height / (uint64_t)num_rb;
   }
};

#endif

