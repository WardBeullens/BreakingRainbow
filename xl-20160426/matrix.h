#ifndef _MATRIX_H
#define _MATRIX_H

#include <assert.h>

#include "util.h"

#include "gf/gfv.h"

template <unsigned M,unsigned N>
struct matrix
{
   static const unsigned n = N;
   static const unsigned m = M;

   gfv<m> L[n];

   void read_text(FILE *fd)
   {
      unsigned int v;

      int ret = 0;

      for (unsigned i = 0; i < n; i++)
      {
         for (unsigned j = 0; j < m; j++)
         {
            ret = fscanf(fd, GF_FMT " ", &v);
            this->set(i, j, v);
         }
      }
   }

   void write_text(FILE *fd)
   {
      for (unsigned i = 0; i < n; i++)
      {
         for (unsigned j = 0; j < m; j++)
         {
            fprintf(fd, GF_FMT " ", this->get(i, j).v);
         }
         fprintf(fd,"\n");
      }
   }

   void read_challenge(FILE *fd)
   {
       unsigned int v;

        for (unsigned int i = 0; i < n; i++)
        {
           for (unsigned int j = 0; j < m; j++)
           {
              fscanf(fd, GF_FMT " ", &v);
              L[i].set(m-1-j, v);
           }

           fscanf(fd, ";\n");
        }
   }

   inline void set(unsigned row, unsigned col, const gf &b)
   {
       this->L[ row ].set(col, b);
   }

   inline const gf get(unsigned row, unsigned col) const
   {
       return this->L[row][col];
   }

   inline void set_zero(unsigned start=0, unsigned rows=n)
   {
#ifdef _OPENMP
        #pragma omp parallel for
#endif
       for(unsigned i = start; i < start + rows; i++) 
           L[i].set_zero();
   }

   inline bool is_zero()
   {
      for(unsigned i = 0; i < N; i++) 
         if( !L[i].is_zero() )
            return false;

      return true;
   }

   inline void rand()
   { 
      for(unsigned i=n; i--;)
            this->L[i].rand(); 
   }

   inline void rand(uint32_t row_weight)
   { 
      for(unsigned i=n; i--;)
            this->L[i].rand(row_weight); 
   }

   void transpose(matrix<N, M> &mat) const 
   { 
      for(unsigned i = 0; i < M; i++) 
         for (unsigned j = 0; j < N; j++) 
            mat.L[i].set(j, this->L[j][i]);
   }
  
   inline void dump( FILE * fp ) const 
   { 
      for(int i = 0; i < (int)n; i++) 
         this->L[i].dump(fp);
   }

   inline const gf hash()
   {
      gf ret;

      ret.v = 0;
      for(unsigned i = 0; i < N; i++)
         ret += L[i].hash();
   
      return ret;
   }

};

/////////////////////////////// the lowest level functions //////////////////////////

#define MAX_UNITS 8

template <unsigned ma, unsigned mb, unsigned nc, unsigned na, unsigned units>
inline void matrix_mad_special(matrix<mb, nc> &c, 
                               const matrix<ma, na> &a, 
                               const gfv<mb> *b, 
                               unsigned start_row, unsigned rows,
                               unsigned start_col, unsigned cols,
                               unsigned start_unit)
{
   typedef gfv<units * gfv_unit::N> gfv_type;

   gf_buf<gfv_type> buf;

   uint8_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;

   for(unsigned i = start_row; i < start_row + rows; i++)
   {
      for(unsigned j = 1; j < gfv<mb>::GF; j++)
            buf[j].set_zero();

      for(unsigned j = start_col; j < start_col + cols; j += 16)   
      {
         a.L[i].get(j, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);

         buf[ v0 ]  += *((gfv_type *) (b[j+0].c  + start_unit));
         buf[ v1 ]  += *((gfv_type *) (b[j+1].c  + start_unit));
         buf[ v2 ]  += *((gfv_type *) (b[j+2].c  + start_unit));
         buf[ v3 ]  += *((gfv_type *) (b[j+3].c  + start_unit));
         buf[ v4 ]  += *((gfv_type *) (b[j+4].c  + start_unit));
         buf[ v5 ]  += *((gfv_type *) (b[j+5].c  + start_unit));
         buf[ v6 ]  += *((gfv_type *) (b[j+6].c  + start_unit));
         buf[ v7 ]  += *((gfv_type *) (b[j+7].c  + start_unit));
         buf[ v8 ]  += *((gfv_type *) (b[j+8].c  + start_unit));
         buf[ v9 ]  += *((gfv_type *) (b[j+9].c  + start_unit));
         buf[ v10 ] += *((gfv_type *) (b[j+10].c + start_unit));
         buf[ v11 ] += *((gfv_type *) (b[j+11].c + start_unit));
         buf[ v12 ] += *((gfv_type *) (b[j+12].c + start_unit));
         buf[ v13 ] += *((gfv_type *) (b[j+13].c + start_unit));
         buf[ v14 ] += *((gfv_type *) (b[j+14].c + start_unit));
         buf[ v15 ] += *((gfv_type *) (b[j+15].c + start_unit));
      }

#if QQ == 2
      *((gfv_type *)(c.L[i].c + start_unit)) += buf[1];
#else
      buf.reduce(*((gfv_type *)(c.L[i].c + start_unit)), true);
#endif

   }
}

/////////////////////////// this function is meant to ///////////////////////////
/////////////////////////// reduce the working set    ///////////////////////////

template <unsigned ma, unsigned mb, unsigned nc, unsigned na>
void matrix_mad_special(matrix<mb, nc> &c, 
                        const matrix<ma, na> &a, 
                        const gfv<mb> *b, 
                        unsigned n)
{
   //const unsigned block_size = (1 << 15); // ???
#if QQ == 16
   //const unsigned w = max(block_size/sizeof(gfv<mb>)/32*32, 32);
   const unsigned w = 256;
#elif QQ == 2
   //const unsigned w = max(block_size/sizeof(gfv<mb>)/64*64, 64);
   const unsigned w = 256;
#else
   const unsigned w = 256;
#endif   

   for(unsigned unit = 0; unit + MAX_UNITS <= gfv<mb>::M; unit += MAX_UNITS)
   {
      for(unsigned col = 0; col < ma; col += w)
      {
         matrix_mad_special<ma, mb, nc, na, MAX_UNITS>
                           (c, a, b, 0, n, col, min(w, ma-col), unit);
      }   
   }

   const unsigned tail = gfv<mb>::M % MAX_UNITS;

   if(tail > 0)
   {
      for(unsigned col = 0; col < ma; col += w)
      {
         matrix_mad_special<ma, mb, nc, na, tail>
                           (c, a, b, 0, n, col, min(w, ma-col), gfv<mb>::M - tail);
      }   
   }
}

/////////////////////////////// the lowest level functions //////////////////////////

#define MAX_UNITS 8

template <unsigned units, unsigned mb, unsigned nb, unsigned na>
void matrix_mad(matrix<mb, na> &c, 
                const matrix<nb, na> &a, 
                const matrix<mb, nb> &b,
                unsigned start_row, unsigned rows,
                unsigned start_col, unsigned cols,
                unsigned start_unit)
{
    typedef gfv<units * gfv_unit::N> gfv_type;


#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
    for(unsigned i = start_row; i < start_row + rows; i++)
    {
        uint8_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;

        gf_buf<gfv_type> buf;

        for(unsigned j = 1; j < gfv<mb>::GF; j++)
                buf[j].set_zero();

        for(unsigned j = start_col; j < start_col + cols; j += 16)    
        {
            a.L[i].get(j, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);

            buf[ v0 ]  += *((gfv_type *) (b.L[j+0].c  + start_unit));
            buf[ v1 ]  += *((gfv_type *) (b.L[j+1].c  + start_unit));
            buf[ v2 ]  += *((gfv_type *) (b.L[j+2].c  + start_unit));
            buf[ v3 ]  += *((gfv_type *) (b.L[j+3].c  + start_unit));
            buf[ v4 ]  += *((gfv_type *) (b.L[j+4].c  + start_unit));
            buf[ v5 ]  += *((gfv_type *) (b.L[j+5].c  + start_unit));
            buf[ v6 ]  += *((gfv_type *) (b.L[j+6].c  + start_unit));
            buf[ v7 ]  += *((gfv_type *) (b.L[j+7].c  + start_unit));
            buf[ v8 ]  += *((gfv_type *) (b.L[j+8].c  + start_unit));
            buf[ v9 ]  += *((gfv_type *) (b.L[j+9].c  + start_unit));
            buf[ v10 ] += *((gfv_type *) (b.L[j+10].c + start_unit));
            buf[ v11 ] += *((gfv_type *) (b.L[j+11].c + start_unit));
            buf[ v12 ] += *((gfv_type *) (b.L[j+12].c + start_unit));
            buf[ v13 ] += *((gfv_type *) (b.L[j+13].c + start_unit));
            buf[ v14 ] += *((gfv_type *) (b.L[j+14].c + start_unit));
            buf[ v15 ] += *((gfv_type *) (b.L[j+15].c + start_unit));
        }

#if QQ == 2

        *((gfv_type *)(c.L[i].c + start_unit)) += buf[1];
#else
        buf.reduce(*((gfv_type *)(c.L[i].c + start_unit)), true);
#endif

    }
}

/////////////////////////// this function is meant to ///////////////////////////
/////////////////////////// reduce the working set    ///////////////////////////

template <unsigned mb, unsigned nb, unsigned na>
inline void matrix_mad(matrix<mb, na> &c, 
                const matrix<nb, na> &a, 
                const matrix<mb, nb> &b,
                unsigned start_row=0, unsigned rows=na)
{
#if QQ == 16
    const unsigned w = 256;
#elif QQ == 2
    const unsigned w = 512;
#else
    const unsigned w = 256;
#endif    

    const unsigned ma = nb;

    for(unsigned unit = 0; unit + MAX_UNITS <= gfv<mb>::M; unit += MAX_UNITS)
    {
        for(unsigned col = 0; col < ma; col += w)
        {
            matrix_mad<MAX_UNITS>
                (c, a, b, start_row, rows, col, min(w, ma-col), unit);
        }    
    }

    const unsigned tail = gfv<mb>::M % MAX_UNITS;

    if(tail > 0)
    {
        for(unsigned col = 0; col < ma; col += w)
        {
            matrix_mad<tail>
                (c, a, b, start_row, rows, col, min(w, ma-col), gfv<mb>::M - tail);
        }    
    }
}

template <unsigned mb, unsigned nb, unsigned na>
inline void matrix_prod(matrix<mb, na> &c, 
                const matrix<nb, na> &a, 
                const matrix<mb, nb> &b,
                unsigned start_row=0, unsigned rows=na)
{
#ifdef _OPENMP
        #pragma omp parallel for
#endif
    for(unsigned i = start_row; i < start_row + rows; i++)
        c.L[i].set_zero();

    matrix_mad(c, a, b, start_row, rows);
}

#endif // _MATRIX_H

