
#ifndef MACAULAY_MATRIX_H
#define MACAULAY_MATRIX_H

#include <stdint.h>
#include <sys/mman.h>
#include <set>

#include "row_mask.h"
#include "sp_matrix.h"

#include "gf/gf_buf.h"

#include "binomial.h"
#include "monomial.h"
#include "xl_mem.h"

#define MAX_BLOCKNUM 128

#ifdef MPI
typedef struct
{
   unsigned start;
   unsigned end;
   unsigned width;
} range_t;
#endif

template <class Sys, unsigned num_var, unsigned ext_d>
class macaulay_matrix
{
    public:

#if QQ == 2
        const static uint64_t height = Sum_Binomial<num_var, ext_d>::value;
        static const uint64_t num_rb = Sum_Binomial<num_var, ext_d - 2>::value;
#else // #if QQ == 2
        const static uint64_t height = Binomial<num_var + ext_d, ext_d>::value;
        static const uint64_t num_rb = Binomial<num_var + ext_d - 2, ext_d - 2>::value;
#endif // #if QQ == 2

        const static unsigned width = height;

        static const unsigned m = height;
        static const unsigned n = width;

        static const unsigned sys_m = Sys::m;
        static const unsigned sys_n = Sys::n;

        static const bool is_sparse = (width > 10 * sys_n);

        Sys sys;
        uint32_t col_idx[num_rb * Sys::m];

        double time;

        row_mask<height, num_rb, Sys::n> mask;

        macaulay_matrix(Sys sys, FILE *fd = NULL)
        {
            this->sys = sys;

            this->gen_col_idx();

            time = 0;
        }

        void gen_col_idx()
        {	
            // for enumerating the extension monomials
            monomial<num_var> ext_mon;

            for (uint32_t i = 0; i < num_rb; i++)
            {
                // for enumerating monomials in the original system
                monomial<num_var> sys_mon;

                if (i % 100000 == 0) 
                {
                    ECHO_R("generating col. indices: (%u/%lu)", i, num_rb);
                    fflush(stdout);
                }

                for (uint32_t j = 0; j < sys.m; j++)
                {
                    monomial<num_var> prod_mon = sys_mon * ext_mon;

                    col_idx[(num_rb - 1 - i) * sys.m + j] = prod_mon.monomial_to_index();

                    sys_mon.step();
                }
		
                ext_mon.step();
            }

            ECHO_NL();
        }

        void get_time(std::set<double> &s) { s.insert(time); }

        uint64_t num_entries() const
        {
           return (uint64_t) this->m * 
                  (uint64_t) this->sys_m;
        }

        void copy2sparse(sparse_matrix<height, height, Sys::m> *mat)
        {
            for(unsigned r = 0; r < height; r++)
            {
                unsigned rb = this->mask->block_idx(r);
                unsigned block_row = this->mask->blockrow_idx(r);

                for (unsigned col = 0; col < Sys::m; col++)
                    mat->L[r].insert(this->col_idx[rb * this->sys.m + col],
                            this->sys.L[block_row][col]);
            }
        }

        void col_idx_dump()
        {
            for(unsigned i = 0; i < num_rb; i++)
            {
                printf("col_idx[%u] = [ ", i);

                for(unsigned j = 0; j < sys.m; j++)
                    printf("%u ", col_idx[i * sys.m + j]);

                printf("]\n");
            }
        }

        void get_col_range(std::set<unsigned> &l)
        {
           l.insert(0);
           l.insert(width);
        }
};


#define TEMPLATE_MAC class Sys, unsigned num_var, unsigned ext_d
#define TYPE_MAC macaulay_matrix<Sys, num_var, ext_d>

///////////////////////////////// the lowest-level functions ////////////////////////////////

#if QQ != 2

template <unsigned M, unsigned Nb, TEMPLATE_MAC>
inline void sparse_matrix_mad(gfv<M> *c, 
                       const TYPE_MAC &a, 
                       const matrix<M, Nb> &b,
                       unsigned start_row, unsigned rows,
                       unsigned start_col, unsigned cols,
                       bool add)
{
    unsigned end_row = start_row + rows;
    unsigned end_col = start_col + cols;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const uint32_t *ptr;

        gf_buf<gfv<M> > buf;

        gfv<M> * vecs = new gfv<M> [cols];
        gfv<M> * vec_ptr;

        unsigned count = 0;

#ifdef _OPENMP
#pragma omp for schedule(static) 
#endif
        for(unsigned r = start_row; r < end_row; r++)
        {
            for(unsigned i = 1; i < gfv<M>::GF; i++)
                buf[i].set_zero();

            unsigned rb = a.mask.block_idx(r);
            unsigned block_row = a.mask.blockrow_idx(r);
            unsigned order = a.mask.blockrow_order(r);

            if(order == 0 || count == 0)
            {
                ptr = &a.col_idx[rb * a.sys.m + start_col];

                for(unsigned i = 0; i < cols; i++)
                    vecs[i] = b.L[*ptr++];
            }

            vec_ptr = vecs;

              unsigned col = start_col;

            while (((col & 0xf) > 0) && (col < end_col))
            {
                buf[ a.sys.L[block_row][col] ] += *vec_ptr++;
                col++;
            }

            while (col < (end_col & (~0xf)))
            {
                uint8_t v0, v1, v2, v3, v4, v5, v6, v7, 
                        v8, v9, v10, v11, v12, v13, v14, v15;

                a.sys.L[block_row].get(col, 
                        v0, v1, v2, v3, v4, v5, v6, v7, 
                        v8, v9, v10, v11, v12, v13, v14, v15);

                buf[ v0 ] += *vec_ptr++;
                buf[ v1 ] += *vec_ptr++;
                buf[ v2 ] += *vec_ptr++;
                buf[ v3 ] += *vec_ptr++;
                buf[ v4 ] += *vec_ptr++;
                buf[ v5 ] += *vec_ptr++;
                buf[ v6 ] += *vec_ptr++;
                buf[ v7 ] += *vec_ptr++;
                buf[ v8 ] += *vec_ptr++;
                buf[ v9 ] += *vec_ptr++;
                buf[ v10 ] += *vec_ptr++;
                buf[ v11 ] += *vec_ptr++;
                buf[ v12 ] += *vec_ptr++;
                buf[ v13 ] += *vec_ptr++;
                buf[ v14 ] += *vec_ptr++;
                buf[ v15 ] += *vec_ptr++;

                col += 16;
            }

            while (col < end_col)
            {
                buf[ a.sys.L[block_row][col] ] += *vec_ptr++;
                col++;
            }

            buf.reduce(c[r], add);

            count++;
        }

        delete [] vecs;
    }
}

#elif QQ == 2

template <unsigned M, unsigned Nb, TEMPLATE_MAC>
inline void sparse_matrix_mad(gfv<M> *c, 
                       const TYPE_MAC &a, 
                       const matrix<M, Nb> &b,
                       unsigned start_row, unsigned rows,
                       unsigned start_col, unsigned cols, 
                       bool add)
{
    unsigned end_row = start_row + rows;
    unsigned end_col = start_col + cols;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const uint32_t *ptr;

        gfv<M> buf[gfv<M>::GF];

        gfv<M> * vecs = new gfv<M> [cols];
        gfv<M> * vec_ptr;

        unsigned count = 0;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(unsigned r = start_row; r < end_row; r++)
        {
            buf[1].set_zero();

            unsigned rb = a.mask.block_idx(r);
            unsigned block_row = a.mask.blockrow_idx(r);
            unsigned order = a.mask.blockrow_order(r);

            if(order == 0 || count == 0)
            {
                ptr = &a.col_idx[rb * a.sys.m + start_col];

                for(unsigned i = 0; i < cols; i++)
                    vecs[i] = b.L[*ptr++];
            }

            vec_ptr = vecs;

            unsigned end_head = min(ceil2(start_col, 64) << 6, end_col);
            unsigned start_tail = (end_col >> 6) << 6;

            unsigned block_num = start_col >> 6;


            // handle the first, not completely filled 64-bit block
            if (end_head - start_col)
            {
                uint64_t val = ((uint64_t*)(&a.sys.L[block_row]))[block_num++];

                val >>= start_col & 0x3f;

                for (unsigned col = start_col; col < end_head; col++)
                {
                    if(val & 1) buf[1] += *vec_ptr;

                    vec_ptr++;
                    val >>= 1;
                }
            }

            // go over all full 64-bit blocks
            for (unsigned col = end_head; col < start_tail; col += 64)
            {
                uint64_t val = ((uint64_t*)(&a.sys.L[block_row]))[block_num++];

                for (uint64_t i = 0; i < 64; i++)
                {
                    if(val & 1) buf[1] += *vec_ptr;

                    vec_ptr++;
                    val >>= 1;
                }
            }

            // handle the last, not completely filled 64-bit block
            if (end_col - start_tail)
            {
                uint64_t val = ((uint64_t*)(&a.sys.L[block_row]))[block_num++];

                for (uint64_t i = start_tail; i < end_col; i++)
                {
                    if(val & 1) buf[1] += *vec_ptr;

                    vec_ptr++;
                    val >>= 1;
                }
            }

            if(add)
                c[r] += buf[1];
            else
                c[r] = buf[1];

            count++;
        }

        delete [] vecs;
    }
}

#endif // QQ 

/////////////////////////// this function is meant to reduce ///////////////////////////
/////////////////////////// the working set                  ///////////////////////////

template <unsigned M, unsigned Nb, TEMPLATE_MAC>
void sparse_matrix_mad(gfv<M> *c, 
                       TYPE_MAC &a, 
                       const matrix<M, Nb> &b,
                       unsigned start_row, unsigned rows,
                       bool add)
{
    unsigned col_batch = 512; // ???
    unsigned row_batch = rows; // ???

    unsigned end_row = start_row + rows;

    for(unsigned i = start_row; i < end_row; i += row_batch)
    {
        unsigned nrows = min(row_batch, end_row - i);

        for(unsigned j = 0; j < Sys::m; j += col_batch)
        {
            unsigned ncols = min(col_batch, Sys::m - j);

            sparse_matrix_mad(c, a, b, i, nrows, j, ncols, j == 0 ? add : true);
        }
    }
}

/////////////// mult(cL, a, b): cL = gfv* : a = mac_matrix : b = matrix ////////////

template <unsigned M, unsigned Nb, TEMPLATE_MAC>
inline void sparse_matrix_mad(gfv<M> *c, 
                               TYPE_MAC &a, 
                               matrix<M, Nb> &b)
{
    sparse_matrix_mad(c, a, b, 0, a.height, true);
}

/////////////////// mult(c, a, b): c, b = matrix : a = mac_matrix ///////////////

template <unsigned M, unsigned Nc, unsigned Nb, TEMPLATE_MAC>
inline void sparse_matrix_mad(matrix<M, Nc> &c, 
                               const TYPE_MAC &a,
                               matrix<M, Nb> &b)
{
    sparse_matrix_mad(c.L, a, b);
}

template <unsigned M, unsigned Nb, TEMPLATE_MAC>
void sparse_matrix_prod(gfv<M> *c, 
                       const TYPE_MAC &a, 
                       const matrix<M, Nb> &b,
                       unsigned start_row, unsigned rows)
{
    sparse_matrix_mad(c, a, b, start_row, rows, false);
}

template <unsigned M, unsigned Nc, unsigned Nb, TEMPLATE_MAC>
inline void sparse_matrix_prod(matrix<M, Nc> &c, 
                               TYPE_MAC &a, 
                               matrix<M, Nb> &b)
{
    sparse_matrix_mad(c.L, a, b, 0, a.height, false);
}

#endif //ifndef MACAULAY_MATRIX_H

