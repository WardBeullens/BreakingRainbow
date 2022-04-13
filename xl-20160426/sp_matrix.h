#ifndef _SP_MATRIX_B_H_
#define _SP_MATRIX_B_H_

#include "gf/gf_buf.h"
#include "matrix.h"

template <unsigned M, unsigned N, unsigned W>
class sparse_matrix
{
   public:
    static const unsigned m = M;
    static const unsigned n = N;
    static const unsigned w = W;

    static const unsigned max_entries = (N * w);

    struct entry
    {
        unsigned idx;
        gf val;
    };
	
    struct sparse_vector
    {
        entry ent[w];	
	
        sparse_vector(): _sz(0) {};
        ~sparse_vector() {};

        inline unsigned get_sz() const
        {
            return _sz;
        }

        inline void set_zero()
        {
            _sz = 0;
        }

        inline void rand(unsigned num)
        {
            unsigned idx;
            gf e;		

            for(unsigned j = 0; j < num; j++)
            {
                idx = m;

                while(idx == m)
                {
                    idx = ::rand() % m;

                    for(unsigned k = 0; k < _sz; k++)
                    {
                        if(ent[k].idx == idx)
                            idx = m;
                    }
                }

                do e = gf::rand();
                while( !e );

                insert(idx, e);
            }

            sort();
        }

        inline void insert(unsigned i, gf e)
        {
            assert(_sz < w);

            ent[ _sz ].idx = i;
            ent[ _sz ].val = e;
		
            _sz++;
        }

        void dump(FILE * fp) const
        {
            for(unsigned i = 0; i < _sz; i++)
                fprintf(fp, "(%u, " GF_FMT ") ", ent[i].idx, (int)(ent[i].val));

            printf("\n");
        }

        void sort()
        {
            for(unsigned i = 1; i < _sz; i++)
            {
                entry tmp = ent[i];
                unsigned j;

                for(j = i; j >= 1; j--)
                {
                    if(ent[j-1].idx > tmp.idx) 
                    {
                        ent[j] = ent[j-1];
                    }
                    else 
                    {                    
                        break;
                    }
                }

                ent[j] = tmp;
            }
        }

        void reduce();

    private:

        unsigned _sz;
    };

    unsigned idx[n*w];

    gfv<w> val[n];

    void dump(FILE * fp) const
    {
       for (int i = 0; i < 1; i++)
         this->val[i].dump(fp);       
    }

    inline unsigned num_entries()
    {
       return (N * w);
    }

    inline void set_zero()
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif 
       for (unsigned i = 0; i < n; i++)
       {
          for (unsigned j = 0; j < w; j++)
             idx[i*w+j] = 0;
          val[i].set_zero();
       }
    }

    inline void rand()
    { 
       this->set_zero();

       sparse_vector tmp;

       for (unsigned i = 0; i < n; i++)
       {
          tmp.rand(w);

          for (unsigned j = 0; j < w; j++)
          {
             val[i].set(j, tmp.ent[j].val);
             idx[i*w + j] = tmp.ent[j].idx;
          }

          tmp.set_zero();
       }
    }

    inline void dense_copy(matrix<m, n> &mat) const
    {
        mat.set_zero();

        for(unsigned i = 0; i < n; i++)
            for(unsigned j = 0; j < w; j++)
                mat.L[i].set( idx[i*w + j], val[i].get(j) );
    }

    inline void dense_copy_transpose(matrix<n, m> &mat) const
    {
        mat.set_zero();

        for(unsigned i = 0; i < n; i++)
            for(unsigned j = 0; j < w; j++)
                mat.L[idx[i*w + j]].set(i, val[i].get(j) );
    }
};

template<unsigned m, unsigned nc, unsigned nb, unsigned w>
inline void sparse_matrix_mad(matrix<m, nc> &c, 
                              const sparse_matrix<nb, nc, w> &a, 
                              matrix<m, nb> &b)
{
    sparse_matrix_mad(c, a, b, 0, nc);
}

template<unsigned m, unsigned nc, unsigned nb, unsigned w>
inline void sparse_matrix_prod(matrix<m, nc> &c, 
                              const sparse_matrix<nb, nc, w> &a, 
                              matrix<m, nb> &b)
{
    c.set_zero();

    sparse_matrix_mad(c, a, b);
}

template<unsigned m, unsigned nc, unsigned nb, unsigned w>
inline void sparse_matrix_prod(matrix<m, nc> &c, 
                              const sparse_matrix<nc, nb, w> &a, 
                              matrix<m, nb> &b)
{
#ifndef _OPENMP
    c.set_zero();

    gfv<m> buf[gfv<m>::GF];
    buf[0].set_zero();

    for (unsigned i = 0; i < nb; i++)
    {
       b.L[ i ].gf16_expand(buf);

       for (unsigned j = 0; j < a.w; j++)
          c.L[ a.idx[i*a.w + j] ] += buf[ a.val[i][j] ];
    }
#else
    matrix<m, nc> *loc;

#pragma omp parallel
    {
       unsigned threads = omp_get_num_threads();
       unsigned tid = omp_get_thread_num();

#pragma omp master
       if (threads > 1)
          loc = new matrix<m, nc>[threads];
       else
          loc = &c;

#pragma omp barrier
       gfv<m> buf[gfv<m>::GF];

       buf[0].set_zero();

       matrix<m, nc> *tmp = &loc[tid];
       tmp->set_zero();

#pragma omp for schedule(static)
       for (unsigned i = 0; i < nb; i++)
       {
          b.L[ i ].gf16_expand(buf);

          for(unsigned j = 0; j < a.w; j+=16)
          {
             uint8_t v0, v1, v2, v3, v4, v5, v6, v7, 
                     v8, v9, va, vb, vc, vd, ve, vf;

             a.val[i].get(j,
                   v0, v1, v2, v3, v4, v5, v6, v7, 
                   v8, v9, va, vb, vc, vd, ve, vf);

             tmp->L[ a.idx[i*a.w + j + 0x0] ] += buf[ v0 ];
             tmp->L[ a.idx[i*a.w + j + 0x1] ] += buf[ v1 ];
             tmp->L[ a.idx[i*a.w + j + 0x2] ] += buf[ v2 ];
             tmp->L[ a.idx[i*a.w + j + 0x3] ] += buf[ v3 ];
             tmp->L[ a.idx[i*a.w + j + 0x4] ] += buf[ v4 ];
             tmp->L[ a.idx[i*a.w + j + 0x5] ] += buf[ v5 ];
             tmp->L[ a.idx[i*a.w + j + 0x6] ] += buf[ v6 ];
             tmp->L[ a.idx[i*a.w + j + 0x7] ] += buf[ v7 ];
             tmp->L[ a.idx[i*a.w + j + 0x8] ] += buf[ v8 ];
             tmp->L[ a.idx[i*a.w + j + 0x9] ] += buf[ v9 ];
             tmp->L[ a.idx[i*a.w + j + 0xa] ] += buf[ va ];
             tmp->L[ a.idx[i*a.w + j + 0xb] ] += buf[ vb ];
             tmp->L[ a.idx[i*a.w + j + 0xc] ] += buf[ vc ];
             tmp->L[ a.idx[i*a.w + j + 0xd] ] += buf[ vd ];
             tmp->L[ a.idx[i*a.w + j + 0xe] ] += buf[ ve ];
             tmp->L[ a.idx[i*a.w + j + 0xf] ] += buf[ vf ];
          }
       }

       if (threads > 1)
       {
#pragma omp for schedule(static)
          for (unsigned i = 0; i < nc; i++)
          {
             // sumup locally to avoid numa access
             for (unsigned j = 1; j < (threads-1); j++)
                loc[tid].L[i] += loc[(tid+j) % threads].L[i];

             c.L[i] = loc[tid].L[i] + loc[(tid+threads-1) % threads].L[i];
          }

#pragma omp master
          delete [] loc;
       }
    }

#endif

}

#if QQ == 31

template<unsigned m, unsigned nc, unsigned nb, unsigned w>
inline void sparse_matrix_mad(matrix<m, nc> &c, 
                              const sparse_matrix<nb, nc, w> &a, 
                              matrix<m, nb> &b,
                              const unsigned start_row, const unsigned rows)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
   {
        gfv<m> buf[4];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for(unsigned i = start_row; i < (start_row + rows); i++)
      {
         for (unsigned k = 0; k < b.L[0].M; k++ ) 
         {
            __m128i mask = _mm_set1_epi32(0x00FF);

            buf[0].c[k].v = _mm_and_si128(c.L[i].c[k].v, mask);

            buf[1].c[k].v = _mm_srli_epi32(c.L[i].c[k].v, 8);
            buf[1].c[k].v = _mm_and_si128(c.L[i].c[k].v, mask);

            buf[2].c[k].v = _mm_srli_epi32(c.L[i].c[k].v, 16);
            buf[2].c[k].v = _mm_and_si128(c.L[i].c[k].v, mask);

            buf[3].c[k].v = _mm_srli_epi32(c.L[i].c[k].v, 24);
            buf[3].c[k].v = _mm_and_si128(c.L[i].c[k].v, mask);
         }

         for(unsigned j = 0; j < a.w; j++)
         {
            uint8_t v0 = a.val[i].get(j);

            for (unsigned k = 0; k < b.L[0].M; k++ ) 
               b.L[ a.idx[i*a.w + j] ].c[k].mad(
                     buf[0].c[k].v, buf[1].c[k].v, buf[2].c[k].v, buf[3].c[k].v, 
                     v0);
         }

         for (unsigned k = 0; k < c.L[0].M; k++ ) 
            c.L[i].c[k].reduce(
                  buf[0].c[k].v, buf[1].c[k].v, buf[2].c[k].v, buf[3].c[k].v);

      }
   }
};


#else

template<unsigned m, unsigned nc, unsigned nb, unsigned w>
inline void sparse_matrix_mad(matrix<m, nc> &c, 
                              const sparse_matrix<nb, nc, w> &a, 
                              matrix<m, nb> &b,
                              const unsigned start_row, const unsigned rows)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
   {
#if QQ != 2
        gf_buf<gfv<m> > buf;
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for(unsigned i = start_row; i < (start_row + rows); i++)
      {
#if QQ == 2
         for(unsigned j = 0; j < a.w; j++)
            c.L[i] += b.L[ a.idx[i*a.w + j] ];
#else
         for(unsigned j = 1; j < gfv<m>::GF; j++)
            buf[j].set_zero();

         for(unsigned j = 0; j < a.w; j+=16)
         {
            uint8_t v0, v1, v2, v3, v4, v5, v6, v7, 
                    v8, v9, va, vb, vc, vd, ve, vf;

            a.val[i].get(j,
                  v0, v1, v2, v3, v4, v5, v6, v7, 
                  v8, v9, va, vb, vc, vd, ve, vf);

            buf[ v0 ] += b.L[ a.idx[i*a.w + j + 0x0] ];
            buf[ v1 ] += b.L[ a.idx[i*a.w + j + 0x1] ];
            buf[ v2 ] += b.L[ a.idx[i*a.w + j + 0x2] ];
            buf[ v3 ] += b.L[ a.idx[i*a.w + j + 0x3] ];
            buf[ v4 ] += b.L[ a.idx[i*a.w + j + 0x4] ];
            buf[ v5 ] += b.L[ a.idx[i*a.w + j + 0x5] ];
            buf[ v6 ] += b.L[ a.idx[i*a.w + j + 0x6] ];
            buf[ v7 ] += b.L[ a.idx[i*a.w + j + 0x7] ];
            buf[ v8 ] += b.L[ a.idx[i*a.w + j + 0x8] ];
            buf[ v9 ] += b.L[ a.idx[i*a.w + j + 0x9] ];
            buf[ va ] += b.L[ a.idx[i*a.w + j + 0xa] ];
            buf[ vb ] += b.L[ a.idx[i*a.w + j + 0xb] ];
            buf[ vc ] += b.L[ a.idx[i*a.w + j + 0xc] ];
            buf[ vd ] += b.L[ a.idx[i*a.w + j + 0xd] ];
            buf[ ve ] += b.L[ a.idx[i*a.w + j + 0xe] ];
            buf[ vf ] += b.L[ a.idx[i*a.w + j + 0xf] ];
         }

         buf.reduce(c.L[i], true);
#endif
      }
   }
};

#endif

template <unsigned _m, unsigned _n>
class sparse_matrix<_m, _n, 0>
{
   public:

    static const unsigned m = _m;
    static const unsigned n = _n;

    unsigned *idx;
    gf *val;

    unsigned row[n+1];

    template <unsigned _w>
    sparse_matrix(sparse_matrix<n, m, _w> &a)
    {
       for (unsigned i = 1; i <= n; i++)
          row[i] = 0;

       for (unsigned i = 0; i < m; i++)
          for (unsigned j = 0; j < _w; j++)
             row[a.idx[i*_w + j] + 1]++;

       for (unsigned i = 1; i <= n; i++)
          row[i] += row[i-1];

       DUMP(m);
       DUMP(_w);
       DUMP(row[n]);

       val = (gf*)malloc(sizeof(gf)*row[n]);
       idx = (unsigned*)malloc(sizeof(unsigned)*row[n]);

       unsigned cnt[n];

       for (unsigned i = 0; i < n; i++)
          cnt[i] = 0;

       for (unsigned i = 0; i < m; i++)
          for (unsigned j = 0; j < _w; j++)
          {
             unsigned r = a.idx[i*_w + j];

             val[ row[r] + cnt[r] ] = a.val[i][j];
             idx[ row[r] + cnt[r] ] = i;

             cnt[r]++;
          }

    }

    ~sparse_matrix()
    {
       free(val);
       free(idx);
    }

};

template<unsigned m, unsigned nc, unsigned nb>
inline void sparse_matrix_prod(matrix<m, nc> &c, 
                              const sparse_matrix<nb, nc, 0> &a, 
                              matrix<m, nb> &b)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
   {
      gfv<m> buf[gfv<m>::GF];


#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (unsigned i = 0; i < nc; i++)
      {
         for(unsigned j = 1; j < gfv<m>::GF; j++)
            buf[j].set_zero();

         for (unsigned j = a.row[i]; j < a.row[i+1]; j++)
            buf[a.val[j]] += b.L[ a.idx[j] ];

         c.L[i].gf16_reduce(buf, false);
      }

   }
}

#endif //ifndef _SP_MATRIX_B_H_

