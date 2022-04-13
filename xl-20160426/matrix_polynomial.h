#ifndef _MATRIX_POLYNOMIAL_H
#define _MATRIX_POLYNOMIAL_H

#include <stdio.h>
#include <math.h>

template <unsigned m, unsigned n, unsigned deg_coef>
class matrix_polynomial
{
#ifdef OPEN_MPI
    unsigned start_ci;

    unsigned num_coef;
#endif

   public:

    static const unsigned deg = deg_coef;

    unsigned max_nom_deg;

    matrix<n-m, n> fta;

    unsigned nom_deg[n];

    matrix_array<m, n, deg_coef> coef;

    template <unsigned deg_ai>
    matrix_polynomial(const matrix_array<n-m, m, deg_ai> &ai)
    {
        max_nom_deg = ceil(((float)(n-m) + (float)m)/(float)m);

#ifdef OPEN_MPI
        unsigned all_num_coef[mpi_size];

        // determining the workload
        for (unsigned r = 0; r < mpi_size; r++)
        {
            all_num_coef[r] = (max_nom_deg+1) / mpi_size;

            if(((max_nom_deg+1) % mpi_size) > r)
                all_num_coef[r]++;
        }

        num_coef = all_num_coef[mpi_rank];

        start_ci = 0;

        for (unsigned r = 0; r < mpi_rank; r++)
            start_ci += all_num_coef[r];
#endif

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (unsigned i = 0; i < coef.deg; i++)
            coef[i].set_zero();

        for (unsigned i = 0; i < n; i++)
            nom_deg[i] = max_nom_deg;

        initialize(ai);
    }

    const matrix<m, n>& operator [](int i) const
    {
       return this->coef[i];
    }

    matrix<m, n>& operator [](int i)
    {
       return this->coef[i];
    }

    void allgather()
    {
#ifdef OPEN_MPI
        unsigned num_coefs[mpi_size];
        unsigned start_ci[mpi_size];

        num_coefs[mpi_rank] = this->num_coef;
        start_ci[mpi_rank] = this->start_ci ;

        for (unsigned i = 0; i < mpi_size; i++)
        {
            MPI_Bcast(&num_coefs[i], sizeof(unsigned), MPI_CHAR, i, MPI_COMM_WORLD);
            MPI_Bcast(&start_ci[i],  sizeof(unsigned), MPI_CHAR, i, MPI_COMM_WORLD);
        }

        ///////////////////////////////////////////////////////////////////////////

        matrix<m, n> * comm_ptr = coef;

        size_t matrix_size = sizeof(matrix<m, n>);

        for (unsigned i = 0; i < mpi_size; i++)
        {
           MPI_Bcast(comm_ptr,
                 num_coefs[i] * matrix_size,
                 MPI_CHAR, i, MPI_COMM_WORLD);

           comm_ptr += num_coefs[i];
        }
#endif
    }

    inline void getRowDeg(unsigned rd[n])
    {
        for(unsigned i = 0; i < n; i++) 
        {
            rd[i] = 0;

            for(unsigned d = deg; d--; ) 
            {
                if(!coef[d].L[i].is_zero())
                { 
                    rd[i] = d; 
                    break; 
                }
            }
        }
    }


    bool isFullRank(const matrix<n-m, n> &fta)
    {
       // get a local copy
       matrix<n-m, n> t = fta;

       gf piv;

       for (unsigned i = 0; i < (n-m); i++)
       {
          piv = t.L[i][i];

          if (!piv)
             for (unsigned j = i + 1; j < t.n; j++)
                if (t.L[j][i])
                {
                   piv = t.L[j][i];
                   swap(t.L[i], t.L[j]);
                   break;
                }

          if (!piv) return false;

          t.L[i] *= piv.inv();

          for (unsigned j = i+1; j < t.n; j++)
             if (t.L[j][i])
                t.L[j] -= t.L[i] * t.L[j][i];
       }

       return true;
    }


    template <unsigned deg_ai>
    void countCoffMat(const matrix_array<n-m, m, deg_ai> &ai, const unsigned t_max)
    {

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            boost::scoped_ptr<matrix<n-m, m> > buf(new matrix<n-m, m>);

            buf->set_zero();

#ifdef _OPENMP
            #pragma omp for
#endif
            for (unsigned i = 0; i < m; i++)
                fta.L[i].set_zero();

#ifdef _OPENMP
            #pragma omp for //nowait
#endif
#ifdef OPEN_MPI
            for (unsigned i = start_ci; i <= min(t_max, start_ci + num_coef); i++)
#else
            for (unsigned i = 0; i <= min(t_max, (max_nom_deg + 1)); i++)
#endif
               matrix_mad_special(*buf, coef[i], ai[t_max-i].L, m);

#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                for (unsigned i = 0; i < m; i++)
                    fta.L[i] += buf->L[i];
            }
        }

#ifdef OPEN_MPI

#if (QQ % 2) == 0

        if(num_coef == 0)
            for (unsigned i = 0; i < m; i++)
                assert(fta.L[i].is_zero());

        MPI_Allreduce (MPI_IN_PLACE, fta.L, sizeof(gfv<n-m>) * m,
               MPI_BYTE, MPI_BXOR, MPI_COMM_WORLD);

#else
        matrix<n-m, m> * tmp = new matrix<n-m, m> [mpi_size];

        for (unsigned i = 0; i < mpi_size; i++)
        {
            if(i == mpi_rank)
                for (unsigned j = 0; j < m; j++)
                    tmp[i].L[j] = fta.L[j];

            MPI_Bcast(&(tmp[i]), sizeof(gfv<n-m>) * m, MPI_BYTE,
                      i,
                      MPI_COMM_WORLD);
        }

        for (unsigned i = 0; i < mpi_size; i++)
        {
            if(i != mpi_rank)
                for (unsigned j = 0; j < m; j++)
                    fta.L[j] += tmp[i].L[j];
        }

        delete [] tmp;

#endif // (QQ % 2) == 0

#endif // ifdef OPENMPI
    }

    template <unsigned deg_ai>
    void initialize(const matrix_array<n-m, m, deg_ai> &ai)
    {
        static unsigned count = 0;

        unsigned t0 = max_nom_deg;

        for (unsigned j = 0; j < (n-m); j++)
            coef[t0].set_zero();

        for (unsigned i = (n-m); i < n; i++)
            coef[t0].L[i].set(i - (n-m), gf(1));

        for (unsigned i = t0; i-- > 0; )
            for (unsigned j = (n-m); j < n; j++)
                coef[i].L[j].set_zero();

        do
        {
            assert(count <= 100);

            for (unsigned i = t0; i-- > 0; )
                for (unsigned j = 0; j < (n-m); j++)
                    coef[i].L[j].rand();

            for (unsigned i = 0; i < n; i++)
                fta.L[i].set_zero();

            for (unsigned i = 0; i <= t0; i++)
                matrix_mad(fta, coef[i], ai[t0-i], 0, n-m);

            count++;
        }
        while (!isFullRank(fta));

        // pre-compute the lower part
        for (unsigned i = 0; i < m; i++)
            fta.L[(n-m) + i] = ai[0].L[i];
    }

    // generate information of Tau matrix
    // per: permutation of rows
    // tau: upper right part of the "not-so-dense" matrix
    void gen_tau(matrix<n-m, n> &tau, unsigned *per)
    {
        for (unsigned i = 0; i < n; i++)
            per[i] = i;

        // insertion sort by nominal degrees
        unsigned tmp;

        for (unsigned i = 1; i < n; i++)
        {
            tmp = per[i];

            unsigned j;
            for (j = i; j >= 1; j--)
                if (nom_deg[per[j-1]] < nom_deg[tmp])
                   per[j] = per[j-1];
                else
                   break;

            per[j] = tmp;
        }

        // exchanging rows of fta
        boost::scoped_ptr<matrix<n-m, n> > t(new matrix<n-m, n>);

        for (unsigned i = 0; i < n; i++)
            t->L[i] = fta.L[per[i]];

        // gaussian elim.
        gfv<n-m> tmp_v, tmp_w;

        tau.set_zero();

        for (unsigned i = 0; i < (n-m); i++)
        {
            for (unsigned j = n-i; j-- > 0; )
            {
                if(t->L[j][i])
                {
#if QQ != 2
                    gf p = t->L[j][i].inv();
#endif

                    // moving the pivot row backward
                    tmp = per[j];
                    tmp_v = t->L[j];
                    tmp_w = tau.L[j];

                    for (unsigned k = j+1; k <= n-1-i; k++)
                    {
                        per[k-1] = per[k];
                        t->L[k-1] = t->L[k];
                        tau.L[k-1] = tau.L[k];
                    }

                    per[n-1-i] = tmp;
                    t->L[n-1-i] = tmp_v;
                    tau.L[n-1-i] = tmp_w;
                    tau.L[n-1-i].set((n-m) - 1 - i, gf(1));

#if QQ == 16
                    gfv<n-m> buf_t[16], buf_tau[16];

                    t->L[n-1-i].gf16_expand(buf_t);
                    tau.L[n-1-i].gf16_expand(buf_tau);
#endif
                    // elim
#ifdef _OPENMP
                    #pragma omp parallel for
#endif
                    for (unsigned k = 0; k < j; k++)
                    {
                        if(t->L[k][i])
                        {
#if QQ == 16
                            gf q(p * t->L[k][i]);

                            t->L[k] -= buf_t[q.v];
                            tau.L[k] -= buf_tau[q.v];
#elif QQ == 2
                            t->L[k] -= t->L[n-1-i];
                            tau.L[k] -= tau.L[n-1-i];
#else
                            gf q(p * t->L[k][i]);

                            t->L[k] -= t->L[n-1-i] * q;
                            tau.L[k] -= tau.L[n-1-i] * q;
#endif
                        }
                    }

                    break;
                }
            }
        }
    }

    void update()
    {
        matrix<n-m, n> Tau;
        unsigned order[n];

        // generating information of tau matrix
        gen_tau(Tau, order);

        // updating nd
        unsigned nd_tmp[n];

#ifdef OPEN_MPI
        bool deg_increased = false;
#endif

        /////////////// updating nominal degrees //////////////

        for (unsigned i = 0; i < n; i++)
            nd_tmp[i] = nom_deg[order[i]];

        for (unsigned i = 0; i < n; i++)
            nom_deg[i] = nd_tmp[i];

        for (unsigned i = m; i < n; i++)
            nom_deg[i]++;

        //////////////// updating coef //////////////////

#ifdef OPEN_MPI
        assert(num_coef < coef.deg);
#else
        assert((max_nom_deg + 1) < coef.deg);
#endif

#ifdef OPEN_MPI
        // there might be empty ranges when the coefficients are distributed over MPI nodes
        if (num_coef > 0)
#endif
        {
#ifdef _OPENMP
            unsigned max_threads = omp_get_max_threads();
#else
            unsigned max_threads = 1;
#endif
            boost::scoped_array<matrix<m, n> > buf(new matrix<m, n>[max_threads]);

            int start[max_threads];

            for (unsigned i = 0; i < max_threads; i++)
                start[i] = -1;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
#ifdef OPEN_MPI
            for (int i = (start_ci + num_coef - 1); i >= (int)start_ci; i--)
#else
            for (int i = (max_nom_deg + 1) - 1; i >= 0; i--)
#endif
            {
#ifdef _OPENMP
                unsigned tid = omp_get_thread_num();
#else
                unsigned tid = 0;
#endif

                if (start[tid] == -1) //first iteration
                {
                    start[tid] = i;

                    // applying permutation
                    for (unsigned j = 0; j < n; j++)
                        buf[tid].L[j] = coef[i].L[order[j]];

                    // applying the "not-so-dense" matrix
                    matrix_mad_special(buf[tid], Tau, &(buf[tid].L[m]), m);

                    // copy upper part to coef[i]
                    for (unsigned j = 0; j < m; j++)
                        coef[i].L[j] = buf[tid].L[j];

                    // copy lower part to coef[start[tid]+1] when all threads are finished
                }
                else
                {
                    matrix<m, n> tmp_coef;

                    // applying permutation
                    for (unsigned j = 0; j < n; j++)
                        tmp_coef.L[j] = coef[i].L[order[j]];

                    // applying the "not-so-dense" matrix to top-m rows
                    matrix_mad_special(tmp_coef, Tau, &(tmp_coef.L[m]), m);

                    // copy upper part to coef[i]
                    for (unsigned j = 0; j < m; j++)
                        coef[i].L[j] = tmp_coef.L[j];

                    // copy lower part to coef[i+1]
                    for (unsigned j = m; j < n; j++)
                        coef[i+1].L[j] = tmp_coef.L[j];

                }
            }

            unsigned count = 0;

            // compute the actual number of threads involved
            for (count = 0; (count < max_threads); count++)
               if (start[count] == -1)
                  break;

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (unsigned tid = 0; tid < count; tid++)
               // copy lower part to coef[start[tid]+1]
               for (unsigned j = m; j < n; j++)
                  coef[start[tid] + 1].L[j] = buf[tid].L[j];

        }

        for (unsigned j = 0; j < m; j++)
#ifdef OPEN_MPI
            coef[start_ci + num_coef].L[j].set_zero();
#else
            coef[max_nom_deg + 1].L[j].set_zero();
#endif

#ifdef OPEN_MPI
        if (mpi_rank == 0)
#endif
            for (unsigned j = m; j < n; j++)
                coef[0].L[j].set_zero();

/////////////////////////////////////////////////////

        for (unsigned i = m; i < n; i++)
        {
            if(nom_deg[i] > max_nom_deg)
            {
#ifdef OPEN_MPI
                deg_increased = true; // max_nom_deg increased
#endif
                max_nom_deg = nom_deg[i];

                break;
            }
        }

#ifdef OPEN_MPI
        unsigned end_ci = start_ci + num_coef;

        unsigned new_end_ci = end_ci;
        unsigned new_start_ci = start_ci;

        if(deg_increased == true)
        {
            if (mpi_rank == max_nom_deg % mpi_size)
            {
                new_end_ci = end_ci + 1;
            }

            if (mpi_rank > max_nom_deg % mpi_size)
            {
                new_end_ci = end_ci + 1;
                new_start_ci = start_ci + 1;
            }
        }

        unsigned _mpi_size = min(mpi_size, max_nom_deg+1);


        if(mpi_rank < _mpi_size)
        {
           enum MPI_TAGS{TAG_FULL_COFF, TAG_HALF_COFF};

            MPI_Request send1;
            MPI_Request send2;

            gfv<m> * start_ptr = coef[start_ci].L;
            gfv<m> * end_ptr   = coef[end_ci].L;

            if(new_end_ci == end_ci && mpi_rank < _mpi_size-1 && end_ci > start_ci)
            {
                MPI_Isend(end_ptr + m, sizeof(gfv<m>)*(n-m), MPI_CHAR,
                          mpi_rank+1, TAG_HALF_COFF, MPI_COMM_WORLD, &send1);
            }

            if(new_start_ci == start_ci+1 && mpi_rank > 0 && end_ci > start_ci)
            {
                MPI_Isend(start_ptr, sizeof(gfv<m>)*m, MPI_CHAR,
                          mpi_rank-1, TAG_FULL_COFF, MPI_COMM_WORLD, &send2);
            }

            if(new_start_ci == start_ci && mpi_rank > 0 && new_end_ci > new_start_ci)
            {
                MPI_Recv(start_ptr + m, sizeof(gfv<m>)*(n-m), MPI_CHAR,
                         mpi_rank-1, TAG_HALF_COFF, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }

            if(new_end_ci == end_ci+1 && mpi_rank < _mpi_size-1 && new_end_ci > new_start_ci)
            {
                MPI_Recv(end_ptr, sizeof(gfv<m>)*m, MPI_CHAR,
                         mpi_rank+1, TAG_FULL_COFF, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }

            if(new_end_ci == end_ci && mpi_rank < _mpi_size-1 && end_ci > start_ci)
            {
                MPI_Wait(&send1, MPI_STATUS_IGNORE);
            }

            if(new_start_ci == start_ci+1 && mpi_rank > 0 && end_ci > start_ci)
            {
                MPI_Wait(&send2, MPI_STATUS_IGNORE);
            }
        }


        start_ci = new_start_ci;

        num_coef = new_end_ci - start_ci;
#endif

        // generating lower part of the next fta
        // extremely important !!!!!
        matrix<n-m, n> copy_fta;

        copy_fta = fta;

        for (unsigned j = m; j < n; j++)
           fta.L[j] = copy_fta.L[order[j]];
    }
};

#endif // _MATRIX_POLYNOMIAL_H

