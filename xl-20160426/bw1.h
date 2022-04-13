#ifndef _BW1_h_
#define _BW1_h_

#ifdef IBV
#include "ibv/ibv_buffer.h"
#endif

#include <float.h>

#include "util.h"
#include "matrix.h"
#include "sp_matrix.h"
#include "xl_mem.h"

#define OUTPUT_PERIOD_MS 5000

#ifdef MPI
#include "xl_comm.h"
#endif

class BW1_base
{
   // returns true every OUTPUT_PERIOD_MS ms
   static bool time_to_output(double period)
   {
      static double last_t = 0;

      double t = get_ms_time();

      if(t - last_t >= period) 
      {
         last_t = t;
         return true;    
      }
      else 
         return false;
   }

   public:

   static void show_speed_stats(unsigned i, unsigned max, double num_mults, double max_mults,
         const char *part = NULL)
   {
      static double start_mults, last_mults = DBL_MAX;
      static double start_t = 0;
      static double period = 0;

      // meant to print stats of bw_i
      if(i > max)
      {
         double t = get_ms_time();
         double period = t - start_t;

         ECHO("%s time: %.2f ms, mults/cycle: %.2f\n\n", part, period, 
               num_mults / (period * (double)CPU_FREQ * 1000.0));

         fflush(stdout);

         return;
      }

      // start of new loop
      if(num_mults <= last_mults)
      {
         last_mults = start_mults = num_mults;
         start_t = get_ms_time();
         period = OUTPUT_PERIOD_MS;

         time_to_output(0);

         ECHO("starting computation...\n");

         return;
      }

      if( time_to_output(max(period,OUTPUT_PERIOD_MS)) )
      {
         double t = get_ms_time();
         double delta_t = t - start_t;

         ECHO("(%i/%i), remaining: %.2f s, mults/cycle: %.2f\n", i, max, 
               double(max_mults-num_mults) * delta_t / (double)(num_mults-start_mults) / 
               1000.0, 
               num_mults / (delta_t * (double)CPU_FREQ * 1000.0));

         period = (((double)max_mults / (double)num_mults) * delta_t) / 100.0;

         last_mults = num_mults;

         fflush(stdout);
      }
   }
};


class BW1 : BW1_base
{

	public:

   template <unsigned m, unsigned n, unsigned N, unsigned w, class Mac, unsigned deg_ai>
   static double bw_1(matrix_array<m, n, deg_ai> &ai , const unsigned num_iter, 
           Mac &B, const sparse_matrix<n, N, w> &z_sp)
   {
       ECHO("BW_1 no mpi\n");
   
       void *map[2];
   
       for (int i = 0; i < 2; i++)
           map[i] = XL_malloc(sizeof(matrix<n, N>));
   
       matrix<n, N> * BiyNew = new(map[0]) matrix<n, N>;
       matrix<n, N> * BiyOld = new(map[1]) matrix<n, N>;
   
       boost::scoped_ptr<matrix<n, m> > ai_tr(new matrix<n, m>);
   
       DUMP(sizeof(matrix<n, N>));
   
#ifdef _OPENMP
#pragma omp parallel for schedule(static) 
           for(unsigned r = 0; r < N; r++)
           {
              BiyNew->L[r].set_zero();
              BiyOld->L[r].set_zero();
           }
   
#pragma omp parallel for schedule(static) 
           for(unsigned r = 0; r < m; r++)
              ai_tr->L[r].set_zero();
#endif
   
       z_sp.dense_copy(*BiyOld);
   
       uint64_t num_mults = 0;
   
       double t_sp = 0;
       double t_mac = 0;
       double t_trans = 0;
   
       for(unsigned i = 0; i < num_iter; i++) 
       {
           show_speed_stats(i, num_iter, num_mults, (B.num_entries() + n*n) * n * num_iter);
   
           num_mults += (B.num_entries() + n*n) * n;
   
           t_mac -= get_ms_time();
           sparse_matrix_prod(*BiyNew, B, *BiyOld);
           t_mac += get_ms_time();
   
           t_sp -= get_ms_time();
           memcpy(ai_tr->L, &(BiyNew->L), sizeof(gfv<m>) * n);
           t_sp += get_ms_time();
   
           swap(BiyOld, BiyNew);
   
           t_trans -= get_ms_time();

#ifdef _OPENMP
#pragma omp parallel for 
#endif
           // transpose
           for (unsigned k = 0; k < m; k++)
              for (unsigned j = 0; j < n; j++)
                   ai[i].L[k].set(j, ai_tr->get(j, k));
           t_trans += get_ms_time();
       }
   
       show_speed_stats(1, 0, num_mults, num_mults, "BW_1");   
   
       ECHO_NL();
   
       BiyOld->~matrix<n, N>();
       BiyNew->~matrix<n, N>();
   
       for (int i = 0; i < 2; i++)
           XL_free(map[i], sizeof(matrix<n, N>));
   
       return num_mults;
   }

};

#ifdef OPEN_MPI
#ifdef MPI_SIZE

class BW1_mpi_size_blocks : BW1_base
{

	public:

   template <unsigned m, unsigned n, unsigned N, unsigned w, class Mac, unsigned deg_ai>
   static double bw_1(matrix_array<m, n, deg_ai> &ai , const unsigned num_iter, 
           Mac &B, const sparse_matrix<n, N, w> &z_sp)
   {
       ECHO("BW_1 mpi_size blocks mpi\n");
       
       void *map[2];
   
       const unsigned int n_mpi = n / MPI_SIZE;
   
       for (int i = 0; i < 2; i++)
           map[i] = XL_malloc(sizeof(matrix<n_mpi, N>));
   
       matrix<n_mpi, N> * BiyNew = new(map[0]) matrix<n_mpi, N>;
       matrix<n_mpi, N> * BiyOld = new(map[1]) matrix<n_mpi, N>;
   
       BiyOld->set_zero();
   
       for(unsigned i = 0; i < N; i++)
           for(unsigned j = 0; j < z_sp.w; j++)
               if (((mpi_rank*n_mpi) <= z_sp.idx[i*z_sp.w+j]) &&
                       (z_sp.idx[i*z_sp.w+j] < ((mpi_rank+1)*n_mpi)))
                   BiyOld->set(
                           i,
                           z_sp.idx[i*z_sp.w+j] - (mpi_rank*n_mpi),
                           z_sp.val[i][j] );
   
       matrix<n_mpi, m> *ai_tr;
       ai_tr = new matrix<n_mpi, m>[MPI_SIZE*(num_iter+2)];
   
       double num_mults = 1;
   
       for(unsigned i = 0; i < num_iter + 1; i++) 
       {
           show_speed_stats(i, num_iter + 2, num_mults,
                B.num_entries() * n * num_iter);
           num_mults += B.num_entries() * n;
   
           sparse_matrix_prod(*BiyNew, B, *BiyOld);
   
           memcpy(ai_tr[mpi_rank * (num_iter+2) + i].L, BiyOld->L, sizeof(ai_tr[mpi_rank * (num_iter+2) + i].L[0]) * n);
   
           swap(BiyOld, BiyNew);
       }
   
       show_speed_stats(1, 0, num_mults, num_mults, "BW1");   
   
       MPI_Barrier(MPI_COMM_WORLD);
   
       MPI_Allgather
           (
            ai_tr + mpi_rank * (num_iter+2), 
            sizeof(matrix<n_mpi, m>) * (num_iter+2), 
            MPI_CHAR,
            ai_tr,
            sizeof(matrix<n_mpi, m>) * (num_iter+2), 
            MPI_CHAR,
            MPI_COMM_WORLD
           );
   
       // transpose
#ifdef _OPENMP
#pragma omp parallel for 
#endif
       for(unsigned i = 1; i < num_iter + 1; i++) 
           for (unsigned rank = 0; rank < MPI_SIZE; rank++)
               for(unsigned j = 0; j < m; j++)
                   for(unsigned k = 0; k < n_mpi; k++)
                       ai[i-1].L[k + rank*n_mpi].set(j, 
                             ai_tr[rank * (num_iter+2) + i].get(j, k));
   
       BiyOld->~matrix<n_mpi, N>();
       BiyNew->~matrix<n_mpi, N>();
   
       for (int i = 0; i < 2; i++)
           XL_free(map[i], sizeof(matrix<n_mpi, N>));
   
       delete [] ai_tr;
   
      return num_mults;
   }

};

#endif // ifdef MPI_SIZE

#ifdef IBV

class BW1_two_blocks_ibv : BW1_base
{

   public:

   template <unsigned m, unsigned n, unsigned N, unsigned w, class Mac, unsigned deg_ai>
   static double bw_1(matrix_array<m, n, deg_ai> &ai , const unsigned num_iter, 
           Mac &B, const sparse_matrix<n, N, w> &z_sp)
   {
      ECHO("BW_1 two blocks mpi\n");

      const uint32_t n_mpi = n >> 1;

      ibv_buffer<matrix<n_mpi, N> > *map = 
         (ibv_buffer<matrix<n_mpi, N> > *)XL_malloc(
               4 * sizeof(ibv_buffer<matrix<n_mpi, N> >));

      ibv_buffer<matrix<n_mpi, N> > *BiyNew[2];
      ibv_buffer<matrix<n_mpi, N> > *BiyOld[2];

      DUMP(sizeof(ibv_buffer<matrix<n_mpi, N> >));

      BiyNew[0] = new(&map[0]) ibv_buffer<matrix<n_mpi, N> >;
      BiyOld[0] = new(&map[1]) ibv_buffer<matrix<n_mpi, N> >;

      BiyNew[1] = new(&map[2]) ibv_buffer<matrix<n_mpi, N> >;
      BiyOld[1] = new(&map[3]) ibv_buffer<matrix<n_mpi, N> >;

      BiyOld[0]->set_zero();
      BiyOld[1]->set_zero();

      for(unsigned i = 0; i < N; i++)
         for(unsigned j = 0; j < z_sp.w; j++)
            if (z_sp.idx[i*z_sp.w+j] < n_mpi)
               BiyOld[0]->set(
                     i, z_sp.idx[i*z_sp.w+j], z_sp.val[i][j] );
            else
               BiyOld[1]->set(
                     i, z_sp.idx[i*z_sp.w+j] - n_mpi, z_sp.val[i][j] );

      matrix<n_mpi, m> *ai_tr = new matrix<n_mpi, m>[2];

      double num_mults = 0;

      uint32_t mpi_start[mpi_size + 1];

      mpi_start[0] = 0;

      for (unsigned i = 0; i < mpi_size; i++)
         mpi_start[i] = (double)N / (double)mpi_size * (double)i;

      mpi_start[mpi_size] = N;


      double time_comm;
      double time_wait;
      double time_comp;

      time_comm = 0;
      time_wait = 0;
      time_comp = 0;

      // get communication ranges 

      uint32_t *cnt = (uint32_t*)malloc(Mac::width * mpi_size * sizeof(*cnt));

      for (unsigned i = 0; i < Mac::width * mpi_size; i++)
         cnt[i] = 0;

      for (unsigned rank = 0; rank < mpi_size; rank++)
      {
         for(unsigned r = mpi_start[rank]; r < mpi_start[rank + 1]; r++)
         {
            unsigned rb = B.mask.block_idx(r);

            for (unsigned col = 0; col < Mac::sys_m; col++)
               cnt[ B.col_idx[rb * Mac::sys_m + col] + Mac::width * rank] += 1;
         }

      }

      for (unsigned rank = (mpi_rank + 1) % mpi_size; rank != mpi_rank;
            rank = (rank + 1) % mpi_size)
      {
         {
            // send

            std::vector<ibv_range*> vec;

            unsigned i = mpi_start[mpi_rank];

            while (i < mpi_start[mpi_rank + 1])
            {
               while ((cnt[Mac::width * rank + i] == 0) && 
                     (i < mpi_start[mpi_rank + 1]))
                  i++;

               if (i < mpi_start[mpi_rank + 1])
               {
                  unsigned start = i;

                  while ((cnt[Mac::width * rank + i] > 0) && 
                        (i < mpi_start[mpi_rank + 1]))
                     i++;

                  vec.push_back(new ibv_range(start * sizeof(gfv<n_mpi>),
                           (i - start) * sizeof(gfv<n_mpi>)));
               }
            }

            for (int buf = 0; buf < 4; buf++)
               map[buf].set_range_send(rank, vec);

            vec.clear();
         }

         {
            // recv

            std::vector<ibv_range*> vec;

            unsigned i = mpi_start[rank];

            while (i < mpi_start[rank + 1])
            {
               while ((cnt[Mac::width * mpi_rank + i] == 0) 
                     && (i < mpi_start[rank + 1]))
                  i++;

               if (i < mpi_start[rank + 1])
               {
                  unsigned start = i;

                  while ((cnt[Mac::width * mpi_rank + i] > 0) 
                        && (i < mpi_start[rank + 1]))
                     i++;

                  vec.push_back(new ibv_range(start * sizeof(gfv<n_mpi>),
                           (i - start) * sizeof(gfv<n_mpi>)));
               }
            }

            for (int buf = 0; buf < 4; buf++)
               map[buf].set_range_recv(rank, vec);

            vec.clear();
         }
      }

      free(cnt);

      // comm stats
      {
         uint64_t sends[mpi_size];
         uint64_t recvs[mpi_size];

         uint64_t num_sends[mpi_size];
         uint64_t num_recvs[mpi_size];

         MPI_Gather(&map[0].send, sizeof(map[0].send), MPI_BYTE,
               sends, sizeof(map[0].send), MPI_BYTE, 0, MPI_COMM_WORLD);

         MPI_Gather(&map[0].recv, sizeof(map[0].recv), MPI_BYTE,
               recvs, sizeof(map[0].recv), MPI_BYTE, 0, MPI_COMM_WORLD);

         MPI_Gather(&map[0].num_send, sizeof(map[0].num_send), MPI_BYTE,
               num_sends, sizeof(map[0].num_send), MPI_BYTE, 0, MPI_COMM_WORLD);

         MPI_Gather(&map[0].num_recv, sizeof(map[0].num_recv), MPI_BYTE,
               num_recvs, sizeof(map[0].num_recv), MPI_BYTE, 0, MPI_COMM_WORLD);

         uint64_t sum_send = 0;
         uint64_t sum_recv = 0;

         uint64_t max_send = 0;
         uint64_t max_recv = 0;

         ECHO("\n");

         ECHO("node: | send:    | recv:    | num_send: | num_recv:\n");
         ECHO("--------------------------------------------------\n");   

         for (unsigned i = 0; i < mpi_size; i++)
         {
            ECHO("%2i    | %8lu | %8lu | %3lu       | %3lu\n", 
                  i, sends[i], recvs[i],
                  num_sends[i], num_recvs[i]);

            sum_send += sends[i];
            sum_recv += recvs[i];

            if (sends[i] > max_send)
            {
               max_send = sends[i];
            }
            max_recv = max(max_recv, recvs[i]);
         }

         ECHO("\n");

         ECHO("sum send: %lu  avg: %f  max: %lu\n",
               sum_send, (double)sum_send/mpi_size, max_send);
         ECHO("sum recv: %lu  avg: %f  max: %lu\n",
               sum_recv, (double)sum_recv/mpi_size, max_recv);

         ECHO("\n");
      }
      
      MPI_Barrier(MPI_COMM_WORLD);

      for (unsigned id = 0; id < 2; id++)
      {
         time_comp -= get_ms_time();
         sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
               mpi_start[mpi_rank], 
               mpi_start[mpi_rank+1] - mpi_start[mpi_rank],
               false);
         time_comp += get_ms_time();

         time_comm -= get_ms_time();
         BiyNew[id]->allgather();
         time_comm += get_ms_time();
      }

      for(unsigned i = 0; i < num_iter; i++) 
      {
         show_speed_stats(i, num_iter, num_mults, (B.num_entries() + n*n) * n * num_iter);
         num_mults += (B.num_entries() + n*n) * n;

         for (unsigned id = 0; id < 2; id++)
         {
            time_wait -= get_ms_time();
            BiyNew[id]->wait();
            time_wait += get_ms_time();


            time_comp -= get_ms_time();
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
//            for(unsigned j = 0; j < m; j++)
//            {
//               ai_tr[id].L[j].set_zero();
//               for(unsigned i = 0; i < x->w; i++)
//               {
//                  unsigned col = x->idx[j*z_sp.w+i];
//
//                  if ((mpi_start[mpi_rank] <= col)
//                        && (col < mpi_start[mpi_rank + 1]))
//                     ai_tr[id].L[j] += (BiyNew[id]->L[col]) * x->val[j][i];
//               }
//            }

            memcpy(ai_tr[id].L, BiyNew[i]->L, sizeof(ai_tr[id].L[0]) * n);


#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(unsigned k = 0; k < ai_tr[id].m; k++)
               for(unsigned j = 0; j < ai_tr[id].n; j++)
                  if (id == 0)
                     ai[i].L[k].set(j, ai_tr[0].get(j, k));
                  else
                     ai[i].L[k + ai_tr[0].m].set(j, ai_tr[1].get(j, k));
            time_comp += get_ms_time();

            swap(BiyOld[id], BiyNew[id]);

            if (i < num_iter)
            {
               time_comp -= get_ms_time();
               sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
                     mpi_start[mpi_rank], 
                     mpi_start[mpi_rank+1] - mpi_start[mpi_rank],
                     false);
               time_comp += get_ms_time();

               MPI_Barrier(MPI_COMM_WORLD);

               time_comm -= get_ms_time();
               BiyNew[id]->allgather();
               time_comm += get_ms_time();
            }
         }
      }


      show_speed_stats(1, 0, num_mults, num_mults, "BW_1");

      MPI_Barrier(MPI_COMM_WORLD);

      ECHO("\ncollect ai (msg size: %f MB)\n", 
            (double)(sizeof(matrix<m, n>) * (num_iter + 1)) / (1024*1024));

      for(unsigned i = 0; i < num_iter + 1; i++)
          MPI_Allreduce(MPI_IN_PLACE, &ai[i], sizeof(matrix<m, n>), MPI_BYTE,
            MPI_BXOR, MPI_COMM_WORLD);

      ECHO("collect done\n\n");

      MPI_Barrier(MPI_COMM_WORLD);

      for (unsigned rank = 0; rank < mpi_size; rank++)
      {
         MPI_Status status;

         if (rank > 0)
         {
            if (mpi_rank == 0)
            {
               MPI_Recv(&time_comm, 1, MPI_DOUBLE,
                     rank, 0, MPI_COMM_WORLD, &status);
               MPI_Recv(&time_wait, 1, MPI_DOUBLE,
                     rank, 0, MPI_COMM_WORLD, &status);
               MPI_Recv(&time_comp, 1, MPI_DOUBLE,
                     rank, 0, MPI_COMM_WORLD, &status);
            }
            if (mpi_rank == rank)
            {
               MPI_Send(&time_comm, 1, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
               MPI_Send(&time_wait, 1, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
               MPI_Send(&time_comp, 1, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
            }
         }

         ECHO("%i - time_comm: %8.2f  time_wait: %8.2f  time_comp: %8.2f\n", 
               rank, time_comm, time_wait, time_comp);
      }

      ECHO_NL();

      MPI_Barrier(MPI_COMM_WORLD);

      BiyOld[0]->~ibv_buffer<matrix<n_mpi, N> >();
      BiyNew[0]->~ibv_buffer<matrix<n_mpi, N> >();
      BiyOld[1]->~ibv_buffer<matrix<n_mpi, N> >();
      BiyNew[1]->~ibv_buffer<matrix<n_mpi, N> >();

      XL_free(map, 4*sizeof(matrix<n_mpi, N>));

      delete [] ai_tr;

      return num_mults;
   }

};

#endif //ifdef IBV


class BW1_two_blocks : BW1_base
{

   public:


   template <unsigned m, unsigned n, unsigned N, unsigned w, 
            class Mac, unsigned deg_ai>
   static double bw_1(matrix_array<m, n, deg_ai> &ai , const unsigned num_iter, 
         Mac &B, const sparse_matrix<n, N, w> &z_sp)
   {
      ECHO("BW_1 two blocks mpi\n");

      const uint32_t n_mpi = n >> 1;

      matrix<n_mpi, N> *map = 
         (matrix<n_mpi, N> *)XL_malloc(4 * sizeof(matrix<n_mpi, N>));

      matrix<n_mpi, N> *BiyNew[2];
      matrix<n_mpi, N> *BiyOld[2];

      BiyNew[0] = new(&map[0]) matrix<n_mpi, N>;
      BiyOld[0] = new(&map[1]) matrix<n_mpi, N>;

      BiyNew[1] = new(&map[2]) matrix<n_mpi, N>;
      BiyOld[1] = new(&map[3]) matrix<n_mpi, N>;

      BiyOld[0]->set_zero();
      BiyOld[1]->set_zero();

      for(unsigned i = 0; i < N; i++)
         for(unsigned j = 0; j < z_sp.w; j++)
            if (z_sp.idx[i*z_sp.w+j] < n_mpi)
               BiyOld[0]->set(
                     i, z_sp.idx[i*z_sp.w+j], z_sp.val[i][j] );
            else
               BiyOld[1]->set(
                     i, z_sp.idx[i*z_sp.w+j] - n_mpi, z_sp.val[i][j] );

      matrix<n_mpi, m> *ai_tr = new matrix<n_mpi, m>[2];

      double num_mults = 0;

      XL_Handle handle[2];

      uint32_t mpi_work = N / mpi_size;

      double time_comm[2];
      double time_comp[2];

      time_comm[0] = 0;
      time_comm[1] = 0;
      time_comp[0] = 0;
      time_comp[1] = 0;

      unsigned step = mpi_work*1/(mpi_size*mpi_size);
      unsigned row;

      DUMP(N - mpi_work * mpi_size);


      for (unsigned id = 0; id < 2; id++)
      {
         time_comp[id] -= get_ms_time();

         sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
               mpi_work * mpi_rank, mpi_work, false);
         sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
               mpi_work * mpi_size, N - mpi_work * mpi_size, false);

         time_comp[id] += get_ms_time();

         time_comm[id] -= get_ms_time();
         XL_Iallgather(&(BiyNew[id]->L[mpi_work*mpi_rank]), 
               sizeof(BiyNew[id]->L[0]) * mpi_work, MPI_CHAR,
               BiyNew[id]->L, sizeof(BiyNew[id]->L[0]) * mpi_work, MPI_CHAR,
               MPI_COMM_WORLD, &handle[id]);
         time_comm[id] += get_ms_time();
      }

      for(unsigned i = 0; i < num_iter; i++) 
      {
         show_speed_stats(i, num_iter, num_mults, (B.num_entries() + n*n) * n * num_iter);
         num_mults += (B.num_entries() + n*n) * n;

         for (unsigned id = 0; id < 2; id++)
         {
            time_comm[id] -= get_ms_time();
            XL_Wait(&handle[id]);
            time_comm[id] += get_ms_time();


            time_comp[id] -= get_ms_time();
            memcpy(ai_tr[id].L, BiyNew[id]->L, sizeof(ai_tr[id].L[0]) * n);
            time_comp[id] += get_ms_time();


            time_comp[id] -= get_ms_time();
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(unsigned k = 0; k < ai_tr[id].m; k++)
               for(unsigned j = 0; j < ai_tr[id].n; j++)
                  if (id == 0)
                     ai[i].L[k].set(j, ai_tr[0].get(j, k));
                  else
                     ai[i].L[k + ai_tr[0].m].set(j, ai_tr[1].get(j, k));
            time_comp[id] += get_ms_time();


            swap(BiyOld[id], BiyNew[id]);

            if (i < num_iter)
            {
               for (row = mpi_work * mpi_rank; 
                     row + step < mpi_work * (mpi_rank + 1);
                     row += step)
               {
                  time_comp[id] -= get_ms_time();
                  sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
                        row, step, false);
                  time_comp[id] += get_ms_time();

                  time_comm[id] -= get_ms_time();
                  XL_Test(&handle[(id+1)%2]);
                  time_comm[id] += get_ms_time();
               }

               time_comp[id] -= get_ms_time();
               sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id], 
                     row, mpi_work * (mpi_rank + 1) - row, false);
               sparse_matrix_mad(&(BiyNew[id]->L[0]), B, *BiyOld[id],
                     mpi_work * mpi_size, N - mpi_work * mpi_size, false);
               time_comp[id] += get_ms_time();


               time_comm[id] -= get_ms_time();
               XL_Iallgather(&(BiyNew[id]->L[mpi_work*mpi_rank]), 
                     sizeof(BiyNew[id]->L[0]) * mpi_work, MPI_CHAR,
                     BiyNew[id]->L, 
                     sizeof(BiyNew[id]->L[0]) * mpi_work, MPI_CHAR, 
                     MPI_COMM_WORLD, &handle[id]);
               time_comm[id] += get_ms_time();
            }
         }
      }

      show_speed_stats(1, 0, num_mults, num_mults, "BW_1");


      MPI_Barrier(MPI_COMM_WORLD);

      for (unsigned rank = 0; rank < mpi_size; rank++)
      {
         MPI_Status status;

         if (rank > 0)
         {
            if (mpi_rank == 0)
            {
               MPI_Recv(time_comm, 2, MPI_DOUBLE,
                     rank, 0, MPI_COMM_WORLD, &status);
               MPI_Recv(time_comp, 2, MPI_DOUBLE,
                     rank, 0, MPI_COMM_WORLD, &status);
            }
            if (mpi_rank == rank)
            {
               MPI_Send(time_comm, 2, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
               MPI_Send(time_comp, 2, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
            }
         }

         ECHO("%i - time_comm: %8.2f  time_comp: %8.2f\n", 
               rank, time_comm[0]+time_comm[1], 
               time_comp[0]+time_comp[1]);
         //ECHO("    time_comm[1]: %8.2f  time_comp[1]: %8.2f\n", 
         //      time_comm[1], time_comp[1]);

      }

      ECHO_NL();

      MPI_Barrier(MPI_COMM_WORLD);

      BiyOld[0]->~matrix<n_mpi, N>();
      BiyNew[0]->~matrix<n_mpi, N>();
      BiyOld[1]->~matrix<n_mpi, N>();
      BiyNew[1]->~matrix<n_mpi, N>();

      XL_free(map, 4*sizeof(matrix<n_mpi, N>));

      delete [] ai_tr;

      return num_mults;
   }

};

class BW1_one_block : BW1_base
{

	public:

   template <unsigned m, unsigned n, unsigned N, unsigned w, class Mac, unsigned deg_ai>
   static double bw_1(matrix_array<m, n, deg_ai> &ai , const unsigned num_iter, 
           Mac &B, const sparse_matrix<n, N, w> &z_sp)
   {
       ECHO("BW_1 one block mpi\n");
   
       void *map[2];
   
       for (int i = 0; i < 2; i++)
           map[i] = XL_malloc(sizeof(matrix<n, N>));
   
       matrix<n, N> * BiyNew = new(map[0]) matrix<n, N>;
       matrix<n, N> * BiyOld = new(map[1]) matrix<n, N>;
   
       DUMP(sizeof(matrix<n, N>));
   
       z_sp.dense_copy(*BiyOld);
   
       matrix<n, m> *ai_tr = new matrix<n, m>;
   
       unsigned block_size = 1024*8;
       unsigned step = block_size*mpi_size;
   
       unsigned num_handle = 8;
   
       XL_Handle handle[num_handle];
   
       double num_mults = 0;
   
       for(unsigned i = 0; i < num_iter; i++) 
       {
           show_speed_stats(i, num_iter, num_mults, (B.num_entries() + n*n) * n * num_iter);
   
           num_mults += (B.num_entries() + n*n) * n;
   
           unsigned row;
           unsigned h = 0;
   
           for (row = 0; (row+step) < N; row += step)
           {
               sparse_matrix_mad(BiyNew->L, B, *BiyOld,
                       row + block_size*mpi_rank, block_size, false);
   
               if (row != 0)
                   XL_Wait(&handle[h]);
   
               XL_Iallgather(&BiyNew->L[row + block_size*mpi_rank], 
                       sizeof(BiyNew->L[0]) * block_size, MPI_CHAR,
                       &BiyNew->L[row], 
                       sizeof(BiyNew->L[0]) * block_size, MPI_CHAR,
                       MPI_COMM_WORLD, &handle[h]);
   
               h = (h + 1) % num_handle;
           }
   
           sparse_matrix_mad(BiyNew->L, B, *BiyOld,
                   row, N - row, false);
   
           for (h = 0; h < num_handle; h++)
               XL_Wait(&handle[h]);
   
           memcpy(ai_tr->L, BiyNew->L, sizeof(ai_tr->L[0]) * n);
   
           swap(BiyOld, BiyNew);
   
#ifdef _OPENMP
#pragma omp parallel for 
#endif
           // transpose
           for(unsigned j = 0; j < ai_tr->n; j++)
               for(unsigned k = 0; k < ai_tr->m; k++)
                   ai[i].L[k].set(j, ai_tr->get(j, k));
       }
   
   
       BiyOld->~matrix<n, N>();
       BiyNew->~matrix<n, N>();
   
       for (int i = 0; i < 2; i++)
           XL_free(map[i], sizeof(matrix<n, N>));
   
       delete ai_tr;
   
       show_speed_stats(1, 0, num_mults, num_mults, "BW_1");   
   
       return num_mults;
   }
};

#endif // ifndef MPI

#endif // ifndef _BW1_h_

