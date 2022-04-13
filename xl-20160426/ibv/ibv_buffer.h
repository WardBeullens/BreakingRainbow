#ifndef __BW_COMM_H_
#define __BW_COMM_H_

#include <infiniband/verbs.h>

#include "xl_ibv_comm.h"

class ibv_range
{
public:
   unsigned offset;
   unsigned length;

   ibv_range(unsigned o, unsigned l) : offset(o), length(l) {}
};

template <class T>
class ibv_buffer : public T
{
   public:
      void *data;

      static struct xl_ibv_context *ctx;
      static unsigned inst_cnt;
      static bool init_qp;

      struct ibv_send_wr **send_wr;
      struct ibv_recv_wr **recv_wr;

      uint64_t num_send;
      uint64_t num_recv;

      uint64_t send;
      uint64_t recv;

      struct ibv_handle handle;

      std::vector<struct ibv_mr*> mr;

      ibv_buffer()
      {
         send_wr = (struct ibv_send_wr**)malloc(mpi_size * sizeof(struct ibv_send_wr **));
         recv_wr = (struct ibv_recv_wr**)malloc(mpi_size * sizeof(struct ibv_recv_wr **));

         this->data = (T*)this;

         num_send = 0;
         num_recv = 0;

         send = 0;
         recv = 0;

         if (ctx == NULL)
         {
            ECHO("init ctx\n");

            ctx = xl_ibv_init_ctx();
            if (!ctx)
            {
               fprintf(stderr, "Couldn't initialize Infiniband!\n");
               ABORT;
            }
         }

         inst_cnt++;
      };

      ~ibv_buffer()
      {
         std::vector<struct ibv_mr*>::const_iterator it;

         for (it = this->mr.begin(); it != this->mr.end(); it++)
         {
            if (ibv_dereg_mr(*it)) 
               fprintf(stderr, "Couldn't deregister MR\n");
         }

         inst_cnt--;

         if (inst_cnt == 0)
         {
            MPI_Barrier(MPI_COMM_WORLD);

            ECHO("kill ctx\n");

            xl_ibv_close_ctx(ctx);

            init_qp = true;

            ctx = NULL;
         }
      }

      void init_wr(struct ibv_sge &wr, unsigned offset,
            unsigned length)
      {
         struct ibv_mr* mr = ibv_reg_mr(ctx->pd, 
               (void*)((uint64_t)(this->data) + offset),
               length,
               IBV_ACCESS_LOCAL_WRITE);
         if (!mr) {
            fprintf(stderr, "Couldn't register MR\n");
            ABORT;
         }

         wr.addr   = (uint64_t)(this->data) + offset;
         wr.length = length;
         wr.lkey   = mr->lkey;

         this->mr.push_back(mr);

      }

      void set_range_send(unsigned rank, std::vector<ibv_range*> &vec)
      {
         this->set_range_cnt_send(rank, vec.size());

         std::vector<ibv_range*>::const_iterator it;

         for (unsigned j = 0; j < vec.size(); j++)
         {
            this->init_wr(this->send_wr[rank][j >> 5].sg_list[j & 0x1f],
                  vec[j]->offset, vec[j]->length); 

            send += vec[j]->length;
         }
      }

      void set_range_recv(unsigned rank, std::vector<ibv_range*> &vec)
      {
         this->set_range_cnt_recv(rank, vec.size());

         std::vector<ibv_range*>::const_iterator it;

         for (unsigned j = 0; j < vec.size(); j++)
         {
            this->init_wr(this->recv_wr[rank][j >> 5].sg_list[j & 0x1f],
                  vec[j]->offset, vec[j]->length); 
            recv += vec[j]->length;
         }
      }

      void set_range_cnt_send(unsigned rank, int range_cnt)
      {
         {
            size_t size = ((range_cnt >> 5) + 1) * sizeof(struct ibv_send_wr);
            this->send_wr[rank] = (struct ibv_send_wr*)malloc(size);
            memset(this->send_wr[rank], 0, size);

            for (int j = 0; (j*32) < range_cnt; j++)
            {
               this->send_wr[rank][j].next = NULL;

               int num_sge = 32;
               if ((j+1)*32 > range_cnt)
                  num_sge = (range_cnt - j*32);

               this->send_wr[rank][j].opcode     = IBV_WR_SEND;
               this->send_wr[rank][j].send_flags = IBV_SEND_SIGNALED;

               this->send_wr[rank][j].num_sge = num_sge;
               this->send_wr[rank][j].sg_list = 
                  (struct ibv_sge*)malloc(num_sge * sizeof(struct ibv_sge));

               if (j > 0)
                  this->send_wr[rank][j-1].next = &(this->send_wr[rank][j]);
            }
         }

         num_send += range_cnt;
      }

      void set_range_cnt_recv(unsigned rank, int range_cnt)
      {
         {
            size_t size = ((range_cnt >> 5) + 1) * sizeof(struct ibv_recv_wr);
            this->recv_wr[rank] = (struct ibv_recv_wr*)malloc(size);
            memset(this->recv_wr[rank], 0, size);

            for (int j = 0; (j*32) < range_cnt; j++)
            {
               this->recv_wr[rank][j].next = NULL;

               int num_sge = 32;
               if ((j+1)*32 > range_cnt)
                  num_sge = (range_cnt - j*32);

               this->recv_wr[rank][j].num_sge = num_sge;
               this->recv_wr[rank][j].sg_list = 
                  (struct ibv_sge*)malloc(num_sge * sizeof(struct ibv_sge));

               if (j > 0)
                  this->recv_wr[rank][j-1].next = &(this->recv_wr[rank][j]);
            }

         }

         num_recv += range_cnt;
      }

      void allgather()
      {
         if (init_qp)
         {
            xl_ibv_init_qp(ctx, send_wr, recv_wr);
            init_qp = false;
         }

         ibv_allgather_blocks(
               ctx,
               send_wr,
               recv_wr,
               &handle
               );
      }

      void wait()
      {
         ibv_wait(ctx, &handle);
      }

};

template <class T>
struct xl_ibv_context *ibv_buffer<T>::ctx = NULL;

template <class T>
unsigned ibv_buffer<T>::inst_cnt = 0;

template <class T>
bool ibv_buffer<T>::init_qp = true;

#endif

