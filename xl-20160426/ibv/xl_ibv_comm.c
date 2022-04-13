#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>

#include <infiniband/verbs.h>

#include <mpi.h>

#include "xl_ibv_comm.h"

#ifdef MAIN
int mpi_size;
int mpi_rank;
#else
extern int mpi_size;
extern int mpi_rank;
#endif

#define ABORT exit(-1);

#define XL_IBV_MTU IBV_MTU_1024
#define XL_IBV_SL 0

uint16_t xl_ibv_get_local_lid(struct ibv_context *context, int port)
{
   struct ibv_port_attr attr;

   if (ibv_query_port(context, port, &attr))
      return 0;

   return attr.lid;
}

int xl_ibv_connect_ctx(struct ibv_qp *qp, int port, 
           int my_psn,
           enum ibv_mtu mtu, int sl,
           struct xl_ibv_dest *dest)
{
   struct ibv_qp_attr attr = {
      .qp_state           = IBV_QPS_RTR,
      .path_mtu           = mtu,
      .dest_qp_num        = dest->qpn,
      .rq_psn             = dest->psn,
      .max_dest_rd_atomic = 1,
      .min_rnr_timer      = 12,
      .ah_attr            = {
         .is_global     = 0,
         .dlid          = dest->lid,
         .sl            = sl,
         .src_path_bits = 0,
         .port_num      = port
      }
   };
   if (ibv_modify_qp(qp, &attr,
           IBV_QP_STATE              |
           IBV_QP_AV                 |
           IBV_QP_PATH_MTU           |
           IBV_QP_DEST_QPN           |
           IBV_QP_RQ_PSN             |
           IBV_QP_MAX_DEST_RD_ATOMIC |
           IBV_QP_MIN_RNR_TIMER)) {
      fprintf(stderr, "Failed to modify QP to RTR\n");
      return 1;
   }

   attr.qp_state      = IBV_QPS_RTS;
   attr.timeout       = 14;
   attr.retry_cnt     = 7;
   attr.rnr_retry     = 7;
   attr.sq_psn        = my_psn;
   attr.max_rd_atomic = 1;
   if (ibv_modify_qp(qp, &attr,
           IBV_QP_STATE              |
           IBV_QP_TIMEOUT            |
           IBV_QP_RETRY_CNT          |
           IBV_QP_RNR_RETRY          |
           IBV_QP_SQ_PSN             |
           IBV_QP_MAX_QP_RD_ATOMIC)) {
      fprintf(stderr, "Failed to modify QP to RTS\n");
      return 1;
   }

   return 0;
}

void xl_ibv_exch_dest(const struct xl_ibv_dest *loc_dest,
                   struct xl_ibv_dest *rem_dest)
{
   MPI_Status stat;

   MPI_Request *send = (MPI_Request*)malloc(sizeof(MPI_Request) * mpi_size);
   MPI_Request *recv = (MPI_Request*)malloc(sizeof(MPI_Request) * mpi_size);

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      MPI_Isend((void*)&loc_dest[i], sizeof(struct xl_ibv_dest), MPI_BYTE, 
            i, 1, MPI_COMM_WORLD, &send[i]);

      MPI_Irecv((void*)&rem_dest[i], sizeof(struct xl_ibv_dest), MPI_BYTE, 
            i, 1, MPI_COMM_WORLD, &recv[i]);
   }

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      MPI_Wait(&send[i], &stat);
      MPI_Wait(&recv[i], &stat);
   }

   free(send);
   free(recv);
}

struct xl_ibv_context *xl_ibv_init_ctx()
{
   struct xl_ibv_context *ctx;

   ctx = malloc(sizeof *ctx);
   if (!ctx)
      return NULL;

   struct ibv_device   *ib_dev;

   ctx->dev_list = ibv_get_device_list(NULL);
   if (!ctx->dev_list) {
      perror("Failed to get IB devices list");
      return NULL;
   }

   ib_dev = *(ctx->dev_list);
   if (!ib_dev) {
      fprintf(stderr, "No IB devices found\n");
      return NULL;
   }


   ctx->context = ibv_open_device(ib_dev);
   if (!ctx->context) {
      fprintf(stderr, "Couldn't get context for %s\n",
            ibv_get_device_name(ib_dev));
      return NULL;
   }

   /*  get all active ports  */
   ctx->dev_attr = (struct ibv_device_attr*)malloc(sizeof(struct ibv_device_attr));

   ibv_query_device(ctx->context, ctx->dev_attr);

   ctx->port = (int*)malloc(ctx->dev_attr->phys_port_cnt * sizeof(int));
   ctx->port_cnt = 0;

   for (int j = 1; j <= ctx->dev_attr->phys_port_cnt; j++)
   {
      struct ibv_port_attr port_attr;
      ibv_query_port(ctx->context, j, &port_attr);

      if (port_attr.state == IBV_PORT_ACTIVE)
      {
         ctx->port[ctx->port_cnt] = j;
         ctx->port_cnt++;
      }
   }

   ctx->pd = ibv_alloc_pd(ctx->context);
   if (!ctx->pd) {
      fprintf(stderr, "Couldn't allocate PD\n");
      return NULL;
   }

   return ctx;
}


struct xl_ibv_context * xl_ibv_init_qp(struct xl_ibv_context *ctx, 
      struct ibv_send_wr **send_wr, struct ibv_recv_wr **recv_wr)
{

   int num_send_wr[mpi_size];
   int num_recv_wr[mpi_size];

   int sum_wr = 0;

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      num_send_wr[i] = 1;
      num_recv_wr[i] = 1;

      {
         struct ibv_send_wr *tmp = send_wr[i];
         while (tmp != NULL) { num_send_wr[i]++; tmp = tmp->next; }
      }

      {
         struct ibv_recv_wr *tmp = recv_wr[i];
         while (tmp != NULL) { num_recv_wr[i]++; tmp = tmp->next; }
      }

      sum_wr = sum_wr + num_send_wr[i] + num_recv_wr[i];
   }

    ctx->cq = ibv_create_cq(ctx->context, sum_wr*2, NULL, NULL, 0);
   if (!ctx->cq) {
      fprintf(stderr, "Couldn't create CQ\n");
      perror("doof");
      return NULL;
   }

   struct xl_ibv_dest *loc_dest = 
      (struct xl_ibv_dest*)malloc(mpi_size * sizeof(struct xl_ibv_dest));
   struct xl_ibv_dest *rem_dest = 
      (struct xl_ibv_dest*)malloc(mpi_size * sizeof(struct xl_ibv_dest));

   ctx->qp = (struct ibv_qp**)malloc(mpi_size * sizeof(struct ibv_qp*));

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      {
              

         struct ibv_qp_init_attr attr = {
            .send_cq = ctx->cq,
            .recv_cq = ctx->cq,
            .cap     = {
               .max_send_wr  = num_send_wr[i]*2,
               .max_recv_wr  = num_recv_wr[i]*2,
               .max_send_sge = send_wr[i]->num_sge,
               .max_recv_sge = recv_wr[i]->num_sge,
            },
            .qp_type = IBV_QPT_RC
         };


         ctx->qp[i] = ibv_create_qp(ctx->pd, &attr);
         if (!ctx->qp[i])  {
            fprintf(stderr, "Couldn't create QP!\n");
            return NULL;
         }
      }

      {
         struct ibv_qp_attr attr = {
            .qp_state        = IBV_QPS_INIT,
            .pkey_index      = 0,
            .port_num        = ctx->port[i % ctx->port_cnt],
            .qp_access_flags = 0
         };

         if (ibv_modify_qp(ctx->qp[i], &attr,
                  IBV_QP_STATE              |
                  IBV_QP_PKEY_INDEX         |
                  IBV_QP_PORT               |
                  IBV_QP_ACCESS_FLAGS)) {
            fprintf(stderr, "Failed to modify QP to INIT\n");
            return NULL;
         }
      }


      loc_dest[i].lid = xl_ibv_get_local_lid(ctx->context, ctx->port[i % ctx->port_cnt]);
      loc_dest[i].qpn = ctx->qp[i]->qp_num;
      loc_dest[i].psn = lrand48() & 0xffffff;
      if (!loc_dest[i].lid) {
         fprintf(stderr, "Couldn't get local LID\n");
         return NULL;
      }
   }

   xl_ibv_exch_dest(loc_dest, rem_dest);

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      if (xl_ibv_connect_ctx(ctx->qp[i], ctx->port[i % ctx->port_cnt], 
               loc_dest[i].psn, XL_IBV_MTU, XL_IBV_SL, 
               &rem_dest[i]))
      {
         fprintf(stderr, "Couldn't connect to remote QP\n");
         return NULL;
      }
   }

   free(loc_dest);
   free(rem_dest);

   return ctx;
}

int xl_ibv_close_ctx(struct xl_ibv_context *ctx)
{
   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
   {
      if (ibv_destroy_qp(ctx->qp[i])) {
         fprintf(stderr, "Couldn't destroy QP\n");
         return 1;
      }
   }

   free(ctx->qp);

   free(ctx->dev_attr);

   free(ctx->port);

   if (ibv_destroy_cq(ctx->cq)) {
      fprintf(stderr, "Couldn't destroy CQ\n");
      return 1;
   }

   if (ibv_dealloc_pd(ctx->pd)) {
      fprintf(stderr, "Couldn't deallocate PD\n");
      return 1;
   }

   if (ibv_close_device(ctx->context)) {
      fprintf(stderr, "Couldn't release context\n");
      return 1;
   }

   ibv_free_device_list(ctx->dev_list);

   free(ctx);

   return 0;
}

int xl_ibv_post_recv(struct xl_ibv_context *ctx, void *buf, size_t size, int target,
      struct ibv_handle *handle)
{
   struct ibv_sge list = {
      .addr   = (uintptr_t) buf,
      .length = size,
   };
   struct ibv_recv_wr wr = {
      .next       = NULL,
      .wr_id      = (uint64_t)handle,
      .sg_list    = &list,
      .num_sge    = 1,
   };
   struct ibv_recv_wr *bad_wr;
   int i;

   handle->cnt = 0;
   handle->num = 1;

   if (ibv_post_recv(ctx->qp[target], &wr, &bad_wr) != 0)
   {
      printf("could not post recv!\n");
      ABORT;
   }

   return 0;
}

int xl_ibv_post_send(struct xl_ibv_context *ctx, void *buf, size_t size, int target,
      struct ibv_handle *handle)
{
   struct ibv_sge list = {
      .addr   = (uintptr_t) buf,
      .length = size,
   };
   struct ibv_send_wr wr = {
      .next       = NULL,
      .wr_id      = (uint64_t)handle,
      .sg_list    = &list,
      .num_sge    = 1,
      .opcode     = IBV_WR_SEND,
      .send_flags = IBV_SEND_SIGNALED,
   };
   struct ibv_send_wr *bad_wr;

   handle->cnt = 0;
   handle->num = 1;

   if (ibv_post_send(ctx->qp[target], &wr, &bad_wr) != 0)
   {
      perror("Could not post send!");
      ABORT;
   }
}

void ibv_wait(struct xl_ibv_context *ctx, struct ibv_handle *handle)
{
   struct ibv_wc wc;
   int ne = 0;

   while (handle->cnt != handle->num)
   {
      do {
         ne = ibv_poll_cq(ctx->cq, 1, &wc);
         if (ne < 0) {
            fprintf(stderr, "poll CQ failed %d\n", ne);
            ABORT;
         }
      } while (ne < 1);

      if (wc.status != IBV_WC_SUCCESS)
      {
         printf("%s:%i -- comm error: %s\n", __FILE__, __LINE__, 
               ibv_wc_status_str(wc.status));
         ABORT;
      }

      ((struct ibv_handle*)wc.wr_id)->cnt += 1;
   } 
}

void ibv_allgather(struct xl_ibv_context *ctx, void *buf, size_t size, 
      uint32_t *mpi_start,
      struct ibv_handle *handle)
{
   // ensure to post recv before send! 
   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
      xl_ibv_post_recv(ctx, 
            (void*)((uint64_t)buf + (mpi_start[i]*size)),
            (mpi_start[i+1] - mpi_start[i])*size,
            i, handle);

   MPI_Barrier(MPI_COMM_WORLD);

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
      xl_ibv_post_send(ctx, 
            (void*)((uint64_t)buf + mpi_start[mpi_rank]*size),
            (mpi_start[mpi_rank+1] - mpi_start[mpi_rank])*size,
            i, handle);

   handle->cnt = 0;
   handle->num = 2 * (mpi_size - 1);

#ifdef XL_IBV_BLOCKING
   ibv_wait(ctx, handle);
#endif
}

void ibv_allgather_blocks(struct xl_ibv_context *ctx,
      struct ibv_send_wr **send_wr,
      struct ibv_recv_wr **recv_wr,
      struct ibv_handle *handle)
{
   handle->cnt = 0;
   handle->num = 0;

   /* ensure to post recv before send! */
   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
      if (recv_wr[i]->num_sge > 0)
      {
         struct ibv_recv_wr *tmp = recv_wr[i];

         while (tmp != NULL)
         {
            tmp->wr_id = (uint64_t)handle;
            handle->num++;

            tmp = tmp->next;
         }

         struct ibv_recv_wr *bad_wr;

         if (ibv_post_recv(ctx->qp[i], recv_wr[i], &bad_wr) != 0)
         {
            perror("could not post recv");
            ABORT;
         }
      }

   MPI_Barrier(MPI_COMM_WORLD);

   for (int i = (mpi_rank + 1) % mpi_size; i != mpi_rank; i = (i + 1) % mpi_size)
      if (send_wr[i]->num_sge > 0)
      {
         struct ibv_send_wr *tmp = send_wr[i];

         while (tmp != NULL)
         {
            tmp->wr_id = (uint64_t)handle;
            handle->num++;

            tmp = tmp->next;
         }

         struct ibv_send_wr *bad_wr;

         if (ibv_post_send(ctx->qp[i], send_wr[i], &bad_wr) != 0)
         {
            perror("could not post send");
            ABORT;
         }
      }

#ifdef XL_IBV_BLOCKING
   ibv_wait(ctx, handle);
#endif
}

