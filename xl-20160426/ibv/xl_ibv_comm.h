#ifdef __cplusplus
extern "C" {
#endif 

struct xl_ibv_context {
	struct ibv_device **dev_list;
	struct ibv_context *context;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_qp **qp;
   struct ibv_device_attr *dev_attr;

   int *port;
   int port_cnt;
};

struct xl_ibv_dest {
	int lid;
	int qpn;
	int psn;
};

struct ibv_handle {
   int cnt;
   int num;
};


struct xl_ibv_context *xl_ibv_init_ctx();
struct xl_ibv_context *xl_ibv_init_qp(struct xl_ibv_context *ctx,
      struct ibv_send_wr **send_wr, struct ibv_recv_wr **recv_wr);

int xl_ibv_close_ctx(struct xl_ibv_context *ctx);

int xl_ibv_post_recv(struct xl_ibv_context *ctx, void *buf, size_t size, int target,
      struct ibv_handle *handle);
int xl_ibv_post_send(struct xl_ibv_context *ctx, void *buf, size_t size, int target,
      struct ibv_handle *handle);

void ibv_wait(struct xl_ibv_context *ctx, struct ibv_handle *handle);

void ibv_allgather(struct xl_ibv_context *ctx, void *buf, size_t size, 
      uint32_t *mpi_start,
      struct ibv_handle *handle);

void ibv_allgather_blocks(struct xl_ibv_context *ctx,
      struct ibv_send_wr **send_wr,
      struct ibv_recv_wr **recv_wr,
      struct ibv_handle *handle);

#ifdef __cplusplus
}
#endif 

