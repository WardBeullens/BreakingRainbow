#ifndef XL_COMM_H
#define XL_COMM_H

struct timeval start, end;

#ifdef ISEND
class XL_Handle
{
public:
    std::deque<MPI_Request*> req;
};
#else
typedef struct
{
} XL_Handle;
#endif

typedef struct
{
    bool end;

    void *sendbuf;
    int  sendcount;
    MPI_Datatype sendtype;
    
    void *recvbuf;
    int recvcount;
    MPI_Datatype recvtype;
    
    MPI_Comm comm;

    XL_Handle *handle;
} mpi_msg_t;


void XL_Wait(XL_Handle *handle)
{
#ifdef ISEND
    MPI_Status stat;

    while (handle->req.size() > 0)
    {
        MPI_Request *tmp;

        tmp = handle->req.front();
        handle->req.pop_front();

        MPI_Wait(tmp, &stat);

        delete tmp;
    }
#endif
}

void XL_Test(XL_Handle *handle)
{
#ifdef ISEND
    MPI_Status stat;

    int flag = 0;

    std::deque<MPI_Request*>::iterator it = handle->req.begin();

    while (it != handle->req.end())
       MPI_Test(*it++, &flag, &stat);
#endif
}

void XL_Iallgather(void *sendbuf, int  sendcount,
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        MPI_Datatype recvtype, MPI_Comm comm, XL_Handle *handle)
{
#ifdef BLOCKING
    MPI_Allgather
        (sendbuf, sendcount, sendtype,
         recvbuf, recvcount, recvtype,
         comm
        );
#endif
        
#ifdef ISEND
    for (unsigned j = 0; j < mpi_size; j++)
    {
        unsigned i = (j + mpi_rank) % mpi_size;
        if (i != mpi_rank)
        {
            MPI_Request *rreq = new MPI_Request;
            MPI_Request *sreq = new MPI_Request;

            MPI_Irecv((uint8_t*)recvbuf + (recvcount * i), recvcount, recvtype,
                    i, 0, comm, rreq);

            MPI_Isend((uint8_t*)sendbuf, sendcount, sendtype,
                    i, 0, comm, sreq);

            handle->req.push_back(rreq);
            handle->req.push_back(sreq);
        }
    }
#endif
}

#endif //define XL_COMM_H

