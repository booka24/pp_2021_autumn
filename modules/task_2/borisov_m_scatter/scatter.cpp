// Copyright [2021] <Borisov Maxim>
#include "../../../modules/task_2/borisov_m_scatter/scatter.h"

int my_scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
    int proc_count;
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    int send_type_size;
    int recv_type_size;

    if (sendcount != recvcount)
        return MPI_ERR_COUNT;
    if (sendcount <= 0 || recvcount <= 0)
        return MPI_ERR_COUNT;
    if (root < 0)
        return MPI_ERR_COUNT;
    if (MPI_Type_size(sendtype, &send_type_size) == MPI_ERR_TYPE)
        return MPI_ERR_TYPE;
    if (MPI_Type_size(recvtype, &recv_type_size) == MPI_ERR_TYPE)
        return MPI_ERR_TYPE;

    if (proc_rank == root) {
        memcpy(recvbuf, reinterpret_cast<char*>(sendbuf) +
                proc_rank * sendcount * send_type_size,
                sendcount * send_type_size);
        for (int i = 0; i < proc_count; i++) {
            MPI_Send(reinterpret_cast<char*>(sendbuf) +
                    i * sendcount * send_type_size,
                    sendcount, sendtype, i, 0, comm);
        }
    } else {
        MPI_Status status;
        MPI_Recv(recvbuf, recvcount, recvtype, root, 0 , comm, &status);
    }

    return MPI_SUCCESS;
}
