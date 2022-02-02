// Copyright 2021 Abuyassen Albara

#include "../../../modules/task_2/Abuyassen_A_reduce/reduce.h"

#include <mpi.h>

#include <algorithm>

#include <cstring>

int MyMpiReduce(void * source, void * dist, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  if (source == NULL || dist == NULL)
    return MPI_ERR_BUFFER;
  if (count <= 0)
    return MPI_ERR_COUNT;
  if (op != MPI_MAX && op != MPI_MIN && op != MPI_SUM && op != MPI_PROD)
    return MPI_ERR_OP;

  int size, rank;
  MPI_Comm_size(comm, & size);
  MPI_Comm_rank(comm, & rank);

  if (rank == root) {
    void * recived;
    if (datatype == MPI_INT) {
      recived = new int[count];
      std::memcpy(dist, source, count * sizeof(int));
    } else if (datatype == MPI_FLOAT) {
      recived = new float[count];
      std::memcpy(dist, source, count * sizeof(float));
    } else if (datatype == MPI_DOUBLE) {
      recived = new double[count];
      std::memcpy(dist, source, count * sizeof(double));
    } else {
      return MPI_ERR_TYPE;
    }

    for (int proc = 0; proc < size; proc++) {
      if (proc != root) {
        MPI_Status status;
        MPI_Recv(recived, count, datatype, proc, 0, comm, & status);
        if (datatype == MPI_INT) {
          if (op == MPI_SUM) {
            for (int i = 0; i < count; i++) {
              static_cast < int * > (dist)[i] += static_cast < int * > (recived)[i];
            }
          } else if (op == MPI_PROD) {
            for (int i = 0; i < count; i++) {
              static_cast < int * > (dist)[i] *= static_cast < int * > (recived)[i];
            }
          } else if (op == MPI_MAX) {
            for (int i = 0; i < count; i++)
              if (static_cast < int * > (dist)[i] < static_cast < int * > (recived)[i])
                static_cast < int * > (dist)[i] = static_cast < int * > (recived)[i];
          } else if (op == MPI_MIN) {
            for (int i = 0; i < count; i++)
              if (static_cast < int * > (dist)[i] > static_cast < int * > (recived)[i])
                static_cast < int * > (dist)[i] = static_cast < int * > (recived)[i];
          }
        } else if (datatype == MPI_FLOAT) {
          if (op == MPI_MAX) {
            for (int i = 0; i < count; i++)
              if (static_cast < float * > (dist)[i] < static_cast < float * > (recived)[i])
                static_cast < float * > (dist)[i] = static_cast < float * > (recived)[i];
          } else if (op == MPI_MIN) {
            for (int i = 0; i < count; i++)
              if (static_cast < float * > (dist)[i] > static_cast < float * > (recived)[i])
                static_cast < float * > (dist)[i] = static_cast < float * > (recived)[i];
          } else if (op == MPI_SUM) {
            for (int i = 0; i < count; i++)
              static_cast < float * > (dist)[i] += static_cast < float * > (recived)[i];
          } else if (op == MPI_PROD) {
            for (int i = 0; i < count; i++)
              static_cast < float * > (dist)[i] *= static_cast < float * > (recived)[i];
          }
        } else if (datatype == MPI_DOUBLE) {
          if (op == MPI_MAX) {
            for (int i = 0; i < count; i++)
              if (static_cast < double * > (dist)[i] < static_cast < double * > (recived)[i])
                static_cast < double * > (dist)[i] = static_cast < double * > (recived)[i];
          } else if (op == MPI_MIN) {
            for (int i = 0; i < count; i++)
              if (static_cast < double * > (dist)[i] > static_cast < double * > (recived)[i])
                static_cast < double * > (dist)[i] = static_cast < double * > (recived)[i];
          } else if (op == MPI_SUM) {
            for (int i = 0; i < count; i++)
              static_cast < double * > (dist)[i] += static_cast < double * > (recived)[i];
          } else if (op == MPI_PROD) {
            for (int i = 0; i < count; i++)
              static_cast < double * > (dist)[i] *= static_cast < double * > (recived)[i];
          }
        }
      }
    }
  } else {
    MPI_Send(source, count, datatype, root, 0, comm);
  }
  return MPI_SUCCESS;
}
