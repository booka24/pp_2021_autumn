// Copyright 2021 Abuyassen Albara

#ifndef MODULES_TASK_2_ABUYASSEN_A_REDUCE_REDUCE_H_
#define MODULES_TASK_2_ABUYASSEN_A_REDUCE_REDUCE_H_

#include <mpi.h>

int MyMpiReduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

#endif  // MODULES_TASK_2_ABUYASSEN_A_REDUCE_REDUCE_H_
