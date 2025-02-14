// Copyright 2021 Pyatkin Nicolai
#ifndef MODULES_TASK_2_PYATCKIN_N_SCATTER_SCATTER_H_
#define MODULES_TASK_2_PYATCKIN_N_SCATTER_SCATTER_H_
#include <string>

int gen_int(int arr[], int len);
double gen_double(double arr[], int len);
float gen_float(float arr[], int len);
int MyScatter(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf
    , int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

#endif  // MODULES_TASK_2_PYATCKIN_N_SCATTER_SCATTER_H_
