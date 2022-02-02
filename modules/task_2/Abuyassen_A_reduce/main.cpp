// Copyright 2021 Abuyassen Albara

#include <gtest/gtest.h>

#include <random>

#include <iostream>

#include <algorithm>

#include "../../../modules/task_2/Abuyassen_A_reduce/reduce.h"

#include <gtest-mpi-listener.hpp>

TEST(Parallel_Operations_MPI, Sum_Of_Intgeres) {
  std::mt19937 gen;
  gen.seed(static_cast < unsigned int > (time(0)));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  const int root = 0;
  const int size = 10;
  int * buffer = new int[size];
  int * first_receiver = new int[size];
  int * second_receiver = new int[size];

  for (int i = 0; i < size; i++)
    buffer[i] = gen() % 10;

  double start = MPI_Wtime();
  MyMpiReduce(buffer, first_receiver, size, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
  if (rank == root) {
    double end = MPI_Wtime();
    std::cout << "My MPI: " << end - start << " s\n";
  }
  double s_start = MPI_Wtime();
  MPI_Reduce(buffer, second_receiver, size, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
  if (rank == root) {
    double end = MPI_Wtime();
    std::cout << "MPI: " << end - s_start << " s\n";
    for (int i = 0; i < size; i++)
      ASSERT_EQ(first_receiver[i], second_receiver[i]);
  }
}

TEST(Parallel_Operations_MPI, Prod_Of_Floats) {
  std::mt19937 gen;
  gen.seed(static_cast < unsigned int > (time(0)));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  const int root = 0;
  const int size = 10;
  float * buffer = new float[size];
  float * first_receiver = new float[size];
  float * second_receiver = new float[size];

  for (int i = 0; i < size; i++)
    buffer[i] = (gen() % 10) * 0.366f;

  double start = MPI_Wtime();
  MyMpiReduce(buffer, first_receiver, size, MPI_FLOAT, MPI_PROD, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "My MPI: " << end - start << " s\n";
  }
  double s_start = MPI_Wtime();
  MPI_Reduce(buffer, second_receiver, size, MPI_FLOAT, MPI_PROD, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "MPI: " << end - s_start << " s\n";
    for (int i = 0; i < size; i++)
      ASSERT_EQ(first_receiver[i], second_receiver[i]);
  }
}

TEST(Parallel_Operations_MPI, Max_Of_Doubles) {
  std::mt19937 gen;
  gen.seed(static_cast < unsigned int > (time(0)));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  const int root = 0;
  const int size = 10;
  double * buffer = new double[size];
  double * first_receiver = new double[size];
  double * second_receiver = new double[size];

  for (int i = 0; i < size; i++)
    buffer[i] = (gen() % 10) * 0.366;

  double start = MPI_Wtime();
  MyMpiReduce(buffer, first_receiver, size, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "My MPI: " << end - start << " s\n";
  }
  double s_start = MPI_Wtime();
  MPI_Reduce(buffer, second_receiver, size, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "MPI: " << end - s_start << " s\n";
    for (int i = 0; i < size; i++)
      ASSERT_EQ(first_receiver[i], second_receiver[i]);
  }
}

TEST(Parallel_Operations_MPI, Min_Of_Intgeres) {
  std::mt19937 gen;
  gen.seed(static_cast < unsigned int > (time(0)));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  const int root = 0;
  const int size = 10;
  int * buffer = new int[size];
  int * first_receiver = new int[size];
  int * second_receiver = new int[size];

  for (int i = 0; i < size; i++)
    buffer[i] = gen() % 10;

  double start = MPI_Wtime();
  MyMpiReduce(buffer, first_receiver, size, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "My MPI: " << end - start << " s\n";
  }
  double s_start = MPI_Wtime();
  MPI_Reduce(buffer, second_receiver, size, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);
  if (rank == 0) {
    double end = MPI_Wtime();
    std::cout << "MPI: " << end - s_start << " s\n";
    for (int i = 0; i < size; i++)
      ASSERT_EQ(first_receiver[i], second_receiver[i]);
  }
}

TEST(Parallel_Operations_MPI, Get_Errors) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  int * buffer = nullptr;
  int * receiver = nullptr;
  const int count = -1;

  ASSERT_EQ(MyMpiReduce(buffer, receiver, count, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD), MPI_ERR_BUFFER);

  buffer = new int[1];
  receiver = new int[1];
  buffer[0] = 1;
  ASSERT_EQ(MyMpiReduce(buffer, receiver, count, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD), MPI_ERR_COUNT);

  ASSERT_EQ(MyMpiReduce(buffer, receiver, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD), MPI_ERR_OP);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(& argc, argv);
  MPI_Init(& argc, & argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners & listeners = ::testing::UnitTest::GetInstance() -> listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
