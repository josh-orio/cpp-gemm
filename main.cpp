#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <iostream>
#include <random>

// BLAS C++ API
#include <cblas.h>
////

// C/C++ GEMM ROUTINES
#include "cxx.hpp"
////

// ARM NEON GEMM ROUTINES
#ifdef __APPLE__
#include "neon.hpp"
#endif
////

// FORTRAN GEMM ROUTINES
extern "C" {
void ft_gemm(float *a, float *b, float *c, int *c_rows, int *c_cols,
             int *depth);
void ft_ca_gemm(float *a, float *b, float *c, int *c_rows, int *c_cols,
                int *depth, int *block_size);
}
////

// RANDOM NUMBER GENERATOR
std::random_device rd;
std::default_random_engine re = std::default_random_engine(rd());
std::uniform_real_distribution<double> urd =
    std::uniform_real_distribution<double>(0, 1);
float rng() { return (int)(urd(re) * 100) % 10; }
////

// PRINT MATRIX (FOR DEBUGGING)
void print_matrix(float *a, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int ii = 0; ii < cols; ii++) {
      std::cout << std::format("{} ", a[(i * cols) + ii]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
////

int main() {
  // std::system("clear");
  // std::system("python3 ../np.py");

  int dim = 2048;
  int c_rows = dim, c_cols = dim, depth = dim;

  float *a_p = new float[depth * c_rows];
  float *b_p = new float[c_cols * depth];
  float *c_p = new float[c_cols * c_rows];

  // load a and b matrices with random values
  std::generate(a_p, a_p + (depth * c_rows), rng);
  std::generate(b_p, b_p + (c_cols * depth), rng);

  // std::cout << "MATRIX A" << std::endl;
  // print_matrix(a_p, c_rows, depth);

  // std::cout << "MATRIX B" << std::endl;
  // print_matrix(b_p, depth, c_cols);

  // timing counter
  double duration = 0;

  auto start = std::chrono::high_resolution_clock::now(),
       end = std::chrono::high_resolution_clock::now();

  long flops = (long)c_rows * (long)c_cols * (long)depth *
               2; /* flops used for one full matmul (1x mul, 1x add)*/
                  // std::cout << flops << std::endl;

  start = std::chrono::high_resolution_clock::now();
  gemm(a_p, depth, c_rows, b_p, c_cols, depth, c_p);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  // std::cout << std::format("{}", flops) << std::endl;
  std::cout << std::format("GFLOPS: {} (Unoptimized)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  start = std::chrono::high_resolution_clock::now();
  tp_gemm(a_p, depth, c_rows, b_p, c_cols, depth, c_p);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Transposed)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  start = std::chrono::high_resolution_clock::now();
  ca_gemm(a_p, b_p, c_p, c_rows, c_cols, depth, 32);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  start = std::chrono::high_resolution_clock::now();
  ca_gemm_c(a_p, b_p, c_p, c_rows, c_cols, depth, 128);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware (C Version))",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  start = std::chrono::high_resolution_clock::now();
  ca_gemm_c2(a_p, b_p, c_p, c_rows, c_cols, depth, 128);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware (C Version 2))",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

#ifdef NEON_GEMM_HPP

  start = std::chrono::high_resolution_clock::now();
  neon_gemm(a_p, b_p, c_p, c_rows, c_cols, depth);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (NEON)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

#endif

  start = std::chrono::high_resolution_clock::now();
  ft_gemm(a_p, b_p, c_p, &c_rows, &c_cols, &depth);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Fortran)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  int block_size = 128; // can only pass pointers to fortran - eye roll.
  start = std::chrono::high_resolution_clock::now();

  // memset(c_p, 0, c_rows * c_cols * sizeof(float));
  ft_ca_gemm(a_p, b_p, c_p, &c_rows, &c_cols, &depth, &block_size);
  // ^ this implementation is slightly wrong

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Fortran Blocked)",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  start = std::chrono::high_resolution_clock::now();

  //  float alpha = 1.0f; // Scaling factor for A*B
  //   float beta = 1.0f;  // Scaling factor for C

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1,
              &a_p[0], dim, &b_p[0], dim, 1, &c_p[0], dim);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (OpenBLAS)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  print_matrix(c_p, /*c_rows*/ 1, /*c_cols*/ 16);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);


  // #include <cblas.h>
  // int M = 2; // Number of rows in A and C
  //   int N = 2; // Number of columns in B and C
  //   int K = 2; // Number of columns in A and rows in B

  //   // Define matrices A, B, and C
  //   float A[2][2] = { {1.0f, 2.0f}, {3.0f, 4.0f} }; // 2x2 matrix
  //   float B[2][2] = { {2.0f, 3.0f}, {4.0f, 5.0f} }; // 2x2 matrix
  //   float C[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} }; // Result matrix
  //   initialized to zero

  //   // Set scalar values for the operation
  //   float alpha = 1.0f; // Scaling factor for A*B
  //   float beta = 1.0f;  // Scaling factor for C

  //   // Call sgemm: C := alpha * A * B + beta * C

  //   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
  //               M, N, K,
  //               alpha,
  //               &A[0][0], K,
  //               &B[0][0], N,
  //               beta,
  //               &C[0][0], N);

  //   // Print the result
  //   std::cout << "Result matrix A:\n";
  //   for (int i = 0; i < M; ++i) {
  //       for (int j = 0; j < N; ++j) {
  //           std::cout << A[i][j] << " ";
  //       }
  //       std::cout << "\n";
  //   }
  //   std::cout << "Result matrix B:\n";
  //   for (int i = 0; i < M; ++i) {
  //       for (int j = 0; j < N; ++j) {
  //           std::cout << B[i][j] << " ";
  //       }
  //       std::cout << "\n";
  //   }
  //   std::cout << "Result matrix C:\n";
  //   for (int i = 0; i < M; ++i) {
  //       for (int j = 0; j < N; ++j) {
  //           std::cout << C[i][j] << " ";
  //       }
  //       std::cout << "\n";
  //   }

  return 0;
}