#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <iostream>
#include <random>

// C/C++ GEMM ROUTINES
#include "gemms.hpp"
////

// BLAS C++ API
// #include "blas.hh"
#ifdef __APPLE__
#include <vecLib/vecLib.h> /* for some reason this file has sgemm_() ?? */
#endif
////

// ARM NEON INSTRUCTIONS & GEMM ROUTINE
#ifdef __APPLE__
#include "n_gemm.hpp"
#include <arm_neon.h>
#endif
////

extern "C" {
void ft_gemm(float *a, float *b, float *c, int *c_rows, int *c_cols,
             int *depth);
void ft_ca_gemm(float *a, float *b, float *c, int *c_rows, int *c_cols,
                int *depth, int *block_size);
}

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

// CACHE AWARE GEMM
void ca_gemm(float *a, float *b, float *c, uint c_rows, uint c_cols, uint depth,
             uint block_size) {
  // int n = depth;

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, c_rows * c_cols * sizeof(float));

  // Iterate over the blocks of the matrices
  for (int i = 0; i < c_rows; i += block_size) {
    for (int ii = 0; ii < c_cols; ii += block_size) {
      for (int iii = 0; iii < depth; iii += block_size) {

        // Multiply the blocks
        for (int iv = i; iv < std::min(i + block_size, c_rows); ++iv) {
          for (int v = ii; v < std::min(ii + block_size, c_cols); ++v) {
            double sum = 0.0;
            for (int vi = iii; vi < std::min(iii + block_size, depth); ++vi) {
              sum += a[(iv * depth) + vi] * b[(vi * c_cols) + v];
            }
            c[(iv * c_cols) + v] += sum;

            // unopt implem.
            // c[(i * c_cols) + ii] += a[(i * depth) + iii] * b[(iii * c_cols) +
            // ii];
          }
        }
      }
    }
  }

  // print_matrix(c, c_rows, c_cols);
}
////

// CACHE AWARE GEMM
void ca_gemm_c(float *a, float *b, float *c, uint c_rows, uint c_cols,
               uint depth, uint block_size) {
  // int n = depth;

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, c_rows * c_cols * sizeof(float));

  for (int ii = 0; ii < c_rows; ii += block_size) {
    for (int jj = 0; jj < c_cols; jj += block_size) {
      for (int kk = 0; kk < depth; kk += block_size) {
        // Multiply block A[ii:ii+BLOCK_SIZE, kk:kk+BLOCK_SIZE]
        //          by block B[kk:kk+BLOCK_SIZE, jj:jj+BLOCK_SIZE]
        //          and accumulate into block C[ii:ii+BLOCK_SIZE,
        //          jj:jj+BLOCK_SIZE]
        for (int i = ii; i < ii + block_size && i < c_rows; i++) {
          for (int j = jj; j < jj + block_size && j < c_cols; j++) {
            double sum = c[(i * c_rows) + j]; // Initial value of C(i, j)
            for (int k = kk; k < kk + block_size && k < depth; k++) {
              sum += a[(i * c_rows) + k] * b[(k * c_cols) + j];
            }
            c[(i * c_rows) + j] = sum; // Update C(i, j)
          }
        }
      }
    }
  }
}
////

// CACHE AWARE GEMM
void ca_gemm_c2(float *a, float *b, float *c, uint c_rows, uint c_cols,
                uint depth, uint block_size) {
  // int n = depth;

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, c_rows * c_cols * sizeof(float));

  for (int ii = 0; ii < c_rows; ii += block_size) {
    for (int jj = 0; jj < c_cols; jj += block_size) {
      for (int kk = 0; kk < depth; kk += block_size) {
        // Multiply block A[ii:ii+BLOCK_SIZE, kk:kk+BLOCK_SIZE]
        //          by block B[kk:kk+BLOCK_SIZE, jj:jj+BLOCK_SIZE]
        //          and accumulate into block C[ii:ii+BLOCK_SIZE,
        //          jj:jj+BLOCK_SIZE]
        for (int i = ii; i < std::min(ii + block_size, c_rows); i++) {
          for (int j = jj; j < std::min(jj + block_size, c_cols); j++) {
            double sum = c[(i * c_rows) + j]; // Initial value of C(i, j)
            for (int k = kk; k < std::min(kk + block_size, depth); k++) {
              sum += a[(i * c_rows) + k] * b[(k * c_cols) + j];
            }
            c[(i * c_rows) + j] = sum; // Update C(i, j)
          }
        }
      }
    }
  }
}
////

// CACHE AWARE GEMM
void ca_gemm_c3(float *a, float *b, float *c, uint c_rows, uint c_cols,
                uint depth, uint block_size) {
  // int n = depth;

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, c_rows * c_cols * sizeof(float));

  for (int ii = 0; ii < c_rows; ii += block_size) {
    for (int jj = 0; jj < c_cols; jj += block_size) {
      for (int kk = 0; kk < depth; kk += block_size) {
        // Multiply block A[ii:ii+BLOCK_SIZE, kk:kk+BLOCK_SIZE]
        //          by block B[kk:kk+BLOCK_SIZE, jj:jj+BLOCK_SIZE]
        //          and accumulate into block C[ii:ii+BLOCK_SIZE,
        //          jj:jj+BLOCK_SIZE]
        for (int i = ii; i < std::min(ii + block_size, c_rows); i++) {
          for (int j = jj; j < std::min(jj + block_size, c_cols); j++) {
            double sum = c[(i * c_rows) + j]; // Initial value of C(i, j)
            for (int k = kk; k < std::min(kk + block_size, depth); k++) {
              sum += a[(i * c_rows) + k] * b[(k * c_cols) + j];
            }
            c[(i * c_rows) + j] = sum; // Update C(i, j)
          }
        }
      }
    }
  }
}
////

int main() {
  std::system("clear");
  std::system("python3 ../np.py");

  int dim = 512;
  // dim = 4;

  int c_rows = dim, c_cols = dim, depth = dim;

  float *a_p = new float[depth * c_rows];
  float *b_p = new float[c_cols * depth];
  float *c_p = new float[c_cols * c_rows];

  long flops = c_rows * c_cols * depth *
               2; /* flops used for one full matmul (1x mul, 1x add)*/

  // timing counter
  double duration = 0;

  auto start = std::chrono::high_resolution_clock::now(),
       end = std::chrono::high_resolution_clock::now();

  // load a and b with random values
  std::generate(a_p, a_p + (depth * c_rows), rng);
  std::generate(b_p, b_p + (c_cols * depth), rng);

  // std::cout << "MATRIX A" << std::endl;
  // print_matrix(a_p, c_rows, depth);

  // std::cout << "MATRIX B" << std::endl;
  // print_matrix(b_p, depth, c_cols);

  start = std::chrono::high_resolution_clock::now();
  gemm(a_p, depth, c_rows, b_p, c_cols, depth, c_p);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  // std::cout << std::format("{}", flops) << std::endl;
  std::cout << std::format("GFLOPS: {} (Unoptimized)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, c_rows, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  tp_gemm(a_p, depth, c_rows, b_p, c_cols, depth, c_p);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Transposed)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, /*c_rows*/ 1, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  ca_gemm(a_p, b_p, c_p, c_rows, c_cols, depth, 32);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, /*c_rows*/ 1, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  ca_gemm_c(a_p, b_p, c_p, c_rows, c_cols, depth, 128);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware (C Version))",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, /*c_rows*/ 1, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  ca_gemm_c2(a_p, b_p, c_p, c_rows, c_cols, depth, 128);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Cache Aware (C Version 2))",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, /*c_rows*/ 1, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  neon_gemm(a_p, b_p, c_p, c_rows, c_cols, depth);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (NEON)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, /*c_rows*/ 1, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  start = std::chrono::high_resolution_clock::now();
  ft_gemm(a_p, b_p, c_p, &c_rows, &c_cols, &depth);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Fortran)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, c_rows, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  int block_size = 128;

  start = std::chrono::high_resolution_clock::now();
  memset(c_p, 0, c_rows * c_cols * sizeof(double));
  ft_ca_gemm(a_p, b_p, c_p, &c_rows, &c_cols, &depth, &block_size);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (Fortran Blocked)",
                           (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(c_p, c_rows, c_cols);
  std::fill(c_p, c_p + (c_rows * c_cols), 0);

  // float alpha = 1, beta = 0;
  // char N = 'N', T = 'T';

  // float *ta = new float[depth * c_rows];
  // float *tb = new float[c_cols * depth];

  // // TRANSPOSE A
  // for (int i = 0; i < c_rows; i++) {
  //   for (int ii = 0; ii < depth; ii++) {
  //     ta[(ii * c_rows) + i] = a_p[(i * depth) + ii];
  //   }
  // }

  // // TRANSPOSE B
  // for (int i = 0; i < depth; i++) {
  //   for (int ii = 0; ii < c_cols; ii++) {
  //     tb[(ii * depth) + i] = b_p[(i * c_cols) + ii];
  //   }
  // }

  // start = std::chrono::high_resolution_clock::now();
  // // sgemm_("N", "T", &c_rows, &c_cols, &depth, &alpha, a_p, &depth, b_p,
  // // &c_cols,
  // //        &beta, c_p, &c_cols);

  // // sgemm_("N", "T", &c_rows, &c_cols, &depth, &alpha, a_p, &depth, b_p,
  // // &c_cols,
  // //        &beta, c_p, &c_cols);

  // /* THIS BLAS ROUTINE DOESNT MULTIPLY PROPERLY */
  // /* REFER TO NUMPY GFLOPS INSTEAD */

  // sgemm_(&N, &T, &c_rows, &c_cols, &depth, &alpha, a_p, &c_rows, b_p, &depth,
  //        &beta, c_p, &c_rows);
  // end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration<double>(end - start).count();

  // float *tc = new float[c_rows * c_cols];

  // // TRANSPOSE A
  // for (int i = 0; i < c_rows; i++) {
  //   for (int ii = 0; ii < depth; ii++) {
  //     tc[(ii * c_rows) + i] = c_p[(i * depth) + ii];
  //   }
  // }

  // // sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha,
  // // float *a, int *lda, float *b, int *ldb, float *beta, float *c__, int
  // *ldc)

  // std::cout << std::format("GFLOPS: {} (BLAS)", (flops / 1e9) / duration)
  //           << std::endl;
  // // std::cout << "MATRIX C" << std::endl;
  // // print_matrix(tc, c_rows, c_cols);
  // std::fill(c_p, c_p + (c_rows * c_cols), 0);

  return 0;
}