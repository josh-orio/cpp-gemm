#ifndef C_GEMM_HPP
#define C_GEMM_HPP

#include <iostream>

/* contains various implementation of C/C++ GEMMs */

// NAIVE MATMUL
void gemm(float *a, long a_cols, long a_rows, float *b, long b_cols,
          long b_rows, float *c) {
  if (a_cols != b_rows) {
    std::cout << "Matrix dimensions do not match." << std::endl;
    return;
  }

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, a_rows * b_cols * sizeof(float));

  long c_cols = b_cols, c_rows = a_rows, depth = a_cols;

  long a_loc, b_loc, c_loc;

  for (int i = 0; i < c_rows; i++) {
    for (int ii = 0; ii < c_cols; ii++) {

      c_loc = (i * c_cols) + ii;

      for (int iii = 0; iii < depth; iii++) {
        a_loc = (i * depth) + iii;
        b_loc = (iii * c_cols) + ii;

        c[c_loc] += a[a_loc] * b[b_loc];

        // c[(i * c_cols) + ii] += a[(i * depth) + iii] * b[(iii * c_cols) +
        // ii];

        // c[(i * c_cols) + ii] += a[(i * depth) + iii];
        // c[(i * c_cols) + ii] += b[(iii * c_cols) + ii];
      }
    }
  }
}
////

// TRANSPOSED MATMUL
void tp_gemm(float *a, long a_cols, long a_rows, float *b, long b_cols,
             long b_rows, float *c) {
  if (a_cols != b_rows) {
    std::cout << "Matrix dimensions do not match." << std::endl;
    return;
  }

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, a_rows * b_cols * sizeof(float));

  float *t = new float[b_cols * b_rows];

  // TRANSPOSE B
  //   float tmp;
  for (int i = 0; i < b_rows; i++) {
    for (int ii = 0; ii < b_cols; ii++) {

      t[(ii * b_rows) + i] = b[(i * b_cols) + ii];
      // std::println("{0} -> {1}", (i * b_cols) + ii, (ii * b_rows) + i);

      // if (ii > i) {
      // tmp = b[(ii * b_rows) + i];
      // b[(ii * b_rows) + i] = b[(i * b_cols) + ii];
      // b[(i * b_cols) + ii] = tmp;
      // }
    }
  }

  //   std::println("MATRIX A");
  //   print_matrix(a, a_rows, a_cols);

  //   std::println("MATRIX B");
  //   print_matrix(b, b_rows, b_cols);

  //   std::println("MATRIX T");
  //   print_matrix(t, b_cols, b_rows);

  long c_cols = b_cols, c_rows = a_rows, depth = a_cols;

  long a_loc, t_loc, c_loc;

  for (int i = 0; i < c_rows; i++) {
    for (int ii = 0; ii < c_cols; ii++) {

      c_loc = (i * c_cols) + ii;

      for (int iii = 0; iii < depth; iii++) {
        a_loc = (i * depth) + iii;
        t_loc = (ii * depth) + iii;

        c[c_loc] += a[a_loc] * t[t_loc];

        // c[(i * c_cols) + ii] += a[(i * depth) + iii] * t[(ii * depth) + iii];

        // unopt implem.
        // c[(i * c_cols) + ii] += a[(i * depth) + iii] * b[(iii * c_cols) +
        // ii];

        // c[(i * c_rows) + ii] += t[(ii * depth) + iii]; // works
      }
      // std::println("");
    }
  }

  delete [] t;
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

#endif