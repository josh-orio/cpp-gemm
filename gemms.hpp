#ifndef C_GEMM_HPP
#define C_GEMM_HPP

#include <iostream>

/* contains various implementation of C/C++ GEMMs */

// UNOPTIMIZED MATMUL
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

  free(t);
}
////

#endif