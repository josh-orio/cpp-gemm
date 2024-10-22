#ifndef NEON_GEMM_HPP
#define NEON_GEMM_HPP

/* contains GEMMs that use NEON intrinsics */

#include <algorithm>
#include <arm_neon.h>

// NEON GEMM
void neon_gemm(float *a, float *b, float *c, int c_rows, int c_cols,
               int depth) {
  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, c_rows * c_cols * sizeof(float));

  float *t = new float[c_cols * depth];

  // TRANSPOSE B
  //   float tmp;
  for (int i = 0; i < depth; i++) {
    for (int ii = 0; ii < c_cols; ii++) {
      t[(ii * c_rows) + i] = b[(i * c_cols) + ii];
    }
  }

  long a_row, t_row;
  long a_loc, t_loc, c_loc;
  // float32x4_t n1, n2, acc = vdupq_n_f32(0.0f);
  float32x4_t n1, n2, acc;

  int diff;
  float32x4_t mask;

  for (int i = 0; i < c_rows * c_cols; i++) {
    a_row = i / c_cols;
    t_row = i % c_cols;
    // std::cout << a_row << " " << t_row << std::endl;

    for (int ii = 0; ii < depth; ii += 4) {
      a_loc = (depth * a_row) + ii;
      t_loc = (depth * t_row) + ii;

      // std::cout << a_loc << " " << t_loc << std::endl;

      n1 = vld1q_f32(&a[a_loc]);
      n2 = vld1q_f32(&t[t_loc]);
      acc = {0, 0, 0, 0};

      if (ii + 4 > depth) {
        // std::cout << "partial" << std::endl;
        diff = (ii + 4) - depth;
        // std::cout << "diff " << diff << std::endl;
        mask = {1, 1, 1, 1};

        for (; diff > 0; diff--) {
          mask[4 - diff] = 0;
        }

        n2 = vmulq_f32(n2, mask);
      }

      // std::cout << "n1 " << n1[0] << n1[1] << n1[2] << n1[3] << std::endl;
      // std::cout << "n2 " << n2[0] << n2[1] << n2[2] << n2[3] << std::endl;
      // std::cout << "mk " << mask[0] << mask[1] << mask[2] << mask[3] <<
      // std::endl;

      acc = vmlaq_f32(acc, n1, n2);

      // std::cout << i << std::endl;
      // std::cout << n1[0] << " " << n1[1] << " " << n1[2] << " " << n1[3] <<
      // std::endl; std::cout << n2[0] << " " << n2[1] << " " << n2[2] << " " <<
      // n2[3] << std::endl; std::cout << acc[0] << " " << acc[1] << " " <<
      // acc[2] << " " << acc[3] << std::endl; std::cout << acc[0] + acc[1] +
      // acc[2] + acc[3] << std::endl; std::cout << std::endl;

      // vst1q_f32(&c[i], acc);
      c[i] += vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
              vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);

      // c[i] = i;
    }
  }
}
////

#endif