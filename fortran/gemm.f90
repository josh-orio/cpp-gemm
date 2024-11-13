module sum_module
    use iso_c_binding
    implicit none
  contains
    ! Fortran function to add two arrays with C binding
    subroutine ft_gemm(a, b, c, c_rows, c_cols, depth) bind(C, name="ft_gemm")
      implicit none
      ! input dimensions
      integer(c_int), intent(in) :: c_rows
      integer(c_int), intent(in) :: c_cols
      integer(c_int), intent(in) :: depth
      ! input arrays
      real(c_float), intent(in) :: a(c_rows * depth)
      real(c_float), intent(in) :: b(depth * c_cols)
      ! output array
      real(c_float), intent(out) :: c(c_rows * c_cols)
      real(c_float) :: sum
      ! caclculation loops
      integer(c_int) :: i, ii, iii
      do i = 1, c_rows
        do ii = 1, c_cols
          do iii = 1, depth
            c((i-1)*c_cols + ii) = c((i-1)*c_cols + ii) + a((i-1)*depth + iii) * b((iii-1)*c_cols + ii)
          end do
        end do
      end do
    end subroutine ft_gemm

    subroutine ft_ca_gemm(a, b, c, c_rows, c_cols, depth, block_size) bind(C, name="ft_ca_gemm")
      implicit none
      ! input dimensions
      integer(c_int), intent(in) :: c_rows
      integer(c_int), intent(in) :: c_cols
      integer(c_int), intent(in) :: depth
      integer(c_int), intent(in) :: block_size! input arrays
      real(c_float), intent(in) :: a(c_rows * depth)
      real(c_float), intent(in) :: b(depth * c_cols)
      ! output array
      real(c_float), intent(out) :: c(c_rows * c_cols)
      real(c_float) :: sum
      ! caclculation loops
      integer(c_int) :: ii, jj, kk, i, j, k

      do ii = 0, c_rows, block_size
        do jj = 0, c_cols, block_size
          do kk = 0, depth, block_size

            do i = ii, MIN(ii + block_size, c_rows)
              do j = jj, MIN(jj + block_size, c_cols)
                ! sum = c((i * c_rows) + j)
                do k = kk, MIN(kk + block_size, depth)
                  ! sum = sum + (a((i * c_rows) + k) * b((k * c_cols) + j))
                  c((i-1)*c_cols + j) = c((i-1)*c_cols + j) + a((i-1)*depth + k) * b((k-1)*c_cols + j)
          
                end do
                ! c((i * c_rows) + j) = sum
              end do
            end do

          end do
        end do
      end do
    end subroutine ft_ca_gemm

  !     for (int ii = 0; ii < c_rows; ii += block_size) {
  !   for (int jj = 0; jj < c_cols; jj += block_size) {
  !     for (int kk = 0; kk < depth; kk += block_size) {
  ! 
  !       for (int i = ii; i < ii + block_size && i < c_rows; i++) {
  !         for (int j = jj; j < jj + block_size && j < c_cols; j++) {
  !           double sum = c[(i * c_rows) + j]; // Initial value of C(i, j)
  !           for (int k = kk; k < kk + block_size && k < depth; k++) {
  !             sum += a[(i * c_rows) + k] * b[(k * c_cols) + j];
  !           }
  !           c[(i * c_rows) + j] = sum; // Update C(i, j)
  !         }
  !       }
  !     }
  !   }
  ! }
  end module sum_module