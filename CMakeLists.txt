cmake_minimum_required(VERSION 3.28.0)
project(gemm LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate")


# set(CMAKE_CXX_FLAGS_RELEASE "-g -march=apple-latest -mtune=apple-latest")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=apple-latest -mtune=apple-latest")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

add_executable(gemm main.cpp gemm.f90)

set_target_properties(gemm PROPERTIES LINKER_LANGUAGE Fortran)

# find_package(BLAS REQUIRED)
# target_link_libraries(gemm PRIVATE BLAS::BLAS)

# target_include_directories(gemm PRIVATE blas/)
