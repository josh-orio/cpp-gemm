cmake_minimum_required(VERSION 3.28.0)

project(gemm LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_CXX_FLAGS_RELEASE "-g -march=apple-latest -mtune=apple-latest")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=apple-latest -mtune=apple-latest")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

include_directories(gemm/)

add_executable(gemm main.cpp fortran/gemm.f90)

set_target_properties(gemm PROPERTIES LINKER_LANGUAGE Fortran)

# OpenBLAS detection & linking
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/usr/local/include/openblas/)
    target_link_libraries(gemm ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "OpenBLAS not found.")
endif()
