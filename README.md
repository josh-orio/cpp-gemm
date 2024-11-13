# Fast Matrix Multiplication (Not that fast)

Went down a geohot rabbit hole, now I'm trying to make fast matmuls

:0

How did he do it in a 6hr livestream?

## Installing OpenBLAS and Linking

```
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
mkdir build
cd build
cmake ..
sudo make install
```

Should install to /usr/local/include/openblas

```
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/usr/local/include/openblas/)
    target_link_libraries(executable_name ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "OpenBLAS not found.")
endif()
```

## Other Requirements

- CMake
- G++/Clang++
- Gfortran