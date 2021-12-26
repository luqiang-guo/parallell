clang++-13 relu_1.cpp -o relu_1 -O3

clang++-13 thread_loop.cpp relu_parallel.cpp -o relu_parallel -O3 \
    -fopenmp \


clang++-13 -g --std=c++11 thread_loop.cpp test_relu_parallel.cpp -o test_relu_parallel -O3 \
    -fopenmp -lnuma  \
