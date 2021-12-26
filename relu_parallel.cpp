#include <cstdint>
#include <iostream>
#include <chrono>
#include "thread_loop.h"
#include <omp.h>


template<typename T>
void relu(int64_t begine, int64_t end, T *src, T *dst)
{
    const T zero_val = static_cast<T>(0);
    for(int64_t i = begine; i < end; i++)
    {
        if(src[i] > zero_val)
        {
            dst[i] = src[i];
        }
        else 
        {
             dst[i] = zero_val;
        }
           
    }
}


template<typename T>
void test_relu(T *src, size_t len, T *dst)
{
    parallel_loop(0, len, [=](int64_t begin, int64_t end){
        relu(begin, end, src, dst);
    }, 32768);
}


#define TEST_NUM        500
#define TEST_SIZE       4
#define TEST_LEN        (244*244*3)
#define TEST_TOTAL_LEN (TEST_NUM*TEST_LEN*TEST_SIZE)

int main(void)
{

    init_num_threads(omp_get_max_threads());
    double sum=0.0;
    void * src = malloc(TEST_TOTAL_LEN + 1024);
    void * dst = malloc(TEST_TOTAL_LEN);

    for(int i = 0; i < TEST_NUM; i ++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();   
        float * tmp_src = (float *)((unsigned char *)src + TEST_LEN*TEST_SIZE*i);
        float * tmp_dst = (float *)((unsigned char *)dst + TEST_LEN*TEST_SIZE*i);

        test_relu(tmp_src, TEST_LEN, tmp_dst);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        sum += time;
        // printf("t1 = %ld \n", t1.time_since_epoch().count());
        // printf("i=%d, relu time=%ld\n", i, time);
        printf("%ld\n", time);

        // test_copy(tmp_src,  TEST_LEN*TEST_SIZE, tmp_dst);
    }   

    printf("avg = %lf \n", sum/TEST_NUM);
}
