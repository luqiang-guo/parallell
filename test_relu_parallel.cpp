#include <cstdint>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include <utmpx.h>
#include <random>

#include "numa.h"
#include "thread_loop.h"

template<typename T>
inline void relu(int64_t begine, int64_t end, T *src, T *dst)
{
    // int cpu = sched_getcpu();
    // int node = numa_node_of_cpu(cpu);
    // printf("node = %d, cpu = %d \n", node, cpu);
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
void test_relu_naive(T *src, size_t len, T *dst)
{
    // printf("relu naive \n");
    relu(0, len, src, dst);
}

template<typename T>
void test_relu_static_schedule(T *src, size_t len, T *dst)
{
    parallel_loop_test(0, len, [=](int64_t begin, int64_t end){
        relu(begin, end, src, dst);
    }, 32768);
}

template<typename T>
void test_relu_dynamic_schedule(T *src, size_t len, T *dst)
{
    parallel_dynamic_loop(0, len, [=](int64_t begin, int64_t end){
        relu(begin, end, src, dst);
    }, 8192);
}

template<typename T>
void test_relu_task_loop(T *src, size_t len, T *dst)
{
    parallel_task_loop(0, len, [=](int64_t begin, int64_t end){
        relu(begin, end, src, dst);
    }, 8192);
}

void worker_loop(int i)
{
    while(1)
    {
        sleep(1);
    }
}

void bazy_loop(void* arg)
{
    while(1)
    {

    }
}

#define NUM_TID 400

void test_thread()
{
    std::vector<std::thread*> tid(NUM_TID);
    for(int64_t i=0; i < NUM_TID -2; i++)
    {
        tid[i] = new std::thread(worker_loop, 2);
    }

    tid[NUM_TID-2] = new std::thread(bazy_loop, nullptr);
    // tid[NUM_TID-1] = new std::thread(bazy_loop, nullptr);

}

#define TEST_NUM        100
#define TEST_SIZE       4
#define TEST_LEN        (244*244*40)
#define TEST_TOTAL_LEN (TEST_NUM*TEST_LEN*TEST_SIZE)


template<typename func>
void  timing(func& f)
{
    double sum=0.0;
    void * src = malloc(TEST_TOTAL_LEN + 1024);
    void * dst = malloc(TEST_TOTAL_LEN);
    std::mt19937 rng;

    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> distribution(-1, 1);
    // std::cout << distribution(rng) << std::endl;

    for(int64_t i; i < TEST_NUM*TEST_LEN; i++)
    {
        ((float*)src)[i] = distribution(rng);
    }

    for(int i = 0; i < TEST_NUM; i ++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();   
        float * tmp_src = (float *)((unsigned char *)src + TEST_LEN*TEST_SIZE*i);
        float * tmp_dst = (float *)((unsigned char *)dst + TEST_LEN*TEST_SIZE*i);

        f(tmp_src, TEST_LEN, tmp_dst);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        sum += time;
        // printf("t1 = %ld \n", t1.time_since_epoch().count());
        // printf("i=%d, relu time=%ld\n", i, time);
        // printf("%ld\n", time);

        // test_copy(tmp_src,  TEST_LEN*TEST_SIZE, tmp_dst);
    }   

    printf("avg = %lf \n", sum/TEST_NUM);
}

int main(void)
{

    // init_num_threads(omp_get_max_threads()-4);
    init_num_threads(64);
    test_thread();

    timing(test_relu_naive<float>);
    timing(test_relu_static_schedule<float>);
    timing(test_relu_dynamic_schedule<float>);
    timing(test_relu_task_loop<float>);
}
