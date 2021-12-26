#include <cstdint>
#include <stdio.h>
#include "thread_loop.h"
#include "omp.h"

#define NUM_THREADS 8

void init_num_threads(size_t num) 
{
    printf("thread num = %ld\n", num);
    omp_set_num_threads(num);
}


inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

void parallel_loop(int64_t begin, int64_t end, std::function< void(int64_t, int64_t)> f, int64_t grain_size)
{
    // std::cout << "begin = " << begin << " end = " << end << " grain_size = " << grain_size << std::endl;

    #pragma omp parallel
    {
        int64_t num_threads = omp_get_num_threads();
        // std::cout << "omp_get_num_threads() = " << num_threads << std::endl;
        if (grain_size > 0) {
            num_threads = std::min(num_threads, divup((end - begin), grain_size));
        }
        // std::cout << "num_threads = " << num_threads << std::endl;
        int64_t chunk_size = divup((end - begin), num_threads);
        // std::cout << "chunk_size = " << chunk_size << std::endl;

        int64_t tid = omp_get_thread_num();
        // std::cout << "omp_get_thread_num() = " << tid << std::endl;
        
        int64_t begin_tid = begin + tid * chunk_size;
        int64_t end_tid = std::min(end, chunk_size + begin_tid);
        if (begin_tid < end) {
            f(begin_tid, end_tid);
        }
    }
}


void parallel_loop_test(int64_t begin, int64_t end, std::function< void(int64_t, int64_t)> f, int64_t grain_size)
{

    int64_t nthr = omp_get_max_threads();
    nthr = std::min(nthr, divup((end - begin), grain_size));
    int64_t chunk_size = divup((end - begin), nthr);
    // printf("nthr = %ld\n", nthr);

    #pragma omp parallel num_threads(nthr)
    {     
        int64_t tid = omp_get_thread_num();
        int64_t begin_tid = begin + tid * chunk_size;
        int64_t end_tid = std::min(end, chunk_size + begin_tid);

        f(begin_tid, end_tid);
    }
}

void parallel_dynamic_loop(int64_t begin, int64_t end, std::function< void(int64_t, int64_t)> f, int64_t grain_size)
{
    int task_num = divup((end - begin), grain_size);
    int64_t chunk_size = divup((end - begin), task_num);

    // int64_t nthr = task_num/4 + 1;
    // printf("nthr = %ld\n", nthr);
    #pragma omp parallel for schedule(dynamic) //num_threads(nthr)
    for(int64_t i = 0; i < task_num; i++){  
        int64_t begin_tid = begin + i * chunk_size;
        int64_t end_tid = std::min(end, chunk_size + begin_tid);
        // printf("i = %ld, thread num = %d \n", i, omp_get_thread_num());
        f(begin_tid, end_tid);
    }
}

void parallel_task_loop(int64_t begin, int64_t end, std::function< void(int64_t, int64_t)> f, int64_t grain_size)
{
    int64_t nthr = omp_get_max_threads();
    int task_num = divup((end - begin), grain_size);
    int64_t chunk_size = divup((end - begin), task_num);

    #pragma omp parallel
    #pragma omp single
    #pragma omp taskloop num_tasks(nthr)
    for (int64_t i=0; i<task_num; i++)
    {
        int64_t begin_tid = begin + i * chunk_size;
        int64_t end_tid = std::min(end, chunk_size + begin_tid);
        // printf("i = %ld, thread num = %d \n", i, omp_get_thread_num());
        f(begin_tid, end_tid);
    } 
}
