#pragma once
#include <sched.h>
#include <stdlib.h>
#include <time.h>

extern "C" void wan21_gemm_pool_run(int workers, int base_numa,
                                    void (*task)(int, int, void*), void* context);
extern "C" void wan21_gemm_pool_barrier(int workers);

namespace wan21_gemm_pool {
template <class Fn>
inline void run(int workers, int base_numa, Fn fn)
{
    auto invoke = [](int rank, int total, void* ctx) {
        (*static_cast<Fn*>(ctx))(rank, total);
    };
    wan21_gemm_pool_run(workers, base_numa, invoke, &fn);
}

inline void barrier(int workers) { wan21_gemm_pool_barrier(workers); }
inline int base_numa()
{
    const char* rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!rank || !*rank) rank = getenv("OMPI_COMM_WORLD_RANK");
    if (rank && *rank) return (int)(strtol(rank, nullptr, 10) % 16);
    const int cpu = sched_getcpu();
    return cpu < 0 ? 0 : cpu / 38;
}
inline double walltime()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
} // namespace wan21_gemm_pool
