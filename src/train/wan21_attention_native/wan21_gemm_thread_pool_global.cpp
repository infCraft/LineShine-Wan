// Lock-free task dispatch for persistent NUMA-local GEMM workers.
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "wan21_gemm_thread_pool.h"

namespace {
class Wan21GemmPool {
    static constexpr int kMaxWorkers = 36;
    using Task = void (*)(int, int, void*);
    struct alignas(64) Slot {
        unsigned epoch = 0;
        Task task = nullptr;
        void* context = nullptr;
        int total = 0;
        int base_numa = 0;
    };
    struct WorkerInfo { Wan21GemmPool* pool; int rank; };
    pthread_mutex_t run_lock_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_t threads_[kMaxWorkers - 1] = {};
    WorkerInfo info_[kMaxWorkers - 1] = {};
    Slot slots_[kMaxWorkers - 1] = {};
    bool initialized_ = false;
    unsigned wake_epoch_ = 0;
    unsigned global_epoch_ = 0;
    Task global_task_ = nullptr;
    void* global_context_ = nullptr;
    int global_base_numa_ = 0;
    int sleepers_ = 0;
    int waiting_ = 0;
    int stopping_ = 0;
    int remaining_ = 0;
    unsigned barrier_epoch_ = 0;
    int barrier_arrivals_ = 0;

    static void check(int rc) {
        if (rc) { fprintf(stderr, "wan21 GEMM pool: pthread error %d\n", rc); abort(); }
    }
    static void bind_rank(int base_numa, int rank) {
        static thread_local int last_base = -1;
        if (last_base == base_numa) return;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(base_numa * 38 + rank, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask)) {
            perror("wan21 GEMM pool sched_setaffinity"); abort();
        }
        last_base = base_numa;
    }
    static void* entry(void* arg) {
        auto* info = static_cast<WorkerInfo*>(arg);
        info->pool->worker_main(info->rank);
        return nullptr;
    }
    void worker_main(int rank) {
        Slot& slot = slots_[rank - 1];
        unsigned seen_slot = 0, seen_global = 0;
        for (;;) {
            if (__atomic_load_n(&stopping_, __ATOMIC_ACQUIRE)) return;
            const unsigned global_now = __atomic_load_n(&global_epoch_, __ATOMIC_ACQUIRE);
            const unsigned slot_now = __atomic_load_n(&slot.epoch, __ATOMIC_ACQUIRE);
            if (global_now == seen_global && slot_now == seen_slot) {
                const double until = wan21_gemm_pool::walltime() + 0.0002;
                int spin = 0;
                while (__atomic_load_n(&global_epoch_, __ATOMIC_ACQUIRE) == seen_global &&
                       __atomic_load_n(&slot.epoch, __ATOMIC_ACQUIRE) == seen_slot &&
                       !__atomic_load_n(&stopping_, __ATOMIC_RELAXED)) {
                    if ((++spin & 63) == 0 && wan21_gemm_pool::walltime() >= until) break;
                    asm volatile("yield" ::: "memory");
                }
                if (__atomic_load_n(&global_epoch_, __ATOMIC_ACQUIRE) != seen_global ||
                    __atomic_load_n(&slot.epoch, __ATOMIC_ACQUIRE) != seen_slot) continue;
                const unsigned wake_seen = __atomic_load_n(&wake_epoch_, __ATOMIC_ACQUIRE);
                __atomic_add_fetch(&sleepers_, 1, __ATOMIC_ACQ_REL);
                if (__atomic_load_n(&global_epoch_, __ATOMIC_ACQUIRE) == seen_global &&
                    __atomic_load_n(&slot.epoch, __ATOMIC_ACQUIRE) == seen_slot &&
                    !__atomic_load_n(&stopping_, __ATOMIC_ACQUIRE) &&
                    __atomic_load_n(&wake_epoch_, __ATOMIC_ACQUIRE) == wake_seen)
                    syscall(SYS_futex, &wake_epoch_, FUTEX_WAIT_PRIVATE, wake_seen,
                            nullptr, nullptr, 0);
                __atomic_sub_fetch(&sleepers_, 1, __ATOMIC_ACQ_REL);
                continue;
            }
            if (global_now != seen_global) {
                seen_global = global_now;
                const Task task = global_task_;
                void* const context = global_context_;
                bind_rank(global_base_numa_, rank);
                task(rank, 36, context);
            } else {
                seen_slot = slot_now;
                const Task task = slot.task;
                void* const context = slot.context;
                bind_rank(slot.base_numa, rank);
                task(rank, slot.total, context);
            }
            if (__atomic_sub_fetch(&remaining_, 1, __ATOMIC_ACQ_REL) == 0 &&
                __atomic_load_n(&waiting_, __ATOMIC_ACQUIRE))
                syscall(SYS_futex, &remaining_, FUTEX_WAKE_PRIVATE, 1,
                        nullptr, nullptr, 0);
        }
    }
    void wake_sleepers() {
        __atomic_add_fetch(&wake_epoch_, 1, __ATOMIC_RELEASE);
        if (__atomic_load_n(&sleepers_, __ATOMIC_ACQUIRE))
            syscall(SYS_futex, &wake_epoch_, FUTEX_WAKE_PRIVATE, INT_MAX,
                    nullptr, nullptr, 0);
    }
public:
    ~Wan21GemmPool() {
        if (!initialized_) return;
        __atomic_store_n(&stopping_, 1, __ATOMIC_RELEASE);
        wake_sleepers();
        for (auto& thread : threads_) check(pthread_join(thread, nullptr));
        check(pthread_mutex_destroy(&run_lock_));
    }
    void run(int n, int numa, Task task, void* context) {
        if (n <= 0 || n > kMaxWorkers || numa < 0 || numa >= 16 || !task) {
            fputs("wan21 GEMM pool: invalid workers/NUMA/task\n", stderr); abort();
        }
        check(pthread_mutex_lock(&run_lock_));
        if (n == 1) {
            bind_rank(numa, 0);
            task(0, 1, context);
            check(pthread_mutex_unlock(&run_lock_));
            return;
        }
        if (!initialized_) {
            initialized_ = true;
            for (int i = 1; i < kMaxWorkers; ++i) {
                info_[i - 1] = {this, i};
                check(pthread_create(&threads_[i - 1], nullptr, entry, &info_[i - 1]));
            }
        }
        __atomic_store_n(&remaining_, n - 1, __ATOMIC_RELEASE);
        if (n == 36) {
            global_task_ = task;
            global_context_ = context;
            global_base_numa_ = numa;
            __atomic_add_fetch(&global_epoch_, 1, __ATOMIC_RELEASE);
        } else {
            for (int rank = 1; rank < n; ++rank) {
                Slot& slot = slots_[rank - 1];
                slot.task = task;
                slot.context = context;
                slot.total = n;
                slot.base_numa = numa;
                __atomic_add_fetch(&slot.epoch, 1, __ATOMIC_RELEASE);
            }
        }
        wake_sleepers();
        bind_rank(numa, 0);
        task(0, n, context);
        int left = __atomic_load_n(&remaining_, __ATOMIC_ACQUIRE);
        for (int spin = 0; left && spin < 2048; ++spin) {
            asm volatile("yield" ::: "memory");
            left = __atomic_load_n(&remaining_, __ATOMIC_ACQUIRE);
        }
        if (left) {
            __atomic_store_n(&waiting_, 1, __ATOMIC_RELEASE);
            while ((left = __atomic_load_n(&remaining_, __ATOMIC_ACQUIRE)) != 0)
                syscall(SYS_futex, &remaining_, FUTEX_WAIT_PRIVATE, left,
                        nullptr, nullptr, 0);
            __atomic_store_n(&waiting_, 0, __ATOMIC_RELEASE);
        }
        check(pthread_mutex_unlock(&run_lock_));
    }
    void barrier(int n) {
        if (n == 1) return;
        const unsigned seen = __atomic_load_n(&barrier_epoch_, __ATOMIC_ACQUIRE);
        if (__atomic_add_fetch(&barrier_arrivals_, 1, __ATOMIC_ACQ_REL) == n) {
            __atomic_store_n(&barrier_arrivals_, 0, __ATOMIC_RELAXED);
            __atomic_add_fetch(&barrier_epoch_, 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&barrier_epoch_, __ATOMIC_ACQUIRE) == seen)
                asm volatile("yield" ::: "memory");
        }
    }
};
Wan21GemmPool& pool() { static Wan21GemmPool singleton; return singleton; }
} // namespace
extern "C" void wan21_gemm_pool_run(int n, int base_numa,
                                    void (*task)(int, int, void*), void* context) {
    pool().run(n, base_numa, task, context);
}
extern "C" void wan21_gemm_pool_barrier(int n) { pool().barrier(n); }
