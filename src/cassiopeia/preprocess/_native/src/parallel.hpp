// Minimal fork-join helpers built on std::thread.
#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace cas {

// Run fn(thread_index, begin, end) over [0, n) split into contiguous
// blocks, one per thread. Exceptions thrown by workers are rethrown.
template <typename Fn>
void parallel_blocks(size_t n, int n_threads, Fn &&fn) {
    if (n == 0) return;
    int t = std::max(1, std::min<int>(n_threads, static_cast<int>(n)));
    if (t == 1) {
        fn(0, size_t(0), n);
        return;
    }
    std::vector<std::thread> threads;
    std::exception_ptr error;
    std::mutex error_mutex;
    size_t step = (n + t - 1) / t;
    for (int i = 0; i < t; ++i) {
        size_t b = i * step, e = std::min(n, b + step);
        if (b >= e) break;
        threads.emplace_back([&, i, b, e]() {
            try {
                fn(i, b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        });
    }
    for (auto &th : threads) th.join();
    if (error) std::rethrow_exception(error);
}

// Run fn(thread_index, item) for every item in [0, n) with dynamic
// scheduling (good for items of uneven cost).
template <typename Fn>
void parallel_dynamic(size_t n, int n_threads, Fn &&fn, size_t grain = 1) {
    if (n == 0) return;
    int t = std::max(1, std::min<int>(n_threads, static_cast<int>((n + grain - 1) / grain)));
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&](int tid) {
        try {
            while (true) {
                size_t b = next.fetch_add(grain);
                if (b >= n) break;
                size_t e = std::min(n, b + grain);
                for (size_t i = b; i < e; ++i) fn(tid, i);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    if (t == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        for (int i = 0; i < t; ++i) threads.emplace_back(worker, i);
        for (auto &th : threads) th.join();
    }
    if (error) std::rethrow_exception(error);
}

}  // namespace cas
