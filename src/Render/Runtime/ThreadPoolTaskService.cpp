#include "Render/Runtime/ThreadPoolTaskService.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace org::runtime {

namespace {

class PoolScope final : public ITaskScope {
public:
    void Cancel() noexcept override { m_cancelled.store(true, std::memory_order_release); }
    void Wait() override {
        std::unique_lock lock(m_mutex);
        m_idle.wait(lock, [&] { return m_outstanding == 0; });
    }
    void CancelAndWait() override { Cancel(); Wait(); }

    bool Cancelled() const noexcept { return m_cancelled.load(std::memory_order_acquire); }
    void Enter() { std::lock_guard lock(m_mutex); ++m_outstanding; }
    void Leave() {
        std::lock_guard lock(m_mutex);
        if (--m_outstanding == 0) m_idle.notify_all();
    }

private:
    std::atomic_bool m_cancelled{false};
    std::mutex m_mutex;
    std::condition_variable m_idle;
    size_t m_outstanding = 0;
};

struct Task {
    std::shared_ptr<PoolScope> scope;
    std::function<void()> work;
};

} // namespace

struct ThreadPoolTaskService::Impl {
    std::mutex mutex;
    std::condition_variable ready;
    std::array<std::deque<Task>, 3> queues; // indexed by TaskPriority
    struct Delayed {
        std::chrono::steady_clock::time_point due;
        TaskPriority priority;
        Task task;
    };
    std::vector<Delayed> delayed;
    std::vector<std::thread> threads;
    bool stopping = false;

    // Caller holds the lock. Moves due delayed tasks into their queues.
    void PromoteDue(std::chrono::steady_clock::time_point now) {
        for (auto it = delayed.begin(); it != delayed.end();) {
            if (it->due <= now) {
                queues[static_cast<size_t>(it->priority)].push_back(std::move(it->task));
                it = delayed.erase(it);
            } else {
                ++it;
            }
        }
    }

    void Run() {
        std::unique_lock lock(mutex);
        while (true) {
            PromoteDue(std::chrono::steady_clock::now());
            Task task;
            bool found = false;
            for (auto& queue : queues) {
                if (!queue.empty()) {
                    task = std::move(queue.front());
                    queue.pop_front();
                    found = true;
                    break;
                }
            }
            if (found) {
                lock.unlock();
                if (!task.scope || !task.scope->Cancelled()) {
                    try { task.work(); } catch (...) {}
                }
                if (task.scope) task.scope->Leave();
                lock.lock();
                continue;
            }
            if (stopping) return;
            if (delayed.empty()) {
                ready.wait(lock);
            } else {
                const auto next = std::min_element(delayed.begin(), delayed.end(),
                    [](const Delayed& a, const Delayed& b) { return a.due < b.due; })->due;
                ready.wait_until(lock, next);
            }
        }
    }

    // Removes the scope's queued (not yet started) tasks. Started ones keep running.
    void Retract(const std::shared_ptr<PoolScope>& scope) {
        size_t removed = 0;
        {
            std::lock_guard lock(mutex);
            for (auto& queue : queues) {
                for (auto it = queue.begin(); it != queue.end();) {
                    if (it->scope == scope) { it = queue.erase(it); ++removed; }
                    else ++it;
                }
            }
        }
        for (size_t i = 0; i < removed; ++i) scope->Leave();
    }

    bool Enqueue(const std::shared_ptr<ITaskScope>& scope, TaskPriority priority,
        std::chrono::steady_clock::duration delay, std::function<void()>&& func) {
        auto pool = std::dynamic_pointer_cast<PoolScope>(scope);
        if (scope && !pool) return false; // a scope from another service
        if (pool && pool->Cancelled()) return false;
        if (pool) pool->Enter();
        {
            std::lock_guard lock(mutex);
            if (stopping) {
                if (pool) pool->Leave();
                return false;
            }
            Task task{std::move(pool), std::move(func)};
            if (delay <= std::chrono::steady_clock::duration::zero())
                queues[static_cast<size_t>(priority)].push_back(std::move(task));
            else
                delayed.push_back({std::chrono::steady_clock::now() + delay, priority, std::move(task)});
        }
        ready.notify_one();
        return true;
    }
};

ThreadPoolTaskService::ThreadPoolTaskService(size_t workerCount) : m_impl(std::make_shared<Impl>()) {
    if (workerCount == 0) {
        const unsigned hardware = std::thread::hardware_concurrency();
        workerCount = hardware > 1 ? hardware - 1 : 1;
    }
    m_impl->threads.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i)
        m_impl->threads.emplace_back([impl = m_impl] { impl->Run(); });
}

ThreadPoolTaskService::~ThreadPoolTaskService() {
    {
        std::lock_guard lock(m_impl->mutex);
        m_impl->stopping = true;
        // Delayed work that has not come due is dropped; queued work drains.
        for (auto& entry : m_impl->delayed)
            if (entry.task.scope) entry.task.scope->Leave();
        m_impl->delayed.clear();
    }
    m_impl->ready.notify_all();
    for (auto& thread : m_impl->threads)
        if (thread.joinable()) thread.join();
}

size_t ThreadPoolTaskService::WorkerCount() const noexcept { return m_impl->threads.size(); }

void ThreadPoolTaskService::ParallelFor(std::string_view taskName, size_t itemCount, std::function<void(size_t)> func) {
    ParallelForLimited(taskName, itemCount, m_impl->threads.size() + 1, std::move(func));
}

void ThreadPoolTaskService::ParallelForLimited(std::string_view, size_t itemCount, size_t maximumConcurrency,
    std::function<void(size_t)> func) {
    if (itemCount == 0) return;
    const size_t helpers = (std::min)({maximumConcurrency > 0 ? maximumConcurrency - 1 : 0, itemCount - 1, m_impl->threads.size()});
    if (helpers == 0) {
        for (size_t i = 0; i < itemCount; ++i) func(i);
        return;
    }
    struct Shared {
        std::atomic_size_t next{0};
        std::mutex failureMutex;
        std::exception_ptr failure;
    };
    auto shared = std::make_shared<Shared>();
    auto body = [shared, &func, itemCount] {
        while (true) {
            const size_t index = shared->next.fetch_add(1, std::memory_order_relaxed);
            if (index >= itemCount) return;
            try { func(index); }
            catch (...) {
                std::lock_guard lock(shared->failureMutex);
                if (!shared->failure) shared->failure = std::current_exception();
            }
        }
    };
    // Helpers run under a private scope so this call can join exactly them.
    auto scope = std::make_shared<PoolScope>();
    for (size_t i = 0; i < helpers; ++i)
        m_impl->Enqueue(scope, TaskPriority::FrameCritical, {}, body);
    body(); // the caller participates until every item is claimed
    // Helpers still queued would find nothing to do; retract them rather than wait
    // for a worker (all workers may be blocked in nested ParallelFor calls).
    m_impl->Retract(scope);
    scope->Wait(); // func is borrowed by reference: started helpers must finish first
    if (shared->failure) std::rethrow_exception(shared->failure);
}

std::shared_ptr<ITaskScope> ThreadPoolTaskService::CreateScope(std::string_view) {
    return std::make_shared<PoolScope>();
}

bool ThreadPoolTaskService::Submit(const std::shared_ptr<ITaskScope>& scope, TaskPriority priority, std::string_view,
    std::function<void()>&& func) {
    return m_impl->Enqueue(scope, priority, {}, std::move(func));
}

bool ThreadPoolTaskService::ScheduleAfter(const std::shared_ptr<ITaskScope>& scope, std::chrono::steady_clock::duration delay,
    TaskPriority priority, std::string_view, std::function<void()>&& func) {
    return m_impl->Enqueue(scope, priority, delay, std::move(func));
}

} // namespace org::runtime
