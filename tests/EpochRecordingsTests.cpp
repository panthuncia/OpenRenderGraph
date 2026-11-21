// Epoch recording outcome rules (ResultOf) and the recording cache, without a device.
//
// The cache posts through a fake requester that keeps each request's done. The tests complete them by hand, the way
// the host's thread would.

#include "OpenRenderGraph/EpochRecordings.h"
#include "TestSupport.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace er = org::epoch_recordings;
using Recording = std::shared_ptr<const org::GraphHost::EpochRecording>;

struct Key {
    int id = 0;
    int variant = 0;
    friend bool operator==(const Key&, const Key&) = default;
};
// Hashes the id only, so variants of an id share a hash.
struct KeyHash {
    std::uint64_t operator()(const Key& a_key) const { return static_cast<std::uint64_t>(a_key.id) * 0x9E3779B97F4A7C15ull; }
};

Recording MakeRecording(std::uint64_t a_build) {
    auto recording = std::make_shared<org::GraphHost::EpochRecording>();
    recording->buildGeneration = a_build;
    return recording;
}

er::Result Ready(std::uint64_t a_build) {
    er::Result result;
    result.outcome = er::Outcome::Ready;
    result.recording = MakeRecording(a_build);
    return result;
}

er::Result NotReady(er::Outcome a_outcome, std::string a_error) {
    er::Result result;
    result.outcome = a_outcome;
    result.error = std::move(a_error);
    return result;
}

// Keeps every posted request's done. Refuses from request refuseFrom on.
struct FakeHost {
    std::vector<er::Done> pending;
    std::size_t refuseFrom = SIZE_MAX;
    std::size_t posted = 0;
    er::Posting Post(er::Done a_done) {
        if (posted >= refuseFrom)
            return er::Posting::Refused;
        ++posted;
        pending.push_back(std::move(a_done));
        return er::Posting::Posted;
    }
};

struct Failures {
    std::vector<std::string> what;
    std::vector<std::string> error;
    er::OnFailure Hook() {
        return [this](std::string_view a_what, std::string_view a_error) {
            what.emplace_back(a_what);
            error.emplace_back(a_error);
        };
    }
};

template <class C>
er::Posting PostTo(C& a_cache, FakeHost& a_host, const typename C::EntryPtr& a_entry, std::size_t a_count, typename C::Settled a_settled = {}) {
    return a_cache.Post(a_entry, a_count, "the test epoch", [&](std::size_t, er::Done a_done) { return a_host.Post(std::move(a_done)); },
        std::move(a_settled));
}

const auto kUsable = [](const auto&) { return false; };
const auto kDescribe = [] { return std::string("the test entry"); };

int TestResultOf() {
    {
        const auto r = er::ResultOf(MakeRecording(5), nullptr, 5);
        CHECK(r.outcome == er::Outcome::Ready);
        CHECK(r.recording && r.error.empty());
    }
    {
        // Build 0 takes a recording of any build.
        const auto r = er::ResultOf(MakeRecording(9), nullptr, 0);
        CHECK(r.outcome == er::Outcome::Ready);
    }
    {
        const auto r = er::ResultOf(MakeRecording(4), nullptr, 5);
        CHECK(r.outcome == er::Outcome::Dropped);
        CHECK(!r.recording && !r.error.empty());
    }
    {
        const auto r = er::ResultOf(nullptr, std::make_exception_ptr(org::GraphHost::EpochRecordingDropped(3)), 3);
        CHECK(r.outcome == er::Outcome::Dropped);
    }
    {
        const auto r = er::ResultOf(nullptr, std::make_exception_ptr(std::runtime_error("boom")), 3);
        CHECK(r.outcome == er::Outcome::Failed);
        CHECK(r.error == "boom");
    }
    {
        const auto r = er::ResultOf(nullptr, std::make_exception_ptr(42), 3);
        CHECK(r.outcome == er::Outcome::Failed);
        CHECK(r.error == "an unknown error");
    }
    {
        const auto r = er::ResultOf(nullptr, nullptr, 3);
        CHECK(r.outcome == er::Outcome::Failed);
        CHECK(!r.error.empty());
    }
    return 0;
}

// Find hashes the key itself, finds by hash and equality, and records the use.
int TestFindHashesKey() {
    er::Cache<Key, 4, er::AfterDrop::AskAgain, KeyHash> cache;
    CHECK(!cache.Find(Key{1, 0}, 1, kUsable, kDescribe));
    auto entry = cache.Add({7, 0}, Key{1, 0}, 1);
    CHECK(entry->hash == KeyHash{}(Key{1, 0}));
    CHECK(cache.Find(Key{1, 0}, 2, kUsable, kDescribe) == entry);
    CHECK(entry->lastUse == 2);
    // Same hash, other key.
    CHECK(!cache.Find(Key{1, 1}, 3, kUsable, kDescribe));
    CHECK(entry->lastUse == 2);
    return 0;
}

// Requests of one entry settle together, and the settled callback runs once with every recording.
int TestSettleTogether() {
    er::Cache<Key, 4, er::AfterDrop::AskAgain, KeyHash> cache;
    FakeHost host;
    int calls = 0;
    std::size_t recordings = 0;
    auto entry = cache.Add({7, 0}, Key{1, 0}, 1, 2);
    CHECK(PostTo(cache, host, entry, 2, [&](const auto& a_status) {
        ++calls;
        recordings = a_status.recordings.size();
        ENSURE(a_status.Ready());
    }) == er::Posting::Posted);
    CHECK(host.pending.size() == 2);
    host.pending[1](Ready(7));
    CHECK(!cache.IsSettled(*entry));
    CHECK(!cache.RecordingOf(*entry, 1));
    host.pending[0](Ready(7));
    CHECK(cache.IsSettled(*entry));
    CHECK(calls == 1 && recordings == 2);
    CHECK(cache.RecordingOf(*entry, 0) && cache.RecordingOf(*entry, 1));
    CHECK(!cache.RecordingOf(*entry, 2));
    CHECK(cache.Find(Key{1, 0}, 2, kUsable, kDescribe) == entry);
    return 0;
}

// A drop: AskAgain misses at the next lookup, KeepUntilNewKey finds it dropped. Neither reports a failure.
int TestAfterDrop() {
    Failures failures;
    {
        er::Cache<Key, 4, er::AfterDrop::AskAgain, KeyHash> cache(failures.Hook());
        FakeHost host;
        auto entry = cache.Add({7, 0}, Key{1, 0}, 1, 2);
        CHECK(PostTo(cache, host, entry, 2) == er::Posting::Posted);
        host.pending[0](Ready(7));
        host.pending[1](NotReady(er::Outcome::Dropped, "dropped"));
        const auto status = cache.Read(*entry);
        CHECK(status.settled && status.dropped && !status.Ready());
        CHECK(status.recordings.empty());  // used whole or not at all
        CHECK(!cache.Find(Key{1, 0}, 2, kUsable, kDescribe));
        CHECK(!cache.AnyOf([](const auto&) { return true; }));
    }
    {
        er::Cache<Key, 4, er::AfterDrop::KeepUntilNewKey, KeyHash> cache(failures.Hook());
        FakeHost host;
        auto entry = cache.Add({7, 0}, Key{1, 0}, 1);
        CHECK(PostTo(cache, host, entry, 1) == er::Posting::Posted);
        host.pending[0](NotReady(er::Outcome::Dropped, "dropped"));
        auto found = cache.Find(Key{1, 0}, 2, kUsable, kDescribe);
        CHECK(found == entry);
        CHECK(cache.Read(*found).dropped);
    }
    CHECK(failures.what.empty());
    return 0;
}

// A failed request goes to OnFailure when it settles, named with its request. The kept entry fails again at its lookup,
// which then misses.
int TestFailure() {
    Failures failures;
    er::Cache<Key, 4, er::AfterDrop::KeepUntilNewKey, KeyHash> cache(failures.Hook());
    FakeHost host;
    bool settled = false;
    auto entry = cache.Add({7, 0}, Key{1, 0}, 1, 2);
    CHECK(PostTo(cache, host, entry, 2, [&](const auto& a_status) { settled = !a_status.Ready(); }) == er::Posting::Posted);
    host.pending[1](NotReady(er::Outcome::Failed, "boom"));
    CHECK(failures.what.size() == 1);
    CHECK(failures.what[0] == "the recording of the test epoch (request 1)");
    CHECK(failures.error[0] == "boom");
    host.pending[0](Ready(7));
    CHECK(settled);
    CHECK(cache.Read(*entry).error == "boom");
    CHECK(!cache.Find(Key{1, 0}, 2, kUsable, kDescribe));
    CHECK(failures.what.size() == 2);
    CHECK(failures.what[1] == "the recording of the test entry");
    CHECK(failures.error[1] == "boom");
    CHECK(!cache.AnyOf([](const auto&) { return true; }));
    return 0;
}

// A refused post settles the requests not posted with an error and posts no more. Its lookup reports a failure.
int TestRefused() {
    Failures failures;
    er::Cache<Key, 4, er::AfterDrop::AskAgain, KeyHash> cache(failures.Hook());
    FakeHost host;
    host.refuseFrom = 1;
    int calls = 0;
    auto entry = cache.Add({7, 0}, Key{1, 0}, 1, 3);
    CHECK(PostTo(cache, host, entry, 3, [&](const auto&) { ++calls; }) == er::Posting::Refused);
    CHECK(host.pending.size() == 1);
    CHECK(!cache.IsSettled(*entry));
    CHECK(failures.what.empty());
    host.pending[0](Ready(7));
    CHECK(cache.IsSettled(*entry));
    CHECK(calls == 1);
    CHECK(!cache.Read(*entry).Ready());
    CHECK(!cache.Find(Key{1, 0}, 2, kUsable, kDescribe));
    CHECK(failures.what.size() == 1);
    return 0;
}

// Unusable ready entries miss. Adding drops ready entries of builds not kept, then the least recently used settled one.
int TestKeepRules() {
    er::Cache<Key, 2, er::AfterDrop::AskAgain, KeyHash> cache;
    FakeHost host;
    auto a = cache.Add({7, 0}, Key{1, 0}, 1);
    auto b = cache.Add({7, 0}, Key{2, 0}, 2);
    CHECK(PostTo(cache, host, a, 1) == er::Posting::Posted);
    CHECK(PostTo(cache, host, b, 1) == er::Posting::Posted);
    host.pending[0](Ready(7));
    host.pending[1](Ready(7));
    CHECK(cache.Find(Key{1, 0}, 3, kUsable, kDescribe) == a);  // a is now the most recently used
    // Full: the least recently used settled entry (b) goes.
    auto c = cache.Add({7, 0}, Key{3, 0}, 4);
    CHECK(cache.Find(Key{1, 0}, 5, kUsable, kDescribe) == a);
    CHECK(!cache.Find(Key{2, 0}, 5, kUsable, kDescribe));
    // Build 7 is no longer kept, so the ready a goes. The unsettled c stays.
    auto d = cache.Add({8, 9}, Key{4, 0}, 6);
    CHECK(!cache.Find(Key{1, 0}, 7, kUsable, kDescribe));
    CHECK(cache.Find(Key{3, 0}, 7, kUsable, kDescribe) == c);
    CHECK(cache.Find(Key{4, 0}, 7, kUsable, kDescribe) == d);
    // A ready entry the caller cannot use misses and goes.
    CHECK(PostTo(cache, host, d, 1) == er::Posting::Posted);
    host.pending.back()(Ready(9));
    CHECK(!cache.Find(Key{4, 0}, 8, [](const auto& a_status) { return a_status.Ready(); }, kDescribe));
    CHECK(!cache.Find(Key{4, 0}, 8, kUsable, kDescribe));
    CHECK(cache.EraseIf([](const auto& a_entry) { return !a_entry.LockedStatus().settled; }) == 1);
    // Unsettled entries are never dropped to make room, so the cache may hold more than Keep.
    for (int id = 10; id < 13; ++id)
        (void)cache.Add({8, 9}, Key{id, 0}, 9);
    CHECK(cache.EraseIf([](const auto&) { return true; }) == 3);
    return 0;
}

// A request may settle after its entry was erased and the cache is gone.
int TestSettleAfterCache() {
    FakeHost host;
    bool settled = false;
    {
        er::Cache<Key, 2, er::AfterDrop::AskAgain, KeyHash> cache([](std::string_view, std::string_view) {});
        auto entry = cache.Add({7, 0}, Key{1, 0}, 1);
        CHECK(PostTo(cache, host, entry, 1, [&](const auto& a_status) { settled = a_status.Ready(); }) == er::Posting::Posted);
        CHECK(cache.EraseIf([](const auto&) { return true; }) == 1);
    }
    host.pending[0](Ready(7));
    CHECK(settled);
    return 0;
}
}  // namespace

int main() {
    if (const int line = TestResultOf()) return line;
    if (const int line = TestFindHashesKey()) return line;
    if (const int line = TestSettleTogether()) return line;
    if (const int line = TestAfterDrop()) return line;
    if (const int line = TestFailure()) return line;
    if (const int line = TestRefused()) return line;
    if (const int line = TestKeepRules()) return line;
    if (const int line = TestSettleAfterCache()) return line;
    std::puts("OpenRenderGraphEpochRecordingsTests passed");
    return 0;
}
