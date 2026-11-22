#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "Resources/Relocation.h"

namespace rhi::ma {
class Allocator;
class DefragmentationContext;
}

// Native defragmentation at idle points (docs/development/plans/org-relocation.md in the application, step 5).
//
// The host calls GraphHost::DefragmentAtIdle at a loading screen. Each call waits for the GPU to go idle and runs passes of the
// allocator's Full algorithm over the device-local block vectors. In each pass:
// 1. BeginPass proposes moves. The defragmenter keeps those whose allocation belongs to a graph texture (or a graph buffer, when
//    asked) that may move now, and leaves the rest in place.
// 2. For each kept move it creates the resource again at the destination and hands a relocation to the commit. The commit copies
//    the old resource into the new one in the relocation batch.
// 3. EndPass (CopyRetain) points each allocation at its new resource. The old range comes back as a retained allocation.
// 4. The commit rewrites the descriptors in place, since the GPU is idle, so bindless indices keep their values. A moved buffer's
//    buffer view ref pair goes to the new buffer (rhi::Device::RetargetBufferViewRef), so its ref keeps its slot, and the journal
//    entry carries the old and new device addresses for clients that wrote the address into GPU data.
// 5. The old range is freed once the GPU is past the copy and no recording names the old backing any more.
//
// A campaign is the run of passes from the start rule to the end: it starts when the device-local blocks' free space is large,
// and ends when BeginPass has nothing more to move or after a number of passes that freed no block.
namespace org::relocation {

// Limits and thresholds of defragmentation at an idle point.
struct DefragmentBudget {
    // Passes one call may run.
    uint32_t maxPasses = 4;
    // A call starts no new pass after this many milliseconds, the waits for the GPU included.
    double maxMilliseconds = 40.0;
    // The allocator's limits for one pass. 0 moves: no limit.
    uint64_t maxBytesPerPass = 512ull << 20;
    uint32_t maxMovesPerPass = 0;
    // A campaign starts when the device-local blocks' free space (Allocator::CalculateStatistics: block bytes minus allocation
    // bytes) is over both of these.
    uint64_t minSlackBytes = 64ull << 20;
    double minSlackFraction = 0.25;
    // A campaign ends after this many passes in a row that freed no block.
    uint32_t passesWithoutFreedBlock = 4;
    // After a campaign ends, the start rule is checked again only once this many calls have passed.
    uint32_t recheckCalls = 120;
    // Whether graph buffers move too. A moved buffer's device address changes. Its descriptors and its buffer view ref keep their
    // slots, and the journal names the old and new addresses. The application must pin (Class::Pinned) every buffer whose address
    // it keeps and does not rebase from the journal. Off by default for that reason. A PerFrame buffer never moves. Moves happen
    // only in idle commits, so Written buffers and buffers with view refs move too.
    bool moveBuffers = false;
};

// What one DefragmentAtIdle call did.
struct DefragmentReport {
    uint32_t passes = 0;
    uint64_t moves = 0;
    uint64_t movedBytes = 0;
    double passMilliseconds = 0.0;  // the passes, without the waits for the GPU
    double waitMilliseconds = 0.0;  // the waits for the GPU to go idle
    bool campaignActive = false;    // a later call has more passes to run
};

// The defragmenter. One per process, registered with relocation::Service as a source on first use. Owner thread of the host.
class Defragmenter final : public IRelocationSource {
public:
    static Defragmenter& Get();

    // Whether this call should run passes: a campaign is running, or the start rule says to start one now. Counts the call.
    bool WantsCall(const DefragmentBudget& budget, rhi::ma::Allocator& allocator);
    // Before each pass, with the GPU idle. Counts the blocks freed since the last pass and ends the campaign when too many passes
    // freed none. True when the next commit should run a pass.
    bool ArmPass(const DefragmentBudget& budget, rhi::ma::Allocator& allocator);
    // After the commit that ran the pass. When the commit did not finish the pass (it threw), waitIdle is called and the pass is
    // ended with every move left in place. Logs the pass and adds its numbers to report.
    template <class WaitIdle>
    void FinishPass(double passMilliseconds, DefragmentReport& report, WaitIdle&& waitIdle) {
        if (PassOpen() && !PassEnded()) waitIdle();
        FinishPassAfterWait(passMilliseconds, report);
    }
    // Ends a running campaign (the loading screen ended). Does nothing while a pass is open.
    void EndCampaign(const char* why);
    bool CampaignActive() const noexcept { return m_context != nullptr; }

    // IRelocationSource.
    void BeginCommit(const CommitContext& context) override;
    size_t Poll(std::span<Relocation> out) override;
    void BeforeAdopt(std::span<Relocation> polled) override;
    void AfterAdopt(std::span<Relocation> polled) override;

    ~Defragmenter() override;

private:
    Defragmenter();
    struct Pass;
    struct RetainedRange;
    struct Lingering {
        std::shared_ptr<RetainedRange> range;
        std::weak_ptr<const void> publication;
        uint64_t bytes = 0;
    };

    bool PassOpen() const noexcept;
    bool PassEnded() const noexcept;
    void FinishPassAfterWait(double passMilliseconds, DefragmentReport& report);
    void OpenPass();
    // Looks at one proposed move. Adds it to the pass when it may move now, else returns why not.
    bool Prepare(uint32_t moveIndex, IgnoreReason& why);
    void StartCampaign(rhi::ma::Allocator& allocator, uint64_t slack, uint64_t blockBytes, uint32_t blocks);
    void PruneLingering();

    rhi::ma::DefragmentationContext* m_context = nullptr;
    DefragmentBudget m_budget{};
    std::unique_ptr<Pass> m_pass;
    bool m_armed = false;
    // The current commit's CommitContext::gpuIdle. A pass opens only in an idle commit.
    bool m_commitIdle = false;
    bool m_nothingLeft = false;
    uint64_t m_calls = 0;
    uint64_t m_nextCheck = 0;
    // The campaign.
    uint32_t m_passes = 0;
    uint32_t m_passesWithoutFreedBlock = 0;
    uint32_t m_blocksAtLastPass = 0;
    uint32_t m_blocksFreedThisPass = 0;
    uint32_t m_blocksAtStart = 0;
    uint64_t m_slackAtStart = 0;
    uint64_t m_campaignMoves = 0;
    uint64_t m_campaignBytes = 0;
    double m_campaignMilliseconds = 0.0;
    std::vector<Lingering> m_lingering;
    // Old ranges whose resource could not take its new backing after EndPass. Its descriptors may still name the old resource,
    // so the range is never freed.
    std::vector<std::shared_ptr<RetainedRange>> m_stuck;
};

} // namespace org::relocation
