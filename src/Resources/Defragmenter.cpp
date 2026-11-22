#include "Resources/Defragmenter.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <string>

#include <rhi_allocator.h>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Resources/AllocationRelease.h"
#include "Resources/BackedResource.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Resource.h"

namespace org::relocation {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;

// The device-local block vectors' figures. Committed allocations count as blocks with no free space in them. The PerFrame pool's
// blocks are left out: the defragmenter never moves into or out of them.
struct DeviceLocalBlocks {
    uint32_t blocks = 0;
    uint64_t blockBytes = 0;
    uint64_t slack = 0;
};

DeviceLocalBlocks MeasureDeviceLocal(rhi::ma::Allocator& allocator) {
    rhi::ma::TotalStatistics stats{};
    allocator.CalculateStatistics(&stats);
    const auto& local = stats.heapType[static_cast<size_t>(rhi::HeapType::DeviceLocal)].stats;
    DeviceLocalBlocks out;
    out.blocks = local.blockCount;
    out.blockBytes = local.blockBytes;
    rhi::ma::Statistics perFrame{};
    if (DeviceManager::GetInstance().MemoryGroupStatistics(MemoryGroup::PerFrame, rhi::HeapType::DeviceLocal, perFrame)) {
        out.blocks = out.blocks > perFrame.blockCount ? out.blocks - perFrame.blockCount : 0;
        out.blockBytes = out.blockBytes > perFrame.blockBytes ? out.blockBytes - perFrame.blockBytes : 0;
        const uint64_t allocationBytes = local.allocationBytes > perFrame.allocationBytes ? local.allocationBytes - perFrame.allocationBytes : 0;
        out.slack = out.blockBytes > allocationBytes ? out.blockBytes - allocationBytes : 0;
        return out;
    }
    out.slack = local.blockBytes > local.allocationBytes ? local.blockBytes - local.allocationBytes : 0;
    return out;
}

} // namespace

// An old range a CopyRetain move handed back. Freed through the release gate when the last holder lets go: the descriptor
// manager's retirement (the GPU is past the copy) and the defragmenter's lingering list (no recording names the old backing).
struct Defragmenter::RetainedRange {
    rhi::ma::AllocationPtr allocation;
    uint64_t bytes = 0;
    ~RetainedRange() { ReleaseAllocation(std::move(allocation)); }
};

// One pass, from BeginPass to AfterAdopt.
struct Defragmenter::Pass {
    // A move handed to the commit.
    struct Prepared {
        uint32_t moveIndex = 0;
        // Held from Poll until the backing switched, so nothing can change the backing between the copy and the switch.
        std::unique_lock<std::recursive_mutex> backingLock;
        // The publication snapshot of the old backing. Recordings that bound the old backing hold it.
        std::weak_ptr<const void> publication;
        Relocation relocation;
        uint64_t bytes = 0;
        // Set in BeforeAdopt when EndPass moved it.
        std::shared_ptr<RetainedRange> retained;
    };
    // No frees while the pass names allocations by pointer.
    std::optional<ReleaseHold> hold;
    rhi::ma::DefragmentationPassMoveInfo info{};
    std::vector<rhi::ma::AllocationPtr> retained;
    std::vector<Prepared> prepared;
    size_t handedOut = 0;
    bool ended = false;  // EndPass ran
    uint64_t moves = 0, movedBytes = 0;
    std::array<uint64_t, static_cast<size_t>(IgnoreReason::Count)> ignored{};
    uint64_t ignoredTotal = 0;
};

Defragmenter& Defragmenter::Get() {
    // Never destroyed, like the relocation service that polls it.
    static auto* instance = [] {
        auto defragmenter = std::shared_ptr<Defragmenter>(new Defragmenter());
        Service::Get().AddSource(defragmenter);
        return new std::shared_ptr<Defragmenter>(std::move(defragmenter));
    }();
    return **instance;
}

Defragmenter::Defragmenter() = default;
Defragmenter::~Defragmenter() = default;

bool Defragmenter::PassOpen() const noexcept { return m_pass != nullptr; }
bool Defragmenter::PassEnded() const noexcept { return m_pass && m_pass->ended; }

bool Defragmenter::WantsCall(const DefragmentBudget& budget, rhi::ma::Allocator& allocator) {
    ++m_calls;
    m_budget = budget;
    PruneLingering();
    if (m_context) return true;
    if (m_calls < m_nextCheck) return false;
    m_nextCheck = m_calls + (std::max)(budget.recheckCalls, 1u);
    const auto measured = MeasureDeviceLocal(allocator);
    const bool start = measured.slack > budget.minSlackBytes
        && static_cast<double>(measured.slack) > budget.minSlackFraction * static_cast<double>(measured.blockBytes);
    if (!start) {
        // Say why no campaign starts, a few times, so a run with the switch on shows the slack it measured.
        static uint32_t logged = 0;
        if (logged < 8) {
            ++logged;
            spdlog::info("ORG defragmenter: no campaign. Device-local blocks: {}, {:.1f} MB, {:.1f} MB free in them ({:.0f} %, starts above {:.0f} MB and {:.0f} %)",
                measured.blocks, static_cast<double>(measured.blockBytes) / kMiB, static_cast<double>(measured.slack) / kMiB,
                measured.blockBytes ? 100.0 * static_cast<double>(measured.slack) / static_cast<double>(measured.blockBytes) : 0.0,
                static_cast<double>(budget.minSlackBytes) / kMiB, 100.0 * budget.minSlackFraction);
        }
        return false;
    }
    StartCampaign(allocator, measured.slack, measured.blockBytes, measured.blocks);
    return m_context != nullptr;
}

void Defragmenter::StartCampaign(rhi::ma::Allocator& allocator, uint64_t slack, uint64_t blockBytes, uint32_t blocks) {
    rhi::ma::DefragmentationDesc desc{};
    desc.flags = rhi::ma::DefragmentationFlagsAlgorithmFull;
    desc.maxBytesPerPass = m_budget.maxBytesPerPass;
    desc.maxAllocationsPerPass = m_budget.maxMovesPerPass;
    allocator.BeginDefragmentation(&desc, &m_context);
    if (!m_context) {
        spdlog::error("ORG defragmenter: the allocator did not start a defragmentation");
        return;
    }
    m_passes = 0;
    m_passesWithoutFreedBlock = 0;
    m_blocksAtLastPass = blocks;
    m_blocksFreedThisPass = 0;
    m_blocksAtStart = blocks;
    m_slackAtStart = slack;
    m_campaignMoves = 0;
    m_campaignBytes = 0;
    m_campaignMilliseconds = 0.0;
    m_nothingLeft = false;
    detail::NoteDefragCampaign();
    spdlog::info("ORG defragmenter: campaign started. Device-local blocks: {}, {:.1f} MB, {:.1f} MB free in them ({:.0f} %)", blocks,
        static_cast<double>(blockBytes) / kMiB, static_cast<double>(slack) / kMiB,
        blockBytes ? 100.0 * static_cast<double>(slack) / static_cast<double>(blockBytes) : 0.0);
}

void Defragmenter::EndCampaign(const char* why) {
    if (!m_context || m_pass) return;
    rhi::ma::DefragmentationStats stats{};
    m_context->GetStats(&stats);
    m_context->ReleaseThis();
    m_context = nullptr;
    m_armed = false;
    m_nextCheck = m_calls + (std::max)(m_budget.recheckCalls, 1u);
    spdlog::info("ORG defragmenter: campaign ended ({}). {} passes, {} moves, {:.1f} MB moved, {:.1f} ms in passes; blocks {} -> {}; "
        "free space at the start {:.1f} MB; {} old ranges still held by recordings",
        why, m_passes, m_campaignMoves, static_cast<double>(m_campaignBytes) / kMiB, m_campaignMilliseconds, m_blocksAtStart,
        m_blocksAtLastPass, static_cast<double>(m_slackAtStart) / kMiB, m_lingering.size());
}

bool Defragmenter::ArmPass(const DefragmentBudget& budget, rhi::ma::Allocator& allocator) {
    m_budget = budget;
    if (!m_context || m_pass) return false;
    if (m_nothingLeft) {
        EndCampaign("nothing left to move");
        return false;
    }
    // Retained ranges free their blocks only when they go, which is a pass or more after their move. So a pass is credited with
    // the blocks gone since the last pass started.
    const auto measured = MeasureDeviceLocal(allocator);
    m_blocksFreedThisPass = m_blocksAtLastPass > measured.blocks ? m_blocksAtLastPass - measured.blocks : 0;
    if (m_passes > 0) m_passesWithoutFreedBlock = m_blocksFreedThisPass ? 0 : m_passesWithoutFreedBlock + 1;
    m_blocksAtLastPass = measured.blocks;
    if (m_passesWithoutFreedBlock >= (std::max)(budget.passesWithoutFreedBlock, 1u)) {
        EndCampaign("passes freed no block");
        return false;
    }
    m_armed = true;
    return true;
}

void Defragmenter::PruneLingering() {
    if (m_pass) return;
    std::erase_if(m_lingering, [](const Lingering& entry) {
        if (!entry.publication.expired()) return false;
        detail::NoteDefragLingering(-1, -static_cast<int64_t>(entry.bytes));
        return true;
    });
}

void Defragmenter::BeginCommit(const CommitContext& context) {
    m_commitIdle = context.gpuIdle;
}

size_t Defragmenter::Poll(std::span<Relocation> out) {
    if (!m_pass) {
        PruneLingering();
        // Passes run only in DefragmentAtIdle's idle commits. A frame's own commit leaves an armed pass for the idle one.
        if (!m_armed || !m_context || !m_commitIdle) return 0;
        m_armed = false;
        OpenPass();
        if (!m_pass) return 0;
    }
    auto& pass = *m_pass;
    size_t written = 0;
    while (written < out.size() && pass.handedOut < pass.prepared.size())
        out[written++] = std::move(pass.prepared[pass.handedOut++].relocation);
    return written;
}

void Defragmenter::OpenPass() {
    BT_ZONE_SCOPE("ORG.Defragmenter.BeginPass");
    auto pass = std::make_unique<Pass>();
    // Before BeginPass: from here until EndPass the pass names allocations by pointer.
    pass->hold.emplace();
    const auto result = m_context->BeginPass(&pass->info);
    if (result == rhi::Result::Ok) {
        // Nothing to move. The campaign ends at the next ArmPass.
        m_nothingLeft = true;
        ++m_passes;
        return;
    }
    if (result != rhi::Result::False) {
        spdlog::error("ORG defragmenter: BeginPass failed: {}", rhi::ResultName(result));
        m_nothingLeft = true;
        return;
    }
    pass->retained.resize(pass->info.moveCount);
    m_pass = std::move(pass);
    auto& current = *m_pass;
    for (uint32_t i = 0; i < current.info.moveCount; ++i) {
        current.info.pMoves[i].operation = rhi::ma::DefragmentationMoveOperation::Ignore;
        IgnoreReason why = IgnoreReason::Failed;
        if (Prepare(i, why)) continue;
        ++current.ignored[static_cast<size_t>(why)];
        ++current.ignoredTotal;
    }
    if (!current.prepared.empty()) return;
    // Nothing may move. The pass ends here with every move left in place, and the commit sees no relocation from this source.
    current.info.pRetained = current.retained.data();
    if (const auto ended = m_context->EndPass(&current.info); ended != rhi::Result::Ok && ended != rhi::Result::False)
        spdlog::error("ORG defragmenter: EndPass failed: {}", rhi::ResultName(ended));
    current.ended = true;
    current.hold.reset();
}

bool Defragmenter::Prepare(uint32_t moveIndex, IgnoreReason& why) {
    auto& pass = *m_pass;
    auto& move = pass.info.pMoves[moveIndex];
    auto* record = move.pSrcAllocation ? move.pSrcAllocation->GetOwnerRecord<RelocationRecord>() : nullptr;
    // A null record: an allocation ORG does not track (a host's AllocatedResource, such as a streaming window).
    if (!MayMove(record, m_commitIdle, &why)) return false;
    auto owner = record->Owner();
    if (!owner) {
        why = IgnoreReason::NoRecord;
        return false;
    }
    auto* backed = dynamic_cast<BackedResource*>(owner.get());
    auto* texture = dynamic_cast<PixelBuffer*>(owner.get());
    auto* buffer = dynamic_cast<BufferBase*>(owner.get());
    if (!backed || (!texture && !buffer)) {
        why = IgnoreReason::Unsupported;
        return false;
    }
    if (buffer && !m_budget.moveBuffers) {
        why = IgnoreReason::Pinned;
        return false;
    }
    // PerFrame buffers never move, and buffers with a view ref pair move only at idle.
    if (buffer && !BufferMayMove(*buffer, m_commitIdle, &why)) return false;
    // An externally immutable resource gets no barriers from the ledgers, so the copy could not move it to CopySource. An attached
    // representation replaces the backing.
    if (owner->GetGraphOwnership() == Resource::GraphOwnership::ExternalImmutableShaderResource
        || owner->GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid()) {
        why = IgnoreReason::Unsupported;
        return false;
    }
    if (texture) {
        // A copy cannot carry a multisampled texture. On D3D12 it copies one plane, so a depth-stencil texture would lose its stencil.
        const auto& description = texture->GetDescription();
        if (description.sampleCount > 1 || description.hasDSV) {
            why = IgnoreReason::Unsupported;
            return false;
        }
    }
    auto* mutex = backed->BackingMutex();
    if (!mutex) {
        why = IgnoreReason::Unsupported;
        return false;
    }
    // Never waits: another thread holding it may be waiting for something this thread holds. A busy backing stays this pass.
    std::unique_lock<std::recursive_mutex> lock(*mutex, std::try_to_lock);
    if (!lock) {
        why = IgnoreReason::Stale;
        return false;
    }
    // The record belongs to the current backing. Otherwise the allocation is an old backing that leases still hold.
    if (backed->CurrentRelocationRecord() != record) {
        why = IgnoreReason::Stale;
        return false;
    }
    auto publication = backed->CapturePublishedBackingAllocation();
    if (!publication) {
        why = IgnoreReason::Unsupported;
        return false;
    }
    rhi::ResourcePtr made;
    const auto result = texture ? texture->MakeDefragmentationResource(*move.pSrcAllocation, *move.pDstTmpAllocation, made)
                                : buffer->MakeDefragmentationResource(*move.pSrcAllocation, *move.pDstTmpAllocation, made);
    if (!rhi::IsOk(result) || !made) {
        spdlog::warn("ORG defragmenter: making '{}' at its new place failed: {}", owner->GetName(), rhi::ResultName(result));
        why = IgnoreReason::Failed;
        return false;
    }
    move.pDstTmpAllocation->SetResource(std::move(made));
    Pass::Prepared prepared;
    prepared.moveIndex = moveIndex;
    prepared.publication = publication;
    prepared.bytes = record->SizeBytes();
    auto& relocation = prepared.relocation;
    relocation.resource = owner;
    relocation.resourceID = owner->GetGlobalResourceID();
    relocation.fromGeneration = backed->GetBackingGeneration();
    relocation.sizeBytes = prepared.bytes;
    relocation.source = Source::Defragmenter;
    // After EndPass the allocation names the new buffer. The switch needs the old one for the view ref pair and the old address.
    if (buffer) relocation.fromBuffer = move.pSrcAllocation->GetResource().GetHandle();
    try {
        // The new resource belongs to the destination allocation until EndPass, then to the backing's allocation. The lease keeps the
        // backing's allocation alive while the copy target lives.
        relocation.copyTarget = detail::MakeCopyTarget(move.pDstTmpAllocation->GetResource(), *owner, publication->lease);
    } catch (const std::exception& error) {
        spdlog::warn("ORG defragmenter: the copy target of '{}' failed: {}", owner->GetName(), error.what());
        why = IgnoreReason::Failed;
        return false;
    }
    prepared.backingLock = std::move(lock);
    pass.prepared.push_back(std::move(prepared));
    return true;
}

void Defragmenter::BeforeAdopt(std::span<Relocation> polled) {
    if (!m_pass || m_pass->ended) return;
    BT_ZONE_SCOPE("ORG.Defragmenter.EndPass");
    auto& pass = *m_pass;
    const size_t count = (std::min)(polled.size(), pass.prepared.size());
    for (size_t k = 0; k < count; ++k) {
        auto& move = pass.info.pMoves[pass.prepared[k].moveIndex];
        move.operation = polled[k].skipped ? rhi::ma::DefragmentationMoveOperation::Ignore
                                           : rhi::ma::DefragmentationMoveOperation::CopyRetain;
    }
    pass.info.pRetained = pass.retained.data();
    const auto result = m_context->EndPass(&pass.info);
    pass.ended = true;
    if (result != rhi::Result::Ok && result != rhi::Result::False) {
        // EndPass changed nothing. No allocation names its new resource, so no backing may switch.
        spdlog::error("ORG defragmenter: EndPass failed: {}. The pass's moves are called off", rhi::ResultName(result));
        for (auto& relocation : polled) relocation.skipped = true;
        m_nothingLeft = true;
        pass.hold.reset();
        return;
    }
    for (size_t k = 0; k < count; ++k) {
        if (polled[k].skipped) continue;
        auto& prepared = pass.prepared[k];
        auto& allocation = pass.retained[prepared.moveIndex];
        if (!allocation) {
            // A CopyRetain move always hands its old range back. Without it there is nothing to retire.
            spdlog::error("ORG defragmenter: EndPass handed no old range back for '{}'", polled[k].resource ? polled[k].resource->GetName() : "?");
            continue;
        }
        auto range = std::make_shared<RetainedRange>();
        range->allocation = std::move(allocation);
        range->bytes = prepared.bytes;
        prepared.retained = range;
        polled[k].retired = range;
        // Kept until no recording names the old backing. A revision ticket keeps binding a recording made with the old backing, and
        // that recording still uses the old resource.
        m_lingering.push_back({range, prepared.publication, prepared.bytes});
        detail::NoteDefragLingering(1, static_cast<int64_t>(prepared.bytes));
    }
    // EndPass is done with the allocation pointers. Frees parked during the pass run now.
    pass.hold.reset();
}

void Defragmenter::AfterAdopt(std::span<Relocation> polled) {
    if (!m_pass) return;
    auto& pass = *m_pass;
    const size_t count = (std::min)(polled.size(), pass.prepared.size());
    for (size_t k = 0; k < count; ++k) {
        auto& prepared = pass.prepared[k];
        if (!polled[k].skipped) {
            ++pass.moves;
            pass.movedBytes += prepared.bytes;
            continue;
        }
        if (!prepared.retained) continue;
        // EndPass moved it, and the switch failed. The allocation names the new resource, but the descriptors may still name the
        // old one, so the old range must never be freed.
        spdlog::error("ORG defragmenter: '{}' moved in the allocator but did not take its new backing. Its old memory stays allocated",
            polled[k].resource ? polled[k].resource->GetName() : "?");
        m_stuck.push_back(prepared.retained);
    }
    // The backing locks go here.
    pass.prepared.clear();
}

void Defragmenter::FinishPassAfterWait(double passMilliseconds, DefragmentReport& report) {
    // A commit that never polled must not leave the pass armed: the frame's own commit runs without the GPU idle.
    m_armed = false;
    if (!m_pass) {
        // The pass ended in OpenPass (nothing to move), or no pass ran.
        return;
    }
    auto& pass = *m_pass;
    if (!pass.ended) {
        // The commit stopped before BeforeAdopt. The GPU is idle now, so the copies it may have submitted are done.
        spdlog::error("ORG defragmenter: a pass did not reach its end in the commit. Its moves are left in place");
        for (uint32_t i = 0; i < pass.info.moveCount; ++i) pass.info.pMoves[i].operation = rhi::ma::DefragmentationMoveOperation::Ignore;
        pass.info.pRetained = pass.retained.data();
        (void)m_context->EndPass(&pass.info);
        pass.ended = true;
        m_nothingLeft = true;
    }
    pass.hold.reset();
    ++m_passes;
    m_campaignMoves += pass.moves;
    m_campaignBytes += pass.movedBytes;
    m_campaignMilliseconds += passMilliseconds;
    report.moves += pass.moves;
    report.movedBytes += pass.movedBytes;
    detail::NoteDefragPass(pass.moves, pass.movedBytes, pass.ignoredTotal, m_blocksFreedThisPass,
        static_cast<uint64_t>(passMilliseconds * 1000.0));
    const auto ignored = [&](IgnoreReason reason) { return pass.ignored[static_cast<size_t>(reason)]; };
    spdlog::info("ORG defragmenter: pass {}: {} moves, {:.1f} MB, in {:.2f} ms. Left in place: {} (no owner {}, pinned {}, written {}, "
        "unsupported {}, busy {}, failed {}). Device-local blocks at its start: {}, {} freed since the pass before",
        m_passes, pass.moves, static_cast<double>(pass.movedBytes) / kMiB, passMilliseconds, pass.ignoredTotal,
        ignored(IgnoreReason::NoRecord), ignored(IgnoreReason::Pinned), ignored(IgnoreReason::WrittenNotIdle),
        ignored(IgnoreReason::Unsupported), ignored(IgnoreReason::Stale), ignored(IgnoreReason::Failed), m_blocksAtLastPass,
        m_blocksFreedThisPass);
    m_pass.reset();
}

} // namespace org::relocation
