#include "Managers/Singletons/StatisticsManager.h"
#include "Utilities/EnvironmentFlag.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <rhi_helpers.h>
#include <spdlog/spdlog.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"


namespace org {

StatisticsManager& StatisticsManager::GetInstance() {
    static StatisticsManager inst;
    return inst;
}

namespace {
void UpdateEma(double& value, double sample) {
    value = value * (1.0 - PassStats::alpha) + sample * PassStats::alpha;
}

bool MemoryIntrospectionLoggingEnabled() {
    static const bool enabled = org::detail::EnvironmentFlagEnabled("SARP_MEMORY_INTROSPECTION_LOG");
    return enabled;
}

void LogMemoryIntrospectionAccounting(
    uint64_t frameSerial,
    const rhi::ma::Budget& localBudget)
{
    // Keep the benchmark log readable while still sampling growth during loading.
    if (!MemoryIntrospectionLoggingEnabled() || (frameSerial != 1 && frameSerial % 120 != 0)) {
        return;
    }

    spdlog::info(
        "Memory diagnostics allocator: frame={} dxgi_usage={} dxgi_budget={} "
        "allocator_blocks={} allocator_allocations={} dxgi_minus_allocator_blocks={} "
        "allocator_block_count={} allocator_allocation_count={}",
        frameSerial,
        localBudget.usageBytes,
        localBudget.budgetBytes,
        localBudget.stats.blockBytes,
        localBudget.stats.allocationBytes,
        localBudget.usageBytes > localBudget.stats.blockBytes
            ? localBudget.usageBytes - localBudget.stats.blockBytes
            : 0,
        localBudget.stats.blockCount,
        localBudget.stats.allocationCount);
}
}

void StatisticsManager::Initialize() {
    m_numFramesInFlight = org::runtime::GetOpenRenderGraphSettings().numFramesInFlight;
	auto device = DeviceManager::GetInstance().GetDevice();
	m_gpuTimestampFreq = device.GetTimestampCalibration(rhi::QueueKind::Graphics).ticksPerSecond;
    m_getCollectPassStatistics = []() {
        return org::runtime::GetOpenRenderGraphSettings().collectPassStatistics;
    };
    m_getCollectPipelineStatistics = []() {
        return org::runtime::GetOpenRenderGraphSettings().collectPipelineStatistics;
    };
    m_collectPassStatistics = m_getCollectPassStatistics();
    m_collectPipelineStatistics = m_getCollectPipelineStatistics();
}

void StatisticsManager::RegisterPasses(const std::vector<std::string>& passNames) {
    m_passNames = passNames;
    m_numPasses = static_cast<unsigned>(passNames.size());
    m_passTechniquePaths.assign(m_numPasses, {});
    m_stats.assign(m_numPasses, {});
    m_isGeometryPass.assign(m_numPasses, false);
    m_meshStatsEma.assign(m_numPasses, {});
    m_passLastExecutionFrame.assign(m_numPasses, kNeverSeenFrame);
    m_passNameToIndex.clear();
    for (unsigned i = 0; i < m_numPasses; ++i) {
        if (!m_passNames[i].empty()) {
            m_passNameToIndex.emplace(m_passNames[i], i);
        }
    }
}

unsigned StatisticsManager::RegisterPass(const std::string& passName, bool isGeometryPass, std::string_view techniquePath) {
    if (!passName.empty()) {
        auto it = m_passNameToIndex.find(passName);
        if (it != m_passNameToIndex.end()) {
            if (isGeometryPass) {
                m_isGeometryPass[it->second] = true;
            }
            if (!techniquePath.empty()) {
                auto& existingTechniquePath = m_passTechniquePaths[it->second];
                if (existingTechniquePath.empty()) {
                    existingTechniquePath.assign(techniquePath);
                }
                else if (existingTechniquePath != techniquePath) {
                    spdlog::warn(
                        "Pass '{}' was registered with conflicting technique paths '{}' and '{}'. Keeping the first one.",
                        passName,
                        existingTechniquePath,
                        techniquePath);
                }
            }
            return it->second;
        }
    }

    std::string resolvedName = passName;
    if (resolvedName.empty()) {
        resolvedName = "UnnamedPass#" + std::to_string(m_unnamedPassCounter++);
    }

    const unsigned index = static_cast<unsigned>(m_passNames.size());
    m_passNames.push_back(resolvedName);
    m_passNameToIndex[resolvedName] = index;

    m_numPasses = static_cast<unsigned>(m_passNames.size());
    m_passTechniquePaths.emplace_back(techniquePath);
    m_stats.emplace_back();
    m_isGeometryPass.push_back(isGeometryPass);
    m_meshStatsEma.emplace_back();
    m_passLastExecutionFrame.push_back(kNeverSeenFrame);
    return index;
}

void StatisticsManager::BeginFrame() {
    ++m_frameSerial;

    org::runtime::MemoryBudgetStats memoryBudgetStats{};
    memoryBudgetStats.sampleFrameSerial = m_frameSerial;
    if (auto* allocator = DeviceManager::GetInstance().GetAllocator()) {
        rhi::ma::Budget localBudget{};
        allocator->GetBudget(&localBudget, nullptr);
        memoryBudgetStats.usageBytes = localBudget.usageBytes;
        memoryBudgetStats.budgetBytes = localBudget.budgetBytes;
        memoryBudgetStats.valid = true;
        LogMemoryIntrospectionAccounting(m_frameSerial, localBudget);
    }
    m_memoryBudgetStats = memoryBudgetStats;
}

void StatisticsManager::RecordCpuUpdateTime(unsigned passIndex, double milliseconds) {
    RecordCpuTimeSample(passIndex, milliseconds, true);
}

void StatisticsManager::RecordCpuExecuteTime(unsigned passIndex, double milliseconds) {
    RecordCpuTimeSample(passIndex, milliseconds, false);
}

void StatisticsManager::RecordCpuTimeSample(unsigned passIndex, double milliseconds, bool isUpdate) {
    if (!m_collectPassStatistics) {
        return;
    }

    std::scoped_lock lock(m_cpuStatsMutex);
    if (passIndex >= m_stats.size()) {
        return;
    }

    auto& passStats = m_stats[passIndex];
    if (isUpdate) {
        UpdateEma(passStats.cpuUpdateTimeEma, milliseconds);
    }
    else {
        UpdateEma(passStats.cpuExecuteTimeEma, milliseconds);
        if (passIndex < m_passLastExecutionFrame.size()) {
            m_passLastExecutionFrame[passIndex] = m_frameSerial;
        }
    }
}

void StatisticsManager::RebuildVisiblePassIndices(uint64_t maxStaleFrames, std::vector<unsigned>& out) const {
    out.clear();
    out.reserve(m_passNames.size());

    const bool includeNeverSeen = (maxStaleFrames == (std::numeric_limits<uint64_t>::max)());

    for (unsigned i = 0; i < m_passNames.size(); ++i) {
        const uint64_t lastExecution = (i < m_passLastExecutionFrame.size()) ? m_passLastExecutionFrame[i] : kNeverSeenFrame;
        if (lastExecution == kNeverSeenFrame) {
            if (includeNeverSeen) {
                out.push_back(i);
            }
            continue;
        }

        const uint64_t missingFrames = (m_frameSerial >= lastExecution) ? (m_frameSerial - lastExecution) : 0;
        if (missingFrames <= maxStaleFrames) {
            out.push_back(i);
        }
    }
}

const std::vector<unsigned>& StatisticsManager::GetVisiblePassIndices() const {
    if (m_getCollectPassStatistics && !m_getCollectPassStatistics()) {
        m_visiblePassIndices.clear();
        return m_visiblePassIndices;
    }

    RebuildVisiblePassIndices(m_defaultMaxStaleFrames, m_visiblePassIndices);
    return m_visiblePassIndices;
}

const std::vector<unsigned>& StatisticsManager::GetVisiblePassIndices(uint64_t maxStaleFrames) const {
    if (m_getCollectPassStatistics && !m_getCollectPassStatistics()) {
        m_visiblePassIndices.clear();
        return m_visiblePassIndices;
    }

    RebuildVisiblePassIndices(maxStaleFrames, m_visiblePassIndices);
    return m_visiblePassIndices;
}

void StatisticsManager::MarkGeometryPass(const std::string& passName) {
    auto it = std::find(m_passNames.begin(), m_passNames.end(), passName);
    if (it != m_passNames.end()) {
        m_isGeometryPass[it - m_passNames.begin()] = true;
    }
}

void StatisticsManager::RegisterQueue(rhi::QueueKind queueKind) {
    m_timestampBuffers[queueKind];
    m_meshStatsBuffers[queueKind];
    m_recordedQueries[queueKind];
    m_pendingResolves[queueKind];
    if (m_timestampPool && m_pipelineStatsPool) {
        EnsureQueueBuffers(queueKind);
    }
}

void StatisticsManager::EnsureQueueBuffers(rhi::QueueKind queueKind) {
    if (!m_timestampPool) {
        return;
    }

    auto device = DeviceManager::GetInstance().GetDevice();

    rhi::ResourceDesc tsRb = rhi::helpers::ResourceDesc::Buffer(
        static_cast<uint64_t>(m_timestampQueryInfo.elementSize) * m_timestampQueryInfo.count,
        rhi::HeapType::Readback);
    auto result = device.CreateCommittedResource(tsRb, m_timestampBuffers[queueKind]);
    if (m_pipelineStatsPool) {
        rhi::ResourceDesc psRb = rhi::helpers::ResourceDesc::Buffer(
            static_cast<uint64_t>(m_pipelineStatsQueryInfo.elementSize) * m_pipelineStatsQueryInfo.count,
            rhi::HeapType::Readback);
        result = device.CreateCommittedResource(psRb, m_meshStatsBuffers[queueKind]);
    }
    (void)result;
}

void StatisticsManager::SetupQueryHeap() {
    auto device = DeviceManager::GetInstance().GetDevice();
    if (!m_collectPassStatistics) {
        return;
    }

    if (m_numPasses == 0) {
        return;
    }

    if (m_queryPoolPassCapacity >= m_numPasses && m_timestampPool && (m_pipelineStatsPool || !m_collectPipelineStatistics)) {
        return;
    }

    m_queryPoolPassCapacity = m_numPasses;

    // Defer destruction of old pools/buffers so the GPU can finish in-flight work.
    auto& deletionMgr = DeletionManager::GetInstance();
    if (m_timestampPool) {
        deletionMgr.MarkForDelete(std::move(m_timestampPool));
    }
    if (m_pipelineStatsPool) {
        deletionMgr.MarkForDelete(std::move(m_pipelineStatsPool));
    }
    for (auto& kv : m_timestampBuffers) {
        if (kv.second) {
            deletionMgr.MarkForDelete(std::move(kv.second));
        }
    }
    for (auto& kv : m_meshStatsBuffers) {
        if (kv.second) {
            deletionMgr.MarkForDelete(std::move(kv.second));
        }
    }

    // Timestamp heap: 2 queries/pass/frame. The pass timings need nothing else.
    rhi::QueryPoolDesc tq;
    tq.type = rhi::QueryType::Timestamp;
    tq.count = m_queryPoolPassCapacity * 2 * m_numFramesInFlight;
    if (device.CreateQueryPool(tq, m_timestampPool) != rhi::Result::Ok || !m_timestampPool) {
        spdlog::warn("StatisticsManager: could not create a timestamp query pool for {} passes; pass GPU timings are off", m_queryPoolPassCapacity);
        m_timestampPool = {};
        m_collectPassStatistics = false;
        return;
    }

    // Mesh-shader pipeline statistics are optional. A device without mesh-shader queries (or an adopted
    // device that did not enable them) fails this pool, and until now the null pool was dereferenced right
    // after; the timestamps must not go down with it.
    if (m_collectPipelineStatistics) {
        rhi::QueryPoolDesc sq;
        sq.type = rhi::QueryType::PipelineStatistics;
        sq.count = m_queryPoolPassCapacity * m_numFramesInFlight;
        sq.statsMask = rhi::PipelineStatBits::PS_MeshInvocations | rhi::PipelineStatBits::PS_MeshPrimitives;
        if (device.CreateQueryPool(sq, m_pipelineStatsPool) != rhi::Result::Ok || !m_pipelineStatsPool) {
            spdlog::info("StatisticsManager: mesh pipeline statistics unavailable on this device; collecting pass timestamps only");
            m_pipelineStatsPool = {};
            m_collectPipelineStatistics = false;
        }
    }

    m_timestampQueryInfo = m_timestampPool->GetQueryResultInfo();
    if (m_pipelineStatsPool) {
        m_pipelineStatsQueryInfo = m_pipelineStatsPool->GetQueryResultInfo();
        m_pipelineStatsFields.resize(2);
        m_pipelineStatsFields[0].field = rhi::PipelineStatTypes::MeshInvocations;
        m_pipelineStatsFields[1].field = rhi::PipelineStatTypes::MeshPrimitives;
        m_pipelineStatsLayout = m_pipelineStatsPool->GetPipelineStatsLayout(m_pipelineStatsFields.data(), static_cast<uint32_t>(m_pipelineStatsFields.size()));
    }

    // Readback buffers for each registered queue.
    for (auto& kv : m_timestampBuffers) {
        EnsureQueueBuffers(kv.first);
    }

    m_recordedQueries.clear();
    m_pendingResolves.clear();
    for (auto& kv : m_timestampBuffers) {
        const auto qk = kv.first;
        m_recordedQueries[qk];
        m_pendingResolves[qk];
    }
}

void StatisticsManager::BeginQuery(
    unsigned passIndex,
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd)
{
    if (!m_collectPassStatistics) return;
    if (!m_timestampPool || passIndex >= m_numPasses || passIndex >= m_queryPoolPassCapacity) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;

    // Timestamp "begin" marker = write a timestamp at index 2*N
    const uint32_t tsIdx = (frameBase + passIndex) * 2u;
    cmd.ResetQueries(m_timestampPool->GetHandle(), tsIdx, 1);
    cmd.WriteTimestamp(m_timestampPool->GetHandle(), tsIdx, rhi::Stage::Top); // RHI: EndQuery on a Timestamp pool writes a timestamp

    // Begin pipeline stats for geometry passes
    if (m_collectPipelineStatistics && m_isGeometryPass[passIndex]) {
        const uint32_t psIdx = frameBase + passIndex;
        cmd.ResetQueries(m_pipelineStatsPool->GetHandle(), psIdx, 1);
        cmd.BeginQuery(m_pipelineStatsPool->GetHandle(), psIdx);
    }
    m_recordedQueries[queueKind][frameIndex].push_back(tsIdx);
}

void StatisticsManager::EndQuery(
    unsigned passIndex,
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd)
{
    if (!m_collectPassStatistics) return;
    if (!m_timestampPool || passIndex >= m_numPasses || passIndex >= m_queryPoolPassCapacity) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;

    // Timestamp "end" marker = write a timestamp at index 2*N + 1
    const uint32_t tsIdx = (frameBase + passIndex) * 2u + 1u;
    cmd.ResetQueries(m_timestampPool->GetHandle(), tsIdx, 1);
    cmd.WriteTimestamp(m_timestampPool->GetHandle(), tsIdx, rhi::Stage::Bottom); // RHI: EndQuery on a Timestamp pool writes a timestamp

    // End pipeline stats for geometry passes
    if (m_collectPipelineStatistics && m_isGeometryPass[passIndex]) {
        const uint32_t psIdx = frameBase + passIndex;
        cmd.EndQuery(m_pipelineStatsPool->GetHandle(), psIdx);
    }
    m_recordedQueries[queueKind][frameIndex].push_back(tsIdx);
}

void StatisticsManager::ResolveQueries(
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd)
{
    if (!m_collectPassStatistics) return;
    if (!m_timestampPool || m_timestampQueryInfo.elementSize == 0) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;
    auto psIt = m_meshStatsBuffers.find(queueKind);
    const bool havePipelineStats = m_collectPipelineStatistics && psIt != m_meshStatsBuffers.end() && psIt->second;

    auto& rec = m_recordedQueries[queueKind][frameIndex];
    if (rec.empty()) return;

    // Collapse timestamp indices into contiguous ranges
    std::sort(rec.begin(), rec.end());
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    uint32_t start = rec[0], prev = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        if (rec[i] == prev + 1) {
            prev = rec[i];
        }
        else {
            ranges.emplace_back(start, prev - start + 1);
            start = rec[i];
            prev = rec[i];
        }
    }
    ranges.emplace_back(start, prev - start + 1);

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;
    const uint64_t tsStride = m_timestampQueryInfo.elementSize; // usually 8
    const uint64_t psStride = m_pipelineStatsQueryInfo.elementSize; // backend-dependent

    auto& tsBuf = tsIt->second;   // rhi::ResourcePtr

    // Resolve timestamp data and remember what to read on frame complete
    for (auto& r : ranges) {
        // Write timestamp results starting at byte offset = stride * firstIndex
        cmd.ResolveQueryData(
            m_timestampPool->GetHandle(),
            r.first, r.second,
            tsBuf->GetHandle(),
            tsStride * uint64_t(r.first)
        );

        m_pendingResolves[queueKind][frameIndex].push_back(r);

        if (!havePipelineStats) continue;

        // For each stamped pass in this range, resolve pipeline stats if it's a geometry pass
        for (uint32_t idx = r.first; idx < r.first + r.second; idx += 2) {
            const uint32_t encoded = idx / 2; // frameBase + passIndex
            if (encoded < frameBase) {
                continue;
            }
            const uint32_t pi = encoded - frameBase;
            if (pi >= m_numPasses) {
                continue;
            }
            if (!m_isGeometryPass[pi]) continue;

            const uint32_t psIdx = frameBase + pi;

            cmd.ResolveQueryData(
                m_pipelineStatsPool->GetHandle(),
                psIdx, 1,
                psIt->second->GetHandle(),
                psStride * uint64_t(psIdx)
            );
        }
    }

    rec.clear();
}

void StatisticsManager::BeginQuery(
    unsigned passIndex,
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd,
    QueryRecordingContext& ctx)
{
    if (!m_collectPassStatistics) return;
    if (ctx.owned) {
        if (passIndex < m_passNames.size())
            BeginOwnedRange(*ctx.owned, m_passNames[passIndex], static_cast<int>(passIndex), org::runtime::TimestampRangeKind::Pass, cmd);
        return;
    }
    if (!m_timestampPool || passIndex >= m_numPasses || passIndex >= m_queryPoolPassCapacity) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;
    const uint32_t tsIdx = (frameBase + passIndex) * 2u;
    cmd.ResetQueries(m_timestampPool->GetHandle(), tsIdx, 1);
    cmd.WriteTimestamp(m_timestampPool->GetHandle(), tsIdx, rhi::Stage::Top);

    if (m_collectPipelineStatistics && m_isGeometryPass[passIndex]) {
        const uint32_t psIdx = frameBase + passIndex;
        cmd.ResetQueries(m_pipelineStatsPool->GetHandle(), psIdx, 1);
        cmd.BeginQuery(m_pipelineStatsPool->GetHandle(), psIdx);
    }
    ctx.recordedIndices.push_back(tsIdx);
}

void StatisticsManager::EndQuery(
    unsigned passIndex,
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd,
    QueryRecordingContext& ctx)
{
    if (!m_collectPassStatistics) return;
    if (ctx.owned) {
        EndOwnedRange(*ctx.owned, cmd);
        return;
    }
    if (!m_timestampPool || passIndex >= m_numPasses || passIndex >= m_queryPoolPassCapacity) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;
    const uint32_t tsIdx = (frameBase + passIndex) * 2u + 1u;
    cmd.ResetQueries(m_timestampPool->GetHandle(), tsIdx, 1);
    cmd.WriteTimestamp(m_timestampPool->GetHandle(), tsIdx, rhi::Stage::Bottom);

    if (m_collectPipelineStatistics && m_isGeometryPass[passIndex]) {
        const uint32_t psIdx = frameBase + passIndex;
        cmd.EndQuery(m_pipelineStatsPool->GetHandle(), psIdx);
    }
    ctx.recordedIndices.push_back(tsIdx);
}

void StatisticsManager::ResolveQueries(
    unsigned frameIndex,
    rhi::Queue& queue,
    rhi::CommandList& cmd,
    QueryRecordingContext& ctx)
{
    if (!m_collectPassStatistics) return;
    if (ctx.owned) {
        // Every range the list recorded, into the recording's own buffer: each submission resolves them again.
        auto& owned = *ctx.owned;
        EndOwnedRange(owned, cmd);
        if (!owned.resolved && !owned.entries.empty())
            cmd.ResolveQueryData(owned.pool->GetHandle(), 0, static_cast<uint32_t>(owned.entries.size() * 2u), owned.readback->GetHandle(), 0);
        owned.resolved = true;
        return;
    }
    if (!m_timestampPool || m_timestampQueryInfo.elementSize == 0) return;

    auto queueKind = queue.GetKind();
    auto tsIt = m_timestampBuffers.find(queueKind);
    if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;
    auto psIt = m_meshStatsBuffers.find(queueKind);
    const bool havePipelineStats = m_collectPipelineStatistics && psIt != m_meshStatsBuffers.end() && psIt->second;

    auto& rec = ctx.recordedIndices;
    if (rec.empty()) return;

    std::sort(rec.begin(), rec.end());
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    uint32_t start = rec[0], prev = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        if (rec[i] == prev + 1) {
            prev = rec[i];
        } else {
            ranges.emplace_back(start, prev - start + 1);
            start = rec[i];
            prev = rec[i];
        }
    }
    ranges.emplace_back(start, prev - start + 1);

    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;
    const uint64_t tsStride = m_timestampQueryInfo.elementSize;
    const uint64_t psStride = m_pipelineStatsQueryInfo.elementSize;

    auto& tsBuf = tsIt->second;

    for (auto& r : ranges) {
        cmd.ResolveQueryData(
            m_timestampPool->GetHandle(),
            r.first, r.second,
            tsBuf->GetHandle(),
            tsStride * uint64_t(r.first)
        );

        ctx.pendingRanges.push_back(r);

        if (!havePipelineStats) continue;

        for (uint32_t idx = r.first; idx < r.first + r.second; idx += 2) {
            const uint32_t encoded = idx / 2;
            if (encoded < frameBase) continue;
            const uint32_t pi = encoded - frameBase;
            if (pi >= m_numPasses) continue;
            if (!m_isGeometryPass[pi]) continue;

            const uint32_t psIdx = frameBase + pi;
            cmd.ResolveQueryData(
                m_pipelineStatsPool->GetHandle(),
                psIdx, 1,
                psIt->second->GetHandle(),
                psStride * uint64_t(psIdx)
            );
        }
    }

    rec.clear();
}

void StatisticsManager::MergePendingResolves(
    rhi::QueueKind queueKind,
    unsigned frameIndex,
    QueryRecordingContext& ctx)
{
    if (ctx.owned) {
        // A submission of a kept recording: its slot's completion reads what this one wrote. Never cleared from the
        // context; a slot's pending list is bounded in case a build stopped its completion from coming.
        if (ctx.owned->resolved && !ctx.owned->entries.empty()) {
            auto& pending = m_pendingOwned[frameIndex];
            if (std::ranges::find(pending, ctx.owned) == pending.end() && pending.size() < 256)
                pending.push_back(ctx.owned);
        }
        return;
    }
    auto& dest = m_pendingResolves[queueKind][frameIndex];
    dest.insert(dest.end(), ctx.pendingRanges.begin(), ctx.pendingRanges.end());
    ctx.pendingRanges.clear();
}

void StatisticsManager::OnFrameComplete(
    unsigned frameIndex,
    rhi::Queue& queue)
{
    m_completedTimestamps.clear();
    m_completedOwned.clear();
    ReadOwnedTimestamps(frameIndex);
	if (!m_timestampPool || m_timestampQueryInfo.elementSize == 0) return;

	if (!m_collectPassStatistics) return;
	auto queueKind = queue.GetKind();
	auto tsIt = m_timestampBuffers.find(queueKind);
	auto psIt = m_meshStatsBuffers.find(queueKind);
	if (tsIt == m_timestampBuffers.end() || !tsIt->second) return;
	const bool havePipelineStats = m_collectPipelineStatistics && psIt != m_meshStatsBuffers.end() && psIt->second;

    auto& tsBuf = tsIt->second;
    auto& pending = m_pendingResolves[queueKind][frameIndex];
    if (pending.empty()) return;

    const uint64_t tsStride = m_timestampQueryInfo.elementSize; // usually 8
    const uint64_t psStride = m_pipelineStatsQueryInfo.elementSize; // backend-specific
    const uint32_t frameBase = (frameIndex % (std::max)(1u, m_numFramesInFlight)) * m_queryPoolPassCapacity;
    const double   toMs = 1000.0 / double(m_gpuTimestampFreq);

    auto readU64At = [](const uint8_t* base, uint64_t byteOffset) -> uint64_t {
        uint64_t v = 0;
        std::memcpy(&v, base + byteOffset, sizeof(uint64_t));
        return v;
        };

	// TODO: Avoid searching the field list every time
    auto findFieldOffset = [&](rhi::PipelineStatTypes f, uint32_t& off) -> bool {
        for (const auto& fd : m_pipelineStatsFields) {
            if (fd.field == f && fd.supported) { off = fd.byteOffset; return true; }
        }
        return false;
        };

    for (const auto& r : pending) {
        if (r.second < 2u) {
            continue;
        }

        // Map just the timestamp byte range we resolved this frame
        const uint64_t tsMapOffset = tsStride * uint64_t(r.first);
        const uint64_t tsMapSize = tsStride * uint64_t(r.second);

        void* tsPtrVoid = nullptr;
        tsBuf->Map(&tsPtrVoid, tsMapOffset, tsMapSize);
        if (tsPtrVoid == nullptr) {
            spdlog::warn(
                "StatisticsManager::OnFrameComplete: timestamp readback Map returned null queue={} frameIndex={} firstQuery={} queryCount={}",
                static_cast<int>(queueKind),
                frameIndex,
                r.first,
                r.second);
            continue;
        }
        const uint8_t* tsBase = static_cast<const uint8_t*>(tsPtrVoid);

        for (uint32_t idx = r.first; idx < r.first + r.second; idx += 2) {
            const uint32_t local0 = (idx - r.first);
            const uint32_t local1 = local0 + 1;
            if (local1 >= r.second) {
                spdlog::warn(
                    "StatisticsManager::OnFrameComplete: skipping incomplete timestamp pair queue={} frameIndex={} firstQuery={} queryCount={} localBegin={}",
                    static_cast<int>(queueKind),
                    frameIndex,
                    r.first,
                    r.second,
                    local0);
                break;
            }

            // Read the two timestamps (each element starts at localIndex * tsStride)
            const uint64_t t0 = readU64At(tsBase, uint64_t(local0) * tsStride);
            const uint64_t t1 = readU64At(tsBase, uint64_t(local1) * tsStride);
            const double   ms = double(t1 - t0) * toMs;

            const uint32_t encoded = idx / 2; // frameBase + passIndex
            if (encoded < frameBase) {
                continue;
            }
            const uint32_t pi = encoded - frameBase;
            if (pi >= m_numPasses) {
                continue;
            }

            UpdateEma(m_stats[pi].gpuTimeEma, ms);
            m_stats[pi].gpuTimeMs = ms;
            m_stats[pi].gpuSampleSerial = m_frameSerial;
            m_stats[pi].gpuBeginTick = t0;
            m_stats[pi].gpuEndTick = t1;
            if (pi < m_passLastExecutionFrame.size()) {
                m_passLastExecutionFrame[pi] = m_frameSerial;
            }
            m_completedTimestamps.push_back({m_passNames[pi], static_cast<int>(pi), 0, queueKind,
                org::runtime::TimestampRangeKind::Pass, false, t0, t1, ms});

            if (!havePipelineStats || !m_isGeometryPass[pi]) continue;
            auto& psBuf = psIt->second;

            // Map just this pass's pipeline stat element
            const uint32_t psIdx = frameBase + pi;
            const uint64_t psOffset = psStride * uint64_t(psIdx);
            void* psPtrVoid = nullptr;
            psBuf->Map(&psPtrVoid, psOffset, psStride);
            if (psPtrVoid == nullptr) {
                spdlog::warn(
                    "StatisticsManager::OnFrameComplete: pipeline-stat readback Map returned null queue={} frameIndex={} passIndex={} queryIndex={}",
                    static_cast<int>(queueKind),
                    frameIndex,
                    pi,
                    psIdx);
                continue;
            }

            const uint8_t* psBase = static_cast<const uint8_t*>(psPtrVoid);

            uint64_t inv = 0, prim = 0;
            uint32_t offInv = 0, offPrim = 0;

            if (findFieldOffset(rhi::PipelineStatTypes::MeshInvocations, offInv))
                inv = readU64At(psBase, offInv);
            if (findFieldOffset(rhi::PipelineStatTypes::MeshPrimitives, offPrim))
                prim = readU64At(psBase, offPrim);

            psBuf->Unmap(0, 0);

            auto& mps = m_meshStatsEma[pi];
            UpdateEma(mps.invocationsEma, double(inv));
            UpdateEma(mps.primitivesEma, double(prim));
        }

        tsBuf->Unmap(0, 0);
    }

    pending.clear();
    m_pendingResolves[queueKind].erase(frameIndex);
}

std::shared_ptr<OwnedTimestampQueries> StatisticsManager::CreateOwnedTimestamps(rhi::QueueKind queueKind, uint32_t ranges) {
    if (!m_collectPassStatistics || ranges == 0) return nullptr;
    auto device = DeviceManager::GetInstance().GetDevice();
    auto owned = std::make_shared<OwnedTimestampQueries>();
    rhi::QueryPoolDesc desc;
    desc.type = rhi::QueryType::Timestamp;
    desc.count = ranges * 2u;
    if (device.CreateQueryPool(desc, owned->pool) != rhi::Result::Ok || !owned->pool) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            spdlog::warn("StatisticsManager: could not create a kept recording's timestamp pool ({} ranges); its passes are not timed", ranges);
        return nullptr;
    }
    owned->stride = owned->pool->GetQueryResultInfo().elementSize;
    const auto readback = rhi::helpers::ResourceDesc::Buffer(owned->stride * desc.count, rhi::HeapType::Readback);
    if (owned->stride == 0 || device.CreateCommittedResource(readback, owned->readback) != rhi::Result::Ok || !owned->readback)
        return nullptr;
    const auto ticksPerSecond = device.GetTimestampCalibration(queueKind).ticksPerSecond;
    owned->ticksToMilliseconds = ticksPerSecond ? 1000.0 / double(ticksPerSecond) : GetGpuTicksToMilliseconds();
    owned->queueKind = queueKind;
    owned->capacity = ranges;
    owned->entries.reserve(ranges);
    return owned;
}

bool StatisticsManager::BeginOwnedRange(OwnedTimestampQueries& owned, std::string_view name, int passIndex,
    org::runtime::TimestampRangeKind kind, rhi::CommandList& cmd) {
    // Ranges nest (a pass's sub-ranges inside it); a pass or barrier range opens only at the top level.
    if (owned.resolved || (kind != org::runtime::TimestampRangeKind::Sub && !owned.open.empty())) return false;
    if (owned.entries.size() >= owned.capacity) {
        ++owned.overflow;
        return false;
    }
    // The list's first range resets them all: one reset per submission, outside any render pass.
    if (owned.entries.empty()) cmd.ResetQueries(owned.pool->GetHandle(), 0, owned.capacity * 2u);
    uint32_t occurrence = 0;
    for (const auto& entry : owned.entries) occurrence += entry.name == name;
    owned.entries.push_back({std::string(name), passIndex, kind, occurrence});
    const auto index = static_cast<uint32_t>(owned.entries.size() - 1);
    cmd.WriteTimestamp(owned.pool->GetHandle(), index * 2u, rhi::Stage::Top);
    owned.open.push_back(static_cast<int>(index));
    return true;
}

void StatisticsManager::EndOwnedRange(OwnedTimestampQueries& owned, rhi::CommandList& cmd) {
    if (owned.open.empty()) return;
    cmd.WriteTimestamp(owned.pool->GetHandle(), static_cast<uint32_t>(owned.open.back()) * 2u + 1u, rhi::Stage::Bottom);
    owned.open.pop_back();
}

void StatisticsManager::BeginRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
    if (m_collectPassStatistics && ctx.owned)
        BeginOwnedRange(*ctx.owned, name, -1, org::runtime::TimestampRangeKind::Barrier, cmdList);
}

bool StatisticsManager::BeginSubRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
    return m_collectPassStatistics && ctx.owned && !ctx.owned->open.empty() &&
           BeginOwnedRange(*ctx.owned, name, -1, org::runtime::TimestampRangeKind::Sub, cmdList);
}

void StatisticsManager::EndRangeQuery(rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
    if (ctx.owned) EndOwnedRange(*ctx.owned, cmdList);
}

void StatisticsManager::ReadOwnedTimestamps(unsigned frameIndex) {
    const auto found = m_pendingOwned.find(frameIndex);
    if (found == m_pendingOwned.end()) return;
    auto submitted = std::move(found->second);
    m_pendingOwned.erase(found);
    for (auto& owned : submitted) {
        const auto queries = static_cast<uint64_t>(owned->entries.size()) * 2u;
        void* mapped = nullptr;
        owned->readback->Map(&mapped, 0, owned->stride * queries);
        if (!mapped) continue;
        const auto* base = static_cast<const uint8_t*>(mapped);
        for (size_t i = 0; i < owned->entries.size(); ++i) {
            uint64_t t0 = 0, t1 = 0;
            std::memcpy(&t0, base + owned->stride * (2u * i), sizeof(uint64_t));
            std::memcpy(&t1, base + owned->stride * (2u * i + 1u), sizeof(uint64_t));
            if (!t0 || t1 < t0) continue;
            const auto& entry = owned->entries[i];
            const double ms = double(t1 - t0) * owned->ticksToMilliseconds;
            m_completedTimestamps.push_back({entry.name, entry.passIndex, entry.occurrence, owned->queueKind, entry.kind, true, t0, t1, ms});
            if (entry.passIndex < 0 || static_cast<size_t>(entry.passIndex) >= m_stats.size()) continue;
            auto& stats = m_stats[entry.passIndex];
            UpdateEma(stats.gpuTimeEma, ms);
            stats.gpuTimeMs = ms;
            stats.gpuSampleSerial = m_frameSerial;
            stats.gpuBeginTick = t0;
            stats.gpuEndTick = t1;
            if (static_cast<size_t>(entry.passIndex) < m_passLastExecutionFrame.size())
                m_passLastExecutionFrame[entry.passIndex] = m_frameSerial;
        }
        owned->readback->Unmap(0, 0);
        m_completedOwned.push_back(std::move(owned));
    }
}

void StatisticsManager::ClearAll() {
    m_timestampPool.Reset();
    m_pipelineStatsPool.Reset();
    for (auto& kv:m_timestampBuffers) kv.second.Reset();
    for (auto& kv:m_meshStatsBuffers) kv.second.Reset();
    m_timestampBuffers.clear();
    m_meshStatsBuffers.clear();
    m_passNames.clear();
    m_passTechniquePaths.clear();
    m_stats.clear();
    m_isGeometryPass.clear();
    m_meshStatsEma.clear();
    m_passNameToIndex.clear();
    m_passLastExecutionFrame.clear();
    m_visiblePassIndices.clear();
    m_recordedQueries.clear();
    m_pendingResolves.clear();
    m_pendingOwned.clear();
    m_completedOwned.clear();
    m_completedTimestamps.clear();
    m_numPasses=0;
    m_queryPoolPassCapacity = 0;
    m_unnamedPassCounter = 0;
    m_frameSerial = 0;
    m_collectPassStatistics = true;
    m_getCollectPassStatistics = {};
    m_collectPipelineStatistics = false;
    m_getCollectPipelineStatistics = {};
    m_memoryBudgetStats = {};
}


} // namespace org

namespace org::runtime {

OwnedTimestampQueries::~OwnedTimestampQueries() {
    // A recording is released once its submissions are complete; the deferred deletion keeps the destruction off the thread that
    // drops it, as for the shared pool.
    auto& deletion = org::DeletionManager::GetInstance();
    if (pool) deletion.MarkForDelete(std::move(pool));
    if (readback) deletion.MarkForDelete(std::move(readback));
}

} // namespace org::runtime
