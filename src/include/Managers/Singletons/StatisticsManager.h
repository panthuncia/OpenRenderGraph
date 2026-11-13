#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <functional>
#include <limits>
#include <mutex>
#include <atomic>
#include <memory>
#include <rhi.h>
#include <rhi_helpers.h>
#include "Render/Runtime/IStatisticsService.h"


namespace org {

using QueryRecordingContext = org::runtime::QueryRecordingContext;

using PassStats = org::runtime::PassStats;
using MeshPipelineStats = org::runtime::MeshPipelineStats;
using OwnedTimestampQueries = org::runtime::OwnedTimestampQueries;
using CompletedTimestamp = org::runtime::CompletedTimestamp;

class StatisticsManager {
public:
	static StatisticsManager& GetInstance();

	void Initialize();

	void RegisterPasses(const std::vector<std::string>& passNames);
    
	unsigned RegisterPass(const std::string& passName, bool isGeometryPass = false, std::string_view techniquePath = {});

	void BeginFrame();

	void MarkGeometryPass(const std::string& passName);

	void RegisterQueue(rhi::QueueKind queue);

	void SetupQueryHeap();
	// Query ranges for frame indices 0 .. frames - 1 (a host's run leases): the pools are made again, larger, when they hold
	// fewer. Call it from the thread that records with the shared pools.
	void EnsureFrameCapacity(unsigned frames);
	unsigned FrameCapacity() const { return m_numFramesInFlight; }

	// Timestamp + mesh-stats queries for any pass
	void BeginQuery(unsigned passIndex,
		unsigned frameIndex,
		rhi::Queue& queue,
		rhi::CommandList& cmdList);

	void EndQuery(unsigned passIndex,
		unsigned frameIndex,
		rhi::Queue& queue,
		rhi::CommandList& cmdList);

	// Resolve all queries for a frame before closing
	void ResolveQueries(unsigned frameIndex,
		rhi::Queue& queue,
		rhi::CommandList& cmdList);

	// Thread-safe overloads that use a per-task recording context.
	void BeginQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	void EndQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	void ResolveQueries(unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	void MergePendingResolves(rhi::QueueKind queueKind, unsigned frameIndex, QueryRecordingContext& ctx);
	void PublishAtCompletion(unsigned frameIndex, std::function<void()> publish);

	// Kept recordings' own queries (IStatisticsService::CreateOwnedTimestamps).
	std::shared_ptr<OwnedTimestampQueries> CreateOwnedTimestamps(rhi::QueueKind queueKind, uint32_t ranges);
	void BeginRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	void EndRangeQuery(rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	bool BeginSubRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx);
	const std::vector<CompletedTimestamp>& GetCompletedTimestamps() const { return m_completedTimestamps; }

	void OnFrameComplete(unsigned frameIndex,
		rhi::Queue& queue);
	void RecordCpuUpdateTime(unsigned passIndex, double milliseconds);
	void RecordCpuExecuteTime(unsigned passIndex, double milliseconds);

	void ClearAll();

	const std::vector<bool>& GetIsGeometryPassVector() const { return m_isGeometryPass; }
	const std::vector<unsigned>& GetVisiblePassIndices() const;
	const std::vector<unsigned>& GetVisiblePassIndices(uint64_t maxStaleFrames) const;

	const std::vector<std::string>&        GetPassNames() const { return m_passNames; }
	const std::vector<std::string>&        GetPassTechniquePaths() const { return m_passTechniquePaths; }
	const std::vector<PassStats>&          GetPassStats() const { return m_stats; }
	uint64_t GetFrameSerial() const { return m_frameSerial; }
	double GetGpuTicksToMilliseconds() const { return m_gpuTimestampFreq ? 1000.0 / double(m_gpuTimestampFreq) : 0.0; }
	const std::vector<MeshPipelineStats>&  GetMeshStats() const { return m_meshStatsEma; }
	org::runtime::MemoryBudgetStats GetMemoryBudgetStats() const { return m_memoryBudgetStats; }

private:
	StatisticsManager() = default;
	~StatisticsManager() = default;

	void EnsureQueueBuffers(rhi::QueueKind queueKind);
	bool BeginOwnedRange(OwnedTimestampQueries& owned, std::string_view name, int passIndex, org::runtime::TimestampRangeKind kind, rhi::CommandList& cmd);
	void EndOwnedRange(OwnedTimestampQueries& owned, rhi::CommandList& cmd);
	void ReadOwnedTimestamps(unsigned frameIndex);
	void RecordCpuTimeSample(unsigned passIndex, double milliseconds, bool isUpdate);

	bool m_collectPassStatistics = true;
	std::function<bool()> m_getCollectPassStatistics;
	bool m_collectPipelineStatistics = false;
	std::function<bool()> m_getCollectPipelineStatistics;
	std::mutex m_cpuStatsMutex;

	// The shared pools and readback buffers of one query heap size. Every list that writes or resolves them holds the
	// current one (QueryRecordingContext::sharedQueries). SetupQueryHeap moves the objects it replaces into it, and it
	// hands them to the deletion queue when the last list holding it is released. A deletion queue's frontier covers
	// only the queues ORG submits to, not a host's command stream (in-stream epochs), so it cannot be the only guard.
	struct SharedQueryObjects {
		std::vector<rhi::helpers::AnyObjectPtr> replaced;
		~SharedQueryObjects();
	};
	std::shared_ptr<const void> CurrentSharedQueries() const { return m_sharedQueries.load(std::memory_order_acquire); }
	std::atomic<std::shared_ptr<SharedQueryObjects>> m_sharedQueries{std::make_shared<SharedQueryObjects>()};

	rhi::QueryPoolPtr m_timestampPool;
	rhi::QueryPoolPtr m_pipelineStatsPool;
	rhi::QueryResultInfo m_timestampQueryInfo;
	rhi::QueryResultInfo m_pipelineStatsQueryInfo;
	std::vector<rhi::PipelineStatsFieldDesc> m_pipelineStatsFields;
	rhi::PipelineStatsLayout m_pipelineStatsLayout;

	std::unordered_map<rhi::QueueKind, rhi::ResourcePtr> m_timestampBuffers;
	std::unordered_map<rhi::QueueKind, rhi::ResourcePtr> m_meshStatsBuffers;

	UINT64    m_gpuTimestampFreq = 0;
	unsigned  m_numPasses = 0;
	unsigned  m_numFramesInFlight = 0;
	unsigned  m_queryPoolPassCapacity = 0;
	uint64_t  m_unnamedPassCounter = 0;
	std::unordered_map<std::string, unsigned> m_passNameToIndex;
	uint64_t  m_frameSerial = 0;
	static constexpr uint64_t kNeverSeenFrame = (std::numeric_limits<uint64_t>::max)();
	uint64_t  m_defaultMaxStaleFrames = 240;
	std::vector<uint64_t> m_passLastExecutionFrame;
	mutable std::vector<unsigned> m_visiblePassIndices;

	void RebuildVisiblePassIndices(uint64_t maxStaleFrames, std::vector<unsigned>& out) const;

	// Per-pass data
	std::vector<std::string>        m_passNames;
	std::vector<std::string>        m_passTechniquePaths;
	std::vector<PassStats>          m_stats;
	std::vector<bool>               m_isGeometryPass;
	std::vector<MeshPipelineStats>  m_meshStatsEma;
	org::runtime::MemoryBudgetStats m_memoryBudgetStats{};

	// Recording helpers per queue/frame
	std::unordered_map<rhi::QueueKind,
		std::unordered_map<unsigned, std::vector<unsigned>>> m_recordedQueries;
	std::unordered_map<rhi::QueueKind,
		std::unordered_map<unsigned, std::vector<std::pair<unsigned,unsigned>>>> m_pendingResolves;

	// The queries of kept recordings submitted for each frame slot. They are read when the slot completes.
	// m_completedOwned and m_completedTimestamps hold what the latest completion read. Its owned queries stay alive with
	// it, because the timestamps view their names.
	std::unordered_map<unsigned, std::vector<std::shared_ptr<OwnedTimestampQueries>>> m_pendingOwned;
	// The callbacks given to PublishAtCompletion, by frame slot. The slot's next OnFrameComplete runs them first. Each
	// list has a size limit, like m_pendingOwned.
	std::unordered_map<unsigned, std::vector<std::function<void()>>> m_publishAtCompletion;
	std::vector<std::shared_ptr<OwnedTimestampQueries>> m_completedOwned;
	std::vector<CompletedTimestamp> m_completedTimestamps;
};


} // namespace org
