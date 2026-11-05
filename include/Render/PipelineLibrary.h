#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "Render/PipelineDesc.h"

namespace org::runtime { class ITaskScope; class ITaskService; }

namespace org {

enum class PipelineBuildStatus : uint8_t { Idle, Building, Ready, Failed };

// A pipeline of the library: its description, its handle and its build on each device. Declarations hold it, so it lives as
// long as a pass declares it.
class PipelineLibraryEntry {
public:
	const PipelineDesc& Desc() const noexcept { return m_desc; }
	const std::string& Name() const noexcept { return m_name; }
	const PipelineState& State() const noexcept { return m_state; }
	uint64_t Key() const noexcept { return m_key; }
	PipelineBuildStatus Status(DeviceInstanceId device) const noexcept;
	// Why the device's build failed. Empty unless Status is Failed.
	std::string Error(DeviceInstanceId device) const;

private:
	friend class PipelineLibrary;
	struct DeviceBuild {
		std::atomic<uint8_t> status{static_cast<uint8_t>(PipelineBuildStatus::Idle)};
		std::string error; // Written before a Failed status is published.
	};
	PipelineDesc m_desc;
	std::string m_name;
	uint64_t m_key = 0;
	PipelineState m_state;
	std::array<DeviceBuild, kMaxPipelineDevices> m_builds;
	std::atomic<uint64_t> m_rebuildTicket{0}; // A newer Rebuild supersedes one still in flight.
	struct Generation {
		uint64_t generation = 0;
		std::array<std::shared_ptr<PipelineStatePayload>, kMaxPipelineDevices> payloads;
	};
	std::deque<Generation> m_history; // The retained generations, oldest first. Guarded by the library's mutex.
};
using PipelineLibraryEntryPtr = std::shared_ptr<PipelineLibraryEntry>;

class CommandSignatureLibraryEntry {
public:
	const CommandSignatureDesc& Desc() const noexcept { return m_desc; }
	const CommandSignatureState& State() const noexcept { return m_state; }

private:
	friend class PipelineLibrary;
	CommandSignatureDesc m_desc;
	CommandSignatureState m_state;
};
using CommandSignatureLibraryEntryPtr = std::shared_ptr<CommandSignatureLibraryEntry>;

struct PipelineRebuildOptions {
	std::vector<PipelineShaderDefine> defineOverrides; // Replace a define of the same name, or add one.
	bool bypassCache = true;
	std::string label; // The payloads' label. Empty: "generation-N" ("initial" for a first build).
	// A rebuild the host follows by declaring the passes again (a global reload rebuilds its graphs), so the new shader's
	// resource interface can differ.
	bool allowInterfaceChange = false;
	// Once every device's build ended, on the last build's thread. On success, the result publishes at the next PublishRebuilt.
	// Empty error: success. A superseded rebuild reports nothing.
	std::function<void(const std::string& error)> completed;
};

// Builds pipelines from descriptions, for any device, on the task service. One per Runtime.
//
// Handles are valid from Acquire. A device's payload appears in the handle when its build ends, lock-free, so a recording
// reads it without coordinating with a build. RenderGraph requests each declared pipeline for the device lowering chose and
// does not publish the pass until it is ready.
class PipelineLibrary {
public:
	PipelineLibrary(std::shared_ptr<runtime::ITaskService> tasks, std::shared_ptr<PipelineCompiler> compiler);
	PipelineLibrary(const PipelineLibrary&) = delete;
	PipelineLibrary& operator=(const PipelineLibrary&) = delete;
	// Joins every build.
	~PipelineLibrary();

	// A device pipelines can be built on. Registering one again is harmless.
	void RegisterDevice(DeviceInstanceId id, rhi::Backend backend, rhi::Device device);
	bool HasCompiler() const noexcept { return m_compiler != nullptr; }

	// The pipeline with this description; the same one while it is alive. Compiles nothing. Throws without a compiler.
	PipelineLibraryEntryPtr Acquire(PipelineDesc desc);
	// Starts the device's build unless one started already. A failed build stays failed until a Rebuild succeeds.
	void Request(const PipelineLibraryEntryPtr& entry, DeviceInstanceId device);
	// Blocks until the device's build ended. Idle when it was never requested.
	PipelineBuildStatus Wait(const PipelineLibraryEntryPtr& entry, DeviceInstanceId device) const;

	// The command signature with this description; the same one while it is alive. Creates nothing.
	CommandSignatureLibraryEntryPtr AcquireCommandSignature(CommandSignatureDesc desc);
	// Creates it for the device unless it exists. Quick: nothing compiles. Throws on failure.
	void EnsureCommandSignature(const CommandSignatureLibraryEntryPtr& entry, DeviceInstanceId device);

	// Live entries, for a host's reload (which shader changed is its knowledge).
	std::vector<PipelineLibraryEntryPtr> LiveEntries();
	// Builds the pipeline again on every device it was requested for. Nothing changes until PublishRebuilt, and on all of its
	// devices at once; a failure on any keeps the old payloads and reports through options.completed. So does a changed resource
	// interface: declarations were lowered against the shader's descriptor slots, so that takes a new pipeline, not a rebuild.
	// Returns the generation it will publish as, the same on every device.
	uint64_t Rebuild(const PipelineLibraryEntryPtr& entry, PipelineRebuildOptions options);
	// Queues a retained generation to become active at the next PublishRebuilt. False when it is no longer retained.
	bool Activate(const PipelineLibraryEntryPtr& entry, uint64_t generation);
	std::vector<uint64_t> RetainedGenerations(const PipelineLibraryEntryPtr& entry);
	// Publishes the finished rebuilds and activations into their handles; returns how many. Recordings retain what they bound,
	// so the old payloads go when the last one retires. Called between frames by the host or graph.
	size_t PublishRebuilt();

private:
	struct DeviceInfo {
		rhi::Backend backend = rhi::Backend::Null;
		rhi::Device device{};
	};
	struct FinishedRebuild {
		PipelineLibraryEntryPtr entry;
		uint64_t ticket = 0;
		uint64_t generation = 0;
		bool activation = false; // A retained generation again: already in the history, and no build to be superseded.
		std::array<std::shared_ptr<PipelineStatePayload>, kMaxPipelineDevices> payloads;
	};
	using BuildCompletion = std::function<void(std::shared_ptr<PipelineStatePayload>, std::string error)>;

	DeviceInfo FindDevice(DeviceInstanceId id);
	void StartBuild(PipelineLibraryEntryPtr entry, DeviceInstanceId id, std::vector<PipelineShaderDefine> overrides,
		bool bypassCache, uint64_t generation, std::string label, BuildCompletion completion);
	std::shared_ptr<PipelineStatePayload> CreatePipeline(const PipelineLibraryEntry& entry, const PipelineCompileRequest& request,
		const DeviceInfo& device, const CompiledPipeline& compiled, uint64_t generation, std::string label);
	PipelineLayoutBinding ResolveLayout(const PipelineLayoutRef& layout, DeviceInstanceId id, const DeviceInfo& device);

	std::shared_ptr<runtime::ITaskService> m_tasks;
	std::shared_ptr<PipelineCompiler> m_compiler;
	std::shared_ptr<runtime::ITaskScope> m_scope;
	std::atomic<uint32_t> m_compilesInFlight{0};

	std::mutex m_mutex; // Declarations and device registration, and the build's start. Never held across a compile.
	std::array<DeviceInfo, kMaxPipelineDevices> m_devices;
	std::unordered_map<uint64_t, std::weak_ptr<PipelineLibraryEntry>> m_entries;
	std::unordered_map<uint64_t, std::weak_ptr<CommandSignatureLibraryEntry>> m_commandSignatures;
	std::map<std::pair<uint8_t, uint64_t>, std::weak_ptr<rhi::PipelineLayoutPtr>> m_customLayouts;

	std::atomic<bool> m_rebuildsFinished{false};
	std::vector<FinishedRebuild> m_finishedRebuilds; // Guarded by m_mutex.
};

// A description's identity: everything that affects the pipeline.
uint64_t PipelineContentHash(const PipelineDesc& desc);
uint64_t CommandSignatureContentHash(const CommandSignatureDesc& desc);

} // namespace org
