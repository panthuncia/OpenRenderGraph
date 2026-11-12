#pragma once

// What a graph extension (RenderGraph::RegisterExtension) and the passes it contributes are described with. Kept apart from
// RenderGraph.h so that extension headers need only a forward declaration of the graph.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Render/QueueKind.h"

namespace org {

class RenderGraph;
class RenderGraphPass;
class RenderPass;
class ResourceRegistry;

enum class ExternalInsertKind : uint8_t { Begin, End, Before, After };

struct ExternalInsertPoint {
	int priority = 0;

	// default: preserve extension-local emission order
	bool keepExtensionOrder = true;

	std::vector<std::string> after;   // anchor keys that must precede this pass
	std::vector<std::string> before;  // anchor keys that must follow this pass

	static ExternalInsertPoint Begin(int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.before.push_back("__rg_begin__"); return p; }
	static ExternalInsertPoint End(int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.after.push_back("__rg_end__");   return p; }

	static ExternalInsertPoint After(std::string a, int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.after.push_back(std::move(a)); return p; }
	static ExternalInsertPoint Before(std::string a, int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.before.push_back(std::move(a)); return p; }

	static ExternalInsertPoint Between(std::string a, std::string b, int prio = 0) {
		ExternalInsertPoint p; p.priority = prio;
		p.after.push_back(std::move(a));
		p.before.push_back(std::move(b));
		return p;
	}

	ExternalInsertPoint& AlsoAfter(std::string a) { after.push_back(std::move(a)); return *this; }
	ExternalInsertPoint& AlsoBefore(std::string a) { before.push_back(std::move(a)); return *this; }
};

// Every pass is a RenderPass; its queue is declaration data (ExternalPassDesc::preferredQueueKind, PassBuilder::PreferQueue).
enum class PassType {
	Unknown,
	Render
};

struct ExternalPassDesc {
	PassType type = PassType::Unknown;
	// Made by Copy(), whatever queue it then prefers (ORG_PERSISTENT_COPIES_ON_COPY_QUEUE).
	bool copyPass = false;
	std::string name;
	std::string techniquePath;
	std::optional<ExternalInsertPoint> where;
	std::shared_ptr<RenderPass> pass;
	std::optional<QueueKind> preferredQueueKind;
	std::optional<QueueAssignmentPolicy> queueAssignmentPolicy;
	std::optional<QueueSlotIndex> pinnedQueueSlot; // Target a specific queue slot, bypassing queue preference

	// Optional: if true, the pass will be registered in Get*PassByName().
	bool registerName = true;
	bool isGeometryPass = false; // Optional: opts pass into statistics tracking for rasterization
	bool collectStatistics = true;
	// Persistent execution: the host epoch the pass runs in. persistent::AllEpochs
	// means every epoch. See SetPersistentEpochOrder.
	uint32_t epoch = UINT32_MAX;
	// Persistent execution only, for a pass of an external epoch (SetExternalPersistentEpochs). The pass is recorded when
	// the host executes the epoch, not ahead with the other passes. Recording happens in ExternalEpochExecution::Execute,
	// on the host's thread, into the host's command buffer. Its batch holds only passes recorded at execution. Its barriers
	// come from admission and are recorded around it. Use it for work that is only known at that point, such as a CPU cull's
	// draws. A typed pass recorded so may also skip a run whose work it knows then to be empty (TypedRenderGraphPass's
	// SkipsAtExecution).
	bool recordAtExecution = false;
	// Per-frame passes only. A frame-interrupting pass must run at its insert
	// point even under the persistent graph, which hosts it in the main
	// executable at the cost of a structural rebuild when it appears and when
	// it goes away. Meant for rare debugging work (mid-frame readbacks); other
	// per-frame passes run after the persistent main executable.
	bool interruptsFrame = false;

	static ExternalPassDesc Render(std::string name, std::shared_ptr<RenderPass> renderPass) {
		ExternalPassDesc desc{};
		desc.type = PassType::Render;
		desc.name = std::move(name);
		desc.pass = std::move(renderPass);
		return desc;
	}

	// The same pass on the compute or copy queue: queue semantics are declaration data, not inheritance.
	static ExternalPassDesc Compute(std::string name, std::shared_ptr<RenderPass> computePass) {
		ExternalPassDesc desc = Render(std::move(name), std::move(computePass));
		desc.preferredQueueKind = QueueKind::Compute;
		return desc;
	}

	static ExternalPassDesc Copy(std::string name, std::shared_ptr<RenderPass> copyPass) {
		ExternalPassDesc desc = Render(std::move(name), std::move(copyPass));
		desc.preferredQueueKind = QueueKind::Copy;
		desc.copyPass = true;
		return desc;
	}

	ExternalPassDesc& At(ExternalInsertPoint insertPoint) & {
		where = std::move(insertPoint);
		return *this;
	}

	ExternalPassDesc At(ExternalInsertPoint insertPoint) && {
		where = std::move(insertPoint);
		return std::move(*this);
	}

	ExternalPassDesc& PreferQueue(QueueKind queueKind) & {
		preferredQueueKind = queueKind;
		queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
		return *this;
	}

	ExternalPassDesc PreferQueue(QueueKind queueKind) && {
		preferredQueueKind = queueKind;
		queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
		return std::move(*this);
	}

	ExternalPassDesc& AutomaticQueueAssignment() & {
		queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
		return *this;
	}

	ExternalPassDesc AutomaticQueueAssignment() && {
		queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
		return std::move(*this);
	}

	ExternalPassDesc& RecordAtExecution() & {
		recordAtExecution = true;
		return *this;
	}

	ExternalPassDesc RecordAtExecution() && {
		recordAtExecution = true;
		return std::move(*this);
	}

	ExternalPassDesc& PinToQueue(QueueSlotIndex queueSlot) & {
		pinnedQueueSlot = queueSlot;
		return *this;
	}

	ExternalPassDesc PinToQueue(QueueSlotIndex queueSlot) && {
		pinnedQueueSlot = queueSlot;
		return std::move(*this);
	}

	ExternalPassDesc& RegisterByName(bool enabled = true) & {
		registerName = enabled;
		return *this;
	}

	ExternalPassDesc RegisterByName(bool enabled = true) && {
		registerName = enabled;
		return std::move(*this);
	}

	ExternalPassDesc& GeometryPass(bool enabled = true) & {
		isGeometryPass = enabled;
		return *this;
	}

	ExternalPassDesc GeometryPass(bool enabled = true) && {
		isGeometryPass = enabled;
		return std::move(*this);
	}

	ExternalPassDesc& Technique(std::string path) & {
		techniquePath = std::move(path);
		return *this;
	}

	ExternalPassDesc Technique(std::string path) && {
		techniquePath = std::move(path);
		return std::move(*this);
	}

	ExternalPassDesc& CollectStatistics(bool enabled = true) & {
		collectStatistics = enabled;
		return *this;
	}

	ExternalPassDesc CollectStatistics(bool enabled = true) && {
		collectStatistics = enabled;
		return std::move(*this);
	}

	ExternalPassDesc& Epoch(uint32_t value) & {
		epoch = value;
		return *this;
	}

	ExternalPassDesc Epoch(uint32_t value) && {
		epoch = value;
		return std::move(*this);
	}

	ExternalPassDesc& InterruptsFrame(bool enabled = true) & {
		interruptsFrame = enabled;
		return *this;
	}

	ExternalPassDesc InterruptsFrame(bool enabled = true) && {
		interruptsFrame = enabled;
		return std::move(*this);
	}
};

struct IRenderGraphExtension {
	virtual ~IRenderGraphExtension() = default;
	virtual void PrepareForBuild(RenderGraph& rg) { (void)rg; }

	// Called once from RenderGraph::Setup() after queue registry is populated.
	// Extensions can create queues, allocate upload instances, etc.
	virtual void Initialize(RenderGraph& rg) { (void)rg; }

	// Called before renderer teardown waits on/destroys graph-owned work.
	virtual void Shutdown(RenderGraph& rg) { (void)rg; }

	// lets systems react to registry recreation without RenderGraph including them
	virtual void OnRegistryReset(ResourceRegistry* /*registry*/) {}

	// main hook: inject passes.
	// In a graph of one epoch (rg.PassEpochFilter() is not persistent::AllEpochs) only that epoch's passes and the untagged ones
	// are taken. A host builds one such graph per epoch, so an extension whose passes cost something to make should make only those.
	virtual void GatherStructuralPasses(RenderGraph& rg, std::vector<ExternalPassDesc>& out) = 0;

	// Called after each execution of the graph, on the thread that executed it. Submit GPU work recorded outside the
	// graph here, through rg.GetExternalClient(). It runs after the execution, so it can read what the execution wrote.
	// Not called for tickets (GraphHost's async epochs).
	virtual void AfterExecute(RenderGraph& rg) { (void)rg; }
};

} // namespace org
