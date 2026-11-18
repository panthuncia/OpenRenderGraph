#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/ResourceRequirements.h"

namespace org::persistent { class GraphProgram; class SynchronousAdmission; }
namespace org::compiler { class CompileWorkspace; }

namespace org {

class Runtime;
class Resource;
class CommandListPool;

// Submits GPU work that is recorded outside any graph, such as uploads, readbacks and streaming copies.
// See docs/job-graphs.md, "External clients".
//
// Each submission lists the resources it uses. The client then:
// - finds who used each resource last, and adds the barriers and queue waits needed to continue from there,
// - records the new state of each resource, so the next graph or client starts from it.
//
// Work runs in the order it is submitted, across all graphs and clients on the runtime. So if a client reads
// something a graph writes, submit it after that graph's execution.
//
// Any thread may call it. Submissions are serialized.
class ExternalClient {
public:
	struct Queue {
		rhi::Queue queue;
		rhi::QueueKind kind = rhi::QueueKind::Graphics;
	};
	struct Desc {
		std::shared_ptr<Runtime> runtime;
		// One entry per queue slot. Use the same slot numbers as the graphs on this runtime (QueueRegistry order:
		// graphics, compute, copy, then queues made with CreateQueue). Handoffs between a graph and the client
		// match queues by slot number.
		std::vector<Queue> queues;
		std::string name = "ExternalClient";
		// How many submissions can be in flight. When it is reached, Submit waits for the oldest one to finish.
		size_t capacity = 64;
		// Set this when the host's graphs use closed executions (SynchronousAdmission::SetClosedExecutions).
		// Each submission then starts with a full memory barrier and returns every texture it uses to its home state.
		bool closed = false;
		// Start each submission with a full memory barrier. Set this when the queue is shared with another API whose
		// earlier work the ledgers don't know about (for example DXVK's).
		bool entryFullBarrier = false;
	};
	struct Access {
		Resource* resource = nullptr;
		RangeSpec range{};
		ResourceState state{};
	};
	// The timeline value a submission signals when it finishes. Wait on it or poll it.
	struct Signal {
		rhi::Timeline timeline;
		uint64_t value = 0;
	};

	explicit ExternalClient(Desc desc);
	ExternalClient(const ExternalClient&) = delete;
	ExternalClient& operator=(const ExternalClient&) = delete;
	// Waits for all submissions to finish.
	~ExternalClient();

	// Submits work on queue slot `slot`.
	// `accesses` lists every resource the work uses, with its range and state. Each resource must already have a backing.
	// `record` records the work. It runs on the calling thread before the barriers are computed, so it must not depend on
	// the state a resource is in beforehand. The barriers are recorded into a separate list that runs first.
	// Throws if the submission fails. Nothing is recorded in the ledgers in that case.
	// `signals` are extra timeline points the submission signals when it finishes.
	// `keepAlive` is held until the submission finishes (for example the staging memory its commands read).
	Signal Submit(uint32_t slot, std::span<const Access> accesses, const std::function<void(rhi::CommandList&)>& record,
		std::span<const rhi::TimelinePoint> signals = {}, std::shared_ptr<const void> keepAlive = {});

	// Frees what finished submissions held. Submit also does this.
	void Retire();
	// Stops the background work of its command-list pools. Call it before the task service shuts down.
	void ShutdownTaskWorkers();

private:
	struct SlotState;
	struct PublicationCache;
	void RetireLocked();
	SlotState& Slot(uint32_t slot);

	Desc m_desc;
	std::mutex m_mutex;
	std::unique_ptr<persistent::GraphProgram> m_empty;
	std::unique_ptr<persistent::SynchronousAdmission> m_admission;
	std::unique_ptr<compiler::CompileWorkspace> m_workspace;
	std::vector<std::unique_ptr<SlotState>> m_slots;
	std::unique_ptr<PublicationCache> m_publications;
	struct InFlight {
		uint32_t slot = 0;
		uint64_t value = 0;
	};
	std::deque<InFlight> m_inFlight;
};

} // namespace org
