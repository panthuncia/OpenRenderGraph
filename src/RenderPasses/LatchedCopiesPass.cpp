#include "RenderPasses/LatchedCopiesPass.h"

#include <algorithm>
#include <utility>

#include "Render/PassBuilders.h"

namespace org {

namespace {
struct LatchedCopiesBindings {
	std::vector<std::pair<const void*, ResourceBindingToken>> targets;  // by LatchedTarget::Key
};

class LatchedCopiesPass final : public TypedRenderGraphPass<LatchedCopiesPass, LatchedSource, LatchedCopiesBindings> {
public:
	LatchedCopiesPass(bool copyFixed, std::function<std::vector<LatchedTarget>()> targets,
		std::function<LatchedSource(const PassPrepareContext&)> source,
		std::atomic<std::shared_ptr<const std::vector<const void*>>>* declared)
		: m_targets(std::move(targets)), m_source(std::move(source)), m_declared(declared), m_copyFixed(copyFixed) {}

	LatchedCopiesBindings Declare(PassBuilder& builder) {
		builder.PreferQueue(QueueKind::Graphics);
		LatchedCopiesBindings bindings{};
		auto keys = std::make_shared<std::vector<const void*>>();
		for (const auto& target : m_targets()) {
			bindings.targets.emplace_back(target.Key(),
				target.versioned ? builder.CopyDestination(*target.versioned).Resource() : builder.CopyDestination(target.buffer));
			keys->push_back(target.Key());
		}
		if (m_declared) m_declared->store(std::move(keys), std::memory_order_release);
		return bindings;
	}

	// Empty: the recording depends only on its bindings and the prepared source.
	void Revision(std::vector<uint64_t>&) const {}

	LatchedSource Prepare(const LatchedCopiesBindings&, const PassPrepareContext& preparation) const {
		auto prepared = m_source(preparation);
		prepared.copyFixed = m_copyFixed;
		return prepared;
	}

	static void Record(const LatchedCopiesBindings& bindings, const LatchedSource& prepared, PassRecordContext& recording) {
		RecordLatchedCopies(recording.Commands(), prepared, recording.Run(), [&](const void* key) {
			const auto target = std::find_if(bindings.targets.begin(), bindings.targets.end(),
				[&](const auto& entry) { return entry.first == key; });
			return target != bindings.targets.end() ? recording.Resolve(target->second).GetHandle() : rhi::ResourceHandle{};
		});
	}

private:
	std::function<std::vector<LatchedTarget>()> m_targets;
	std::function<LatchedSource(const PassPrepareContext&)> m_source;
	std::atomic<std::shared_ptr<const std::vector<const void*>>>* m_declared = nullptr;
	bool m_copyFixed = true;
};
} // namespace

std::shared_ptr<RenderPass> MakeLatchedCopiesPass(bool copyFixed, std::function<std::vector<LatchedTarget>()> targets,
	std::function<LatchedSource(const PassPrepareContext&)> source,
	std::atomic<std::shared_ptr<const std::vector<const void*>>>* declared) {
	return std::make_shared<LatchedCopiesPass>(copyFixed, std::move(targets), std::move(source), declared);
}

} // namespace org
