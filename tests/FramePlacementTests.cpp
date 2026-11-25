// Frame placement without a device: how a version matches uses across epochs (FramePlacementVersion::Compute), and when the
// host publishes its first version (FramePlacement, GraphHost's frame-wide placement). No block memory is acquired here.
// Run by PersistentVulkanHostTests' "placement" mode, before its device part.
#include "Render/RenderGraph/FramePlacement.h"
#include "../src/Host/FramePlacement.h"
#include "TestSupport.h"

#include <chrono>
#include <cstdio>
#include <thread>

namespace {
constexpr uint64_t kMB = 1024ull * 1024ull;

org::PlacementUse Use(uint64_t key, uint64_t objectId, uint32_t first, uint32_t last, uint64_t size = 4 * kMB) {
	org::PlacementUse use;
	use.key = key;
	use.objectId = objectId;
	use.name = "test.resource";
	use.firstRank = first;
	use.lastRank = last;
	use.firstUseWrites = true;
	use.writeGuaranteed = true;
	use.eligible = true;
	use.size = size;
	use.alignment = 65536;
	use.memoryClass = 1;
	use.backendMask = 1;
	return use;
}

std::shared_ptr<const org::PlacementSummary> Summary(uint32_t epoch, uint32_t ranks, std::vector<org::PlacementUse> uses) {
	org::PlacementSummary summary;
	summary.epoch = epoch;
	summary.rankCount = ranks;
	summary.memory = org::PlacementMemory::SharedArena;
	summary.uses = std::move(uses);
	summary.Finish();
	return std::make_shared<const org::PlacementSummary>(std::move(summary));
}

org::FramePlacementInputs Inputs(std::vector<std::shared_ptr<const org::PlacementSummary>> summaries) {
	org::FramePlacementInputs inputs;
	inputs.epochOrder = {1, 2};
	inputs.summaries = std::move(summaries);
	return inputs;
}
}  // namespace

int TestFramePlacementCompute() {
	// Same keys in two epochs but different identities: two resources, one per epoch. Each is found by its epoch's key.
	{
		const auto version = org::FramePlacementVersion::Compute(
			Inputs({Summary(1, 4, {Use(7, 100, 0, 3)}), Summary(2, 4, {Use(7, 200, 0, 3)})}), 1);
		ENSURE(version->Entries().size() == 2);
		const auto* first = version->Find(7, 1);
		const auto* second = version->Find(7, 2);
		ENSURE(first && second && first != second);
		ENSURE(!first->MultiEpoch() && !second->MultiEpoch());
		ENSURE(first->placed && second->placed);
		ENSURE(version->FindObject(100) == first && version->FindObject(200) == second);
		// Never live together (different slots), so they share bytes.
		ENSURE(first->offset == second->offset);
	}
	// One identity in two epochs, under different keys (ordinals differ between the graphs): one resource over both, its
	// contents kept from the first to the second.
	{
		const auto version = org::FramePlacementVersion::Compute(
			Inputs({Summary(1, 4, {Use(7, 100, 2, 3), Use(8, 300, 0, 1)}), Summary(2, 4, {Use(9, 100, 0, 1)})}), 1);
		const auto* shared = version->FindObject(100);
		ENSURE(shared && shared->MultiEpoch() && shared->placed);
		ENSURE(version->Find(7, 1) == shared && version->Find(9, 2) == shared && !version->Find(9, 1));
		ENSURE(shared->uses.size() == 2 && shared->uses.front().epoch == 1 && shared->uses.back().epoch == 2);
		// The other resource of epoch 1 ends before the shared one starts: they may share bytes.
		const auto* other = version->FindObject(300);
		ENSURE(other && !other->MultiEpoch());
	}
	// A rebuild that makes an epoch's own resources again (new identities, same keys) keeps the layout. A new object that
	// several epochs share is a new layout.
	{
		const auto before = org::FramePlacementVersion::Compute(
			Inputs({Summary(1, 4, {Use(7, 100, 0, 3), Use(8, 500, 0, 1)}), Summary(2, 4, {Use(8, 500, 2, 3)})}), 1);
		const auto rebuilt = org::FramePlacementVersion::Compute(
			Inputs({Summary(1, 4, {Use(7, 101, 0, 3), Use(8, 500, 0, 1)}), Summary(2, 4, {Use(8, 500, 2, 3)})}), 2);
		ENSURE(before->LayoutHash() == rebuilt->LayoutHash());
		const auto newShared = org::FramePlacementVersion::Compute(
			Inputs({Summary(1, 4, {Use(7, 100, 0, 3), Use(8, 501, 0, 1)}), Summary(2, 4, {Use(8, 501, 2, 3)})}), 3);
		ENSURE(before->LayoutHash() != newShared->LayoutHash());
		// And the entries are in the same order whatever the identities.
		for (size_t i = 0; i < before->Entries().size(); ++i) ENSURE(before->Entries()[i].key == rebuilt->Entries()[i].key);
	}
	// A summary's hash holds the identities, so a new object under a key makes the host compute again.
	{
		const auto a = Summary(1, 4, {Use(7, 100, 0, 3)});
		const auto b = Summary(1, 4, {Use(7, 101, 0, 3)});
		ENSURE(a->hash != b->hash);
	}
	// Repeated keys in one summary still merge and are not eligible (a graph that gave two candidates one key).
	{
		const auto summary = Summary(1, 4, {Use(7, 100, 0, 1), Use(7, 101, 2, 3)});
		ENSURE(summary->uses.size() == 1 && !summary->uses.front().eligible);
	}
	// The first summary with candidates publishes the first version at once (the upload epoch's empty one does not); a later
	// epoch's summary that reports candidates the version leaves uncovered publishes the next one at once.
	{
		org::FramePlacement placement({0, 1, 2});
		org::FramePlacement::Settings settings;
		settings.enabled = true;
		settings.settle = std::chrono::milliseconds(10);
		settings.trimAfter = std::chrono::milliseconds(0);
		placement.SetSettings(settings);
		placement.SetFrameEpochs({1, 2});
		org::PlacementSummary upload;
		upload.epoch = 0;
		upload.Finish();
		placement.Publish(upload);
		ENSURE(!placement.Current() && placement.CurrentId() == 0);
		org::PlacementSummary first = *Summary(1, 4, {Use(7, 100, 0, 3)});
		placement.Publish(first);
		// At once, on this thread, from epoch 1's summary alone.
		const auto provisional = placement.Current();
		ENSURE(provisional && provisional->HasEpoch(1) && !provisional->HasEpoch(2) && provisional->Find(7, 1));
		org::PlacementSummary second = *Summary(2, 4, {Use(9, 200, 0, 3)});
		second.uncovered = 1;
		second.Finish();
		placement.Publish(second);
		// At once again: epoch 2's graph placed what the first version leaves uncovered on its own path.
		const auto current = placement.Current();
		ENSURE(current && current != provisional && placement.CurrentId() == current->Id());
		ENSURE(current->HasEpoch(1) && current->HasEpoch(2) && current->Find(7, 1) && current->Find(9, 2));
		// Epoch 2's graph places from it and reports nothing uncovered: its summary is the one without the count.
		second.uncovered = 0;
		second.Finish();
		placement.Publish(second);
		// The same summary again changes nothing.
		placement.Publish(second);
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		ENSURE(placement.CurrentId() == current->Id());
		// A changed summary whose graph the current version still serves (nothing uncovered, no smaller layout): computed, not
		// published, so graphs do not place again.
		org::PlacementSummary moved = *Summary(2, 6, {Use(9, 200, 2, 5)});
		placement.Publish(moved);
		for (int wait = 0; wait < 200 && placement.Latest() == current; ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(5));
		ENSURE(placement.Latest() != current && placement.CurrentId() == current->Id());
		// One with candidates the version left uncovered gives a new version after the settle time, computed by the worker.
		org::PlacementSummary uncovered = *Summary(2, 6, {Use(9, 200, 2, 5), Use(10, 201, 0, 1)});
		uncovered.uncovered = 1;
		uncovered.Finish();
		placement.Publish(uncovered);
		for (int wait = 0; wait < 200 && placement.CurrentId() == current->Id(); ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(5));
		ENSURE(placement.CurrentId() != current->Id());
	}
	// Without frame epochs, the epoch order is the set.
	{
		org::FramePlacement placement({1});
		org::FramePlacement::Settings settings;
		settings.enabled = true;
		settings.trimAfter = std::chrono::milliseconds(0);
		placement.SetSettings(settings);
		org::PlacementSummary only = *Summary(1, 4, {Use(7, 100, 0, 3)});
		placement.Publish(only);
		ENSURE(placement.Current());
	}
	std::puts("frame placement: identities, keys and the first version");
	return 0;
}
