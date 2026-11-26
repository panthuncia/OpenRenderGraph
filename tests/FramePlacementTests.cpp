// Frame placement without a device: how a version matches uses across epochs (FramePlacementVersion::Compute), and when the
// host publishes its first version (FramePlacement, GraphHost's frame-wide placement). No block memory is acquired here.
// Run by PersistentVulkanHostTests' "placement" mode, before its device part; TestFramePlacementKeeping needs its device.
#include "Render/RenderGraph/FramePlacement.h"
#include "../src/Host/FramePlacement.h"
#include "TestSupport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
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

namespace {
org::FramePlacement::Settings StagingSettings() {
	org::FramePlacement::Settings settings;
	settings.enabled = true;
	settings.settle = std::chrono::milliseconds(10);
	settings.trimAfter = std::chrono::milliseconds(0);
	// Without a device no heap is allocated, so no version can keep resources where they are: every version is packed from scratch.
	settings.keepRunning = false;
	return settings;
}

// Waits for the worker to publish a version other than id.
bool WaitForNewCurrent(const org::FramePlacement& placement, uint64_t id) {
	for (int wait = 0; wait < 200 && placement.CurrentId() == id; ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(5));
	return placement.CurrentId() != id;
}
}  // namespace

// Staged sets (FramePlacement::BeginStaging, GraphHost::PlaceEpochGraphs) and the per-frame freeze, without a device.
int TestFramePlacementStaging() {
	// Shared, as the host owns it: a staging closes through a weak reference to it when its last reference goes.
	const auto owner = std::make_shared<org::FramePlacement>(std::vector<uint32_t>{0, 1, 2});
	auto& placement = *owner;
	placement.SetSettings(StagingSettings());
	placement.SetFrameEpochs({1, 2});
	placement.Publish(*Summary(1, 4, {Use(7, 100, 0, 3)}));
	placement.Publish(*Summary(2, 4, {Use(9, 200, 0, 3)}));
	std::this_thread::sleep_for(std::chrono::milliseconds(60));
	const auto base = placement.Current();
	ENSURE(base && base->HasEpoch(1));
	const auto withUncovered = [](uint32_t epoch, std::vector<org::PlacementUse> uses) {
		org::PlacementSummary summary = *Summary(epoch, 4, std::move(uses));
		summary.uncovered = 1;
		summary.Finish();
		return summary;
	};

	// 1. A staged graph's uncovered candidates make a version for its set only. The running graphs keep the current one, and the
	// worker publishes nothing while the staging is open. The set's attach publishes it.
	{
		const auto staging = placement.BeginStaging({2});
		ENSURE(!staging->Closed() && staging->CurrentId() == base->Id());
		placement.PublishStaged(*staging, withUncovered(2, {Use(9, 201, 0, 3), Use(10, 202, 0, 1, 16 * kMB)}));
		const auto pending = placement.StagedCurrent(*staging);
		ENSURE(pending && pending != base && pending->FindObject(202) && staging->CurrentId() == pending->Id());
		ENSURE(placement.Current() == base && placement.CurrentId() == base->Id());
		// A summary that would make the worker publish (a new epoch's slot) waits while the staging is open.
		placement.Publish(*Summary(0, 2, {Use(11, 300, 0, 1)}));
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		ENSURE(placement.CurrentId() == base->Id());
		ENSURE(placement.CommitStaging(*staging) && staging->Closed());
		ENSURE(placement.Current() == pending);
		const auto summaries = placement.Summaries();
		ENSURE(std::ranges::any_of(summaries, [](const auto& summary) { return summary->epoch == 2 && std::ranges::any_of(summary->uses,
			[](const auto& use) { return use.objectId == 202; }); }));
		// Closed now: the worker publishes what waited.
		ENSURE(WaitForNewCurrent(placement, pending->Id()));
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(60));
	const auto current = placement.Current();

	// 2. A dropped set (its staging closed without its attach, or its last reference gone) publishes nothing.
	{
		const auto staging = placement.BeginStaging({1});
		placement.PublishStaged(*staging, withUncovered(1, {Use(7, 101, 0, 3), Use(12, 102, 0, 1, 32 * kMB)}));
		ENSURE(placement.StagedCurrent(*staging) != current);
		placement.DropStaging(*staging);
		ENSURE(staging->Closed() && placement.Current() == current);
		ENSURE(!placement.CommitStaging(*staging) && placement.Current() == current);
	}
	{
		auto staging = placement.BeginStaging({1});
		placement.PublishStaged(*staging, withUncovered(1, {Use(7, 103, 0, 3), Use(12, 104, 0, 1, 32 * kMB)}));
		staging.reset();
		ENSURE(placement.Current() == current);
		const auto summaries = placement.Summaries();
		ENSURE(std::ranges::none_of(summaries, [](const auto& summary) { return std::ranges::any_of(summary->uses,
			[](const auto& use) { return use.objectId == 104; }); }));
	}

	// 3. Declining: a set may keep the version it had while none of its graphs placed from the pending one. It then computes no
	// other version. Once a graph placed from the pending version, the set keeps it.
	{
		const auto staging = placement.BeginStaging({2});
		placement.PublishStaged(*staging, withUncovered(2, {Use(9, 205, 0, 3), Use(13, 206, 0, 1, 16 * kMB)}));
		ENSURE(placement.StagedCurrent(*staging) != current);
		ENSURE(placement.DeclineStaged(*staging));
		ENSURE(placement.StagedCurrent(*staging) == current && staging->CurrentId() == current->Id());
		placement.PublishStaged(*staging, withUncovered(2, {Use(9, 205, 0, 3), Use(13, 206, 0, 1, 24 * kMB)}));
		ENSURE(placement.StagedCurrent(*staging) == current);
		ENSURE(placement.DeclineStaged(*staging));
		ENSURE(!placement.CommitStaging(*staging) && placement.Current() == current);
	}
	{
		const auto staging = placement.BeginStaging({2});
		placement.PublishStaged(*staging, withUncovered(2, {Use(9, 207, 0, 3), Use(13, 208, 0, 1, 16 * kMB)}));
		const auto pending = placement.StagedCurrent(*staging);
		placement.NoteStagedPlacement(*staging, pending->Id(), 2, true);
		ENSURE(!placement.DeclineStaged(*staging) && placement.StagedCurrent(*staging) == pending);
		ENSURE(placement.CommitStaging(*staging) && placement.Current() == pending);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(60));

	// 4. The freeze: a frame runs one version. One the worker publishes meanwhile is the next frame's; one a plan of the frame needs
	// at once (its uncovered candidates) is the frame's version at once.
	{
		const auto frame = placement.Current();
		placement.Freeze(frame);
		// A new epoch's slot: the worker publishes the next layout (the current one cannot serve it).
		placement.Publish(*Summary(3, 3, {Use(15, 400, 0, 2)}));
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		ENSURE(placement.CurrentId() == frame->Id() && placement.Current() == frame);
		placement.Thaw();
		ENSURE(WaitForNewCurrent(placement, frame->Id()));
		const auto next = placement.Current();
		placement.Freeze(next);
		placement.Publish(withUncovered(1, {Use(7, 105, 0, 3), Use(14, 106, 0, 1, 64 * kMB)}));
		ENSURE(placement.CurrentId() != next->Id() && placement.Current()->FindObject(106));
		placement.Thaw();
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(60));

	// 5. Summaries first (GraphHost::PlaceEpochGraphs, FramePlacementSettings::summariesFirst): while the set collects, its graphs'
	// summaries compute nothing, even with uncovered candidates. The first uncovered one after it computes one version from every
	// summary of the set, so a later graph's new resource (here 302, which also shares 301 with epoch 1) is placed in it already.
	{
		const auto before = placement.Current();
		const auto staging = placement.BeginStaging({1, 2});
		placement.CollectStaged(*staging, true);
		placement.PublishStaged(*staging, withUncovered(1, {Use(7, 301, 0, 3), Use(16, 303, 0, 1, 8 * kMB)}));
		placement.PublishStaged(*staging, withUncovered(2, {Use(9, 301, 0, 1), Use(17, 302, 0, 3, 12 * kMB)}));
		ENSURE(placement.StagedCurrent(*staging) == before && staging->CurrentId() == before->Id());
		placement.CollectStaged(*staging, false);
		// Epoch 1's plan again, with its count of uncovered candidates (a new hash): one version for the whole set.
		org::PlacementSummary again = withUncovered(1, {Use(7, 301, 0, 3), Use(16, 303, 0, 1, 8 * kMB)});
		again.uncovered = 2;
		again.Finish();
		placement.PublishStaged(*staging, std::move(again));
		const auto pending = placement.StagedCurrent(*staging);
		ENSURE(pending && pending != before);
		const auto* shared = pending->FindObject(301);
		ENSURE(shared && shared->placed && shared->MultiEpoch());
		ENSURE(pending->FindObject(302) && pending->FindObject(302)->placed && pending->FindObject(303));
		// Epoch 2's plan finds everything covered: no newer version.
		placement.PublishStaged(*staging, *Summary(2, 4, {Use(9, 301, 0, 1), Use(17, 302, 0, 3, 12 * kMB)}));
		ENSURE(placement.StagedCurrent(*staging) == pending);
		ENSURE(placement.CommitStaging(*staging) && placement.Current() == pending);
	}
	// 6. Offered versions (Settings::offerVersions, M5 step 7): the worker's settled layout is offered, not published, until the host
	// publishes it at a frame boundary (PublishOffered) once the graphs prepared their re-placements. A staged set drops an offer, and
	// an offer nobody publishes is published once it timed out.
	{
		const auto offering = std::make_shared<org::FramePlacement>(std::vector<uint32_t>{0, 1, 2});
		auto& offers = *offering;
		auto settings = StagingSettings();
		settings.offerVersions = true;
		settings.offerTimeout = std::chrono::milliseconds(10000);
		settings.allocateOffered = false;  // no device
		offers.SetSettings(settings);
		offers.SetFrameEpochs({1, 2});
		offers.Publish(*Summary(1, 4, {Use(7, 100, 0, 3, 32 * kMB)}));
		offers.Publish(*Summary(2, 4, {Use(9, 200, 0, 3)}));
		// The first version is published as before: no graph has placed yet.
		for (int wait = 0; wait < 200 && !offers.Current(); ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(5));
		const auto first = offers.Current();
		ENSURE(first && !offers.Offered());
		const auto waitForOffer = [&] {
			for (int wait = 0; wait < 200 && !offers.Offered(); ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(5));
			return offers.Offered();
		};
		// A new epoch's slot: a new layout, offered while the graphs keep the current version.
		offers.Publish(*Summary(0, 2, {Use(11, 300, 0, 1)}));
		const auto offered = waitForOffer();
		ENSURE(offered && offered != first && offered->HasEpoch(0));
		ENSURE(offers.Current() == first && offers.CurrentId() == first->Id());
		ENSURE(!offers.PublishOffered(offered->Id() + 100) && offers.Current() == first);
		ENSURE(offers.PublishOffered(offered->Id()) && offers.Current() == offered && !offers.Offered());
		ENSURE(!offers.PublishOffered(offered->Id()));
		// A staged set drops an offer: the set's attach publishes its own version, and the worker computes again after it. (A layout
		// clearly smaller than the current one is offered; one about as large is served by the current version.)
		offers.Publish(*Summary(1, 4, {Use(7, 101, 0, 3)}));
		const auto stale = waitForOffer();
		ENSURE(stale && offers.Current() == offered);
		{
			const auto staging = offers.BeginStaging({1});
			ENSURE(!offers.Offered() && !offers.PublishOffered(stale->Id()));
			offers.DropStaging(*staging);
		}
		ENSURE(offers.Current() == offered);
		// After the staging closes, the worker offers again; with a short timeout and no host to publish it, it publishes it itself.
		settings.offerTimeout = std::chrono::milliseconds(30);
		offers.SetSettings(settings);
		ENSURE(WaitForNewCurrent(offers, offered->Id()));
		ENSURE(!offers.Offered());
	}
	std::puts("frame placement: staged sets (commit, drop, decline, summaries first), the per-frame freeze and offered versions");
	return 0;
}

// Versions that keep the running graphs' resources where they are (FramePlacementVersion::Keeping). Needs a device: the kept heaps
// are real ones (FramePlacementMemory).
int TestFramePlacementKeeping() {
	auto memory = std::make_shared<org::FramePlacementMemory>();
	const auto compute = [&](std::vector<std::shared_ptr<const org::PlacementSummary>> summaries, uint64_t id) {
		auto version = org::FramePlacementVersion::Compute(Inputs(std::move(summaries)), id);
		version->SetMemory(memory);
		return version;
	};
	// Epoch 1: A (100) then B (101), which share bytes. Epoch 2: C (200).
	const auto epochOne = Summary(1, 4, {Use(1, 100, 0, 1), Use(2, 101, 2, 3)});
	const auto base = compute({epochOne, Summary(2, 4, {Use(3, 200, 0, 3)})}, 1);
	ENSURE(base->Blocks().size() == 1);
	const auto heap = base->AcquireBlock(0);
	const auto* a = base->FindObject(100);
	const auto* b = base->FindObject(101);
	ENSURE(a && b && a->placed && b->placed && a->offset == b->offset);
	// The running graphs' resources, where base put them.
	std::vector<org::FramePlacementPin> pins;
	for (const auto& entry : base->Entries()) pins.push_back({entry.objectId, heap.get(), entry.offset, entry.size});
	const std::vector<uint32_t> keptOne{1};

	// 1. Epoch 2 rebuilt with a larger resource D (201). A and B stay where they are, in base's heap. D does not fit in it, so it
	// goes to a block of a new lane: no heap a pin lies in grows. C (the old graph's) holds no bytes.
	{
		const auto fresh = compute({epochOne, Summary(2, 4, {Use(4, 201, 0, 3, 16 * kMB)})}, 2);
		std::string why;
		const auto kept = org::FramePlacementVersion::Keeping(*fresh, *base, pins, keptOne, 2, &why);
		ENSURE(kept && why.empty() && kept->KeptFrom() == base->Id() && kept->KeptMovedCount() == 0);
		kept->SetMemory(memory);
		const auto* keptA = kept->FindObject(100);
		const auto* keptB = kept->FindObject(101);
		const auto* d = kept->FindObject(201);
		ENSURE(keptA && keptB && d && keptA->placed && keptB->placed && d->placed);
		ENSURE(kept->AcquireBlock(keptA->block) == heap && keptA->offset == a->offset && keptB->offset == b->offset);
		ENSURE(kept->Blocks()[d->block].lane == 1 && kept->AcquireBlock(d->block) != heap);
		ENSURE(!kept->FindObject(200));
	}
	// 2. Epoch 2 now uses A too: A lives until epoch 2, so it is live with B and C in their shared bytes. A, which several epochs
	// use, stays; B and C move (their graphs place again).
	{
		const auto fresh = compute({epochOne, Summary(2, 4, {Use(3, 200, 0, 3), Use(5, 100, 0, 1)})}, 3);
		const auto kept = org::FramePlacementVersion::Keeping(*fresh, *base, pins, keptOne, 3);
		// C (epoch 2, in the same bytes) is live with A now too.
		ENSURE(kept && kept->KeptMovedCount() == 2 && kept->KeptMoved().front() == "test.resource");
		kept->SetMemory(memory);
		const auto* keptA = kept->FindObject(100);
		const auto* keptB = kept->FindObject(101);
		ENSURE(keptA->MultiEpoch() && kept->AcquireBlock(keptA->block) == heap && keptA->offset == a->offset);
		ENSURE(keptB->placed && (keptB->block != keptA->block || keptB->offset != keptA->offset));
	}
	// 3. A pinned resource the new layout does not place (its first use in frame order now reads) stays unplaced, and its bytes are
	// held over the whole frame: nothing else is put there.
	{
		auto read = Use(1, 100, 0, 1);
		read.firstUseWrites = false;
		const auto fresh = compute({Summary(1, 4, {read, Use(2, 101, 2, 3)}), Summary(2, 4, {Use(3, 200, 0, 3), Use(5, 100, 2, 3)})}, 4);
		std::string why;
		const auto kept = org::FramePlacementVersion::Keeping(*fresh, *base, pins, keptOne, 4, &why);
		ENSURE(kept);
		kept->SetMemory(memory);
		const auto* keptA = kept->FindObject(100);
		ENSURE(keptA && !keptA->placed);
		for (const auto& entry : kept->Entries())
			if (entry.placed && kept->AcquireBlock(entry.block) == heap)
				ENSURE(entry.offset >= a->offset + a->size || entry.offset + entry.size <= a->offset);
	}
	// 4. A pin in a heap the base has no block for cannot be kept: the caller packs from scratch.
	{
		const auto fresh = compute({epochOne, Summary(2, 4, {Use(3, 200, 0, 3)})}, 5);
		auto stray = pins;
		const auto other = org::FramePlacementVersion::Compute(Inputs({Summary(1, 4, {Use(1, 100, 0, 3, 8 * kMB)})}), 6);
		other->SetMemory(std::make_shared<org::FramePlacementMemory>());
		const auto otherHeap = other->AcquireBlock(0);
		stray[0].heap = otherHeap.get();
		std::string why;
		ENSURE(!org::FramePlacementVersion::Keeping(*fresh, *base, stray, keptOne, 5, &why) && !why.empty());
	}
	// 5. Lanes: the trim of a version whose lane-1 block went drops that heap; lane 0 stays.
	{
		const auto fresh = compute({epochOne, Summary(2, 4, {Use(4, 201, 0, 3, 16 * kMB)})}, 7);
		const auto kept = org::FramePlacementVersion::Keeping(*fresh, *base, pins, keptOne, 7);
		kept->SetMemory(memory);
		for (uint32_t block = 0; block < kept->Blocks().size(); ++block) (void)kept->AcquireBlock(block);
		ENSURE(std::ranges::count_if(memory->Heaps(), [](const auto& info) { return info.current && info.lane == 1; }) == 1);
		ENSURE(memory->WouldTrim(base->BlockNeeds()));
		memory->ReleaseUnused(base->BlockNeeds());
		ENSURE(std::ranges::none_of(memory->Heaps(), [](const auto& info) { return info.current && info.lane == 1; }));
		ENSURE(std::ranges::any_of(memory->Heaps(), [&](const auto& info) { return info.current && info.identity == heap.get(); }));
	}
	std::puts("frame placement: versions that keep the running graphs' resources (pins, moved pins, held bytes, lanes)");
	return 0;
}
