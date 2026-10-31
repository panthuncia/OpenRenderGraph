// Rebinding safety at the persistent layer.
//
// The compiler treats slots bound to one backing (one physical identity) as one resource: passes that use either slot are
// ordered and transitioned against each other. A binding-only edit keeps the executable, so an edit that changes which slots
// share a backing must not be one. These tests check that:
// - such edits compile a new schedule, with the new orderings and transitions (checked by independent checkers);
// - edits that keep every sharing set (joint rotation, swaps) keep the executable;
// - SplitSharingChanges takes exactly the sharing changes out of an edit, in whole sets, and refuses when it can't;
// - ValidateSharedBackings catches an executable paired with bindings it didn't merge;
// - admission rejects a frame rebinding onto another resource's backing, and an aliased backing ORG didn't place.

#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/PublicationBindingBundle.h"
#include "ScheduleChecks.h"
#include "TestSupport.h"

#include <array>
#include <atomic>
#include <map>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>

using namespace org::persistent;
using namespace org::compiler;

namespace {
template<class F> bool Rejects(F&& fn) { try { fn(); } catch (const std::exception&) { return true; } return false; }

constexpr CompileResourceState kWrite{32, 0, 1, true};
constexpr CompileResourceState kRead{256, 0, 1, false};

// A program over buffer backings named by small handle numbers. Slots bound to the same handle share a backing.
struct Fixture {
    GraphProgram program;
    CompileWorkspace workspace;
    std::atomic_bool cancelled{false};
    std::map<uint32_t, std::shared_ptr<org::ResourceBindingSnapshot>> snapshots;
    uint32_t nextSlot = 0;

    // The binding of `slot` to backing `handle`. Every slot of one handle shares its snapshot, as captures of one backing do.
    BindingVersion Backed(uint32_t handle, uint32_t slot) {
        auto& snapshot = snapshots[handle];
        if (!snapshot) {
            snapshot = std::make_shared<org::ResourceBindingSnapshot>();
            snapshot->resource = rhi::Resource(rhi::ResourceHandle{handle, 1});
            snapshot->backingGeneration = handle;
            snapshot->allocationOwner = std::make_shared<const uint32_t>(handle);
            snapshot->description.type = rhi::ResourceType::Buffer;
            snapshot->description.buffer.sizeBytes = 128;
        }
        PreparedBackingState initial;
        initial.resource = {handle, 1};
        initial.graphResourceID = uint64_t{slot} + 1;
        initial.shape = {1, 1, false};
        initial.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        return BindingVersion::FromSnapshot(snapshot, initial);
    }
    ResourceSlotId Add(GraphEditTransaction& edit, uint32_t handle) {
        const auto slot = edit.AddResource({1, 1, false}, Backed(handle, nextSlot));
        ENSURE(slot.index == nextSlot);
        ++nextSlot;
        return slot;
    }
    void Rebind(GraphEditTransaction& edit, ResourceSlotId slot, uint32_t handle) {
        edit.ReplaceBinding(slot, Backed(handle, slot.index));
    }
    std::shared_ptr<const SelectedPublication> Install(GraphEditTransaction& edit) {
        auto ready = edit.Build(workspace, cancelled);
        ENSURE(ready && program.Install(edit, ready));
        return ready;
    }
};

uint32_t BatchOf(const Schedule& graph, PassId pass) {
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
        for (const auto index : graph.batches[batch].passes)
            if (index == pass.index) return batch;
    return UINT32_MAX;
}
bool Waits(const Schedule& graph, uint32_t producer, uint32_t consumer) {
    for (const auto& wait : graph.relativeWaits)
        if (wait.producerBatch == producer && wait.consumerBatch == consumer) return true;
    return false;
}
// The state step through which `pass` enters `slot`'s compiled resource, if any.
const SymbolicStateStep* EntryStep(const SelectedPublication& selected, ResourceSlotId slot, PassId pass) {
    const auto resource = selected.executable->resourceIndexBySlot.at(slot.index);
    for (const auto& step : selected.executable->graph->states.steps)
        if (step.resource == resource && step.pass == pass.index) return &step;
    return nullptr;
}
bool Valid(const SelectedPublication& selected) {
    try { ValidateSharedBackings(*selected.executable, selected.bindings); } catch (const std::exception&) { return false; }
    return CheckConflicts(*selected.executable->graph) && CheckStates(*selected.executable->graph);
}

// A writer of slot a on queue 0 and a reader of slot b on queue 1, over separate backings.
struct WriterReader {
    Fixture fixture;
    ResourceSlotId a, b;
    PassId writer, reader;
    std::shared_ptr<const SelectedPublication> initial;
    explicit WriterReader(uint32_t aHandle = 1, uint32_t bHandle = 2) {
        auto edit = fixture.program.BeginEdit();
        edit.SetQueues({{0, true}, {0, true}});
        a = fixture.Add(edit, aHandle);
        b = fixture.Add(edit, bHandle);
        writer = edit.AddPass({});
        CompilePass readPass;
        readPass.compatibleQueueSlots = {1};
        readPass.preferredQueueSlot = 1;
        reader = edit.AddPass(readPass);
        edit.Declare(writer, a, {}, kWrite);
        edit.Declare(reader, b, {}, kRead);
        initial = fixture.Install(edit);
    }
};

// Rebinding the reader's slot onto the writer's backing merges the two slots: the reader now waits for the writer across
// queues and enters from the writer's state. The old executable is invalid for the new bindings.
int TestMergeAddsDependency() {
    WriterReader graph;
    const auto& before = *graph.initial->executable->graph;
    CHECK(graph.initial->executable->resourceSlots.size() == 2 && Valid(*graph.initial));
    CHECK(!Waits(before, BatchOf(before, graph.writer), BatchOf(before, graph.reader)));

    auto edit = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(edit, graph.b, 1);
    const auto changed = edit.SharingChanges();
    CHECK(changed == std::vector<uint32_t>{graph.a.index, graph.b.index});
    const auto merged = graph.fixture.Install(edit);
    CHECK(merged->executable != graph.initial->executable);
    CHECK(merged->executable->structuralRevision != graph.initial->executable->structuralRevision);
    CHECK(merged->executable->resourceSlots.size() == 1 && Valid(*merged));
    const auto& after = *merged->executable->graph;
    const auto writerBatch = BatchOf(after, graph.writer), readerBatch = BatchOf(after, graph.reader);
    CHECK(writerBatch != UINT32_MAX && readerBatch != UINT32_MAX);
    CHECK(Waits(after, writerBatch, readerBatch));
    const auto* step = EntryStep(*merged, graph.b, graph.reader);
    CHECK(step && step->before == kWrite && step->after == kRead && step->previousBatch == writerBatch);
    // The executable from before the merge doesn't merge the slots the new bindings share.
    CHECK(Rejects([&] { ValidateSharedBackings(*graph.initial->executable, merged->bindings); }));
    return 0;
}

// Moving one slot of a shared pair to its own backing splits it: two resources, and no ordering between the passes.
int TestSplitRemovesDependency() {
    WriterReader graph(1, 1);
    CHECK(graph.initial->executable->resourceSlots.size() == 1 && Valid(*graph.initial));
    auto edit = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(edit, graph.b, 2);
    CHECK(edit.SharingChanges() == std::vector<uint32_t>{graph.a.index, graph.b.index});
    const auto split = graph.fixture.Install(edit);
    CHECK(split->executable != graph.initial->executable);
    CHECK(split->executable->resourceSlots.size() == 2 && Valid(*split));
    const auto& after = *split->executable->graph;
    CHECK(!Waits(after, BatchOf(after, graph.writer), BatchOf(after, graph.reader)));
    CHECK(Rejects([&] { ValidateSharedBackings(*graph.initial->executable, split->bindings); }));
    return 0;
}

// Rotating every slot of a shared backing together keeps every set, and the executable.
int TestJointRotationKeepsExecutable() {
    WriterReader graph(1, 1);
    auto edit = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(edit, graph.a, 3);
    graph.fixture.Rebind(edit, graph.b, 3);
    CHECK(edit.SharingChanges().empty());
    const auto rotated = graph.fixture.Install(edit);
    CHECK(rotated->executable == graph.initial->executable && Valid(*rotated));
    // Rotating only one of them is a split.
    auto partial = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(partial, graph.a, 4);
    CHECK(partial.SharingChanges().size() == 2);
    const auto split = graph.fixture.Install(partial);
    CHECK(split->executable != rotated->executable && split->executable->resourceSlots.size() == 2 && Valid(*split));
    return 0;
}

// Two unshared slots trading backings each still share with nobody: the executable stays.
int TestSwapKeepsExecutable() {
    WriterReader graph(1, 2);
    auto edit = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(edit, graph.a, 2);
    graph.fixture.Rebind(edit, graph.b, 1);
    CHECK(edit.SharingChanges().empty());
    const auto swapped = graph.fixture.Install(edit);
    CHECK(swapped->executable == graph.initial->executable && Valid(*swapped));
    return 0;
}

// A third slot joining a shared pair, and a writer-writer pair on one queue: every pass of the set is ordered.
int TestThreeWayJoin() {
    Fixture fixture;
    auto edit = fixture.program.BeginEdit();
    edit.SetQueues({{0, true}});
    const auto a = fixture.Add(edit, 1), b = fixture.Add(edit, 1), c = fixture.Add(edit, 2);
    const auto first = edit.AddPass({}), second = edit.AddPass({}), third = edit.AddPass({});
    edit.Declare(first, a, {}, kWrite);
    edit.Declare(second, b, {}, kRead);
    edit.Declare(third, c, {}, kWrite);
    const auto initial = fixture.Install(edit);
    CHECK(initial->executable->resourceSlots.size() == 2 && Valid(*initial));
    auto join = fixture.program.BeginEdit();
    fixture.Rebind(join, c, 1);
    CHECK(join.SharingChanges() == std::vector<uint32_t>{a.index, b.index, c.index});
    const auto joined = fixture.Install(join);
    CHECK(joined->executable->resourceSlots.size() == 1 && Valid(*joined));
    // The third pass writes after the second reads: it enters from the read.
    const auto* step = EntryStep(*joined, c, third);
    CHECK(step && step->before == kRead && step->after == kWrite);
    return 0;
}

// A group member moved onto a direct slot's backing merges with it, like two direct slots.
int TestGroupMemberMerge() {
    Fixture fixture;
    auto edit = fixture.program.BeginEdit();
    edit.SetQueues({{0, true}, {0, true}});
    const auto direct = fixture.Add(edit, 1);
    const auto member = fixture.Add(edit, 2);
    const auto group = edit.AddGroup({1, 1, false}, {member});
    edit.SetGroupResourceClass(group, rhi::ResourceType::Buffer);
    const auto writer = edit.AddPass({});
    CompilePass readPass;
    readPass.compatibleQueueSlots = {1};
    readPass.preferredQueueSlot = 1;
    const auto reader = edit.AddPass(readPass);
    edit.Declare(writer, direct, {}, kWrite);
    edit.DeclareGroupAccess(reader, group, kRead, 0);
    const auto initial = fixture.Install(edit);
    CHECK(initial->executable->resourceSlots.size() == 2 && Valid(*initial));
    auto merge = fixture.program.BeginEdit();
    fixture.Rebind(merge, member, 1);
    CHECK(merge.SharingChanges().size() == 2);
    const auto merged = fixture.Install(merge);
    CHECK(merged->executable->resourceSlots.size() == 1 && Valid(*merged));
    const auto& graph = *merged->executable->graph;
    CHECK(Waits(graph, BatchOf(graph, writer), BatchOf(graph, reader)));
    return 0;
}

// SplitSharingChanges takes the merge out and leaves the unrelated rotation: the rest builds binding-only, and the
// taken bindings build a merged schedule of their own.
int TestSplitTakesSharingChanges() {
    WriterReader graph;
    auto edit = graph.fixture.program.BeginEdit();
    const auto other = graph.fixture.Add(edit, 5);  // a third, unrelated slot (structural)
    edit.Declare(edit.AddPass({}), other, {}, kRead);
    const auto withOther = graph.fixture.Install(edit);
    auto frame = graph.fixture.program.BeginEdit();
    graph.fixture.Rebind(frame, graph.b, 1);      // merges with a
    graph.fixture.Rebind(frame, other, 6);        // rotates alone
    const auto taken = frame.SplitSharingChanges();
    CHECK(taken && taken->size() == 1 && (*taken)[0].first == graph.b && (*taken)[0].second.identity == (uint64_t{1} << 32 | 1));
    CHECK(frame.SharingChanges().empty());
    const auto reduced = graph.fixture.Install(frame);
    CHECK(reduced->executable == withOther->executable && Valid(*reduced));
    CHECK(reduced->bindings.At(other).identity == (uint64_t{1} << 32 | 6));
    CHECK(reduced->bindings.At(graph.b).identity == (uint64_t{1} << 32 | 2));  // still the old backing
    // A later build applies what was taken.
    auto later = graph.fixture.program.BeginEdit();
    for (const auto& [slot, version] : *taken) later.ReplaceBinding(slot, version);
    const auto merged = graph.fixture.Install(later);
    CHECK(merged->executable != reduced->executable && merged->executable->resourceSlots.size() == 2 && Valid(*merged));
    CHECK(merged->executable->resourceIndexBySlot[graph.a.index] == merged->executable->resourceIndexBySlot[graph.b.index]);
    return 0;
}

// Putting a slot back can change another slot's set. Here y joins {w, v}, and x moves alone onto the backing y left. Putting
// y back would make it share with x, so x is taken out too. After the split no set differs from the base.
int TestSplitClosure() {
    Fixture fixture;
    auto edit = fixture.program.BeginEdit();
    edit.SetQueues({{0, true}});
    const auto x = fixture.Add(edit, 2), y = fixture.Add(edit, 1), w = fixture.Add(edit, 3), v = fixture.Add(edit, 3);
    for (const auto slot : {x, y, w, v}) edit.Declare(edit.AddPass({}), slot, {}, kWrite);
    const auto initial = fixture.Install(edit);
    CHECK(Valid(*initial) && initial->executable->resourceSlots.size() == 3);
    auto frame = fixture.program.BeginEdit();
    fixture.Rebind(frame, y, 3);
    fixture.Rebind(frame, x, 1);
    const auto taken = frame.SplitSharingChanges();
    CHECK(taken && taken->size() == 2);
    CHECK(frame.SharingChanges().empty());
    const auto reduced = fixture.Install(frame);
    CHECK(reduced->executable == initial->executable && Valid(*reduced));
    auto later = fixture.program.BeginEdit();
    for (const auto& [slot, version] : *taken) later.ReplaceBinding(slot, version);
    const auto applied = fixture.Install(later);
    CHECK(Valid(*applied) && applied->executable->resourceSlots.size() == 2);
    const auto& index = applied->executable->resourceIndexBySlot;
    CHECK(index[y.index] == index[w.index] && index[w.index] == index[v.index] && index[x.index] != index[y.index]);
    return 0;
}

// A split that can't keep a base binding is refused, and the transaction is left as it was.
int TestSplitRefusals() {
    {
        // A reserved slot bound onto a shared backing has no binding to go back to.
        Fixture fixture;
        auto edit = fixture.program.BeginEdit();
        edit.SetQueues({{0, true}});
        const auto bound = fixture.Add(edit, 1);
        const auto reserved = edit.ReserveResource({1, 1, false});
        fixture.nextSlot = reserved.index + 1;
        edit.Declare(edit.AddPass({}), bound, {}, kWrite);
        edit.Declare(edit.AddPass({}), reserved, {}, kRead);
        fixture.Install(edit);
        auto frame = fixture.program.BeginEdit();
        frame.BindReserved(reserved, fixture.Backed(1, reserved.index));
        const auto before = frame.SharingChanges();
        CHECK(before.size() == 2);
        CHECK(!frame.SplitSharingChanges());
        CHECK(frame.SharingChanges() == before);
        const auto merged = fixture.Install(frame);  // builds structurally, as an inline edit would
        CHECK(Valid(*merged) && merged->executable->resourceSlots.size() == 1);
    }
    {
        // A structural edit compiles a new schedule anyway: nothing to split.
        WriterReader graph;
        auto edit = graph.fixture.program.BeginEdit();
        graph.fixture.Rebind(edit, graph.b, 1);
        edit.AddPass({});
        CHECK(edit.SharingChanges().empty());
        CHECK(!edit.SplitSharingChanges());
        CHECK(Valid(*graph.fixture.Install(edit)));
    }
    {
        // Nothing changes which slots share a backing: an empty split.
        WriterReader graph;
        auto edit = graph.fixture.program.BeginEdit();
        graph.fixture.Rebind(edit, graph.b, 7);
        const auto taken = edit.SplitSharingChanges();
        CHECK(taken && taken->empty());
        CHECK(graph.fixture.Install(edit)->executable == graph.initial->executable);
    }
    return 0;
}

// Admission runs ValidateSharedBackings while sharing validation is on, and rejects a frame rebinding onto a backing
// another resource is bound to.
int TestAdmissionChecks() {
    SetSharingValidation(true);
    CHECK(SharingValidationEnabled());
    Fixture fixture;
    auto edit = fixture.program.BeginEdit();
    edit.SetQueues({{0, true}});
    const auto bound = fixture.Add(edit, 1);
    const auto swapchain = edit.ReserveResource({1, 1, false});
    fixture.nextSlot = swapchain.index + 1;
    edit.Declare(edit.AddPass({}), bound, {}, kWrite);
    edit.Declare(edit.AddPass({}), swapchain, {}, kWrite);
    const auto selected = fixture.Install(edit);
    SynchronousAdmission admission;
    const std::array<ExecutionTimelinePoint, 1> queues{{{21, 0}}};
    auto rebinding = [&](uint32_t handle) {
        const auto binding = fixture.Backed(handle, swapchain.index);
        return FrameRebinding{swapchain, *binding.admission, binding.recording};
    };
    const auto own = rebinding(9);
    auto frame = admission.Prepare(selected, queues, {}, {}, std::span{&own, 1});
    CHECK(frame.rebound.size() == 1);
    admission.Abandon(frame);
    const auto onto = rebinding(1);
    CHECK(Rejects([&] { admission.Prepare(selected, queues, {}, {}, std::span{&onto, 1}); }));
    SetSharingValidation(false);
    CHECK(!SharingValidationEnabled());
    SetSharingValidation(true);
    return 0;
}

// Only ORG's alias pools place backings in shared memory. An aliased backing without a pool is refused.
int TestExternalAliasRejected() {
    Fixture fixture;
    auto storage = std::make_shared<const uint64_t>(1);
    const auto address = reinterpret_cast<const org::AliasHeapGeneration*>(storage.get());
    const auto heap = std::shared_ptr<const org::AliasHeapGeneration>(storage, address);
    auto placed = [&](uint64_t pool) {
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resource = rhi::Resource(rhi::ResourceHandle{40, 1});
        snapshot->backingGeneration = 40;
        snapshot->allocationOwner = std::make_shared<const uint32_t>(40);
        snapshot->description.type = rhi::ResourceType::Buffer;
        snapshot->description.buffer.sizeBytes = 128;
        snapshot->aliasHeap = heap; snapshot->aliasSize = 128; snapshot->aliasPoolID = pool;
        PreparedBackingState initial;
        initial.resource = {40, 1}; initial.graphResourceID = 1; initial.shape = {1, 1, false};
        initial.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        initial.aliasHeap = heap; initial.aliasHeapIdentity = address; initial.aliasSize = 128; initial.aliasPoolID = pool;
        return BindingVersion::FromSnapshot(snapshot, initial);
    };
    {
        auto edit = fixture.program.BeginEdit();
        CHECK(Rejects([&] { edit.AddResource({1, 1, false}, placed(0)); }));
    }
    {
        GraphProgram program;
        auto edit = program.BeginEdit();
        CHECK(!Rejects([&] { edit.AddResource({1, 1, false}, placed(77)); }));
    }
    return 0;
}

// Randomized: rebind slots of a small graph among a few backings, many times. Every build must validate, and an edit with no
// sharing change must keep the executable. Splitting each edit first must give the same final bindings after the taken part
// is applied.
int TestRandomRebinds() {
    std::mt19937 random(1234);
    for (int round = 0; round < 40; ++round) {
        Fixture fixture;
        auto edit = fixture.program.BeginEdit();
        edit.SetQueues({{0, true}, {0, true}});
        std::vector<ResourceSlotId> slots;
        for (uint32_t i = 0; i < 6; ++i) slots.push_back(fixture.Add(edit, 1 + random() % 4));
        for (uint32_t i = 0; i < 6; ++i) {
            CompilePass pass;
            pass.compatibleQueueSlots = {i % 2};
            pass.preferredQueueSlot = i % 2;
            edit.Declare(edit.AddPass(pass), slots[i], {}, random() % 2 ? kWrite : kRead);
        }
        auto current = fixture.Install(edit);
        CHECK(Valid(*current));
        for (int step = 0; step < 12; ++step) {
            auto frame = fixture.program.BeginEdit();
            std::map<uint32_t, uint64_t> wanted;
            for (int n = 1 + random() % 3; n > 0; --n) {
                const auto slot = slots[random() % slots.size()];
                const auto handle = 1 + random() % 5;
                fixture.Rebind(frame, slot, handle);
                wanted[slot.index] = (uint64_t{1} << 32) | handle;
            }
            const bool sharing = !frame.SharingChanges().empty();
            if (step % 2) {
                // Split, build the rest, then apply what was taken.
                const auto taken = frame.SplitSharingChanges();
                CHECK(taken && frame.SharingChanges().empty() && sharing == !taken->empty());
                const auto reduced = fixture.Install(frame);
                CHECK(reduced->executable == current->executable && Valid(*reduced));
                current = reduced;
                if (!taken->empty()) {
                    auto later = fixture.program.BeginEdit();
                    for (const auto& [slot, version] : *taken) later.ReplaceBinding(slot, version);
                    current = fixture.Install(later);
                }
            } else {
                const auto next = fixture.Install(frame);
                CHECK((next->executable == current->executable) == !sharing);
                current = next;
            }
            CHECK(Valid(*current));
            for (const auto& [index, identity] : wanted)
                CHECK(current->bindings.At(current->bindings.CurrentSlot(index)).identity == identity);
        }
    }
    return 0;
}
} // namespace

int main() {
    SetSharingValidation(true);
    if (const int line = TestMergeAddsDependency()) return line;
    if (const int line = TestSplitRemovesDependency()) return line;
    if (const int line = TestJointRotationKeepsExecutable()) return line;
    if (const int line = TestSwapKeepsExecutable()) return line;
    if (const int line = TestThreeWayJoin()) return line;
    if (const int line = TestGroupMemberMerge()) return line;
    if (const int line = TestSplitTakesSharingChanges()) return line;
    if (const int line = TestSplitClosure()) return line;
    if (const int line = TestSplitRefusals()) return line;
    if (const int line = TestAdmissionChecks()) return line;
    if (const int line = TestExternalAliasRejected()) return line;
    if (const int line = TestRandomRebinds()) return line;
    std::puts("rebinding tests passed");
    return 0;
}
