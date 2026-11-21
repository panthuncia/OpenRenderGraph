// Latched copies without a device:
// - LatchedCopyRuns joins a copy that continues the pending one, with the same source and target, and keeps others apart;
// - LatchRegionsHeld keeps a run's held parts while the run and the key stay the same, and clears them otherwise;
// - NullSrvDesc gives every kind the dimension it names, and a texture kind a format, which D3D12's null views need.

#include "Render/LatchedCopies.h"
#include "Render/Runtime/NullViews.h"
#include "TestSupport.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
using Copy = org::LatchedCopyRuns::Copy;

rhi::ResourceHandle Handle(uint32_t index) { return {index, 1}; }

bool SameCopy(const Copy& a, const Copy& b) {
    constexpr rhi::HandleEqual<rhi::ResourceHandle> same{};
    return same(a.target, b.target) && same(a.latch, b.latch) && a.dst == b.dst && a.src == b.src && a.bytes == b.bytes;
}

int TestJoining() {
    std::vector<Copy> out;
    const auto emit = [&](const Copy& copy) { out.push_back(copy); };
    org::LatchedCopyRuns runs;
    runs.Add({Handle(1), Handle(9), 0, 256, 16}, emit);
    runs.Add({Handle(1), Handle(9), 16, 272, 16}, emit);   // continues: joins
    runs.Add({Handle(1), Handle(9), 48, 288, 16}, emit);   // a gap at the target: apart
    runs.Add({Handle(2), Handle(9), 64, 304, 16}, emit);   // another target: apart
    runs.Add({Handle(2), Handle(8), 80, 320, 16}, emit);   // another latch: apart
    runs.Add({Handle(2), Handle(8), 96, 400, 16}, emit);   // a gap at the latch: apart
    CHECK(out.size() == 4);
    runs.Flush(emit);
    CHECK(out.size() == 5);
    CHECK(SameCopy(out[0], {Handle(1), Handle(9), 0, 256, 32}));
    CHECK(SameCopy(out[1], {Handle(1), Handle(9), 48, 288, 16}));
    CHECK(SameCopy(out[2], {Handle(2), Handle(9), 64, 304, 16}));
    CHECK(SameCopy(out[3], {Handle(2), Handle(8), 80, 320, 16}));
    CHECK(SameCopy(out[4], {Handle(2), Handle(8), 96, 400, 16}));
    // Nothing pending: a second flush emits nothing.
    runs.Flush(emit);
    CHECK(out.size() == 5);
    return 0;
}

int TestEmptyCopies() {
    std::vector<Copy> out;
    const auto emit = [&](const Copy& copy) { out.push_back(copy); };
    org::LatchedCopyRuns runs;
    // An empty copy is never emitted, and nothing joins it.
    runs.Add({Handle(1), Handle(9), 0, 0, 0}, emit);
    runs.Add({Handle(1), Handle(9), 0, 0, 8}, emit);
    runs.Flush(emit);
    CHECK(out.size() == 1);
    CHECK(SameCopy(out[0], {Handle(1), Handle(9), 0, 0, 8}));
    return 0;
}

int TestRegionsHeld() {
    org::LatchRegionsHeld held;
    const auto latch = std::make_shared<int>(1), source = std::make_shared<int>(2);
    const org::RunRef run{3, 1};
    auto& slot = held.Of(run, latch, source, 7);
    CHECK(held.slots.size() == 4);
    CHECK(!org::LatchRegionsHeld::Hold(slot, 2));
    CHECK(org::LatchRegionsHeld::Hold(slot, 2));
    // Same run and key: the part stays held.
    CHECK(org::LatchRegionsHeld::Hold(held.Of(run, latch, source, 7), 2));
    // A change of the extra word, the source, or the run clears it.
    CHECK(!org::LatchRegionsHeld::Hold(held.Of(run, latch, source, 8), 2));
    const auto other = std::make_shared<int>(2);
    CHECK(!org::LatchRegionsHeld::Hold(held.Of(run, latch, other, 8), 2));
    CHECK(!org::LatchRegionsHeld::Hold(held.Of(org::RunRef{3, 2}, latch, other, 8), 2));
    // No owner never matches.
    CHECK(!org::LatchRegionsHeld::Same(std::weak_ptr<const void>{}, nullptr));
    return 0;
}

int TestNullDescs() {
    using org::runtime::NullViewKind;
    using org::runtime::NullSrvDesc;
    struct Expected {
        NullViewKind kind;
        rhi::SrvDim dimension;
    };
    const Expected textures[] = {
        {NullViewKind::Texture2D, rhi::SrvDim::Texture2D},
        {NullViewKind::Texture2DArray, rhi::SrvDim::Texture2DArray},
        {NullViewKind::Texture3D, rhi::SrvDim::Texture3D},
        {NullViewKind::TextureCube, rhi::SrvDim::TextureCube},
        {NullViewKind::TextureCubeArray, rhi::SrvDim::TextureCubeArray},
    };
    for (const auto& expected : textures) {
        const auto desc = NullSrvDesc(expected.kind);
        CHECK(desc.dimension == expected.dimension);
        CHECK(desc.formatOverride == rhi::Format::R8G8B8A8_UNorm);
    }
    CHECK(NullSrvDesc(NullViewKind::Texture2D).tex2D.mipLevels == 1);

    const auto typed = NullSrvDesc(NullViewKind::TypedBuffer);
    CHECK(typed.dimension == rhi::SrvDim::Buffer);
    CHECK(typed.buffer.kind == rhi::BufferViewKind::Typed);
    CHECK(typed.formatOverride == rhi::Format::R32_UInt);

    const auto structured = NullSrvDesc(NullViewKind::StructuredBuffer);
    CHECK(structured.dimension == rhi::SrvDim::Buffer);
    CHECK(structured.buffer.kind == rhi::BufferViewKind::Structured);
    CHECK(structured.buffer.structureByteStride == 4);
    CHECK(structured.formatOverride == rhi::Format::Unknown);

    const auto raw = NullSrvDesc(NullViewKind::RawBuffer);
    CHECK(raw.dimension == rhi::SrvDim::Buffer);
    CHECK(raw.buffer.kind == rhi::BufferViewKind::Raw);

    CHECK(NullSrvDesc(NullViewKind::Count).dimension == rhi::SrvDim::Undefined);
    return 0;
}
} // namespace

int main() {
    if (const int line = TestJoining()) return line;
    if (const int line = TestEmptyCopies()) return line;
    if (const int line = TestRegionsHeld()) return line;
    if (const int line = TestNullDescs()) return line;
    std::puts("OpenRenderGraphLatchedCopiesTests passed");
    return 0;
}
