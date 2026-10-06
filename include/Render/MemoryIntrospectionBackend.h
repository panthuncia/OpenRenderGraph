#pragma once

#include <memory>

#include <flecs.h>

#include "Render/MemoryIntrospectionAPI.h"

namespace org::memory {

// A host's own tracking world (its own TrackedEntityToken hooks): the host keeps the world from changing while a
// snapshot reads it.
std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider(flecs::world& world);
// ORG's tracking world, which resources on any thread post their changes to and one owner applies: a snapshot takes
// that ownership, so it may be built on any thread.
std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider();

}
