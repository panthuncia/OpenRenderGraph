#pragma once

#include <memory>

#include <flecs.h>

#include "Render/MemoryIntrospectionAPI.h"

namespace rg::memory {

std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider(flecs::world& world);
std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider();

}
