#pragma once

// The BasicRHI borrowed recorder (rhi_c.h) a contributor's record callback records through: valid while its BorrowState
// is active, for the one call the host makes.

#include <rhi.h>
#include <rhi_c.h>

#include <cstdint>

namespace org::contributor {

struct BorrowState {
    rhi::CommandList* commandList = nullptr;
    uint64_t token = 0;
    bool active = false;
};

basicrhi_c_borrowed_recorder_v1 MakeRecorder(BorrowState& state);

} // namespace org::contributor
