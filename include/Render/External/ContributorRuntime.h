#pragma once

#include "Render/External/ContributorAPI.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace org
{
    class RenderGraph;

namespace external
{
    class ContributorRuntime
    {
    public:
        ContributorRuntime();
        ~ContributorRuntime();
        ContributorRuntime(const ContributorRuntime&) = delete;
        ContributorRuntime& operator=(const ContributorRuntime&) = delete;

        void SetHostAPI(const org_c_host_api& host);
        bool Register(const org_c_contributor_api* contributor, uint64_t& registration);
        bool Unregister(uint64_t registration);
        bool PrepareFrame(const org_c_frame_context& frame) const;
        void BuildPasses(RenderGraph& graph) const;
        void NotifyGraphRebuilt(uint64_t revision) const;
        void NotifyDeviceLost() const;
        bool NotifyDeviceRestored(const org_c_host_api& host) const;
        size_t ContributorCount() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
}
