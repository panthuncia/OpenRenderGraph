#pragma once

#include <memory>
#include <rhi.h>
#include <string>
#include <vector>
#include <OpenRenderGraph/ContributorRegistry.h>

namespace org::contributor {

class ExtensionRegistry;

class ContributorGraphHost
{
public:
	static std::unique_ptr<ContributorGraphHost> Create(rhi::Device device, ExtensionRegistry* extensions,
		std::vector<std::string> anchors);
	~ContributorGraphHost();

	struct FrameInfo
	{
		uint64_t frameIndex{};
		uint64_t generation{};
		uint64_t completionValue{};
		uint32_t frameSlot{};
		uint32_t framesInFlight{};
		uint32_t width{};
		uint32_t height{};
		uint32_t outputWidth{};
		uint32_t outputHeight{};
	};
	void SetStructuralDefinition(const ContributorRegistry::Candidate& candidate);
	void Execute(
		uint32_t frameIndex,
		uint64_t frameFenceValue,
		rhi::Timeline readyTimeline,
		uint64_t readyValue,
		rhi::Timeline completeTimeline,
		uint64_t completeValue,
		const FrameInfo& frame);
	void Retire(uint64_t completedValue) noexcept;

	ContributorGraphHost(const ContributorGraphHost&) = delete;
	ContributorGraphHost& operator=(const ContributorGraphHost&) = delete;

private:
	class Impl;
	explicit ContributorGraphHost(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> impl;
};

} // namespace org::contributor
