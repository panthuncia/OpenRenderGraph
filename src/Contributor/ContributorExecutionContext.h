#pragma once

#include <OpenRenderGraph/ContributorAPI.h>

#include <cstdint>
#include <unordered_map>
#include <memory>

namespace org { class Resource; namespace runtime { class IUploadService; } }

namespace org::contributor {

struct ExecutionContextHost
{
	struct BoundResource { std::shared_ptr<org::Resource> resource; uint64_t baseOffset{}, byteSize{}; };
	static constexpr uint64_t kMagic = 0x4f5247434f4e5458ull; // "ORGCONTX"
	uint64_t magic{ kMagic };
	std::unordered_map<ORGBinding, BoundResource> graphResourcesByBinding;
	std::unordered_map<ORGBinding, ORGBindingInfo> bindingInfo;
	org::runtime::IUploadService* graphUploads{};
	void* uploadUser{};
	ORGStatus (*queueBufferUpload)(void*, ORGBinding, uint64_t, const void*, uint64_t){};
};

} // namespace org::contributor
