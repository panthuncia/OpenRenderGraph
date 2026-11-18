#pragma once

#include <string>

#include "Resources/TrackedAllocation.h"


namespace org {

class IHasMemoryMetadata {
	public:
	virtual ~IHasMemoryMetadata() = default;
	virtual void SetMemoryUsageHint(std::string usage);
	void SetMemoryIdentifier(std::string identifier);
	// Who the memory is for, used for memory attribution. This is a host's subsystem or feature.
	// A '/' separates nested levels.
	void SetMemoryOwner(std::string owner);
private:
	virtual void ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) = 0;
	friend class RenderGraph;
};


} // namespace org
