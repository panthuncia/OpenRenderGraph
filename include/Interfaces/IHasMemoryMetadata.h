#pragma once

#include <string>

#include "Resources/TrackedAllocation.h"

class IHasMemoryMetadata {
	public:
	virtual ~IHasMemoryMetadata() = default;
	virtual void SetMemoryUsageHint(std::string usage);
	void SetMemoryIdentifier(std::string identifier);
private:
	virtual void ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) = 0;
	friend class RenderGraph;
};
