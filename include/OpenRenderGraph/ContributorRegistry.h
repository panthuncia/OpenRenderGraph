#pragma once

#include <OpenRenderGraph/ContributorAPI.h>

#include <cstdint>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace org::contributor {

class ContributorRegistry
{
public:
	struct Resource
	{
		uint64_t handle{};
		uint64_t contributor{};
		std::string id;
		ORGResourceDesc desc{};
	};
	// A host resource contributors may access (ContributorHost's exports).
	struct Export
	{
		std::string id;
		ORGExportDescriptor desc{};
	};
	struct Access
	{
		uint64_t resource{};  // the contributor resource's handle; 0 for an export
		uint32_t exportIndex{ UINT32_MAX };
		std::string resourceId;
		ORGBinding binding{};
		ORGAccessKind kind{};
		ORGSubresourceRange range{};
		ORGViewKind viewKind{};
		ORGViewDimension viewDimension{};
		ORGFormat viewFormat{};
		uint32_t viewFlags{};
		uint64_t firstElement{};
		uint32_t elementCount{};
		uint32_t structureByteStride{};
		ORGBinding counterBinding{};
		float clearValue[4]{};
	};
	struct Pass
	{
		uint64_t contributor{};
		std::string id;
		ORGPassKind kind{};
		ORGQueueAssignment queue{};
		uint32_t flags{};
		int32_t priority{};
		std::string technique;
		std::vector<std::string> featureDomains;
		std::vector<std::string> after;
		std::vector<std::string> before;
		std::vector<Access> accesses;
		std::string epoch;  // empty: every epoch
		ORGPrepareCallback prepare{};
		ORGRevisionCallback invocationRevision{};
		void* userData{};
	};
	struct Candidate
	{
		uint64_t generation{};
		std::vector<Resource> resources;
		std::vector<Pass> passes;
		std::vector<uint64_t> contributors;
	};

	ContributorRegistry();
	~ContributorRegistry();
	// The host's anchors, epochs and exports, which passes may name. Set before the first build.
	void SetHostSymbols(std::vector<std::string> anchors, std::vector<std::string> epochs, std::vector<Export> exports);
	const std::vector<Export>& Exports() const noexcept { return exports_; }
	ORGStatus Register(const ORGContributorDesc*, ORGRegistrationHandle*) noexcept;
	ORGStatus BeginUnregister(ORGRegistrationHandle) noexcept;
	ORGStatus GetRegistrationState(ORGRegistrationHandle, ORGRegistrationState*) const noexcept;
	ORGStatus DeclareResource(ORGBuildHandle, const ORGResourceDesc*) noexcept;
	ORGStatus DeclarePass(ORGBuildHandle, const ORGPassDesc*) noexcept;
	ORGStatus RequestRebuild(ORGRegistrationHandle) noexcept;
	ORGStatus GetDiagnostic(ORGRegistrationHandle, ORGDiagnostic*) const noexcept;
	bool HasPendingRebuild() const noexcept;
	// Whether the last successful compile sized a resource against the render or output extent.
	bool HasRelativeResources() const noexcept { return relativeResources_; }
	ORGStatus Compile(uint64_t generation, uint32_t renderWidth, uint32_t renderHeight,
		uint32_t outputWidth, uint32_t outputHeight, Candidate&) noexcept;
	void Activate(const Candidate&) noexcept;
	void Retire(uint64_t generation) noexcept;
	void NotifyDeviceLost(uint32_t reason) noexcept;
	void Shutdown() noexcept;

private:
	struct Contributor;
	struct Build;
	void SetDiagnostic(ORGRegistrationHandle, ORGStatus, uint32_t phase, uint64_t generation, std::string) noexcept;
	static bool IsNamespaced(const char*) noexcept;
	mutable std::shared_mutex mutex_;
	mutable std::mutex buildMutex_;
	std::unordered_map<uint64_t, Contributor> contributors_;
	std::unordered_map<uint64_t, Contributor> unregistering_;
	std::unordered_map<uint64_t, ORGDiagnostic> diagnostics_;
	std::unordered_map<uint64_t, uint32_t> generationReferences_;
	std::unordered_map<uint64_t, std::vector<uint64_t>> generationContributors_;
	Build* currentBuild_{};
	std::thread::id buildThread_{};
	std::atomic_uint64_t nextHandle_{ 1 };
	uint64_t diagnosticSequence_{};
	bool rebuildRequested_{ true };
	bool relativeResources_{};
	bool open_{ true };
	std::vector<std::string> anchors_;
	std::vector<std::string> epochs_;
	std::vector<Export> exports_;
};

} // namespace org::contributor
