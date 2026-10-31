#include <Render/Runtime/RuntimeDevice.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/ECSManager.h"
#include "Managers/Singletons/ReadbackManager.h"
#include "Managers/Singletons/UploadManager.h"

namespace org::runtime {

void InitializeRuntimeDevice(rhi::Device device)
{
	ECSManager::GetInstance().InstallTrackingHooks();
	DeviceManager::GetInstance().Initialize(device);
	// Descriptor arenas are runtime-wide. Candidate graphs overlap while an old
	// generation is still in flight, so initializing them per graph would
	// invalidate descriptors owned by the retiring generation.
	DescriptorHeapManager::GetInstance().Initialize();
}

void ShutdownRuntimeDevice() noexcept
{
	// Callers have joined CPU work and completed GPU work before shutdown.
	// The order below matters:
	// - Deferred allocations must be destroyed while their allocator is alive.
	// - Releasing descriptor owners can enqueue another wave of backing releases.
	// - The readback manager's pooled buffers and pending requests, and the upload manager's pages and copy queue, go first.
	//   Left to static destruction, they would outlive the device and wait on its destroyed timelines.
	ReadbackManager::GetInstance().Cleanup();
	UploadManager::GetInstance().Cleanup();
	DeletionManager::GetInstance().DrainAll();
	DescriptorHeapManager::GetInstance().Cleanup();
	DeletionManager::GetInstance().DrainAll();
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
	TrackedEntityToken::ResetHooks();
	// Drain what the releases above posted.
	ECSManager::GetInstance().Drain();
}

} // namespace org::runtime
