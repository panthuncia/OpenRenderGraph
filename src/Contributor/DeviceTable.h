#pragma once

// The device table of the pipeline service. It implements basicrhi_c_device_api_v1 on top of rhi::Device.
// Contributors use it to create pipelines and command signatures. Their destruction is deferred until the GPU can no
// longer use them.

#include <rhi.h>
#include <rhi_c.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace org::contributor {

class DeviceTable {
public:
	explicit DeviceTable(rhi::Device device);
	~DeviceTable();
	DeviceTable(const DeviceTable&) = delete;
	DeviceTable& operator=(const DeviceTable&) = delete;

	const basicrhi_c_device_api_v1* Api() const noexcept { return &m_api; }

private:
	static basicrhi_c_result BASICRHI_C_CALL CreatePipeline(void*, const basicrhi_c_pipeline_desc_v1*, basicrhi_c_pipeline*);
	static void BASICRHI_C_CALL DestroyPipeline(void*, basicrhi_c_pipeline, uint64_t);
	static basicrhi_c_result BASICRHI_C_CALL CreateCommandSignature(void*, const basicrhi_c_command_signature_desc_v1*,
		basicrhi_c_command_signature*);
	static void BASICRHI_C_CALL DestroyCommandSignature(void*, basicrhi_c_command_signature, uint64_t);

	rhi::Device m_device;
	basicrhi_c_device_api_v1 m_api{};
	std::mutex m_mutex;
	std::unordered_map<uint64_t, rhi::PipelinePtr> m_pipelines;
	std::unordered_map<uint64_t, rhi::CommandSignaturePtr> m_signatures;
};

} // namespace org::contributor
