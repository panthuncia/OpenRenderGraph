#include "Resources/Buffers/Buffer.h"

#include "Resources/GPUBacking/GpuBufferBacking.h"


namespace org {

void Buffer::OnSetName() {
    if (!m_dataBuffer) {
        return;
    }
    m_dataBuffer->SetName(name.c_str());
}


} // namespace org
