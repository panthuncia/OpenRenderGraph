// Writes input[0] + index into a bindless RWByteAddressBuffer. input[0] is uploaded
// by the host each frame through the graph's upload pass.
cbuffer WriteConstants : register(b0)
{
	uint outputDescriptorIndex;
	uint inputDescriptorIndex;
	uint count;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= count) return;
	ByteAddressBuffer input = ResourceDescriptorHeap[inputDescriptorIndex];
	RWByteAddressBuffer output = ResourceDescriptorHeap[outputDescriptorIndex];
	output.Store(id.x * 4, input.Load(0) + id.x);
}
