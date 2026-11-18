// Writes value + index + WRITE_BIAS into a bindless RWByteAddressBuffer. WRITE_BIAS tells the permutations apart.
#ifndef WRITE_BIAS
#define WRITE_BIAS 0
#endif

cbuffer WriteConstants : register(b0)
{
	uint outputIndex;
	uint value;
	uint count;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= count) return;
	RWByteAddressBuffer output = ResourceDescriptorHeap[outputIndex];
	output.Store(id.x * 4, value + id.x + WRITE_BIAS);
}
