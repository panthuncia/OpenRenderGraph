// A vertex shader fed by the input assembler, for the input layout test.
struct VertexInput
{
	float3 position : POSITION;
	float4 color : COLOR0;
};

struct VertexOutput
{
	float4 position : SV_Position;
	float4 color : COLOR0;
};

VertexOutput VSMain(VertexInput input)
{
	VertexOutput output;
	output.position = float4(input.position, 1.0);
	output.color = input.color;
	return output;
}

float4 PSMain(VertexOutput input) : SV_Target
{
	return input.color;
}
