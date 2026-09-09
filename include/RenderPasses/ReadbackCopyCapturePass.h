#pragma once
#include "RenderPasses/ReadbackCapturePass.h"
namespace org {
using ReadbackCopyCaptureInputs = ReadbackCaptureInputs;
using ReadbackCopyCapturePass = BasicReadbackCapturePass<QueueKind::Copy>;
} // namespace org
