#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <rhi.h>
#include <spdlog/spdlog.h>

namespace ui {

    inline float HalfToFloat(uint16_t h) {
        const uint16_t sign = (h >> 15) & 0x1;
        const uint16_t exp  = (h >> 10) & 0x1F;
        const uint16_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            const float m = static_cast<float>(mant) / 1024.0f;
            const float v = std::ldexp(m, -14);
            return sign ? -v : v;
        }
        if (exp == 31) {
            if (mant == 0) return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            return std::numeric_limits<float>::quiet_NaN();
        }
        const float m = 1.0f + (static_cast<float>(mant) / 1024.0f);
        const float v = std::ldexp(m, static_cast<int>(exp) - 15);
        return sign ? -v : v;
    }

    inline uint8_t FloatToByte(float v) {
        if (std::isnan(v)) return 0;
        return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    inline float UintBitsToFloat(uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    inline uint32_t HashUint32(uint32_t value) {
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        value ^= value >> 16;
        return value;
    }

    inline float Saturate(float value) {
        return std::clamp(value, 0.0f, 1.0f);
    }

    // Decode a single subresource from readback data into tightly-packed RGBA8.
    // 'srcBase' points to the start of the readback buffer data.
    // The footprint describes where and how this subresource is laid out
    // (offset, rowPitch with D3D12 256-byte alignment, width, height).
    // Returns empty vector if the format is unsupported.
    inline std::vector<uint8_t> DecodeSubresourceToRGBA8(
        const std::byte* srcBase,
        size_t srcTotalSize,
        const rhi::CopyableFootprint& fp,
        rhi::Format format)
    {
        const uint32_t w = fp.width;
        const uint32_t h = fp.height;
        if (w == 0 || h == 0) return {};

        std::vector<uint8_t> out(static_cast<size_t>(w) * h * 4);

        auto rowSrc = [&](uint32_t row) -> const std::byte* {
            const size_t off = fp.offset + static_cast<size_t>(row) * fp.rowPitch;
            if (off >= srcTotalSize) return nullptr;
            return srcBase + off;
        };

        using F = rhi::Format;

        switch (format) {

        // RGBA8 direct copy
        case F::R8G8B8A8_UNorm:
        case F::R8G8B8A8_UNorm_sRGB:
        case F::R8G8B8A8_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = rowSrc(y);
                if (!src) break;
                std::memcpy(out.data() + static_cast<size_t>(y) * w * 4, src, static_cast<size_t>(w) * 4);
            }
            return out;

        // BGRA8 swizzle
        case F::B8G8R8A8_UNorm:
        case F::B8G8R8A8_UNorm_sRGB:
        case F::B8G8R8A8_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint8_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = src[2]; // R
                    dst[1] = src[1]; // G
                    dst[2] = src[0]; // B
                    dst[3] = src[3]; // A
                    src += 4; dst += 4;
                }
            }
            return out;

        // R16G16B16A16_Float
        case F::R16G16B16A16_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint16_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = FloatToByte(HalfToFloat(src[0]));
                    dst[1] = FloatToByte(HalfToFloat(src[1]));
                    dst[2] = FloatToByte(HalfToFloat(src[2]));
                    dst[3] = FloatToByte(HalfToFloat(src[3]));
                    src += 4; dst += 4;
                }
            }
            return out;

        // R16G16B16A16_UNorm
        case F::R16G16B16A16_UNorm:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint16_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = static_cast<uint8_t>(src[0] >> 8);
                    dst[1] = static_cast<uint8_t>(src[1] >> 8);
                    dst[2] = static_cast<uint8_t>(src[2] >> 8);
                    dst[3] = static_cast<uint8_t>(src[3] >> 8);
                    src += 4; dst += 4;
                }
            }
            return out;

        // R32G32B32A32_Float
        case F::R32G32B32A32_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const float*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = FloatToByte(src[0]);
                    dst[1] = FloatToByte(src[1]);
                    dst[2] = FloatToByte(src[2]);
                    dst[3] = FloatToByte(src[3]);
                    src += 4; dst += 4;
                }
            }
            return out;

        // R32G32B32_Float (96bpp, no alpha)
        case F::R32G32B32_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const float*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = FloatToByte(src[0]);
                    dst[1] = FloatToByte(src[1]);
                    dst[2] = FloatToByte(src[2]);
                    dst[3] = 255;
                    src += 3; dst += 4;
                }
            }
            return out;

        // R10G10B10A2_UNorm
        case F::R10G10B10A2_UNorm:
        case F::R10G10B10A2_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint32_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    uint32_t p = src[x];
                    dst[0] = static_cast<uint8_t>(((p >>  0) & 0x3FF) * 255 / 1023);
                    dst[1] = static_cast<uint8_t>(((p >> 10) & 0x3FF) * 255 / 1023);
                    dst[2] = static_cast<uint8_t>(((p >> 20) & 0x3FF) * 255 / 1023);
                    dst[3] = static_cast<uint8_t>(((p >> 30) & 0x3)   * 255 / 3);
                    dst += 4;
                }
            }
            return out;

        // R11G11B10_Float
        case F::R11G11B10_Float: {
            // Unpack 11-bit, 11-bit, 10-bit unsigned floats.
            auto unpack11 = [](uint32_t v) -> float {
                uint32_t e = (v >> 6) & 0x1F;
                uint32_t m = v & 0x3F;
                if (e == 0) return (m == 0) ? 0.0f : std::ldexp(static_cast<float>(m) / 64.0f, -14);
                if (e == 31) return (m == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
                return std::ldexp(1.0f + static_cast<float>(m) / 64.0f, static_cast<int>(e) - 15);
            };
            auto unpack10 = [](uint32_t v) -> float {
                uint32_t e = (v >> 5) & 0x1F;
                uint32_t m = v & 0x1F;
                if (e == 0) return (m == 0) ? 0.0f : std::ldexp(static_cast<float>(m) / 32.0f, -14);
                if (e == 31) return (m == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
                return std::ldexp(1.0f + static_cast<float>(m) / 32.0f, static_cast<int>(e) - 15);
            };

            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint32_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    uint32_t p = src[x];
                    dst[0] = FloatToByte(unpack11((p >>  0) & 0x7FF));
                    dst[1] = FloatToByte(unpack11((p >> 11) & 0x7FF));
                    dst[2] = FloatToByte(unpack10((p >> 22) & 0x3FF));
                    dst[3] = 255;
                    dst += 4;
                }
            }
            return out;
        }

        // Single-channel -> grayscale
        case F::R8_UNorm:
        case F::R8_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint8_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = dst[1] = dst[2] = src[x];
                    dst[3] = 255;
                    dst += 4;
                }
            }
            return out;

        case F::R16_UNorm:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint16_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    uint8_t v = static_cast<uint8_t>(src[x] >> 8);
                    dst[0] = dst[1] = dst[2] = v;
                    dst[3] = 255;
                    dst += 4;
                }
            }
            return out;

        case F::R16_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint16_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    uint8_t v = FloatToByte(HalfToFloat(src[x]));
                    dst[0] = dst[1] = dst[2] = v;
                    dst[3] = 255;
                    dst += 4;
                }
            }
            return out;

        case F::R32_Float:
        case F::D32_Float:
        case F::R32_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const float*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    uint8_t v = FloatToByte(src[x]);
                    dst[0] = dst[1] = dst[2] = v;
                    dst[3] = 255;
                    dst += 4;
                }
            }
            return out;

        // Two-channel -> RG
        case F::R8G8_UNorm:
        case F::R8G8_Typeless:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint8_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = 0;
                    dst[3] = 255;
                    src += 2; dst += 4;
                }
            }
            return out;

        case F::R16G16_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint16_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = FloatToByte(HalfToFloat(src[0]));
                    dst[1] = FloatToByte(HalfToFloat(src[1]));
                    dst[2] = 0;
                    dst[3] = 255;
                    src += 2; dst += 4;
                }
            }
            return out;

        case F::R32G32_Float:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const float*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    dst[0] = FloatToByte(src[0]);
                    dst[1] = FloatToByte(src[1]);
                    dst[2] = 0;
                    dst[3] = 255;
                    src += 2; dst += 4;
                }
            }
            return out;

        // R32G32_UInt
        // This is commonly used in the renderer for packed visibility data.
        // Raw integer-to-byte truncation is not readable, so visualize with a
        // stable hash color from the payload and depth-based brightness.
        case F::R32G32_UInt:
            for (uint32_t y = 0; y < h; ++y) {
                const auto* src = reinterpret_cast<const uint32_t*>(rowSrc(y));
                if (!src) break;
                uint8_t* dst = out.data() + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; ++x) {
                    const uint32_t low = src[0];
                    const uint32_t high = src[1];

                    if (low == 0xFFFFFFFFu && high == 0xFFFFFFFFu) {
                        dst[0] = 0;
                        dst[1] = 0;
                        dst[2] = 0;
                        dst[3] = 255;
                        src += 2; dst += 4;
                        continue;
                    }

                    const uint32_t hashed = HashUint32(low ^ (high * 0x9E3779B9u));
                    float brightness = 0.85f;

                    const float depth = UintBitsToFloat(high);
                    if (std::isfinite(depth) && depth >= 0.0f) {
                        brightness = 0.30f + 0.70f * (1.0f / (1.0f + depth));
                    }

                    const float r = ((hashed >> 0) & 0xFFu) / 255.0f;
                    const float g = ((hashed >> 8) & 0xFFu) / 255.0f;
                    const float b = ((hashed >> 16) & 0xFFu) / 255.0f;

                    dst[0] = FloatToByte(Saturate(r * brightness));
                    dst[1] = FloatToByte(Saturate(g * brightness));
                    dst[2] = FloatToByte(Saturate(b * brightness));
                    dst[3] = 255;
                    src += 2; dst += 4;
                }
            }
            return out;

        default:
            spdlog::warn("TextureDecoder: unsupported format {} for RGBA8 decode.", static_cast<int>(format));
            return {};
        }
    }

} // namespace ui
