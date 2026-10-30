#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace org::util {

#ifdef _WIN32
    inline std::wstring s2ws(const std::string_view& utf8)
    {
        if (utf8.empty()) return {};
        int needed = ::MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            utf8.data(),
            static_cast<int>(utf8.size()),
            nullptr,
            0
        );
        if (needed == 0)
            throw std::system_error(::GetLastError(), std::system_category(),
                "MultiByteToWideChar(size)");

        std::wstring out(needed, L'\0');

        int written = ::MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            utf8.data(),
            static_cast<int>(utf8.size()),
            out.data(),
            needed
        );
        if (written == 0)
            throw std::system_error(::GetLastError(), std::system_category(),
                "MultiByteToWideChar(data)");

        return out;
    }

    inline std::string ws2s(const std::wstring_view& wide)
    {
        if (wide.empty()) return {};

        int needed = ::WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            wide.data(),
            static_cast<int>(wide.size()),
            nullptr,
            0,
            nullptr, nullptr
        );
        if (needed == 0)
            throw std::system_error(::GetLastError(), std::system_category(),
                "WideCharToMultiByte(size)");

        std::string out(needed, '\0');

        int written = ::WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            wide.data(),
            static_cast<int>(wide.size()),
            out.data(),
            needed,
            nullptr, nullptr
        );
        if (written == 0)
            throw std::system_error(::GetLastError(), std::system_category(),
                "WideCharToMultiByte(data)");

        return out;
    }
#else
    // wchar_t is UTF-32 off Windows. Invalid input throws, as MB_ERR_INVALID_CHARS /
    // WC_ERR_INVALID_CHARS do on Windows.
    inline std::wstring s2ws(const std::string_view& utf8)
    {
        std::wstring out;
        out.reserve(utf8.size());
        for (size_t i = 0; i < utf8.size();) {
            const auto lead = static_cast<unsigned char>(utf8[i]);
            const size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 0;
            if (length == 0 || i + length > utf8.size())
                throw std::range_error("s2ws: invalid UTF-8");
            char32_t code = length == 1 ? lead : lead & (0x7F >> length);
            for (size_t k = 1; k < length; ++k) {
                const auto next = static_cast<unsigned char>(utf8[i + k]);
                if ((next & 0xC0) != 0x80)
                    throw std::range_error("s2ws: invalid UTF-8");
                code = (code << 6) | (next & 0x3F);
            }
            out.push_back(static_cast<wchar_t>(code));
            i += length;
        }
        return out;
    }

    inline std::string ws2s(const std::wstring_view& wide)
    {
        std::string out;
        out.reserve(wide.size());
        for (const wchar_t character : wide) {
            const auto code = static_cast<char32_t>(character);
            if (code < 0x80) {
                out.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
                out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else if (code < 0x10000) {
                if (code >= 0xD800 && code < 0xE000)
                    throw std::range_error("ws2s: unpaired surrogate");
                out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else if (code < 0x110000) {
                out.push_back(static_cast<char>(0xF0 | (code >> 18)));
                out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
                throw std::range_error("ws2s: invalid code point");
            }
        }
        return out;
    }
#endif

    inline uint16_t CalculateMipLevels(uint16_t width, uint16_t height) {
        return static_cast<uint16_t>(std::floor(std::log2((std::max)(width, height)))) + 1;
    }
}