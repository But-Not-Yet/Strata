// One K/V layer read straight from where it lives (device or pinned host pool) into the file's staging buffer,
// without a full host copy first.  `read(part, offset, dst, n)` copies bytes [offset, offset+n) of part
// 0..4 = k, v, k_scale, v_scale, pooled; after a failed read `error()` (optional) says why.  The file bytes are
// identical to writing a captured ConversationKv.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::core {

struct SessionKvSource {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<size_t, 5> sizes{};
    std::function<bool(size_t part, size_t offset, void* dst, size_t n)> read;
    std::function<std::string()> error;
};

} // namespace strata::core
