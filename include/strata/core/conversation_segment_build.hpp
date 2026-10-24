// The segment-image side of the segmented SSD cache's production wiring (R4 §4.5 阶段 A): host-only glue over
// the already-verified strategy layer (conversation_segment_cache.hpp).  Nothing here changes the cache's
// semantics - it composes the public pieces:
//   * `conversation_kv_slice_prefix` cuts a captured K/V image to a shorter token depth.  Every part a snapshot
//     saves is a byte range copied from its state pool starting at offset 0 (the linear parts are cells*constant,
//     the indexer's pooled part is whole rows in order, the moving spare row included), so the bytes a snapshot
//     at a shallower depth holds are exactly a PREFIX of the deeper image's bytes - the slice is a precise
//     prefix copy, and the geometry fields tighten to what conversation_kv_validate recomputes at the new depth.
//   * `conversation_segment_build` cuts one whole-file snapshot into a self-contained segment image: the
//     checkpoint's own running state becomes the segment's live state, the K/V is sliced to the cut, the image
//     carries no chain (the whole-file tier stays the fallback's owner).  A restore then runs the segment image
//     through the same snapshot validate/restore path a whole file takes.
//   * `conversation_segment_park` is the park-side wiring: the checkpoint chain's stable cuts (the chooser in
//     the strategy layer), dedup-checked BEFORE anything is copied (a cut already on disk costs no K/V copy),
//     admitted append-only - the same-conversation-N-turns bound (criterion 10) and the one-shot prosecutor's
//     eviction (criterion 11) stay the strategy layer's job.
#pragma once

#include "strata/core/conversation_buffer.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_segment_cache.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {

/// Cuts the captured K/V image to `upto` tokens (the segment's own restore depth).  `out` is rebuilt fresh;
/// the part buffers are prefix copies, never aliases.  Fails closed: an extent the image does not cover, a
/// size that is not an exact multiple of its per-cell/row unit, or a page size foreign to the runtime's
/// shapes all refuse (a wrong slice must fail a build, never restore into a session).
inline bool conversation_kv_slice_prefix(const ConversationKv& full, int64_t upto, ConversationKv& out,
                                         std::string& error) {
    out = {};
    const auto shapes = strata::kernels::qsa_real_shapes();
    if (upto <= 0 || full.cells <= 0) {
        error = "segment slice: an empty K/V does not slice";
        return false;
    }
    if (full.page_size != shapes.page_size) {
        error = "segment slice: the image's page size is not this runtime's";
        return false;
    }
    const int64_t cells = ((upto + shapes.page_size - 1) / shapes.page_size) * shapes.page_size;
    if (cells > full.cells) {
        error = "segment slice: the cut extends past the image's cells";
        return false;
    }
    // The linear parts: exactly per-cell units, so the prefix size is a product, never a rounded ratio.
    const auto linear_target = [&](const ConversationBuffer& part, int64_t& target) -> bool {
        if (part.size() == 0) { target = 0; return true; }
        if (part.size() % (size_t) full.cells != 0) {
            error = "segment slice: a part's size is not a whole number of cells";
            return false;
        }
        target = (part.size() / (size_t) full.cells) * (size_t) cells;
        return true;
    };
    int64_t k_bytes = 0, v_bytes = 0, k_scale_bytes = 0, v_scale_bytes = 0;
    if (!linear_target(full.k, k_bytes) || !linear_target(full.v, v_bytes) ||
        !linear_target(full.k_scale, k_scale_bytes) || !linear_target(full.v_scale, v_scale_bytes))
        return false;
    // The indexer's pooled part: whole rows, in order, the moving spare row the deepest row.  A non-indexing
    // layer (the draft) records pooled_rows = 0 and slices to nothing.
    int64_t pooled_bytes = 0;
    int64_t pooled_rows = 0;
    if (full.pooled_rows > 0) {
        if (full.pooled.size() == 0 || full.pooled.size() % (size_t) full.pooled_rows != 0) {
            error = "segment slice: the pooled part's size is not whole rows";
            return false;
        }
        pooled_rows = upto / shapes.idx_block + 1;   // layout()'s rule at the new depth
        if (pooled_rows > full.pooled_rows) {
            error = "segment slice: the cut needs more indexer rows than the image holds";
            return false;
        }
        pooled_bytes = (full.pooled.size() / (size_t) full.pooled_rows) * (size_t) pooled_rows;
    } else if (!full.pooled.empty()) {
        error = "segment slice: a non-indexing layer carries pooled rows";
        return false;
    }
    out = full;
    out.cells = cells;
    out.pooled_rows = pooled_rows;
    const auto copy_prefix = [](const ConversationBuffer& src, size_t bytes, ConversationBuffer& dst) {
        dst.resize(bytes);
        return src.visit(0, bytes, [&](const uint8_t* p, size_t n, size_t at) {
            return dst.visit(at, n, [&](uint8_t* q, size_t m, size_t) {
                std::memcpy(q, p, m);
                return true;
            });
        });
    };
    if (!copy_prefix(full.k, (size_t) k_bytes, out.k) || !copy_prefix(full.v, (size_t) v_bytes, out.v) ||
        !copy_prefix(full.k_scale, (size_t) k_scale_bytes, out.k_scale) ||
        !copy_prefix(full.v_scale, (size_t) v_scale_bytes, out.v_scale) ||
        !copy_prefix(full.pooled, (size_t) pooled_bytes, out.pooled)) {
        error = "segment slice: the prefix copy failed";
        out = {};
        return false;
    }
    return true;
}

/// Builds one segment image from a whole-file snapshot: the checkpoint at `cut_tokens` (whose ids must be the
/// live tokens' own prefix) becomes the segment's live state, the K/V slices to the cut.  The checkpoint's
/// runtime marks (used / pinned) do not carry over - they belong to the session that earned them.  The image
/// carries no checkpoint chain: the whole-file tier keeps the fallback's chain.
inline bool conversation_segment_build(const SavedConversation& parent, int64_t cut_tokens,
                                       const ConversationCheckpoint& ckpt, SavedConversation& out,
                                       std::string& error) {
    out = {};
    error.clear();
    if (!parent.stage_images.empty()) {
        error = "segment build: a layer split's image does not segment";
        return false;
    }
    if (cut_tokens <= 0 || cut_tokens >= (int64_t) parent.live.ids.size() ||
        (int64_t) ckpt.ids.size() != cut_tokens) {
        error = "segment build: the cut is not inside the live state at the checkpoint's depth";
        return false;
    }
    if (!std::equal(ckpt.ids.begin(), ckpt.ids.end(), parent.live.ids.begin())) {
        error = "segment build: the checkpoint's tokens are not the live state's prefix";
        return false;
    }
    if (!ckpt.stage_parts.empty()) {
        error = "segment build: a split checkpoint does not segment";
        return false;
    }
    out.geometry = parent.geometry;
    out.layer_lo = parent.layer_lo;
    out.layer_hi = parent.layer_hi;
    out.cvec = parent.cvec;
    out.live = ckpt;
    out.live.used = 0;
    out.live.pinned = false;
    out.live.imgs.clear();
    for (const auto& k : parent.live.imgs)   // the parent's own image list below the cut, never the chain's copy
        if (k.start < cut_tokens) out.live.imgs.push_back(k);
    out.kv.resize(parent.kv.size());
    for (size_t j = 0; j < parent.kv.size(); ++j)
        if (!conversation_kv_slice_prefix(parent.kv[j], cut_tokens, out.kv[j], error)) {
            out = {};
            return false;
        }
    return true;
}

/// The park-side wiring's result (the counters the log line reads).
struct SegmentParkStats {
    size_t admitted = 0;   // cuts written to the disk this park
    size_t deduped = 0;    // cuts already on disk (touched, never rebuilt: no K/V copy at all)
    size_t failed = 0;     // builds or writes that failed (fail-open, the same loss as any cache write)
};

/// Parks a displaced conversation's stable segments beside its whole file: the checkpoint chain's cuts (the
/// strategy layer's chooser, `min_seg` apart, clipped to the live depth), dedup-checked by key BEFORE the
/// K/V is copied, admitted append-only.  A split's image does not segment (format v1 carries no layer split);
/// the cache being disabled is zero work.  Purely additive: the caller parks its whole file exactly as before.
inline SegmentParkStats conversation_segment_park(ConversationSegmentCache& cache, const SavedConversation& image,
                                                  int64_t min_seg = 4096) {
    SegmentParkStats stats;
    if (!cache.enabled() || !image.stage_images.empty() || image.live.ids.empty()) return stats;
    std::vector<int64_t> lens;
    lens.reserve(image.checkpoints.size());
    for (const auto& c : image.checkpoints)
        if (!c.ids.empty()) lens.push_back((int64_t) c.ids.size());
    for (int64_t n : conversation_segment_bounds(lens, (int64_t) image.live.ids.size(), min_seg)) {
        const ConversationCheckpoint* ck = nullptr;
        for (const auto& c : image.checkpoints)
            if ((int64_t) c.ids.size() == n) { ck = &c; break; }
        if (ck == nullptr) continue;
        if (cache.holds(conversation_spill_key(ck->ids, ck->imgs))) {
            ++stats.deduped;   // the precheck: no build, no K/V copy, no write for a cut already on disk
            continue;
        }
        SavedConversation seg;
        std::string error;
        if (!conversation_segment_build(image, n, *ck, seg, error)) {
            ++stats.failed;
            continue;
        }
        if (cache.admit(std::move(seg))) {
            ++stats.admitted;
        } else {
            ++stats.failed;   // a write failure is fail-open (the strategy layer left no record, no half file)
        }
    }
    return stats;
}

/// The streaming form of `conversation_segment_park`: the segment image is never materialized.  Its meta is the
/// checkpoint's own running state (about 118 MB at this geometry) and its K/V is the prefix of each live source,
/// written through the same aligned block.  Key/dedup/budget rules are unchanged.
inline SegmentParkStats conversation_segment_park_streamed(ConversationSegmentCache& cache,
                                                           const SavedConversation& meta,
                                                           const std::vector<SessionKvSource>& sources,
                                                           int64_t min_seg = 4096) {
    SegmentParkStats stats;
    if (!cache.enabled() || !meta.stage_images.empty() || meta.live.ids.empty()) return stats;
    std::vector<int64_t> lens;
    lens.reserve(meta.checkpoints.size());
    for (const auto& c : meta.checkpoints)
        if (!c.ids.empty()) lens.push_back((int64_t) c.ids.size());
    const auto shapes = strata::kernels::qsa_real_shapes();
    for (int64_t n : conversation_segment_bounds(lens, (int64_t) meta.live.ids.size(), min_seg)) {
        const ConversationCheckpoint* ck = nullptr;
        for (const auto& c : meta.checkpoints)
            if ((int64_t) c.ids.size() == n) { ck = &c; break; }
        if (ck == nullptr) continue;
        if (cache.holds(conversation_spill_key(ck->ids, ck->imgs))) {
            ++stats.deduped;   // no meta copy, no source read, no write
            continue;
        }
        const int64_t cells = ((n + shapes.page_size - 1) / shapes.page_size) * shapes.page_size;
        const int64_t pooled_rows = n / shapes.idx_block + 1;
        SavedConversation seg_meta;
        seg_meta.geometry = meta.geometry;
        seg_meta.layer_lo = meta.layer_lo;
        seg_meta.layer_hi = meta.layer_hi;
        seg_meta.cvec = meta.cvec;
        seg_meta.live = *ck;
        seg_meta.live.used = 0;
        seg_meta.live.pinned = false;
        seg_meta.live.imgs.clear();
        for (const auto& k : meta.live.imgs)
            if (k.start < n) seg_meta.live.imgs.push_back(k);
        seg_meta.checkpoints.clear();
        seg_meta.kv.clear();
        seg_meta.stage_images.clear();
        std::vector<SessionKvSource> seg_sources = sources;
        bool ok = true;
        for (auto& s : seg_sources) {
            if (s.cells <= 0) { ok = false; break; }
            const auto cut = [](size_t full, int64_t full_units, int64_t cut_units) -> size_t {
                if (full == 0 || full_units <= 0) return 0;
                if (full % (size_t) full_units != 0) return std::numeric_limits<size_t>::max();
                return (full / (size_t) full_units) * (size_t) cut_units;
            };
            s.sizes[0] = cut(s.sizes[0], s.cells, cells);
            s.sizes[1] = cut(s.sizes[1], s.cells, cells);
            s.sizes[2] = cut(s.sizes[2], s.cells, cells);
            s.sizes[3] = cut(s.sizes[3], s.cells, cells);
            s.sizes[4] = s.pooled_rows > 0 ? cut(s.sizes[4], s.pooled_rows, pooled_rows) : 0;
            for (size_t part = 0; part < 5; ++part)
                if (s.sizes[part] == std::numeric_limits<size_t>::max()) ok = false;
            if (!ok) break;
            s.cells = cells;
            s.pooled_rows = s.pooled_rows > 0 ? pooled_rows : 0;
        }
        if (!ok) { ++stats.failed; continue; }
        if (cache.admit_streamed(std::move(seg_meta), std::move(seg_sources))) ++stats.admitted;
        else ++stats.failed;   // a write failure is fail-open (the strategy layer left no record, no half file)
    }
    return stats;
}

} // namespace strata::core
