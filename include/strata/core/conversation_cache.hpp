// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

// --- spill-to-session-file (the disk tier of the parked-conversation budget) ---
// A spilled entry lives in a session file (format v1, conversation_file.hpp); the cache keeps only a RAM manifest
// record: the key hashes of the entry's live state and of each checkpoint, their token counts, the geometry and the
// path.  A key hash covers exactly one (ids, filtered images) prefix, value-hashed so an int32 stored id and an int64
// prompt token hash the same; a spilled match is therefore as strong as the RAM prefix match up to hash collisions
// (64-bit, non-adversarial — the same trust the session file's own checksums get).  The engine validates every read
// image (geometry, layer range, sizes) before it is used, so a collision or a stale file is a discard, never wrong
// output.
inline uint64_t conversation_spill_hash64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
inline uint64_t conversation_spill_seed() { return 0x243f6a8885a308d3ull; }
inline uint64_t conversation_spill_fold(uint64_t h, uint64_t v) {
    return (h * 0x100000001b3ull) ^ conversation_spill_hash64(v);
}
// The prefix digest of a token list: dig[i] covers the first i tokens (value-hashed, width-independent).
template<class It>
inline void conversation_spill_prefix_digests(It first, It last, std::vector<uint64_t>& dig) {
    dig.clear();
    dig.push_back(conversation_spill_seed());
    for (It it = first; it != last; ++it) dig.push_back(conversation_spill_fold(dig.back(), (uint64_t)(int64_t) *it));
}
// The key of the first `n` tokens (digest form) with the images that open inside the prefix; both the spill side and
// the prompt side build it the same way.  Mirrors conversation_prefix's image rule: images that open inside the
// prefix must match exactly; images that open later do not invalidate.
template<class Img>
inline uint64_t conversation_spill_key_at(const std::vector<uint64_t>& dig, int64_t n, const Img* imgs, size_t n_imgs) {
    uint64_t k = dig[(size_t) n];
    k = conversation_spill_fold(k, (uint64_t) n);
    size_t used = 0;
    for (size_t j = 0; j < n_imgs; ++j)
        if (imgs[j].start < n) {
            k = conversation_spill_fold(k, (uint64_t) imgs[j].start);
            k = conversation_spill_fold(k, imgs[j].hash);
            ++used;
        }
    return conversation_spill_fold(k, 0x6d61726bull + (uint64_t) used);   // "mark" + the filtered-image count
}
template<class Img>
inline uint64_t conversation_spill_key(const std::vector<int32_t>& ids, const std::vector<Img>& imgs) {
    std::vector<uint64_t> dig;
    conversation_spill_prefix_digests(ids.begin(), ids.end(), dig);
    return conversation_spill_key_at(dig, (int64_t) ids.size(), imgs.data(), imgs.size());
}

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;      // entries_ index; with spilled: the manifest's index
        int64_t tokens = 0;
        bool live = false;
        bool spilled = false;  // the match lives in a session file; restore_spilled reads it back
    };

    // The engine's wiring of the disk tier (generate.cpp fills it).  `write` serializes one entry (session file
    // format v1); `read` refuses a foreign identity or a corrupt file (fail-closed, the caller drops the entry);
    // `remove` deletes the file behind a manifest record.  `admit_write` is the RAM preflight of a spill write
    // (the transient dirty pages of the serialization): false = do not write to disk now.
    struct SpillHooks {
        std::string dir;
        std::function<bool(const SavedConversation&, const std::string& path, std::string& error)> write;
        std::function<bool(const std::string& path, SavedConversation& out, std::string& error)> read;
        std::function<bool(const std::string& path)> remove;
        std::function<bool(size_t need_bytes)> admit_write;
    };

    // The manifest record of one spilled entry: the key hashes (live state first, then the checkpoints), the live
    // state's token count, the geometry it was captured with, and the file.
    struct Spilled {
        std::vector<std::pair<uint64_t, int64_t>> marks;   // (key hash, token count)
        int64_t tokens = 0;                                // the live state's tokens at spill time
        std::array<int64_t, 18> geometry{};
        bool cvec = true;
        size_t bytes = 0;                                  // the RAM the spill freed
        std::string path;
    };

    ConversationCache(size_t budget, size_t slots) : budget_(budget), slots_(slots) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored);
    // spilled entries count too: their K/V must still be restorable (their host RAM is gone, the cells are not)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        for (const auto& s : spilled_) n = std::max<int64_t>(n, s.tokens);
        return n;
    }
    // --- spill state ---
    void set_spill_hooks(SpillHooks hooks) {
        hooks_ = std::move(hooks);
        spill_ready_ = !hooks_.dir.empty() && (bool) hooks_.write && (bool) hooks_.read;
    }
    bool spill_ready() const { return spill_ready_; }
    void set_spill_cooldown(double seconds) { spill_cooldown_s_ = seconds; }
    void set_spill_clock(std::function<double()> now) { spill_clock_ = std::move(now); }
    size_t spills() const { return spills_; }
    size_t spilled_entries() const { return spilled_.size(); }
    size_t spilled_bytes() const { return spilled_bytes_; }
    size_t spilled_restores() const { return spilled_restores_; }
    size_t failed_spills() const { return failed_spills_; }
    double spill_ms() const { return spill_ms_; }
    const std::string& last_spill_error() const { return last_spill_error_; }

    // Spill one eligible entry (oldest first) to disk and free its RAM without an eviction: the gate-refusal loop's
    // unit of work.  False when nothing is eligible (pins, the hysteresis cooldown) or the write failed — nothing
    // is dropped on a failure (fail-open keeps the current behaviour at the gate).
    bool spill_one() {
        if (!spill_ready_) return false;
        for (size_t i = 0; i < entries_.size(); ++i)
            if (spill_write(i)) {
                bytes_ -= entries_[i].bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                metas_.erase(metas_.begin() + (std::ptrdiff_t) i);
                return true;
            }
        return false;
    }
    // Read a spilled entry back; the manifest record goes (the image becomes the caller's active state).  A file
    // that no longer reads (corrupt, identity changed) is dropped from the manifest with its file: fail-open.
    bool restore_spilled(size_t index, SavedConversation& out) {
        out = {};
        if (index >= spilled_.size()) return false;
        std::string error;
        const bool ok = hooks_.read && hooks_.read(spilled_[index].path, out, error);
        if (!ok) {
            last_spill_error_ = error.empty() ? "the spilled session file does not read back" : error;
            if (hooks_.remove) hooks_.remove(spilled_[index].path);
            spilled_bytes_ -= spilled_[index].bytes;
            spilled_.erase(spilled_.begin() + (std::ptrdiff_t) index);
            return false;
        }
        ++spilled_restores_;
        spilled_bytes_ -= spilled_[index].bytes;
        spilled_.erase(spilled_.begin() + (std::ptrdiff_t) index);
        return true;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, false};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        // the disk tier: a spilled entry matches through its manifest marks (the live state and each checkpoint).
        // Only a strictly longer prefix than the best RAM match wins: a restore costs a file read.
        if (!spilled_.empty() && spill_ready_) {
            std::vector<uint64_t> dig;
            conversation_spill_prefix_digests(prompt.begin(), prompt.end(), dig);
            for (size_t i = spilled_.size(); i-- > 0;) {
                const auto& s = spilled_[i];
                if (s.cvec != cvec) continue;
                for (const auto& [key, n] : s.marks) {
                    if (n <= 0 || n >= (int64_t) prompt.size() || n <= best.tokens) continue;
                    if (conversation_spill_key_at(dig, n, images.data(), images.size()) == key)
                        best = {i, n, false, true};
                }
            }
        }
        return best;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        metas_.erase(metas_.begin() + (std::ptrdiff_t) index);
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the pinned prefix is what the queries come back to).  With the disk tier wired,
            // the victim is spilled first (the plan's active path: the budget eviction writes a session file instead
            // of dropping); a spill that fails is refused falls back to the drop.  A victim inside the hysteresis
            // cooldown is not evicted at all — the parking is refused rather than thrashing (or dropping) a fresh
            // conversation; later candidates are still tried.
            bool evicted = false;
            for (size_t i = 0; i < entries_.size(); ++i) {
                if (entries_[i].pinned()) continue;
                if (spill_ready_ && now() - metas_[i].last_activity < spill_cooldown_s_) continue;
                spill_write(i);   // spilled, or failed -> the current behaviour: dropped
                bytes_ -= entries_[i].bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                metas_.erase(metas_.begin() + (std::ptrdiff_t) i);
                ++evictions_;
                evicted = true;
                break;
            }
            if (!evicted) return false;
        }
        return true;
    }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                bytes_ -= e.bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                metas_.erase(metas_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        // the disk tier: a spilled copy whose deepest mark the outgoing chain holds is stale the same way — its
        // file goes with the manifest record (the newer park carries its content forward)
        if (!spilled_.empty()) {
            std::vector<uint64_t> dig;
            conversation_spill_prefix_digests(ids.begin(), ids.end(), dig);
            auto chain_key = [&](int64_t n, const ConversationImageKey* imgs, size_t n_imgs) {
                if (n <= 0 || n > (int64_t) ids.size()) return uint64_t(0);
                return conversation_spill_key_at(dig, n, imgs, n_imgs);
            };
            for (size_t i = spilled_.size(); i-- > 0;) {
                const auto& s = spilled_[i];
                std::pair<uint64_t, int64_t> deepest{0, 0};
                for (const auto& m : s.marks) if (m.second > deepest.second) deepest = m;
                if (deepest.second <= 0 || s.cvec != cvec) continue;
                bool stale = false;
                if (chain_key(deepest.second, images.data(), images.size()) == deepest.first) stale = true;
                if (!stale)
                    for (const auto& k : checkpoints)
                        if ((int64_t) k.ids.size() == deepest.second &&
                            conversation_spill_key(k.ids, k.imgs) == deepest.first) { stale = true; break; }
                if (stale) {
                    if (hooks_.remove) hooks_.remove(s.path);
                    spilled_bytes_ -= s.bytes;
                    spilled_.erase(spilled_.begin() + (std::ptrdiff_t) i);
                    ++dropped;
                }
            }
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        metas_.push_back({now()});
        bytes_ += n;
        return true;
    }

private:
    struct Meta {
        double last_activity = 0;   // the clock at put: the hysteresis never spills the recently active entries
    };

    double now() const { return spill_clock_ ? spill_clock_() : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // Serialize entry `idx` into the spill directory and add the manifest record.  False when the tier is off,
    // the entry is pinned or recently active (hysteresis), the RAM preflight refuses, or the write fails —
    // every refusal leaves the entry in RAM for the caller's fall-back (fail-open).
    bool spill_write(size_t idx) {
        if (!spill_ready_ || idx >= entries_.size()) return false;
        const SavedConversation& e = entries_[idx];
        if (e.pinned()) return false;
        const double t = now();
        if (t - metas_[idx].last_activity < spill_cooldown_s_) return false;
        if (hooks_.admit_write && !hooks_.admit_write(e.bytes() + (size_t(32) << 20))) {
            ++failed_spills_;
            last_spill_error_ = "RAM admission refused the spill write";
            return false;
        }
        Spilled m;
        m.tokens = (int64_t) e.live.ids.size();
        m.geometry = e.geometry;
        m.cvec = e.cvec;
        m.bytes = e.bytes();
        if (!e.live.ids.empty())
            m.marks.push_back({conversation_spill_key(e.live.ids, e.live.imgs), m.tokens});
        for (const auto& c : e.checkpoints)
            if (!c.ids.empty())
                m.marks.push_back({conversation_spill_key(c.ids, c.imgs), (int64_t) c.ids.size()});
        m.path = hooks_.dir + "/spill-" + std::to_string(++spill_seq_) + ".bin";
        std::string error;
        const double t0 = t;
        if (!hooks_.write(e, m.path, error)) {
            ++failed_spills_;
            last_spill_error_ = error.empty() ? "the spill write failed" : error;
            return false;
        }
        last_spill_error_.clear();
        spill_ms_ += (now() - t0) * 1000.0;
        ++spills_;
        spilled_bytes_ += m.bytes;
        spilled_.push_back(std::move(m));
        return true;
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<Meta> metas_;                // parallel to entries_
    ConversationKvReuse reuse_;
    // the disk tier
    SpillHooks hooks_;
    bool spill_ready_ = false;
    double spill_cooldown_s_ = 120.0;
    std::function<double()> spill_clock_;
    std::deque<Spilled> spilled_;           // the RAM manifest: oldest spill first
    uint64_t spill_seq_ = 0;
    size_t spills_ = 0, spilled_bytes_ = 0, spilled_restores_ = 0, failed_spills_ = 0;
    double spill_ms_ = 0;
    std::string last_spill_error_;
};

} // namespace strata::core
