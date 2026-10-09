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

// --- manifest persistence helpers (the restart-survival patch; shared by both tiers) ------------------
// A tiny sidecar blob: magic, version, an auxiliary word, a record count, the records, and an FNV-1a
// checksum over everything before it.  Hand-rolled little-endian; deliberately versioned and checked so a
// malformed or foreign file is refused before anything is adopted.
inline std::string conversation_manifest_leaf(const std::string& path) {
    const size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? path : path.substr(cut + 1);
}
inline uint64_t conversation_manifest_checksum(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
struct ConversationManifestWriter {
    std::vector<uint8_t> bytes;
    void u8(uint8_t v) { bytes.push_back(v); }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) bytes.push_back(uint8_t(v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) bytes.push_back(uint8_t(v >> (8 * i))); }
    void i64(int64_t v) { u64((uint64_t) v); }
    void blob(const void* p, size_t n) {
        const uint8_t* c = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), c, c + n);
    }
};
struct ConversationManifestReader {
    const uint8_t* p = nullptr;
    size_t n = 0, off = 0;
    bool ok = true;
    bool has(size_t k) const { return ok && off + k <= n; }
    uint8_t u8() { if (!has(1)) { ok = false; return 0; } return p[off++]; }
    uint32_t u32() {
        if (!has(4)) { ok = false; return 0; }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= uint32_t(p[off++]) << (8 * i);
        return v;
    }
    uint64_t u64() {
        if (!has(8)) { ok = false; return 0; }
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(p[off++]) << (8 * i);
        return v;
    }
    int64_t i64() { return (int64_t) u64(); }
    const uint8_t* take(size_t k) {
        if (!has(k)) { ok = false; return nullptr; }
        const uint8_t* q = p + off;
        off += k;
        return q;
    }
};

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

// Disk-authoritative conversation cache (park-on-displacement): a parked conversation lives only in a session
// file (format v1); the cache keeps a RAM manifest of key hashes and nothing else.  A conversation is parked
// exactly when it is displaced (the engine parks the outgoing live state BEFORE the incoming one is read - two
// live K/V states must never be held at once), restored when it comes back, and untouched (zero cache IO) while
// it stays active.  A park write that fails is fail-open: the state is dropped, the same loss the pre-disk cache
// had, and no request ever waits on the disk.  A restore read is triaged: the RAM admission's refusal keeps the
// file and its record (refused is not corrupt); corruption or a foreign identity deletes both; a read that
// succeeds is re-verified token-exactly against the prompt before it becomes the active state (a 64-bit key-hash
// collision would otherwise silently poison the K/V).
class ConversationCache {
public:
    struct Match {
        uint64_t id = 0;       // the manifest record's unique id (stable across drops; never an index)
        int64_t tokens = 0;
        bool live = false;     // false: every match is a manifest record
        bool spilled = true;
    };

    // The engine's wiring of the disk (generate.cpp fills it).  `write` serializes one entry (session file
    // format v1) and owns its temporary file; `read` refuses a foreign identity or a corrupt file (fail-closed)
    // and reports whether a false return was the RAM admission's refusal (the file is intact: B-F2); `remove`
    // deletes the file behind a manifest record.
    struct SpillHooks {
        std::string dir;
        std::function<bool(const SavedConversation&, const std::string& path, std::string& error)> write;
        std::function<bool(const std::string& path, SavedConversation& out, std::string& error, bool& admission_refused)> read;
        std::function<bool(const std::string& path)> remove;
        // Reconcile the spill directory against the manifest: remove every spill file not named in `keep`,
        // return how many were removed, and report the total bytes of the files that remain.  Run at park
        // time (the only regular writer), so a file a restore or a supersede ever leaked cannot accumulate.
        std::function<size_t(const std::vector<std::string>& keep, size_t& dir_bytes)> reap;
    };

    // The manifest record of one parked entry: the key hashes (live state first, then the checkpoints), the live
    // state's token count, the geometry it was captured with, and the file.
    struct Spilled {
        uint64_t id = 0;                                   // unique, never reused; restore addresses records by it
        uint64_t refs = 0;                    // live sessions holding this snapshot: never evicted while > 0
                                              // (RAM-only: like the segment tier's, pins do not survive a restart,
                                              //  so the v1 manifest format carries no refs field)
        std::vector<std::pair<uint64_t, int64_t>> marks;   // (key hash, token count)
        int64_t tokens = 0;                                // the live state's tokens at park time
        std::array<int64_t, 18> geometry{};
        bool cvec = true;
        size_t bytes = 0;                                  // the file's payload, for the accounting
        std::string path;
    };

    // `reuse_budget`: the optional cap on the transient K/V retained between a restore and the next park (the
    // --conversation-cache-mib flag's one remaining use); 0 retains nothing.
    // `spill_dir_budget`: the spill directory's byte cap (R4-capacity, mirroring the segment tier's criterion-11
    // eviction); 0 = unbounded (tests and the pre-budget default).  Eviction runs at park time and, when the free
    // space is short, before a park's admission — evict-then-admit, never skip-park.
    explicit ConversationCache(bool enabled, size_t reuse_budget = 0, size_t spill_dir_budget = 0)
        : enabled_(enabled), reuse_budget_(reuse_budget), spill_dir_budget_(spill_dir_budget) {}
    bool enabled() const { return enabled_; }
    size_t bytes() const { return reuse_.bytes(); }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored;
    // the parked states live on disk, their host RAM is gone, the cells are not)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& s : spilled_) n = std::max<int64_t>(n, s.tokens);
        return n;
    }
    void set_spill_hooks(SpillHooks hooks) {
        hooks_ = std::move(hooks);
        spill_ready_ = !hooks_.dir.empty() && (bool) hooks_.write && (bool) hooks_.read;
    }
    bool spill_ready() const { return spill_ready_; }

    // --- counters (the acceptance criteria read these) ---
    size_t parks() const { return parks_; }
    size_t failed_parks() const { return failed_parks_; }
    size_t spilled_entries() const { return spilled_.size(); }
    size_t spilled_bytes() const { return spilled_bytes_; }
    size_t spilled_restores() const { return spilled_restores_; }
    size_t refused_restores() const { return refused_restores_; }
    size_t superseded() const { return superseded_; }
    double park_ms() const { return park_ms_; }
    // the spill directory's total bytes as of the last park's reconciliation (acceptance 9's observable), and
    // the largest manifest-recorded snapshot
    size_t dir_bytes() const { return dir_bytes_; }
    size_t max_snapshot_bytes() const {
        size_t n = 0;
        for (const auto& s : spilled_) n = std::max(n, s.bytes);
        return n;
    }
    // --- the spill tier's directory-byte budget (R4-capacity: mirror the segment tier's criterion 11) ---
    // Candidate = records with refs==0; order = manifest order (spilled_ is oldest-park first, and adoption
    // preserves the manifest's order, so this is oldest-first); evict until the record-byte accounting
    // (spilled_bytes_, the same currency the segment tier's dir_bytes_ keeps) is within the budget.
    // `incoming` is a park's own bytes: evict_for_admission makes room for the file ABOUT to be written,
    // so a park under a tight directory evicts first and then admits — never skips the park.
    size_t evictions() const { return evictions_; }
    size_t ref(uint64_t id) {   // an active session pins its record; RAM-only like the segment tier's pins
        for (auto& s : spilled_)
            if (s.id == id) { ++s.refs; return (size_t) s.refs; }
        return 0;
    }
    void unref(uint64_t id) {
        for (auto& s : spilled_)
            if (s.id == id && s.refs > 0) { --s.refs; return; }
    }
    size_t evict_for_admission(size_t incoming) {
        if (spill_dir_budget_ == 0) return 0;
        size_t evicted = 0;
        while (spilled_bytes_ + incoming > spill_dir_budget_) {
            size_t victim = spilled_.size();
            for (size_t i = 0; i < spilled_.size(); ++i)
                if (spilled_[i].refs == 0) { victim = i; break; }   // oldest first (deque order)
            if (victim == spilled_.size()) break;   // everything left is pinned: the budget yields
            drop_spilled(victim);
            ++evicted;
            ++evictions_;
        }
        return evicted;
    }
    size_t evict_to_budget() { return evict_for_admission(0); }
    const std::string& last_error() const { return last_error_; }

    // The manifest match: the longest recorded prefix that the prompt's digests reproduce.  Pure arithmetic over
    // the manifest - no file is read, nothing is written (an active conversation touches nothing here).
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        if (!spill_ready_ || spilled_.empty()) return best;   // no manifest, no digest work (the active turn's cost)
        std::vector<uint64_t> dig;
        conversation_spill_prefix_digests(prompt.begin(), prompt.end(), dig);
        for (size_t i = spilled_.size(); i-- > 0;) {
            const auto& s = spilled_[i];
            if (s.cvec != cvec) continue;
            for (const auto& [key, n] : s.marks) {
                if (n <= 0 || n >= (int64_t) prompt.size() || n <= best.tokens) continue;
                if (conversation_spill_key_at(dig, n, images.data(), images.size()) == key)
                    best = {s.id, n, false, true};
            }
        }
        return best;
    }

    // Park a displaced conversation: serialize it into the spill directory, add the manifest record, and free its
    // RAM (the image is consumed either way).  False is fail-open: the write failed, the state is dropped (the
    // same loss the pre-disk cache had); no record exists and no half-written file is left behind.
    bool park(SavedConversation&& image) {
        last_error_.clear();
        if (!enabled_ || !spill_ready_) {
            last_error_ = "the conversation cache is disabled";
            return false;
        }
        if (!image.stage_images.empty()) {   // A-F1: format v1 does not carry a layer split's stage images
            last_error_ = "a layer split's stage images do not park";
            ++failed_parks_;   // the refusal drops the state, the same loss as a failed write
            return false;
        }
        Spilled m;
        m.id = ++spill_seq_;
        m.tokens = (int64_t) image.live.ids.size();
        m.geometry = image.geometry;
        m.cvec = image.cvec;
        m.bytes = image.bytes();
        if (!image.live.ids.empty())
            m.marks.push_back({conversation_spill_key(image.live.ids, image.live.imgs), m.tokens});
        for (const auto& c : image.checkpoints)
            if (!c.ids.empty())
                m.marks.push_back({conversation_spill_key(c.ids, c.imgs), (int64_t) c.ids.size()});
        m.path = hooks_.dir + "/spill-" + std::to_string(m.id) + ".bin";
        // Evict-then-admit (requirement 5): before the write, the budget makes room for these bytes (oldest
        // refs==0 first, pinned records never), so a full directory does not skip the park and the free-space
        // admission below stays the last resort (fail-open on a genuinely full disk).
        evict_for_admission(m.bytes);
        const double t0 = now();
        std::string error;
        if (!hooks_.write(image, m.path, error)) {
            ++failed_parks_;
            last_error_ = error.empty() ? "the park write failed" : error;
            if (hooks_.remove) hooks_.remove(m.path);   // never a half-written file at the record's path
            return false;
        }
        park_ms_ += (now() - t0) * 1000.0;
        ++parks_;
        spilled_bytes_ += m.bytes;
        spilled_.push_back(std::move(m));
        // Reconcile the directory against the manifest: a file a restore or a supersede ever left behind
        // cannot accumulate (the acceptance bound is on the directory, not on the manifest).
        if (hooks_.reap) {
            std::vector<std::string> keep;
            keep.reserve(spilled_.size());
            for (const auto& s : spilled_) keep.push_back(s.path);
            size_t bytes = 0;
            hooks_.reap(keep, bytes);
            dir_bytes_ = bytes;
        }
        return true;
    }

    // Restore the record with the unique id `id` (an id survives the manifest shifting under drops, which a
    // capturable index cannot): read its file back, re-verify the state against this prompt token-exactly
    // (B-F3: the key hash matched, the tokens must too), and hand the image to the caller.  The prompt may be
    // a strict prefix of the parked state (a rewind): the shared prefix - min(the prompt's usable length, the
    // state's) tokens - is what is verified and what `*mount_tokens` reports (the deepest checkpoint at or
    // below the prompt boundary, or the whole live state when it fits); the caller mounts the state there.
    // Triage on failure: an admission refusal keeps the file and the record (WARNING only - refused is not
    // corrupt, B-F2); a verification miss or a corrupt/foreign-identity read deletes both, so a poisoned
    // entry cannot match again; a rewind with no mountable checkpoint keeps both too (the file is intact).
    template<class Token>
    bool restore_spilled_exact(uint64_t id, int64_t tokens, const std::vector<Token>& prompt,
                               const std::vector<ConversationImageKey>& images, SavedConversation& out,
                               int64_t* mount_tokens = nullptr) {
        out = {};
        size_t index = spilled_.size();
        for (size_t i = 0; i < spilled_.size(); ++i)
            if (spilled_[i].id == id) { index = i; break; }
        if (index == spilled_.size()) {
            last_error_ = "the manifest record is gone (dropped while this conversation was being parked)";
            return false;
        }
        bool refused = false;
        std::string error;
        const bool ok = hooks_.read && hooks_.read(spilled_[index].path, out, error, refused);
        if (!ok) {
            last_error_ = error.empty() ? "the session file does not read back (corrupt or a foreign identity)"
                                        : error;
            if (refused) {   // B-F2: the RAM admission turned the read down; the file and its record stay
                ++refused_restores_;
                return false;
            }
            drop_spilled(index);
            return false;
        }
        // B-F3: the verified prefix.  The last prompt token starts the next verify window, so a prefix may
        // cover at most prompt.size() - 1 tokens of any state.
        const int64_t cap = (int64_t) prompt.size() - 1;
        auto prefix_ok = [&](const ConversationCheckpoint& c, int64_t limit, int64_t& m) {
            m = std::min<int64_t>((int64_t) c.ids.size(), limit);
            if (m <= 0 || !std::equal(c.ids.begin(), c.ids.begin() + m, prompt.begin())) return false;
            size_t j = 0;
            for (const auto& image : images) {
                if (image.start >= m) continue;
                if (j == c.imgs.size() || !(c.imgs[j++] == image)) return false;
            }
            return j == c.imgs.size();
        };
        int64_t verified = 0;   // the longest token-verified prefix (B-F3's gate)
        int64_t mount = 0;      // the longest verified prefix a mountable state carries
        auto check = [&](const ConversationCheckpoint& c, bool mountable) {
            int64_t m = 0;
            if (!prefix_ok(c, cap, m)) return;
            if (m > verified) verified = m;
            // a state is mountable only whole (its saved running state is): a live state longer than the
            // prompt verifies its shared prefix but cannot mount; a checkpoint at or below the boundary can
            if (mountable && m == (int64_t) c.ids.size() && m > mount) mount = m;
        };
        check(out.live, true);
        for (const auto& c : out.checkpoints) check(c, true);
        (void) tokens;   // the verified prefix stands on its own token equality, on whichever state carries it
        if (verified <= 0) {
            last_error_ = "the restored state does not match the prompt token-exactly (verification miss: "
                          "a key-hash collision or a stale manifest index)";
            drop_spilled(index);
            return false;
        }
        if (mount <= 0) {   // intact file, nothing to mount: keep the file and the record (B-F2)
            last_error_ = "the parked conversation is longer than this prompt and holds no checkpoint at or "
                          "before its boundary";
            return false;
        }
        if (mount_tokens) *mount_tokens = mount;
        ++spilled_restores_;
        hooks_.remove(spilled_[index].path);   // the file was consumed: it must not leak on the disk
        spilled_bytes_ -= spilled_[index].bytes;
        spilled_.erase(spilled_.begin() + (std::ptrdiff_t) index);
        return true;
    }

    // #342: drop the parked records the outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose checkpoints the outgoing chain still holds - any mark (the live
    // state's or a checkpoint's) that the outgoing chain reproduces at that depth makes the record stale, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A record with no matching mark, or with no marks at all (another
    // conversation that only shares the system prompt's root with it), is kept.  Returns
    // how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        if (!spilled_.empty()) {
            std::vector<uint64_t> dig;
            conversation_spill_prefix_digests(ids.begin(), ids.end(), dig);
            auto chain_key = [&](int64_t n, const ConversationImageKey* imgs, size_t n_imgs) {
                if (n <= 0 || n > (int64_t) ids.size()) return uint64_t(0);
                return conversation_spill_key_at(dig, n, imgs, n_imgs);
            };
            for (size_t i = spilled_.size(); i-- > 0;) {
                const auto& s = spilled_[i];
                if (s.cvec != cvec) continue;
                bool stale = false;
                // ANY mark the outgoing chain reproduces at that depth makes the record stale: a thinking
                // conversation re-renders the tail every turn, so the deepest mark sits in the rewritten
                // region and never compares equal - a shallower mark (before the fork) still hits.  Two
                // different conversations cannot share a key on a non-empty prefix, so this cannot drop a
                // record that only shares the system root.
                for (const auto& [key, n] : s.marks) {
                    if (n <= 0) continue;
                    if (chain_key(n, images.data(), images.size()) == key) { stale = true; break; }
                    for (const auto& k : checkpoints)
                        if ((int64_t) k.ids.size() == n &&
                            conversation_spill_key(k.ids, k.imgs) == key) { stale = true; break; }
                    if (stale) break;
                }
                if (stale) {
                    drop_spilled(i);
                    ++dropped;
                }
            }
        }
        superseded_ += dropped;
        return dropped;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints, for the next park's capture.
    // This retention is transient (it dies with the next park or take_reuse) and capped by `reuse_budget`.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (reuse_budget_ != 0 && candidate.bytes() <= reuse_budget_ - reuse_.bytes()) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }

    // ---- manifest persistence: survive a restart (the restart-survival patch) ------------------------
    // Phase one cleared the spill directory at startup and the RAM manifest died with the process; now the
    // manifest is dumped beside the parked files (the engine writes the blob) and adopted at the next start,
    // so a returning prompt can restore a conversation the previous run parked.  Fail-open throughout: a
    // missing, unreadable or malformed dump is "no manifest" (the caller clears the directory as before),
    // and a record whose file is not in `present` (file names) is dropped at adoption.
    std::vector<uint8_t> dump_manifest() const {
        ConversationManifestWriter w;
        static const char kMagic[8] = {'S', 'T', 'C', 'M', 'A', 'N', 'P', '1'};
        w.blob(kMagic, 8);
        w.u32(1);   // version
        w.u32(0);   // reserved
        w.u64(spill_seq_);
        w.u64(spilled_.size());
        for (const auto& s : spilled_) {
            w.u64(s.id);
            w.i64(s.tokens);
            w.u8(s.cvec ? 1 : 0);
            for (int k = 0; k < 7; ++k) w.u8(0);   // pad to 8
            w.u64(s.bytes);
            for (int64_t g : s.geometry) w.i64(g);
            w.u64(s.marks.size());
            for (const auto& [key, n] : s.marks) {
                w.u64(key);
                w.i64(n);
            }
            const std::string leaf = conversation_manifest_leaf(s.path);
            w.u64(leaf.size());
            w.blob(leaf.data(), leaf.size());
        }
        w.u64(conversation_manifest_checksum(w.bytes.data(), w.bytes.size()));
        return std::move(w.bytes);
    }
    bool adopt_manifest(const std::vector<uint8_t>& blob, const std::vector<std::string>& present,
                        std::string& error) {
        ConversationManifestReader r{blob.data(), blob.size()};
        const uint8_t* m = r.take(8);
        if (!m || std::string(reinterpret_cast<const char*>(m), 8) != "STCMANP1") {
            error = "not a parked manifest";
            return false;
        }
        if (r.u32() != 1) { error = "unsupported parked manifest version"; return false; }
        (void) r.u32();
        const uint64_t seq = r.u64();
        const uint64_t count = r.u64();
        if (!r.ok || count > 65536) { error = "malformed parked manifest header"; return false; }
        std::deque<Spilled> records;
        for (uint64_t i = 0; i < count; ++i) {
            Spilled s;
            s.id = r.u64();
            s.tokens = r.i64();
            s.cvec = r.u8() != 0;
            for (int k = 0; k < 7; ++k) (void) r.u8();
            s.bytes = (size_t) r.u64();
            for (int64_t& g : s.geometry) g = r.i64();
            const uint64_t marks = r.u64();
            if (!r.ok || marks > 4096) { error = "malformed parked record"; return false; }
            for (uint64_t k = 0; k < marks; ++k) {
                const uint64_t key = r.u64();
                const int64_t n = r.i64();
                s.marks.push_back({key, n});
            }
            const uint64_t path_len = r.u64();
            if (!r.ok || path_len == 0 || path_len > 4096) { error = "malformed parked record"; return false; }
            const uint8_t* p = r.take((size_t) path_len);
            if (!p) { error = "malformed parked record"; return false; }
            const std::string leaf(reinterpret_cast<const char*>(p), (size_t) path_len);
            if (leaf.find('/') != std::string::npos || leaf.find('\\') != std::string::npos) {
                error = "malformed parked path";
                return false;
            }
            s.path = hooks_.dir.empty() ? leaf : hooks_.dir + "/" + leaf;
            records.push_back(std::move(s));
        }
        const uint64_t sum = r.u64();
        if (!r.ok || r.off != blob.size()) { error = "malformed parked manifest tail"; return false; }
        if (conversation_manifest_checksum(blob.data(), blob.size() - 8) != sum) {
            error = "parked manifest checksum mismatch";
            return false;
        }
        std::deque<Spilled> kept;
        for (auto& s : records) {
            bool here = false;
            for (const auto& p : present)
                if (p == conversation_manifest_leaf(s.path)) { here = true; break; }
            if (here) kept.push_back(std::move(s));
        }
        uint64_t max_id = seq;
        size_t bytes = 0;
        for (const auto& s : kept) {
            max_id = std::max(max_id, s.id);
            bytes += s.bytes;
        }
        spilled_ = std::move(kept);
        spill_seq_ = max_id;   // the next park's id continues past the adopted ones
        spilled_bytes_ = bytes;
        // R4-capacity: an adopted set over the spill budget is evicted at once — nothing is pinned at boot,
        // so refs==0 (all) evict oldest-park-first (the manifest's order), each eviction deleting the file
        // with the record. The caller's keep-list and any future dump then reflect the evicted set.
        evict_to_budget();
        return true;
    }
    // The parked files' names, for the startup keep-list of the directory clear.
    std::vector<std::string> spilled_files() const {
        std::vector<std::string> v;
        v.reserve(spilled_.size());
        for (const auto& s : spilled_) v.push_back(conversation_manifest_leaf(s.path));
        return v;
    }

private:
    void drop_spilled(size_t index) {
        if (hooks_.remove) hooks_.remove(spilled_[index].path);
        spilled_bytes_ -= spilled_[index].bytes;
        spilled_.erase(spilled_.begin() + (std::ptrdiff_t) index);
    }

    double now() const { return spill_clock_ ? spill_clock_() : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    bool enabled_ = false;
    size_t reuse_budget_ = 0;
    size_t spill_dir_budget_ = 0;   // R4-capacity: the spill directory's byte cap (0 = unbounded)
    ConversationKvReuse reuse_;
    SpillHooks hooks_;
    bool spill_ready_ = false;
    std::function<double()> spill_clock_;
    std::deque<Spilled> spilled_;           // the RAM manifest: oldest park first
    uint64_t spill_seq_ = 0;
    size_t parks_ = 0, failed_parks_ = 0, spilled_bytes_ = 0, spilled_restores_ = 0, refused_restores_ = 0,
           superseded_ = 0, evictions_ = 0;
    size_t dir_bytes_ = 0;
    double park_ms_ = 0;
    std::string last_error_;
};

} // namespace strata::core
