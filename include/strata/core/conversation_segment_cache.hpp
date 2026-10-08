// The segmented SSD cache (the benchmark document §4): the cache atom moves from the conversation snapshot
// to the prefix segment.  A segment covers a page-checkpoint-aligned token range [0, n) - a cut of the one
// branch of history whose cells the session holds - and its key is the prefix key at its end depth (the
// existing content-addressed key: digests of the tokens folded with the images that open inside, so the same
// prefix yields the same key whatever the checkpoint cadence was).  Admission is append-only with dedup by
// key; eviction is refs==0 and oldest last_hit first under a directory-byte budget (the spill leak's
// prosecutor conversations are its intended prey - criterion 11); retrieval matches the deepest whole
// segment along the prompt's hash chain, mounts it (a RAM copy of the file's running state - the K/V for the
// matched tokens comes back with it), and the caller re-reads only the fork tail.  The whole-file
// conversation cache (conversation_cache.hpp) stays untouched beside it as the fallback path: the segment
// cache never reads, writes, or deletes anything of the spill tier's.
//
// Like the spill tier, a write that fails is fail-open (the segment is dropped, no record exists, no
// half-written file is left at the record's path), and a mount verifies the segment's tokens against the
// prompt token-exactly before the caller may use it (a 64-bit key collision or a corrupt file is a discard,
// never wrong output).
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// The segment-bound chooser: where a session's verified stable prefix may be cut into segments.  A cut is
// legal only at a checkpoint depth (a mountable state must exist there: the running state is part of the
// segment), never past the verified stable depth, and segments at least `min_seg` apart (the production
// cadence - 4096-page-aligned 16K checkpoints - passes through unchanged).  A remainder shorter than
// min_seg is skipped: its K/V is covered when the chain grows past the next checkpoint.  `checkpoint_lens`
// need not be sorted; duplicates are harmless.
inline std::vector<int64_t> conversation_segment_bounds(const std::vector<int64_t>& checkpoint_lens,
                                                        int64_t stable_depth, int64_t min_seg = 4096) {
    std::vector<int64_t> lens = checkpoint_lens;
    std::sort(lens.begin(), lens.end());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    std::vector<int64_t> cuts;
    int64_t start = 0;
    for (int64_t n : lens) {
        if (n <= 0 || n > stable_depth) break;
        if (n - start >= min_seg) {
            cuts.push_back(n);
            start = n;
        }
    }
    return cuts;
}

class ConversationSegmentCache {
public:
    // The engine's wiring of the disk (generate.cpp fills it).  `write` serializes one segment image
    // (session file format v1: live = the segment-end checkpoint, K/V covering the segment's tokens) and
    // owns its temporary file; `read` refuses a missing or corrupt file (fail-closed); `remove` deletes the
    // file behind a record.  `clock` (optional) is the eviction clock; steady by default.
    struct Hooks {
        std::string dir;
        std::function<bool(const SavedConversation&, const std::string& path, std::string& error)> write;
        std::function<bool(const std::string& path, SavedConversation& out, std::string& error)> read;
        std::function<bool(const std::string& path)> remove;
        std::function<uint64_t()> clock;
    };

    // The manifest record of one segment: content-addressed by its end-depth key.
    struct Segment {
        uint64_t key = 0;     // the prefix key at the segment's end depth (the file's name, too)
        int64_t tokens = 0;   // the segment's end depth (it covers [0, tokens))
        size_t bytes = 0;     // the record's byte accounting (the image's payload at admit time)
        uint64_t last_hit = 0;
        size_t refs = 0;      // active sessions relying on this segment (never evicted while > 0)
        bool cvec = true;
        std::string path;
    };

    struct Match {
        uint64_t key = 0;
        int64_t tokens = 0;
        bool found = false;
    };

    // `dir_budget`: the directory-byte cap (the eviction trigger, criterion 11); 0 = unbounded (tests only).
    explicit ConversationSegmentCache(bool enabled, size_t dir_budget, Hooks hooks)
        : enabled_(enabled), dir_budget_(dir_budget), hooks_(std::move(hooks)) {}

    bool enabled() const { return enabled_; }

    // --- counters (the acceptance criteria read these) ---
    size_t admits() const { return admits_; }
    size_t dedup_hits() const { return dedup_hits_; }
    size_t evictions() const { return evictions_; }
    size_t mounts() const { return mounts_; }
    size_t failed_mounts() const { return failed_mounts_; }
    size_t entries() const { return segments_.size(); }
    size_t dir_bytes() const { return dir_bytes_; }
    size_t max_segment_bytes() const {
        size_t n = 0;
        for (const auto& s : segments_) n = std::max(n, s.bytes);
        return n;
    }
    size_t refs(uint64_t key) const {
        for (const auto& s : segments_) if (s.key == key) return s.refs;
        return 0;
    }
    const std::string& last_error() const { return last_error_; }

    // The manifest match: the deepest recorded segment key that the prompt's digests reproduce.  Pure
    // arithmetic over the manifest - no file is read, nothing is written.  Ties (two records at the same
    // depth cannot share a key on a non-empty prefix) resolve to the first match.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        if (!enabled_ || segments_.empty()) return best;
        std::vector<uint64_t> dig;
        conversation_spill_prefix_digests(prompt.begin(), prompt.end(), dig);
        for (const auto& s : segments_) {
            if (s.cvec != cvec || s.tokens <= 0 || s.tokens >= (int64_t) prompt.size() || s.tokens <= best.tokens)
                continue;
            if (conversation_spill_key_at(dig, s.tokens, images.data(), images.size()) == s.key)
                best = {s.key, s.tokens, true};
        }
        return best;
    }

    // Admit one stable segment (the verified prefix up to a segment end, with its running state and K/V):
    // append-only, deduped by key - a segment already on disk is touched, not rewritten (criterion 10: the
    // same conversation's N turns land as one copy of the shared prefix).  The image is consumed either way.
    // False is a dedup hit (true only when the file was written or the admission was refused); a failed
    // write is fail-open: no record, no half-written file at the record's path.
    bool admit(SavedConversation&& image) {
        last_error_.clear();
        if (!enabled_) {
            last_error_ = "the segment cache is disabled";
            return false;
        }
        if (!image.stage_images.empty()) {   // A-F1's guard carries over: format v1 does not carry a layer split
            last_error_ = "a layer split's stage images do not segment";
            ++failed_admits_;
            return false;
        }
        if (image.live.ids.empty()) {
            last_error_ = "a segment with no tokens has no key";
            ++failed_admits_;
            return false;
        }
        Segment s;
        s.key = conversation_spill_key(image.live.ids, image.live.imgs);
        s.tokens = (int64_t) image.live.ids.size();
        s.cvec = image.cvec;
        s.bytes = image.bytes();
        s.path = hooks_.dir + "/seg-" + hex(s.key) + ".bin";
        for (auto& old : segments_) {
            if (old.key != s.key) continue;
            old.last_hit = tick();   // dedup: touch, never rewrite
            ++dedup_hits_;
            return false;
        }
        s.last_hit = tick();
        std::string error;
        if (!hooks_.write || !hooks_.write(image, s.path, error)) {
            ++failed_admits_;
            last_error_ = error.empty() ? "the segment write failed" : error;
            if (hooks_.remove) hooks_.remove(s.path);   // never a half-written file at the record's path
            return false;
        }
        dir_bytes_ += s.bytes;
        segments_.push_back(std::move(s));
        ++admits_;
        evict_to_budget();
        return true;
    }

    // Mount the segment `key` for this prompt: read the file back, verify the segment's tokens against the
    // prompt token-exactly (a key match is not enough - B-F3's gate carries over), touch the record, and
    // hand the image to the caller.  The file and the record stay: a segment is shared, consumed by copy.
    // On a failed read the record and its file are dropped (a poisoned entry must not match again).
    template<class Token>
    bool mount(uint64_t key, int64_t tokens, const std::vector<Token>& prompt,
               const std::vector<ConversationImageKey>& images, SavedConversation& out, int64_t* seg_tokens) {
        out = {};
        if (seg_tokens) *seg_tokens = 0;
        size_t index = segments_.size();
        for (size_t i = 0; i < segments_.size(); ++i)
            if (segments_[i].key == key) { index = i; break; }
        if (index == segments_.size()) {
            last_error_ = "the segment record is gone";
            ++failed_mounts_;
            return false;
        }
        std::string error;
        if (!hooks_.read || !hooks_.read(segments_[index].path, out, error)) {
            ++failed_mounts_;
            last_error_ = error.empty() ? "the segment file does not read back (corrupt or missing)" : error;
            drop(index);
            return false;
        }
        // B-F3: the verified prefix.  The last prompt token starts the next verify window, so the segment
        // may cover at most prompt.size() - 1 tokens.
        const int64_t cap = (int64_t) prompt.size() - 1;
        const int64_t m = std::min<int64_t>(tokens, cap);
        if (m <= 0 || m > (int64_t) out.live.ids.size() ||
            !std::equal(out.live.ids.begin(), out.live.ids.begin() + m, prompt.begin())) {
            ++failed_mounts_;
            last_error_ = "the mounted segment does not match the prompt token-exactly "
                          "(verification miss: a key-hash collision or a stale record)";
            drop(index);
            return false;
        }
        size_t j = 0;
        for (const auto& image : images) {
            if (image.start >= m) continue;
            if (j == out.live.imgs.size() || !(out.live.imgs[j++] == image)) {
                ++failed_mounts_;
                last_error_ = "the mounted segment's images do not match the prompt's";
                drop(index);
                return false;
            }
        }
        if (j != out.live.imgs.size()) {
            ++failed_mounts_;
            last_error_ = "the mounted segment's images do not match the prompt's";
            drop(index);
            return false;
        }
        segments_[index].last_hit = tick();
        ++mounts_;
        if (seg_tokens) *seg_tokens = m;
        return true;
    }

    // Active-session pinning: a session whose live state extends a segment holds it (never evicted while
    // refs > 0).  ref returns the count after; unref saturates at zero.
    size_t ref(uint64_t key) {
        for (auto& s : segments_)
            if (s.key == key) { ++s.refs; s.last_hit = tick(); return s.refs; }
        return 0;
    }
    void unref(uint64_t key) {
        for (auto& s : segments_)
            if (s.key == key && s.refs > 0) { --s.refs; return; }
    }

    // The eviction pass (criterion 11): delete refs==0 records, oldest last_hit first, until the directory
    // is within the budget (or nothing evictable remains).  Returns how many were deleted.
    size_t evict_to_budget() {
        size_t evicted = 0;
        if (dir_budget_ == 0 || dir_bytes_ <= dir_budget_) return evicted;
        while (dir_bytes_ > dir_budget_) {
            size_t coldest = segments_.size();
            for (size_t i = 0; i < segments_.size(); ++i)
                if (segments_[i].refs == 0 && (coldest == segments_.size() || segments_[i].last_hit < segments_[coldest].last_hit))
                    coldest = i;
            if (coldest == segments_.size()) break;   // everything left is pinned: the budget yields
            drop(coldest);
            ++evicted;
            ++evictions_;
        }
        return evicted;
    }

private:
    void drop(size_t index) {
        if (hooks_.remove) hooks_.remove(segments_[index].path);
        dir_bytes_ -= std::min(dir_bytes_, segments_[index].bytes);
        segments_.erase(segments_.begin() + (std::ptrdiff_t) index);
    }
    uint64_t tick() { return hooks_.clock ? hooks_.clock() : last_hit_seq_++; }
    static std::string hex(uint64_t v) {
        static const char* d = "0123456789abcdef";
        std::string s(16, '0');
        for (int i = 15; i >= 0; --i) { s[(size_t) i] = d[v & 0xf]; v >>= 4; }
        return s;
    }

    bool enabled_ = false;
    size_t dir_budget_ = 0;
    Hooks hooks_;
    std::deque<Segment> segments_;
    uint64_t last_hit_seq_ = 0;
    size_t admits_ = 0, dedup_hits_ = 0, evictions_ = 0, mounts_ = 0, failed_mounts_ = 0, failed_admits_ = 0;
    size_t dir_bytes_ = 0;
    std::string last_error_;
};

} // namespace strata::core
