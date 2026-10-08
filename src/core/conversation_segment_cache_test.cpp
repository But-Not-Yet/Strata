// The segmented SSD cache (R4, the benchmark document §4): behaviour tests over fake spill hooks (the pure
// host policy) plus the segment-bound chooser.  A segment's key is the prefix key at its end depth (content-
// addressed, immune to the checkpoint cadence's drift); admission is append-only with dedup by key; eviction
// is refs==0 and oldest last_hit first under a directory-byte budget; retrieval mounts the deepest hit and
// the caller re-reads only the fork tail.  The whole-file conversation cache stays untouched beside it.
#include "strata/core/conversation_segment_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "strata/core/conversation_cache.hpp"

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}

// A fake disk: path -> payload bytes.  The write serializes the image's payload; a read rebuilds it; a corrupt
// file is simulated by making reads of paths that contain a marker fail.
struct FakeDisk {
    std::map<std::string, std::vector<uint8_t>> files;
    size_t writes = 0;
    std::vector<std::string> removed;
    bool corrupt_reads = false;        // every read fails (a corrupt file)
};

SavedConversation segment_image(const std::vector<int32_t>& ids, uint8_t salt) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.assign(64, salt);
    s.cvec = true;
    return s;
}

ConversationSegmentCache::Hooks fake_hooks(ConversationSegmentCache::Hooks& h, FakeDisk& disk) {
    h.dir = "/fake";
    h.write = [&](const SavedConversation& image, const std::string& path, std::string&) {
        auto& f = disk.files[path];   // a toy serialization: the id list, then the running state
        f.clear();
        for (int32_t id : image.live.ids) {
            f.push_back(uint8_t(id)); f.push_back(uint8_t(id >> 8));
            f.push_back(uint8_t(id >> 16)); f.push_back(uint8_t(id >> 24));
        }
        f.insert(f.end(), image.live.gdn.begin(), image.live.gdn.end());
        disk.writes++;
        return true;
    };
    h.read = [&](const std::string& path, SavedConversation& out, std::string& e) {
        if (disk.corrupt_reads) { e = "corrupt segment file"; return false; }
        auto it = disk.files.find(path);
        if (it == disk.files.end()) { e = "no such segment file"; return false; }
        const size_t n = it->second.size() - 64;
        out.live.ids.resize(n / 4);
        for (size_t i = 0; i < out.live.ids.size(); ++i)
            out.live.ids[i] = int32_t(uint32_t(it->second[4*i]) | uint32_t(it->second[4*i+1]) << 8 |
                                      uint32_t(it->second[4*i+2]) << 16 | uint32_t(it->second[4*i+3]) << 24);
        out.live.gdn.assign(it->second.begin() + (std::ptrdiff_t) n, it->second.end());
        return true;
    };
    h.remove = [&](const std::string& path) { disk.removed.push_back(path); disk.files.erase(path); return true; };
    return h;
}

} // namespace

int main() {
    {
        // §4.1 / §4.2: the key is content-addressed.  Two sessions whose checkpoint cadence drifted (the
        // segment cut at the same depth in both) produce the same key: the second admit is a dedup hit, one
        // file, bytes counted once.  A deeper cut on the same chain is its own key.
        FakeDisk disk;
        ConversationSegmentCache::Hooks h;
        ConversationSegmentCache cache(true, 1 << 20, fake_hooks(h, disk));

        const std::vector<int32_t> tokens(200, 11);
        SavedConversation a = segment_image(std::vector<int32_t>(tokens.begin(), tokens.begin() + 100), 1);
        check(cache.admit(std::move(a)), "the first stable segment admits");
        check(cache.admits() == 1 && cache.entries() == 1 && cache.dir_bytes() > 0, "one record after one admit");
        SavedConversation drift = segment_image(std::vector<int32_t>(tokens.begin(), tokens.begin() + 100), 2);
        check(!cache.admit(std::move(drift)), "the same key's second admit is a dedup skip");
        check(cache.admits() == 1 && cache.dedup_hits() == 1 && cache.entries() == 1,
              "dedup: no second record, no double bytes");
        check(disk.writes == 1, "the dedup skip never touches the disk");
        SavedConversation longer = segment_image(tokens, 3);
        check(cache.admit(std::move(longer)), "a deeper segment of the same chain is its own key");
        check(cache.entries() == 2, "a two-segment chain holds two records");
        check(cache.dir_bytes() == (100 * sizeof(int32_t) + 64) + (200 * sizeof(int32_t) + 64),
              "the directory counts every record's bytes");
    }
    {
        // §4.4 retrieval: a chain A-B-C parked as segments at depths 100/200/300; a new prompt reproduces
        // A-B and forks at 250 -> best() is B (200); mount returns B's state verified token-exactly, and the
        // fork tail (the 50 tokens after 200) is all the caller re-reads.  A rewind inside B falls back to A.
        FakeDisk disk;
        ConversationSegmentCache::Hooks h;
        ConversationSegmentCache cache(true, 1 << 20, fake_hooks(h, disk));
        const std::vector<int32_t> tokens(300, 11);
        for (int64_t depth : {100, 200, 300}) {
            SavedConversation s = segment_image(std::vector<int32_t>(tokens.begin(), tokens.begin() + depth),
                                                uint8_t(depth));
            check(cache.admit(std::move(s)), "every chain segment admits");
        }
        check(cache.entries() == 3, "chain A-B-C: three segment records");

        std::vector<int32_t> prompt(tokens.begin(), tokens.begin() + 250);
        const auto best = cache.best(prompt, {}, true);
        check(best.found && best.tokens == 200 && best.key != 0,
              "the fork at 250 matches the deepest whole segment B (200), never C");
        SavedConversation out;
        int64_t mounted = 0;
        check(cache.mount(best.key, best.tokens, prompt, {}, out, &mounted), "the best segment mounts");
        check(mounted == 200 && out.live.gdn.size() == 64,
              "mount = A+B segment length; the caller re-reads only the fork tail (50 tokens)");
        const auto mid_c = cache.best(std::vector<int32_t>(tokens.begin(), tokens.begin() + 260), {}, true);
        check(mid_c.found && mid_c.tokens == 200, "a prompt inside C's span still mounts B, never a mid-tail");
        const auto best_a = cache.best(std::vector<int32_t>(tokens.begin(), tokens.begin() + 120), {}, true);
        check(best_a.found && best_a.tokens == 100, "a rewind inside B falls back to A");
        SavedConversation out_a;
        check(cache.mount(best_a.key, best_a.tokens,
                          std::vector<int32_t>(tokens.begin(), tokens.begin() + 120), {}, out_a, &mounted) &&
              mounted == 100, "the fallback segment mounts");
    }
    {
        // §4.3 eviction: refs==0 and oldest last_hit first, under a directory-byte budget.  A pinned
        // (refs>0) segment never goes; a one-shot prosecutor conversation's segment does, so the directory
        // stops growing no matter how many prosecutor turns land (criterion 11).
        FakeDisk disk;
        uint64_t clock = 0;
        ConversationSegmentCache::Hooks h;
        fake_hooks(h, disk);
        h.clock = [&clock] { return clock; };
        const size_t record = 64 + 100 * sizeof(int32_t);   // the fake image's gdn plus its id list
        const size_t budget = 5 * record + 16;              // room for five records and no more
        ConversationSegmentCache cache(true, budget, std::move(h));

        const std::vector<int32_t> pinned_ids(100, 1);
        const uint64_t pinned_key = conversation_spill_key(pinned_ids, std::vector<ConversationImageKey>{});
        SavedConversation pinned = segment_image(pinned_ids, 9);
        check(cache.admit(std::move(pinned)), "the shared pinned segment admits");
        check(cache.ref(pinned_key) == 1, "an active session pins the shared segment");

        clock = 10;
        for (int t = 0; t < 12; ++t) {   // twelve one-shot prosecutor conversations, each one fresh segment
            clock += 1;
            std::vector<int32_t> ids(100, int32_t(2 + t));
            SavedConversation s = segment_image(ids, uint8_t(t));
            cache.admit(std::move(s));
            check(cache.dir_bytes() <= budget, "the budget holds after every prosecutor admit");
        }
        check(cache.evictions() >= 8, "the prosecutor segments are evicted (criterion 11)");
        check(cache.entries() <= 5 && cache.dir_bytes() <= budget, "the directory stays at the budget");
        const auto still = cache.best(std::vector<int32_t>(150, 1), {}, true);
        check(still.found && still.key == pinned_key, "the pinned segment survives the prosecutor storm");
        cache.unref(pinned_key);
        check(cache.refs(pinned_key) == 0, "unref releases the pin");
        clock += 5;
        std::vector<int32_t> ids(100, 99);
        SavedConversation s = segment_image(ids, 5);
        cache.admit(std::move(s));   // over budget again: the coldest refs==0 record goes first
        check(cache.dir_bytes() <= budget, "eviction after the pin lifts keeps the budget");
    }
    {
        // triage + coexistence: a corrupt segment file is dropped with its record (a collision must not
        // poison twice); the whole-file cache wired to the same directory is untouched, and the same prompt
        // still restores through it.
        FakeDisk disk;   // corrupted on demand, below: the segment tier's read is made to fail
        ConversationSegmentCache::Hooks h;
        ConversationSegmentCache segments(true, 1 << 20, fake_hooks(h, disk));
        ConversationCache whole(true, 0);
        ConversationCache::SpillHooks sh;
        sh.dir = "/fake";
        sh.write = [&](const SavedConversation& image, const std::string& path, std::string&) {
            auto& f = disk.files[path];
            f.clear();
            for (int32_t id : image.live.ids) {
                f.push_back(uint8_t(id)); f.push_back(uint8_t(id >> 8));
                f.push_back(uint8_t(id >> 16)); f.push_back(uint8_t(id >> 24));
            }
            f.insert(f.end(), image.live.gdn.begin(), image.live.gdn.end());
            return true;
        };
        sh.read = [&](const std::string& path, SavedConversation& out, std::string&, bool& refused) {
            refused = false;
            auto it = disk.files.find(path);
            if (it == disk.files.end()) return false;
            // the same toy serialization the segment hooks use (the id list, then the running state)
            const size_t n = it->second.size() - 64;
            out.live.ids.resize(n / 4);
            for (size_t i = 0; i < out.live.ids.size(); ++i)
                out.live.ids[i] = int32_t(uint32_t(it->second[4*i]) | uint32_t(it->second[4*i+1]) << 8 |
                                          uint32_t(it->second[4*i+2]) << 16 | uint32_t(it->second[4*i+3]) << 24);
            out.live.gdn.assign(it->second.begin() + (std::ptrdiff_t) n, it->second.end());
            return true;
        };
        sh.remove = [&](const std::string& path) { disk.removed.push_back(path); disk.files.erase(path); return true; };
        whole.set_spill_hooks(std::move(sh));

        const std::vector<int32_t> tokens(100, 11);
        SavedConversation seg = segment_image(tokens, 1);
        seg.live.gdn.assign(64, 0);
        check(segments.admit(std::move(seg)), "the segment admits");
        SavedConversation sp = segment_image(tokens, 2);
        sp.live.gdn.assign(64, 0);
        check(whole.park(std::move(sp)), "the whole-file snapshot parks beside the segments");
        check(whole.spilled_entries() == 1 && segments.entries() == 1, "both layers coexist, each with one record");

        const std::vector<int32_t> prompt(tokens.begin(), tokens.begin() + 120);   // strictly past the segment depth
        disk.corrupt_reads = true;   // the segment file goes bad on the disk
        const auto best = segments.best(prompt, {}, true);
        check(best.found, "the segment best() still matches (the manifest is RAM, the corruption is on disk)");
        SavedConversation out;
        int64_t mounted = 0;
        check(!segments.mount(best.key, best.tokens, prompt, {}, out, &mounted),
              "a corrupt segment file refuses to mount");
        check(segments.entries() == 0, "the corrupt segment's record is dropped with its file");
        (void) mounted;
        const auto wb = whole.best(prompt, {}, true);
        check(wb.tokens == 100 && wb.spilled, "the whole-file record is intact after the segment failure");
        SavedConversation wout;
        int64_t wm = 0;
        check(whole.restore_spilled_exact(wb.id, wb.tokens, prompt, {}, wout, &wm),
              "the whole-file fallback restores the same prompt");
        check(wm == 100, "the fallback mounts the full prefix");
    }
    {
        // the segment-bound chooser: cuts only at checkpoint depths (a mountable point must carry running
        // state), clipped to the verified stable depth, at least min_seg apart; a shallow remainder is skipped.
        const std::vector<int64_t> lens = {100, 150, 300, 400};
        const auto full = conversation_segment_bounds(lens, 400, 100);
        check((full == std::vector<int64_t>{100, 300, 400}), "dense checkpoints yield segments min_seg apart");
        const auto shallow = conversation_segment_bounds(lens, 150, 100);
        check((shallow == std::vector<int64_t>{100}), "the bounds clip to the stable depth");
        const auto none = conversation_segment_bounds(lens, 0, 100);
        check(none.empty(), "no verified depth: no segments");
        const auto sparse = conversation_segment_bounds({300}, 300, 4096);
        check(sparse.empty(), "a segment shorter than min_seg is skipped, not written");
        const auto exact = conversation_segment_bounds({4096, 8192, 16384}, 16384, 4096);
        check((exact == std::vector<int64_t>{4096, 8192, 16384}),
              "the production cadence (16K checkpoints) passes through unchanged");
    }

    std::printf("conversation_segment_cache_test: %d checks ok\n", checks);
    return 0;
}
