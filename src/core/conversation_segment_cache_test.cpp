// The segmented SSD cache (R4, the benchmark document §4): behaviour tests over fake spill hooks (the pure
// host policy) plus the segment-bound chooser.  A segment's key is the prefix key at its end depth (content-
// addressed, immune to the checkpoint cadence's drift); admission is append-only with dedup by key; eviction
// is refs==0 and oldest last_hit first under a directory-byte budget; retrieval mounts the deepest hit and
// the caller re-reads only the fork tail.  The whole-file conversation cache stays untouched beside it.
#include "strata/core/conversation_segment_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_segment_build.hpp"

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

    {
        // ---- R4 阶段 A 增量 2：接线（判据 12 的 host 级代理）----
        // K/V 前缀切片：parts 对 cells 线性（data/scales），pooled 按行（idx_block），两者都是池字节的前缀，
        // 所以切片 = 精确的前缀拷贝；几何字段（cells / pooled_rows）随 upto 收紧，validate 才能原样通过。
        {
            ConversationKv kv;
            kv.format = 0;
            kv.cells = 300; kv.heads = 2; kv.head_dim = 8; kv.page_size = 4; kv.pooled_rows = 76; kv.idx_dim = 8;
            const size_t data = 300 * 2 * 8 * 2;      // f16, 32 B per cell
            const size_t pooled = 76 * 8 * 4;         // 76 rows x 128 B  (300/4+1 = 76)
            kv.k.resize(data); kv.v.resize(data); kv.pooled.resize(pooled);
            uint64_t s = 7;
            auto fill = [&s](ConversationBuffer& b) {
                b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t at) {
                    for (size_t i = 0; i < n; ++i) { s = s * 6364136223846793005ull + 1442695040888963407ull;
                                                    p[i] = uint8_t(s >> 33); (void) at; }
                    return true;
                });
            };
            fill(kv.k); fill(kv.v); fill(kv.pooled);

            ConversationKv cut;
            std::string slice_error;
            check(conversation_kv_slice_prefix(kv, 200, cut, slice_error), "a mid prefix slices");
            check(slice_error.empty(), "the slice reports no error");
            check(cut.cells == 200 && cut.pooled_rows == 51 && cut.heads == 2 && cut.head_dim == 8,
                  "the slice's geometry tightens to the cut (cells 200, rows 200/4+1)");
            check(cut.k.size() == 200 * 32 && cut.v.size() == 200 * 32, "linear parts shrink exactly");
            check(cut.pooled.size() == 51 * 32, "the pooled part keeps whole rows only");
            bool prefix_ok = true;
            cut.k.visit(0, cut.k.size(), [&](const uint8_t* p, size_t n, size_t at) {
                return kv.k.visit(at, n, [&](const uint8_t* q, size_t m, size_t) {
                    if (std::memcmp(p, q, m)) { prefix_ok = false; return false; }
                    return true;
                });
            });
            check(prefix_ok, "the sliced K/V is a byte prefix of the whole image's");
            ConversationKv bad;
            check(!conversation_kv_slice_prefix(kv, 0, bad, slice_error), "an empty cut is refused");
            check(!conversation_kv_slice_prefix(kv, 301, bad, slice_error), "a cut past the image is refused");
        }
        // 段镜像构建 + 真实 session 文件往返 + 与整文件路径的挂载深度一致（判据 12）+ 损坏回退。
        // 用真实 conversation_file.cpp 读写（format v1），像生产钩子一样，而不是玩具序列化。
        const std::string seg_dir = "/tmp/strata_segment_cache_test";
        const std::string spill_dir = "/tmp/strata_segment_cache_test_spill";
        {
            std::error_code ec;
            std::filesystem::create_directories(seg_dir, ec);
            std::filesystem::create_directories(spill_dir, ec);
            for (const std::string& d : {seg_dir, spill_dir})
                for (auto& e : std::filesystem::directory_iterator(d, ec)) std::filesystem::remove(e.path(), ec);

            // 父镜像：300 tokens，图在 50；检查点 100 / 200；两层假 K/V（一层带 indexer，一层不带）。
            SavedConversation parent;
            parent.layer_lo = 0; parent.layer_hi = 2;
            parent.live.ids.resize(300);
            for (size_t i = 0; i < parent.live.ids.size(); ++i) parent.live.ids[i] = int32_t(1000 + i);
            parent.live.imgs = {{50, 0xABCD}};
            parent.live.gdn.assign(64, 1);
            parent.cvec = true;
            auto ckpt = [&](int64_t n, uint8_t salt) {
                ConversationCheckpoint c;
                c.ids.assign(parent.live.ids.begin(), parent.live.ids.begin() + (std::ptrdiff_t) n);
                for (const auto& k : parent.live.imgs) if (k.start < n) c.imgs.push_back(k);
                c.gdn.assign(64, salt);
                return c;
            };
            parent.checkpoints.push_back(ckpt(100, 2));
            parent.checkpoints.push_back(ckpt(200, 3));
            auto fake_kv = [&](bool index) {
                ConversationKv kv;
                kv.format = 0;
                kv.cells = 300; kv.heads = 2; kv.head_dim = 8; kv.page_size = 4;
                kv.pooled_rows = index ? 76 : 0; kv.idx_dim = 8;
                kv.k.resize(300 * 32); kv.v.resize(300 * 32);
                if (index) kv.pooled.resize(76 * 32);
                uint64_t s = 11;
                auto fill = [&s](ConversationBuffer& b) {
                    b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t) {
                        for (size_t i = 0; i < n; ++i) { s = s * 6364136223846793005ull + 1442695040888963407ull;
                                                        p[i] = uint8_t(s >> 33); }
                        return true;
                    });
                };
                fill(kv.k); fill(kv.v); fill(kv.pooled);
                return kv;
            };
            parent.kv.push_back(fake_kv(true));
            parent.kv.push_back(fake_kv(false));

            // 真文件钩子（段目录与整文件目录分开；与生产同样的文件格式与身份）。
            auto real_seg_hooks = [&](ConversationSegmentCache::Hooks& h) {
                h.dir = seg_dir;
                h.write = [&](const SavedConversation& image, const std::string& path, std::string& e) {
                    SessionFileIdentity id; std::memset(&id, 0, sizeof id);
                    SessionWriteOptions wo; size_t bytes = 0;
                    return session_file_write(path, image, id, bytes, e, wo, nullptr);
                };
                h.read = [&](const std::string& path, SavedConversation& out, std::string& e) {
                    SessionFileIdentity id; std::memset(&id, 0, sizeof id);
                    SessionReadLimits lim; size_t bytes = 0; SessionStatus st;
                    return session_file_read(path, id, out, bytes, e, lim, &st);
                };
                h.remove = [&](const std::string& path) {
                    std::error_code rm_ec;
                    return std::filesystem::remove(path, rm_ec);
                };
                return h;
            };
            ConversationSegmentCache::Hooks h;
            ConversationSegmentCache segments(true, 1 << 30, real_seg_hooks(h));
            ConversationCache whole(true, 0);
            ConversationCache::SpillHooks sh;
            sh.dir = spill_dir;
            sh.write = [&](const SavedConversation& image, const std::string& path, std::string& e) {
                SessionFileIdentity id; std::memset(&id, 0, sizeof id);
                SessionWriteOptions wo; size_t bytes = 0;
                return session_file_write(path, image, id, bytes, e, wo, nullptr);
            };
            sh.read = [&](const std::string& path, SavedConversation& out, std::string& e, bool& refused) {
                refused = false;
                SessionFileIdentity id; std::memset(&id, 0, sizeof id);
                SessionReadLimits lim; size_t bytes = 0; SessionStatus st;
                return session_file_read(path, id, out, bytes, e, lim, &st);
            };
            sh.remove = [&](const std::string& path) {
                std::error_code rm_ec;
                return std::filesystem::remove(path, rm_ec);
            };
            whole.set_spill_hooks(std::move(sh));

            // park 接线（判据 10/11 的编排）：从父镜像的检查点链切段入段缓存。
            const auto stats1 = conversation_segment_park(segments, parent, 100);
            check(stats1.admitted == 2 && stats1.deduped == 0 && stats1.failed == 0,
                  "the first park admits every cut (100, 200)");
            const auto stats2 = conversation_segment_park(segments, parent, 100);
            check(stats2.deduped == 2 && stats2.admitted == 0,
                  "the same conversation's next park is all dedup (criterion 10: one copy)");
            SavedConversation parent_copy = parent;   // the whole-file tier parks the same conversation beside it
            check(whole.park(std::move(parent_copy)), "the whole-file snapshot parks beside the segments");

            const std::vector<int32_t> prompt(parent.live.ids.begin(), parent.live.ids.begin() + 260);
            const auto seg_best = segments.best(prompt, parent.live.imgs, true);
            check(seg_best.found && seg_best.tokens == 200, "the segment tier matches the deepest whole segment");
            const auto w_best = whole.best(prompt, parent.live.imgs, true);
            check(w_best.tokens == 200, "the whole-file tier matches the same depth");
            SavedConversation seg_out;
            int64_t seg_mount = 0;
            check(segments.mount(seg_best.key, seg_best.tokens, prompt, parent.live.imgs, seg_out, &seg_mount) &&
                  seg_mount == 200, "the segment mounts at its end depth");
            SavedConversation w_out;
            int64_t w_mount = 0;
            check(whole.restore_spilled_exact(w_best.id, w_best.tokens, prompt, parent.live.imgs, w_out, &w_mount) &&
                  w_mount == 200, "the whole-file fallback mounts the same depth (criterion 12: mount parity)");
            check(seg_out.kv.size() == 2 && seg_out.kv[0].cells == 200 && seg_out.kv[0].pooled_rows == 51,
                  "the mounted segment carries its sliced K/V");
            check(seg_out.live.gdn.size() == 64 && seg_out.live.gdn[0] == 3,
                  "the mounted segment carries its checkpoint's running state");

            // holds()：park 接线的 dedup 预检在构建前就跳过（不白拷一份 K/V）。
            const uint64_t key200 = conversation_spill_key(
                std::vector<int32_t>(parent.live.ids.begin(), parent.live.ids.begin() + 200), parent.live.imgs);
            check(segments.holds(key200), "holds() sees an admitted segment's key");
            check(!segments.holds(key200 + 1), "holds() is false for a key nobody admitted");

            // 回退：段文件在盘上损坏 → mount 拒绝并连带记录一起删；整文件路径原样可用（判据 12 的回退半边）。
            SavedConversation parent_copy2 = parent;
            check(whole.park(std::move(parent_copy2)), "a fresh whole-file snapshot parks for the fallback");
            const auto seg_best2 = segments.best(prompt, parent.live.imgs, true);
            check(seg_best2.found, "the segment best() matches again (a fresh file was admitted)");
            std::error_code tr_ec;
            std::filesystem::resize_file(seg_dir + "/seg-" + [&] {
                char b[17]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long) seg_best2.key);
                return std::string(b);
            }() + ".bin", 64, tr_ec);
            SavedConversation seg_out2;
            int64_t seg_mount2 = 0;
            check(!segments.mount(seg_best2.key, seg_best2.tokens, prompt, parent.live.imgs, seg_out2, &seg_mount2),
                  "a truncated segment file refuses to mount");
            check(!segments.holds(seg_best2.key), "the truncated segment's record is dropped with its file");
            SavedConversation w_out2;
            int64_t w_mount2 = 0;
            const auto w_best2 = whole.best(prompt, parent.live.imgs, true);
            check(w_best2.tokens > 0, "the whole-file record survived the segment failure");
            check(whole.restore_spilled_exact(w_best2.id, w_best2.tokens, prompt, parent.live.imgs, w_out2, &w_mount2) &&
                  w_mount2 == 200, "the fallback still restores after the segment path failed");
        }
        // 段构建的拒绝路径：非前缀检查点、带分阶段的父镜像，一律拒（fail-closed）。
        {
            SavedConversation parent;
            parent.live.ids.assign(300, 5);
            parent.live.imgs = {{50, 1}};
            ConversationCheckpoint c;
            c.ids.assign(200, 5);
            c.gdn.assign(64, 0);
            parent.checkpoints.push_back(c);
            SavedConversation seg;
            std::string e;
            check(conversation_segment_build(parent, 200, c, seg, e), "a prefix checkpoint builds");
            SavedConversation split_parent = parent;
            split_parent.stage_images.push_back(SavedConversation{});
            SavedConversation seg2;
            check(!conversation_segment_build(split_parent, 200, c, seg2, e), "a split's image does not segment");
            ConversationCheckpoint foreign;
            foreign.ids.assign(200, 9);   // not a prefix of live
            foreign.gdn.assign(64, 0);
            SavedConversation seg3;
            check(!conversation_segment_build(parent, 200, foreign, seg3, e),
                  "a checkpoint that is not the live prefix is refused");
        }
    }

    {   // the restart-survival patch: the segment manifest dumps and the next cache adopts it over the same
        // fake disk - the mounted segment survives the restart (the file and its verification unchanged)
        FakeDisk disk;
        ConversationSegmentCache::Hooks h;
        ConversationSegmentCache first(true, 1 << 20, fake_hooks(h, disk));
        const std::vector<int32_t> ids(200, 11);
        check(first.admit(segment_image(ids, 1)), "restart: the segment admits");
        const std::vector<uint8_t> blob = first.dump_manifest();
        check(!blob.empty(), "restart: the segment manifest dumps");

        std::vector<std::string> present;   // the files on the fake disk, by name
        for (const auto& entry : disk.files)
            present.push_back(entry.first.substr(entry.first.find_last_of('/') + 1));

        ConversationSegmentCache::Hooks h2;
        ConversationSegmentCache second(true, 1 << 20, fake_hooks(h2, disk));   // a new process, same disk
        std::string err;
        check(second.adopt_manifest(blob, present, err) && err.empty(), "restart: the segment manifest adopts");
        check(second.entries() == 1, "restart: the record survives");
        const std::vector<int32_t> prompt(250, 11);   // strictly longer than the segment (the fork tail is the caller's)
        const auto best = second.best(prompt, {}, true);
        check(best.found && best.tokens == 200, "restart: the segment matches after adoption");
        SavedConversation out;
        int64_t mounted = 0;
        check(second.mount(best.key, best.tokens, prompt, {}, out, &mounted) && mounted == 200,
              "restart: it mounts (the token verification is unchanged)");

        // present filtering + malformed input (fail-open)
        ConversationSegmentCache::Hooks h3;
        ConversationSegmentCache third(true, 1 << 20, fake_hooks(h3, disk));
        check(third.adopt_manifest(blob, {}, err) && third.entries() == 0,
              "restart: without the file present, the record is dropped");
        auto bad = blob;
        bad[2] ^= 0xff;
        check(!third.adopt_manifest(bad, present, err) && third.entries() == 0,
              "restart: a foreign segment manifest is refused");
    }

    std::printf("conversation_segment_cache_test: %d checks ok\n", checks);
    return 0;
}
