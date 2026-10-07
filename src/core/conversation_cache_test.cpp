// The disk-authoritative conversation cache (park-on-displacement): behaviour tests over real session files.
// A parked conversation lives only on disk (format v1); the cache holds a RAM manifest of key hashes.  Park =
// write to disk and free the RAM; restore = read back, re-verify token-exactly, triage the failures; active
// turns touch nothing.
#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "strata/core/conversation_file.hpp"

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

int main() {
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // the transient K/V retention between a restore and the next park, with the later stages of a layer split
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(true, 1 << 20);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
        ConversationCache no_budget(true, 0);
        no_budget.retain(kv(500), 40);
        check(no_budget.retained_bytes() == 0, "without a retention budget nothing is retained");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "strata_conv_cache_disk_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::vector<fs::path> spill_files;   // what park() has written so far
    bool fail_write = false;             // injected write failure
    bool refuse_read = false;            // injected RAM-admission refusal on read
    bool swap_image = false;             // injected: the read returns a state the manifest never recorded
    const SessionFileIdentity file_id{11, 22};
    ConversationCache::SpillHooks hooks;
    hooks.dir = dir.string();
    hooks.write = [&](const SavedConversation& im, const std::string& path, std::string& e) {
        if (fail_write) { e = "injected write failure"; return false; }
        size_t bytes = 0;
        const bool ok = session_file_write(path, im, file_id, bytes, e, {}, nullptr);
        if (ok) spill_files.push_back(path);
        return ok;
    };
    hooks.read = [&](const std::string& path, SavedConversation& out, std::string& e, bool& admission_refused) {
        admission_refused = false;
        if (refuse_read) {
            admission_refused = true;
            e = "injected RAM admission refusal";
            return false;
        }
        if (swap_image) {
            SavedConversation other = image({9, 9, 9});
            other.live.gdn.resize(64, 8);
            out = std::move(other);
            return true;
        }
        size_t bytes = 0;
        return session_file_read(path, file_id, out, bytes, e, {}, nullptr);
    };
    hooks.remove = [](const std::string& path) { std::error_code ec; return fs::remove(path, ec); };
    hooks.reap = [&](const std::vector<std::string>& keep, size_t& dir_bytes) {
        size_t removed = 0;
        dir_bytes = 0;
        for (const auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind("spill-", 0) != 0 || name.size() <= 4 || name.compare(name.size() - 4, 4, ".bin") != 0)
                continue;
            bool kept = false;
            for (const auto& k : keep) if (k == e.path().string()) { kept = true; break; }
            if (kept) dir_bytes += (size_t) e.file_size();
            else if (fs::remove(e.path())) ++removed;
        }
        return removed;
    };
    const auto on_disk = [&] {
        size_t n = 0;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.is_regular_file() && e.path().filename().string()[0] != '.') ++n;
        return n;
    };
    const auto dot_files = [&] {
        size_t n = 0;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.is_regular_file() && e.path().filename().string()[0] == '.') ++n;
        return n;
    };
    const auto fresh = [&]() {
        fs::remove_all(dir);
        fs::create_directories(dir);
        spill_files.clear();
        ConversationCache c(true, 1 << 20);
        c.set_spill_hooks(hooks);
        return c;
    };

    {   // park-on-displacement: park writes one session file and keeps only the manifest; consecutive best()
        // calls on an active conversation write nothing and read nothing (zero cache IO while active)
        auto c = fresh();
        SavedConversation big = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        cp.gdn.resize(64, 3);
        big.checkpoints.push_back(cp);
        big.kv.emplace_back();
        big.kv.back().k.resize(512);
        check(c.park(std::move(big)), "the displaced conversation parks to disk");
        check(c.parks() == 1 && c.failed_parks() == 0 && c.spilled_entries() == 1, "one park, one manifest record");
        check(c.spilled_bytes() >= 512 + 64, "the manifest accounts the file's payload");
        check(on_disk() == 1 && c.bytes() == 0, "the state lives only on disk: no RAM entry, no reuse held");
        const std::vector<int64_t> prompt{1, 2, 3, 4};
        for (int turn = 0; turn < 5; ++turn) {
            auto m = c.best(prompt, {}, true);
            check(m.spilled && m.tokens == 3, "the manifest matches the active conversation");
        }
        check(c.parks() == 1 && c.spilled_restores() == 0 && c.refused_restores() == 0,
              "consecutive turns on the same conversation cause zero cache IO");
        check(c.longest_tokens() == 3, "longest_tokens comes from the manifest");
    }
    {   // the disk round trip is byte-exact: the K/V and the checkpoint chain read back whole
        auto c = fresh();
        SavedConversation big = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        cp.gdn.resize(64, 3);
        cp.tails.resize(32, 4);
        big.checkpoints.push_back(cp);
        big.kv.emplace_back();
        big.kv.back().k.resize(512, 9);
        big.kv.back().v.resize(512, 8);
        ConversationCache keeper(true);
        keeper.set_spill_hooks(hooks);
        check(keeper.park(std::move(big)), "parked");
        const std::vector<int64_t> prompt{1, 2, 3, 4};
        auto m = keeper.best(prompt, {}, true);
        SavedConversation back;
        check(keeper.restore_spilled_exact(m.id, m.tokens, prompt, {}, back) &&
              back.live.ids == std::vector<int32_t>({1, 2, 3}) &&
              back.live.gdn.size() == 64 && back.live.gdn[0] == 7 &&
              back.checkpoints.size() == 1 && back.checkpoints[0].ids == std::vector<int32_t>({1, 2}) &&
              back.checkpoints[0].gdn.size() == 64 && back.checkpoints[0].gdn[10] == 3 &&
              back.checkpoints[0].tails.size() == 32 && back.checkpoints[0].tails[5] == 4 &&
              back.kv.size() == 1 && back.kv[0].k.size() == 512 && back.kv[0].v.size() == 512,
              "the parked state reads back byte-exact: live, checkpoint chain and K/V included");
        check(keeper.spilled_entries() == 0 && keeper.spilled_restores() == 1, "the manifest record goes on restore");
        check(on_disk() == 0, "the restore deletes the file it consumed (no disk leak)");
        check(keeper.best(prompt, {}, true).tokens == 0, "the restored state is the caller's active state now");
    }
    {   // restore through a checkpoint mark (a rewind's resume point) keeps the chain and the K/V consistent
        auto c = fresh();
        SavedConversation big = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        cp.gdn.resize(64, 5);
        big.checkpoints.push_back(cp);
        big.kv.emplace_back();
        big.kv.back().k.resize(256, 2);
        check(c.park(std::move(big)), "parked");
        const std::vector<int64_t> edited{1, 2, 9, 4};   // the suffix the client rewrote
        const std::vector<int64_t> empty;
        auto m = c.best(edited, {}, true);
        check(m.spilled && m.tokens == 2, "the checkpoint boundary matches the edited prompt");
        check(c.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length mark cannot consume the last token");
        check(c.best(std::vector<int64_t>{1}, {}, true).tokens == 0 && c.best(empty, {}, true).tokens == 0,
              "short and empty prompts cannot match");
        SavedConversation back;
        check(c.restore_spilled_exact(m.id, m.tokens, edited, {}, back) &&
              back.checkpoints.size() == 1 && back.checkpoints[0].ids == std::vector<int32_t>({1, 2}) &&
              back.kv[0].k.size() == 256, "the rewind restores from the checkpoint with its K/V");
    }
    {   // steering mode and images isolate the manifest, exactly as the RAM prefix match did
        auto c = fresh();
        SavedConversation s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        s.checkpoints.push_back(ConversationCheckpoint{});
        s.checkpoints.back().ids = {1, 2};
        check(c.park(std::move(s)), "parked");
        const std::vector<int64_t> a{1, 2, 3, 4};
        check(c.best(a, {}, false).tokens == 0, "the other steering mode does not match");
        check(c.best(a, {{1, 123}}, true).tokens == 3, "the same image can resume");
        check(c.best(a, {{1, 124}}, true).tokens == 0, "different image pixels invalidate the same pad tokens");
        check(c.best(a, {}, true).tokens == 2, "a missing later image still resumes from the checkpoint");
        check(c.best(a, {{1, 123}, {2, 45}}, true).tokens == 0, "an additional image inside the prefix invalidates");
        check(c.best(a, {{1, 123}, {3, 45}}, true).tokens == 3, "an image after the prefix does not invalidate");
    }
    {   // B-F3: the exact re-verification refuses a read whose bytes do not hold the recorded state
        auto c = fresh();
        check(c.park(image({1, 2, 3})), "parked");
        swap_image = true;   // the file now yields a state the manifest never recorded
        const std::vector<int64_t> prompt{1, 2, 3, 4};
        auto m = c.best(prompt, {}, true);
        check(m.spilled, "the manifest still matches");
        SavedConversation back;
        check(!c.restore_spilled_exact(m.id, m.tokens, prompt, {}, back), "the foreign state is refused");
        check(c.spilled_entries() == 0 && on_disk() == 0 && c.refused_restores() == 0,
              "a verification miss deletes the file and the record (fail-closed)");
        check(!c.last_error().empty(), "the verification miss is reported");
        swap_image = false;
    }
    {   // B-F2: a read the RAM admission refuses keeps the file and the record - refused is not corrupt
        auto c = fresh();
        check(c.park(image({1, 2, 3})), "parked");
        refuse_read = true;
        const std::vector<int64_t> prompt{1, 2, 3, 4};
        auto m = c.best(prompt, {}, true);
        SavedConversation back;
        check(!c.restore_spilled_exact(m.id, m.tokens, prompt, {}, back), "the refused read does not restore");
        check(c.refused_restores() == 1 && c.spilled_entries() == 1 && on_disk() == 1,
              "the refused entry keeps its file and its manifest record");
        refuse_read = false;
        check(c.restore_spilled_exact(m.id, m.tokens, prompt, {}, back) &&
              back.live.ids == std::vector<int32_t>({1, 2, 3}),
              "once the admission allows it, the same record restores");
    }
    {   // a corrupt file, and a file under another identity, are deleted with their records
        auto c = fresh();
        check(c.park(image({1, 2, 3})), "parked");
        const std::vector<int64_t> prompt{1, 2, 3, 4};
        auto m = c.best(prompt, {}, true);
        check(m.spilled, "the manifest matches");
        fs::resize_file(spill_files.back(), 8);   // truncate the file: the payload hash fails
        SavedConversation back;
        check(!c.restore_spilled_exact(m.id, m.tokens, prompt, {}, back), "the corrupt file refuses to read");
        check(c.spilled_entries() == 0 && on_disk() == 0, "the corrupt entry is deleted and dropped (fail-open)");
        check(c.park(image({7, 8, 9})), "the cache keeps working after a dropped entry");
        auto m2 = c.best(std::vector<int64_t>{7, 8, 9, 1}, {}, true);
        check(m2.spilled && m2.tokens == 3, "the new entry matches");
        // identity flip and unreadable files: the read refuses before any state is used, the entry is deleted
        for (const auto& e : fs::directory_iterator(dir))
            if (e.is_regular_file()) fs::resize_file(e.path(), 0);
        SavedConversation back2;
        check(!c.restore_spilled_exact(m2.id, m2.tokens, std::vector<int64_t>{7, 8, 9, 1}, {}, back2),
              "an unreadable file refuses too");
        check(c.spilled_entries() == 0 && on_disk() == 0, "the foreign entry is deleted and dropped");
    }
    {   // F1: a manifest record is addressed by its unique id, not by a capturable index - park_current's
        // drop_superseded runs between best() and restore_spilled_exact and shifts the manifest's indices
        auto c = fresh();
        check(c.park(image({1, 2, 3})), "A parked");
        check(c.park(image({5, 6, 7})), "B parked");
        check(c.park(image({9, 10, 11})), "C parked");
        const std::vector<int64_t> c_prompt{9, 10, 11, 12};
        const auto m = c.best(c_prompt, {}, true);   // captured before the park-time drop, like the engine does
        SavedConversation newer = image({1, 2, 3, 30});   // A one turn later: its park supersedes A's copy
        newer.checkpoints.push_back(ConversationCheckpoint{});
        newer.checkpoints.back().ids = {1, 2, 3};
        check(c.drop_superseded(newer.live.ids, newer.live.imgs, newer.checkpoints, true) == 1,
              "the park-time drop removes the record in front of the target");
        check(c.spilled_entries() == 2, "two records remain");
        SavedConversation back;
        check(c.restore_spilled_exact(m.id, m.tokens, c_prompt, {}, back) &&
              back.live.ids == std::vector<int32_t>({9, 10, 11}),
              "the restore finds its record by id after the drop shifted the indices");
        // the same when the record the match pointed at is itself dropped: a clear failure, never a wrong file
        const std::vector<int64_t> b_prompt{5, 6, 7, 8};
        const auto m2 = c.best(b_prompt, {}, true);
        SavedConversation newer_b = image({5, 6, 7, 70});
        newer_b.checkpoints.push_back(ConversationCheckpoint{});
        newer_b.checkpoints.back().ids = {5, 6, 7};
        check(c.drop_superseded(newer_b.live.ids, newer_b.live.imgs, newer_b.checkpoints, true) == 1,
              "the target record itself is dropped");
        check(!c.restore_spilled_exact(m2.id, m2.tokens, b_prompt, {}, back) && !c.last_error().empty(),
              "a dropped record fails with a reported reason, not silently");
    }
    {   // the rewind: the prompt is a strict prefix of the parked snapshot - the shared prefix is reused,
        // mounted at the deepest checkpoint at or below the prompt boundary (no truncation of the file)
        auto c = fresh();
        SavedConversation big = image({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        for (int n : {4, 8}) {
            ConversationCheckpoint cp;
            cp.ids.assign(n, 0);
            for (int i = 0; i < n; ++i) cp.ids[(size_t) i] = i + 1;
            cp.gdn.resize(64, n);
            big.checkpoints.push_back(std::move(cp));
        }
        check(c.park(std::move(big)), "the longer conversation parked");
        const std::vector<int64_t> rewind{1, 2, 3, 4, 5, 6, 20};   // a strict prefix plus one new token
        auto m = c.best(rewind, {}, true);
        check(m.spilled && m.tokens == 4, "the checkpoint mark at or below the prompt boundary matches");
        SavedConversation back;
        int64_t mount = -1;
        check(c.restore_spilled_exact(m.id, m.tokens, rewind, {}, back, &mount) &&
              back.live.ids.size() == 10 && back.checkpoints.size() == 2,
              "a snapshot longer than the prompt restores whole instead of being dropped");
        check(mount == 4, "the rewind mounts at the deepest checkpoint at or below the prompt boundary");
        check(c.spilled_entries() == 0 && on_disk() == 0, "the consumed rewind copy leaves no file");
        // the prompt diverges past the checkpoint boundary: the checkpoint state (verified) mounts, the longer
        // live state is not reusable - and none of that touches the fail-closed triage (a verification miss
        // with the file's own bytes is covered by the injected-foreign-state test above)
        SavedConversation other = image({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        for (int n : {4, 8}) {
            ConversationCheckpoint cp;
            cp.ids.assign(n, 0);
            for (int i = 0; i < n; ++i) cp.ids[(size_t) i] = i + 1;
            cp.gdn.resize(64, n);
            other.checkpoints.push_back(std::move(cp));
        }
        check(c.park(std::move(other)), "parked again");
        const std::vector<int64_t> diverged{1, 2, 3, 4, 9, 9};   // diverges past the 4-token checkpoint
        auto m2 = c.best(diverged, {}, true);
        check(m2.spilled && m2.tokens == 4, "the checkpoint boundary still matches past the divergence");
        int64_t mount2 = -1;
        check(c.restore_spilled_exact(m2.id, m2.tokens, diverged, {}, back, &mount2) && mount2 == 4 &&
              back.checkpoints.size() == 2,
              "the divergence mounts at the verified checkpoint below it");
        // a snapshot shorter than the prompt keeps the ordinary mount (its whole live state)
        SavedConversation short_conv = image({1, 2, 3});
        check(c.park(std::move(short_conv)), "the shorter conversation parked");
        const std::vector<int64_t> grown{1, 2, 3, 4};
        auto m3 = c.best(grown, {}, true);
        int64_t mount3 = -1;
        check(c.restore_spilled_exact(m3.id, m3.tokens, grown, {}, back, &mount3) && mount3 == 3 &&
              back.live.ids.size() == 3,
              "a snapshot shorter than the prompt mounts at its live state, as before");
    }
    {   // the spill directory is reconciled against the manifest at park time: orphan files go, the
        // manifest's files stay, and the directory's total bytes are observable (acceptance 9)
        auto c = fresh();
        check(c.park(image({1, 2, 3})), "parked");
        const fs::path orphan = dir / "spill-999.bin";
        { std::FILE* f = std::fopen(orphan.string().c_str(), "wb"); std::fputs("orphan", f); std::fclose(f); }
        check(c.park(image({4, 5, 6})), "the second park runs the reconciliation");
        check(!fs::exists(orphan), "the orphan file is reaped");
        check(on_disk() == 2, "only the manifest's files remain");
        check(c.dir_bytes() > 0, "the directory's total bytes are observable");
    }
    {   // fail-open: a park write that fails drops the state (the same loss as before) and leaves nothing behind
        auto c = fresh();
        fail_write = true;
        check(!c.park(image({1, 2, 3})), "a failed write does not park");
        check(c.failed_parks() == 1 && c.parks() == 0 && c.spilled_entries() == 0,
              "the failure is counted and leaves no record");
        check(!c.last_error().empty(), "the park failure is reported");
        check(on_disk() == 0 && dot_files() == 0, "no half-written file and no temporary orphan is left behind");
        fail_write = false;
        check(c.park(image({1, 2, 3})), "the cache keeps parking after a failure");
        check(c.spilled_entries() == 1, "the next park lands");
    }
    {   // A-F1: a layer split's stage images do not park (format v1 does not carry them)
        auto c = fresh();
        SavedConversation split = image({1, 2, 3});
        split.stage_images.push_back(SavedConversation{});
        check(!c.park(std::move(split)), "stage images are refused");
        check(c.failed_parks() == 1 && c.spilled_entries() == 0 && on_disk() == 0,
              "the refusal drops the state and writes nothing");
    }
    {   // superseded spilled copies leave with their files when the same conversation parks a newer state
        auto c = fresh();
        SavedConversation old = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        old.checkpoints.push_back(cp);
        check(c.park(std::move(old)), "the older copy parked");
        check(c.park(image({7, 8, 9})), "another conversation parked");
        check(on_disk() == 2 && c.spilled_entries() == 2, "two files on disk");
        SavedConversation newer = image({1, 2, 3, 30, 31});   // the same conversation, one turn later
        newer.checkpoints.push_back(ConversationCheckpoint{});
        newer.checkpoints.back().ids = {1, 2, 3};
        newer.checkpoints.push_back(ConversationCheckpoint{});
        newer.checkpoints.back().ids = {1, 2};
        check(c.drop_superseded(newer.live.ids, newer.live.imgs, newer.checkpoints, true) == 1 &&
              c.spilled_entries() == 1 && on_disk() == 1,
              "the newer park supersedes the older copy: file and record go together");
        check(c.park(std::move(newer)), "the newer state parks (the engine drops the stale copy first)");
        auto kept = c.best(std::vector<int64_t>{7, 8, 9, 1}, {}, true);
        check(kept.spilled && kept.tokens == 3, "the unrelated conversation is kept");
        auto bm = c.best(std::vector<int64_t>{1, 2, 3, 30, 31, 40}, {}, true);
        check(bm.tokens == 5, "the newer copy matches through its own marks");
    }
    {   // a disabled cache parks nothing and matches nothing
        ConversationCache off(false);
        const std::vector<int64_t> off_prompt{1, 2, 3, 4};
        check(!off.park(image({1, 2, 3})), "a disabled cache parks nothing");
        check(off.best(off_prompt, {}, true).tokens == 0, "a disabled cache has no matches");
    }
    fs::remove_all(dir);
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
