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
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
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
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
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
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {   // spill-to-session-file: the eviction victim is serialized to disk (format v1) instead of dropped, the
        // RAM manifest keeps (key hash, tokens, geometry, path), best() consults it, and a hit reads the file back
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "strata_conv_cache_spill_test";
        fs::remove_all(dir);
        fs::create_directories(dir);
        double clock_now = 100.0;
        ConversationCache::SpillHooks hooks;
        hooks.dir = dir.string();
        hooks.write = [](const SavedConversation& im, const std::string& path, std::string& e) {
            const SessionFileIdentity id{11, 22};
            size_t bytes = 0;
            return session_file_write(path, im, id, bytes, e, {}, nullptr);
        };
        hooks.read = [](const std::string& path, SavedConversation& out, std::string& e) {
            const SessionFileIdentity id{11, 22};
            size_t bytes = 0;
            return session_file_read(path, id, out, bytes, e, {}, nullptr);
        };
        hooks.remove = [](const std::string& path) { std::error_code ec; fs::remove(path, ec); return !ec; };
        hooks.admit_write = [](size_t) { return true; };
        const auto hooked = [&]() {
            ConversationCache c(1 << 20, 2);
            c.set_spill_hooks(hooks);
            c.set_spill_clock([&] { return clock_now; });
            c.set_spill_cooldown(120.0);
            return c;
        };

        {   // budget-driven eviction spills the oldest entry; a manifest hit reads it back whole
            auto c = hooked();
            SavedConversation big = image({1, 2, 3});
            ConversationCheckpoint cp;
            cp.ids = {1, 2};
            cp.gdn.resize(64, 3);
            big.checkpoints.push_back(cp);
            big.kv.emplace_back();
            big.kv.back().k.resize(512);
            const size_t one = big.bytes();
            const size_t two = image({9, 8, 7}).bytes();
            ConversationCache small(one + two, 8);   // exactly big + one small entry
            small.set_spill_hooks(hooks);
            small.set_spill_clock([&] { return clock_now; });
            small.set_spill_cooldown(0.0);   // the hysteresis has its own block
            check(small.put(std::move(big)) && small.put(image({9, 8, 7})), "two entries fill the budget");
            check(small.spilled_entries() == 0, "nothing spilled yet");
            check(small.put(image({5, 6, 7})), "a third parks by spilling the oldest");
            check(small.evictions() == 1 && small.spills() == 1 && small.failed_spills() == 0,
                  "the budget eviction became a spill, not a drop");
            check(small.spilled_entries() == 1 && small.spilled_bytes() >= one, "the manifest holds the spilled entry");
            check(small.bytes() == two * 2 && small.size() == 2, "the victim's RAM is freed and accounted");
            const std::vector<int64_t> prompt{1, 2, 3, 4};
            auto m = small.best(prompt, {}, true);
            check(m.spilled && m.tokens == 3 && !m.live, "best() finds the spilled entry through the manifest");
            check(small.best(std::vector<int64_t>{9, 8, 7, 5}, {}, true).spilled == false &&
                  small.best(std::vector<int64_t>{9, 8, 7, 5}, {}, true).tokens == 3,
                  "the RAM entry still wins; the non-matching prompt misses the manifest");
            check(small.best(prompt, {}, false).tokens == 0, "steering mode isolates the manifest too");
            SavedConversation back;
            check(small.restore_spilled(m.index, back) && back.live.ids == std::vector<int32_t>({1, 2, 3}) &&
                  back.checkpoints.size() == 1 && back.checkpoints[0].ids == std::vector<int32_t>({1, 2}) &&
                  back.checkpoints[0].gdn.size() == 64 && back.kv.size() == 1 && back.kv[0].k.size() == 512,
                  "the spilled entry reads back whole, checkpoint chain and K/V included");
            check(small.spilled_entries() == 0 && small.spilled_restores() == 1, "the manifest entry goes on restore");
            check(small.best(prompt, {}, true).tokens == 0, "the restored entry is the caller's active state now");
        }
        {   // hysteresis: the entry just hit and re-parked is never spilled inside the cooldown; later it is
            ConversationCache c(1 << 20, 1);
            c.set_spill_hooks(hooks);
            c.set_spill_clock([&] { return clock_now; });
            c.set_spill_cooldown(120.0);
            check(c.put(image({1, 2, 3})), "A parked");
            auto ra = c.take(c.best(a, {}, true).index);   // A is the active conversation
            clock_now += 1.0;
            check(c.put(std::move(ra)), "A parks again right after its turn");
            check(!c.put(image({4, 5, 6})), "inside the cooldown the fresh hit is not spilled: the park is refused");
            check(c.spills() == 0 && c.evictions() == 0 && c.failed_spills() == 0, "no thrash spill happened");
            clock_now += 200.0;
            check(c.put(image({4, 5, 6})), "past the cooldown the stale entry spills");
            check(c.spills() == 1 && c.evictions() == 1, "the eviction after the cooldown wrote a file");
        }
        {   // fail-open: a failed write falls back to the current behaviour (drop), parking still succeeds
            auto c = hooked();
            c.set_spill_cooldown(0.0);
            c.set_spill_hooks([&] {
                ConversationCache::SpillHooks h2 = hooks;
                h2.write = [](const SavedConversation&, const std::string&, std::string& e) {
                    e = "injected failure";
                    return false;
                };
                return h2;
            }());
            c.set_spill_clock([&] { return clock_now; });
            const size_t one = image({1, 2, 3}).bytes();
            check(c.put(image({1, 2, 3})) && c.put(image({7, 8, 9})), "two entries parked");
            check(c.put(image({4, 5, 6})) && c.evictions() == 1 && c.failed_spills() == 1 && c.spills() == 0,
                  "a failed spill still evicts (fail-open drop) and counts the failure");
            check(!c.last_spill_error().empty(), "the spill error is reported");
            check(c.spilled_entries() == 0, "a failed spill leaves no manifest entry");
        }
        {   // the RAM safety hook: a refused admission does not write to disk (the victim is dropped instead)
            auto c = hooked();
            c.set_spill_cooldown(0.0);
            c.set_spill_hooks([&] {
                ConversationCache::SpillHooks h2 = hooks;
                h2.admit_write = [](size_t) { return false; };
                return h2;
            }());
            c.set_spill_clock([&] { return clock_now; });
            check(c.put(image({1, 2, 3})) && c.put(image({7, 8, 9})), "two entries parked");
            check(c.put(image({4, 5, 6})) && c.evictions() == 1 && c.failed_spills() == 1 && c.spills() == 0,
                  "a refused admission falls back to the drop");
        }
        {   // spill_one: the gate-refusal path frees exactly one eligible entry per call; pins and fresh entries stay
            auto c = hooked();
            check(c.put(image({1, 2, 3})), "A parked");
            clock_now += 200.0;
            check(c.put(image({7, 8, 9})), "B parked fresh");
            check(c.spill_one(), "spill_one spills the oldest eligible entry");
            check(c.spills() == 1 && c.evictions() == 0 && c.size() == 1 && c.spilled_entries() == 1,
                  "spill_one frees RAM without counting an eviction");
            check(!c.spill_one(), "the fresh entry is inside the cooldown: nothing more is spillable");
            clock_now += 200.0;
            check(c.spill_one(), "past the cooldown the last entry spills too");
            check(c.size() == 0 && c.spilled_entries() == 2, "an emptied cache");
        }
        {   // a spilled file that no longer reads (corrupt, identity changed) is dropped, not a dead match
            auto c = hooked();
            check(c.put(image({1, 2, 3})) && c.put(image({7, 8, 9})), "two entries parked");
            clock_now += 200.0;
            check(c.put(image({4, 5, 6})) && c.spills() == 1, "the oldest spilled");
            check(c.spilled_entries() == 1, "manifest holds one");
            // corrupt every spill file the earlier blocks left behind (each cache numbers its own spill-1.bin, ...)
            for (const auto& e : fs::directory_iterator(dir))
                if (e.is_regular_file() && e.file_size() > 0) fs::resize_file(e.path(), 8);
            SavedConversation back;
            auto m = c.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true);
            check(m.spilled, "the manifest still matches the corrupt entry");
            check(!c.restore_spilled(m.index, back), "the corrupt file refuses to read");
            check(c.spilled_entries() == 0, "the unreadable entry is dropped from the manifest (fail-open)");
        }
        {   // superseded spilled copies leave the manifest when the same conversation parks a newer state
            auto c = hooked();
            SavedConversation big = image({1, 2, 3});
            ConversationCheckpoint cp;
            cp.ids = {1, 2};
            big.checkpoints.push_back(cp);
            const size_t one = big.bytes();
            ConversationCache small(one + image({9, 8, 7}).bytes(), 8);
            small.set_spill_hooks(hooks);
            small.set_spill_clock([&] { return clock_now; });
            small.set_spill_cooldown(120.0);
            check(small.put(std::move(big)) && small.put(image({9, 8, 7})), "parked");
            clock_now += 200.0;
            check(small.put(image({5, 6, 7})) && small.spills() == 1, "the first copy spilled");
            auto newer = image({1, 2, 3, 30, 31});   // the same conversation, one turn later (longer live state)
            check(small.put(std::move(newer)) && small.spilled_entries() == 0, "the newer park supersedes the spilled copy");
            auto bm = small.best(std::vector<int64_t>{1, 2, 3, 30, 31, 40}, {}, true);
            check(bm.tokens == 5 && !bm.spilled, "the newer copy is in RAM and matches longer");
        }
        fs::remove_all(dir);
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
