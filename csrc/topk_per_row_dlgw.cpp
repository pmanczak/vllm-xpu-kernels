// Many-row topk_per_row for Intel BMG (Xe2): one work-group per row, delegate-
// bound select, used next to the radix kernels of topk_per_row.cpp where it
// measurably beats them (see the gate at the end of namespace dlgw).
//
// Rows of at least minDelegateLen(topK) columns run the delegate-bound select:
// from an SLM copy of the row if it fits the stage (StagedRow), else straight
// from global memory (GlobalRow). Shorter rows, and the rare row on which the
// select gives up (more candidates than slots, e.g. very tie-heavy rows), run
// topKPerRowJob of topk_per_row_dlgw_common.h from the same SLM arena - exact
// either way.
//
// Delegate-bound select. Split the row into groups of G floats and call each
// group's largest key its delegate. Let t be the topK-th largest delegate:
// topK groups hold a key >= t each, so t is at most the row's topK-th largest
// key, and every key > t lies in one of the fewer than topK groups whose
// delegate is > t. So the row's top topK are its keys > t from those groups -
// the candidates - plus, when there are fewer than topK of them, keys == t.
// With G ~ n / (kGroupDiv * topK) the only passes over the whole row are the
// copy into SLM and one max per group; the radix selects run over the
// ~3 * topK delegates and the candidates, not over n keys.
//
// Keys are uint32 in the order of topk_per_row.cpp's (+NaN above +inf, -NaN
// below -inf, +0 above -0), and every comparison is on keys or on the float
// bits, so NaN, signed zeros and infinities rank as they do there.
//
// The host entry points never throw: a call they do not cover returns false
// before any allocation or launch, and the caller runs the original kernels,
// whose argument checks then report anything malformed.

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "topk_per_row_dlgw_common.h"
#include "topk_per_row_dlgw.h"
#include "utils.h"

// Forced inlining of the hot helpers into the kernel body.
#define DLGW_AI inline __attribute__((always_inline))

namespace topk_dlgw {

namespace vllm {

namespace dlgw {

// Threads per work-group, and work-groups a core keeps resident (BMG: one
// 512-wide
// group per Xe core; two do not fit the SLM arena of topK 2048 and gained
// nothing).
#ifndef DLGW_THREADS
  #define DLGW_THREADS 512
#endif
static constexpr int kNumThreads = DLGW_THREADS;
static constexpr int kNumSubGroups = kNumThreads / SUBGROUP_SIZE;
#ifndef DLGW_WG_PER_CORE
  #define DLGW_WG_PER_CORE 1
#endif
static constexpr int kWgPerCore = DLGW_WG_PER_CORE;

// SLM kept free per work-group when sizing the arena against residency.
#ifndef DLGW_SLM_RESERVE
  #define DLGW_SLM_RESERVE 4096
#endif
static constexpr size_t kSlmReserve = DLGW_SLM_RESERVE;

// Largest topK the module can run (the auto gate takes far fewer: topKTuned).
static constexpr int64_t kMaxTopK = 4096;

// Group size G = n / (kGroupDiv * topK), rounded down to a power of two and
// clamped to [kGMin, kGMax]: ~3 * topK delegates, which leaves 2.1-3.1 * topK
// keys >= t on randn and clustered data.
// kGMin: one float4 per lane and group; kGMax bounds the candidates
// ((topK - 1) * G) and the G keys a lane holds in registers.
static constexpr int kGroupDiv = 3;
static constexpr int kGMin = 4;
static constexpr int kGMax = 16;
// The copies move whole float4s between kCopyAlign-aligned addresses: a row
// starts h = (address % kCopyAlign) / 4 floats into its first line, at most
// kMaxShift, and the stage holds it shifted by h. The first kLineFloats floats
// of the stage (the head) are loaded one by one when h > 0.
static constexpr int kCopyAlign = 64;
static constexpr int kLineFloats = kCopyAlign / static_cast<int>(sizeof(float));
static constexpr int kMaxShift = kLineFloats - 1;

// Rows from minDelegateLen(topK) columns on run the delegate select, shorter
// ones the job. kMinLenFactor * topK columns give at least 2 * topK delegates
// at G = kGMin (1.2-1.3 * topK candidates on randn rows). Below
// kMinDelegateCols columns the job is faster still (topK 512: rows of 4096 to
// 6143 columns lost 5 % on B70), and rows from
// kGMin * kGroupDiv * topK columns on always have enough delegates. A
// delegate row is never
// shorter than the stage head (topK 1: 16, not 12 columns), so it never ends
// inside it.
static constexpr int kMinLenFactor = 2 * kGMin;
static constexpr int64_t kMinDelegateCols = 6144;
static constexpr int64_t minDelegateLen(int64_t topK) {
  const int64_t len = kMinLenFactor * topK;
  const int64_t atLeast = len > kMinDelegateCols ? len : kMinDelegateCols;
  const int64_t always = kGMin * kGroupDiv * topK;
  const int64_t shortest = atLeast < always ? atLeast : always;
  return shortest > kLineFloats ? shortest : kLineFloats;
}

// Candidate slots by default (8 B each of SLM): 4096 keeps the arena small
// enough that rows up to 262144 columns still fit the global select on BMG's
// 128 KB SLM. A row with more candidates falls back to the job (exact, slower).
static constexpr int kCandCap = 4096;
// Most candidate slots the global select takes when SLM allows (key + position
// = 8 B each).
static constexpr int64_t kGlobalCandCapMax = 16384;

// Radix selects: 11-bit digits (2048 bins: 13-bit digits made the find step
// bank conflicted), at most three passes; a selected bin of at most kSmallSet
// keys is ranked in one subgroup instead.
static constexpr int kRadixBits = 11;
static constexpr int kHistBins = 1 << kRadixBits;
static constexpr int kSmallSet = SUBGROUP_SIZE;

// Stage lengths are multiples of a chunk of the largest group, 32 lanes x
// kGMax floats, and the stage has one such chunk of slack for the shift that
// aligns the rows and for padding the last chunk.
static constexpr int kStageAlign = SUBGROUP_SIZE * kGMax;
// SLM loads per lane that the passes over the stage and the delegates
// (delegates, group list, candidates) keep in flight: they are latency bound,
// and the compiler does not overlap the loads of different loop iterations.
// Unrolling the radix and emit passes as well measured slower.
static constexpr int kLoadsInFlight = 4;
// The same for the pass that streams a row from global memory (rows past the
// stage cap): LSC bandwidth is bytes in flight over the latency. The pass over
// the listed groups keeps kLoadsInFlight: 8 there (32 keys a lane) spill.
static constexpr int kGlobalLoads = 8;
// Widest launch the module takes: the global select's delegates (n / kGMax
// words) must fit the arena; wider launches are left to _C.
static constexpr int64_t kGlobalMaxWidth = int64_t{1} << 20;

// Stage padding: the bit pattern of the smallest key, 0.
static constexpr uint32_t kPadBits = 0xffffffffu;

// Words of the misc block of the arena.
enum MiscSlot : int {
  kSel = 0,         // find(): selected bin, -1 if none
  kSelAbove,        // keys in the bins above it
  kSelCount,        // keys in it
  kSmallN,          // keys collected for the small-set finish
  kGroupN,          // group list entries
  kCandN,           // candidates (may exceed the slots)
  kTieN,            // ties appended after the candidates
  kEqTaken,         // emit: keys == T taken
  kOutN,            // emit: output slots taken
  kSubTotals = 32,  // find(): one count per subgroup
  kMaxKey = 64,     // largest delegate
  kLowKey,          // a lower bound of t
  kSmall = 96,      // kSmallSet keys of the small-set finish
  kMiscInts = 128
};
static_assert(kSubTotals + kNumSubGroups <= kMaxKey, "misc slots overlap");
static_assert(kSmall + kSmallSet <= kMiscInts, "misc slots overlap");
static_assert(kHistBins % kNumThreads == 0, "find() splits the bins evenly");
static_assert(kNumSubGroups <= SUBGROUP_SIZE, "find() scans one per lane");

// Byte offsets (64 B aligned) of a select's state in the arena.
struct SelectOffsets {
  int delegates;   // uint32 per group, padded to a multiple of 128
  int groups;      // int32 [topK]: groups whose delegate is > t
  int candidates;  // [candCap] of candidates and ties: StagedRow int32 stage
                   // positions, GlobalRow uint32 keys
  int positions;   // GlobalRow: int32 [candCap] stage positions
  int histogram;   // int32 [kHistBins]
  int misc;        // int32 [kMiscInts]
};

// The SLM arena of a work-group, set on the host. Every offset is a multiple of
// 64 B; the 16 B vector accesses also assume the base of the dynamic SLM
// allocation is 16 B aligned, as the runtime provides it (the kernels are
// fuzzed against torch.topk on that assumption). The staged select keeps the
// stage at offset 0, the global one its delegates; topKPerRowJob uses the
// arena from offset 0. One row per work-group: the three never meet.
struct ArenaLayout {
  int stageCap;          // longest row the staged select takes, 0 for none
  int globalCap;         // longest row the global select takes, 0 for none
  int candCap;           // candidate slots of the staged select
  int globalCandCap;     // candidate slots of the global select
  SelectOffsets staged;  // StagedRow
  SelectOffsets global;  // GlobalRow
  size_t bytes;          // the whole arena
};

template <typename T>
using SlmPtr = typename sycl::multi_ptr<
    T,
    sycl::access::address_space::local_space,
    sycl::access::decorated::yes>::pointer;

template <typename T>
using SlmAtomic = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

static DLGW_AI uint32_t ballot(const sycl::sub_group& sg, bool p) {
  uint32_t bits = 0;
  sycl::ext::oneapi::group_ballot(sg, p).extract_bits(bits);
  return bits;
}

static DLGW_AI int popc(uint32_t m) {
  return static_cast<int>(sycl::popcount(m));
}

static DLGW_AI int bitWidth(uint32_t v) {
  return v ? 32 - static_cast<int>(sycl::clz(v)) : 0;
}

static constexpr int divUp(int a, int b) { return (a + b - 1) / b; }
static constexpr int roundUp(int a, int b) { return divUp(a, b) * b; }

// The key of float bits (splitKeyOf of _C: larger float, larger key).
static DLGW_AI uint32_t keyOf(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The float bits of a key.
static DLGW_AI uint32_t bitsOf(uint32_t key) {
  return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

// The largest key of a group from the signed max and min of its float bits:
// non-negative floats order as their bits do, and the negative ones, all
// below them, in reverse.
static DLGW_AI uint32_t groupKey(int32_t smax, int32_t smin) {
  return smax >= 0 ? (static_cast<uint32_t>(smax) | 0x80000000u)
                   : ~static_cast<uint32_t>(smin);
}

// "key > t" and "key == t" on float bits, one compare each. As signed
// integers the bits order like the floats for t >= +0; for t <= -0 the larger
// floats are the unsigned bit patterns below t's. The flip turns either into
// an unsigned compare.
struct Threshold {
  uint32_t bits;
  uint32_t flip;
  uint32_t ref;
  DLGW_AI explicit Threshold(uint32_t key)
      : bits(bitsOf(key)),
        flip((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u),
        ref(bits ^ flip) {}
  DLGW_AI bool above(uint32_t b) const { return (b ^ flip) > ref; }
  DLGW_AI bool equal(uint32_t b) const { return b == bits; }
};

// The keys a radix pass counts: lo <= key <= lo + last, digit
// (key - lo) >> shift.
struct Digit {
  uint32_t lo;
  uint32_t last;
  int shift;
  // Keys whose low shift + width bits are free above lo. Capped at the largest
  // key, so that keys below lo cannot wrap around into the range.
  DLGW_AI Digit(uint32_t lo_, int shift_, int width) : lo(lo_), shift(shift_) {
    const int bits = shift_ + width;
    const uint32_t span = bits >= 32 ? 0xffffffffu : (1u << bits) - 1u;
    last = sycl::min(span, 0xffffffffu - lo_);
  }
  DLGW_AI bool test(uint32_t key, int& bin) const {
    const uint32_t rel = key - lo;
    bin = static_cast<int>(rel >> shift);
    return rel <= last;
  }
};

// The pow2 group size of a staged row of n >= minDelegateLen(topK) columns.
static DLGW_AI int groupSize(int n, int topK) {
  const int ratio = n / (kGroupDiv * topK);
  int g = kGMin;
  while (2 * g <= ratio && g < kGMax)
    g *= 2;
  return g;
}

// A lower bound of t from one lane's delegates, in the order it computes
// them: the lane cuts its delegates of numChunks chunks into S = ceil(topK /
// WG) runs, and the WG * S >= topK run maxima are distinct delegates, so their
// minimum is at most t. Valid only when every run has a delegate (bounded()).
struct RunBound {
  const int runs;
  const int runLen;
  int inRun = 0;
  int run = 0;
  uint32_t runMax = 0;
  uint32_t low = 0xffffffffu;
  DLGW_AI RunBound(int numChunks, int sgid, int topK)
      : runs(divUp(topK, kNumThreads)),
        runLen(sycl::max(1, divUp(numChunks - sgid, kNumSubGroups) / runs)) {}
  static DLGW_AI bool bounded(int numChunks, int topK) {
    return numChunks / kNumSubGroups >= 2 * divUp(topK, kNumThreads);
  }
  DLGW_AI void add(uint32_t d) {
    runMax = sycl::max(runMax, d);
    if (++inRun == runLen && run < runs - 1) {
      low = sycl::min(low, runMax);
      runMax = 0;
      inRun = 0;
      ++run;
    }
  }
  DLGW_AI uint32_t finish() const { return sycl::min(low, runMax); }
};

// The parts of the delegate-bound select that do not read the row, run by the
// whole work-group: the radix selects of t (over the delegates) and of T
// (over the candidates), the group list and the output. StagedRow (row in SLM)
// and GlobalRow (row in global memory) add the passes over the row. Every
// branch on row data is uniform across the work-group, and every exit is taken
// before the first output write: run() returns false (the caller then runs the
// job on the row) if the candidates overflow their slots or if a select finds
// no bin, which the bounds rule out.
//
// Both paths see the row as the stage holds it: stage[p] = row[p - h] for h <=
// p < h + n and the key-0 pattern elsewhere, h = (row address % kCopyAlign) /
// 4, so that float4 v of the stage is a 16 B-aligned float4 of the row (and the
// copies move whole float4s between 64 B-aligned addresses). Chunk c is stage
// floats [32 G c, 32 G (c + 1)); group j = 32 c + l is the G floats of float4s
// 8 G c + 32 q + l, q < G / 4, so that the lanes of a subgroup read a chunk
// conflict-free from SLM and coalesced from global memory. Candidates and ties
// are stage positions; output index = position - h.
class DelegateSelect {
 protected:
  DLGW_AI DelegateSelect(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const SelectOffsets& at,
      const float* row,
      int n,
      int topK,
      int candCap,
      int32_t* out)
      : item_(item),
        sg_(item.get_sub_group()),
        lid_(static_cast<int>(item.get_local_id(0))),
        sgid_(static_cast<int>(sg_.get_group_linear_id())),
        lane_(static_cast<int>(sg_.get_local_linear_id())),
        lower_((1u << lane_) - 1u),
        src_(reinterpret_cast<const uint32_t*>(row)),
        h_(static_cast<int>(
            (reinterpret_cast<uintptr_t>(row) & (kCopyAlign - 1)) /
            sizeof(float))),
        n_(n),
        topK_(topK),
        candCap_(candCap),
        out_(out),
        delegates_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.delegates)),
        groups_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.groups)),
        hist_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.histogram)),
        misc_(reinterpret_cast<SlmPtr<int32_t>>(arena + at.misc)),
        miscU_(reinterpret_cast<SlmPtr<uint32_t>>(arena + at.misc)) {}

  template <typename T, typename P>
  static DLGW_AI sycl::vec<T, 4> load4(P p, int v) {
    return *reinterpret_cast<SlmPtr<const sycl::vec<T, 4>>>(p + 4 * v);
  }

  // First float4 of group j, and the stage position of its u-th float.
  template <int G>
  static DLGW_AI int groupFloat4(int j) {
    return (j >> 5) * 8 * G + (j & 31);
  }
  static DLGW_AI int stagePos(int v, int u) {
    return 4 * (v + (u >> 2) * SUBGROUP_SIZE) + (u & 3);
  }

  // The delegate of a group from its kQ float4s of float bits.
  template <int kQ>
  static DLGW_AI uint32_t groupDelegate(const sycl::vec<int32_t, 4> (&x)[kQ]) {
    sycl::vec<int32_t, 4> hi = x[0];
    sycl::vec<int32_t, 4> lo = x[0];
#pragma unroll
    for (int q = 1; q < kQ; ++q) {
      hi = sycl::max(hi, x[q]);
      lo = sycl::min(lo, x[q]);
    }
    return groupKey(
        sycl::max(sycl::max(hi.x(), hi.y()), sycl::max(hi.z(), hi.w())),
        sycl::min(sycl::min(lo.x(), lo.y()), sycl::min(lo.z(), lo.w())));
  }

  // After the delegate pass: pads the delegates with key 0 to delegateWords
  // and publishes their maximum and the lower bound of t (0 if unbounded).
  // The select state must be initialised and visible.
  DLGW_AI void publishDelegates(
      int numDelegates,
      int delegateWords,
      uint32_t rowMax,
      uint32_t low) const {
    for (int j = numDelegates + lid_; j < delegateWords; j += kNumThreads)
      delegates_[j] = 0;
    rowMax = sycl::reduce_over_group(sg_, rowMax, sycl::maximum<uint32_t>());
    low = sycl::reduce_over_group(sg_, low, sycl::minimum<uint32_t>());
    if (lane_ == 0) {
      SlmAtomic<uint32_t>(miscU_[kMaxKey]).fetch_max(rowMax);
      SlmAtomic<uint32_t>(miscU_[kLowKey]).fetch_min(low);
    }
    sycl::group_barrier(item_.get_group());
  }

  // t, the topK-th largest delegate, over the delegates >= the lower bound.
  DLGW_AI bool delegateThreshold(
      int delegateWords, uint32_t maxKey, uint32_t& t, int& need) const {
    const uint32_t lowKey = miscU_[kLowKey];
    const auto pass = [&](const Digit& d, bool collect) {
      for (int b = sgid_ * SUBGROUP_SIZE; b < delegateWords / 4;
           b += kNumThreads) {
        const auto v = load4<uint32_t>(delegates_, b + lane_);
        const uint32_t key[4] = {v.x(), v.y(), v.z(), v.w()};
        int bin[4];
        bool in[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          in[u] = d.test(key[u], bin[u]);
          if (collect) addSmall(key[u] - lowKey, in[u]);
        }
        if (!collect) histAdd4(bin, in);
      }
    };
    return select(lowKey, bitWidth(maxKey - lowKey), pass, t, need);
  }

  // T, the topK-th largest of numCands > topK candidates (all keys > t).
  // cand(i, pos) returns candidate i's key and sets its stage position.
  template <typename Cand>
  DLGW_AI bool candidateThreshold(
      int numCands,
      uint32_t t,
      uint32_t maxKey,
      const Cand& cand,
      uint32_t& T,
      int& need) const {
    const auto pass = [&](const Digit& d, bool collect) {
      for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
        const int i = b + lane_;
        const bool valid = i < numCands;
        int pos = 0;
        const uint32_t key = cand(valid ? i : 0, pos);
        int bin = 0;
        const bool in = valid && d.test(key, bin);
        if (collect)
          addSmall(key - t, in);
        else
          histAdd(bin, in);
      }
    };
    return select(t, bitWidth(maxKey - t), pass, T, need);
  }

  DLGW_AI void addSmall(uint32_t rel, bool in) const {
    if (in) {
      const int p = SlmAtomic<int32_t>(misc_[kSmallN]).fetch_add(1);
      if (p < kSmallSet) miscU_[kSmall + p] = rel;
    }
  }

  DLGW_AI void histAdd(int bin, bool in) const {
    if (in) SlmAtomic<int32_t>(hist_[bin]).fetch_add(1);
  }

  // Four keys per lane; when all 128 of the subgroup hit one bin (tie-heavy
  // rows), one atomic adds them.
  DLGW_AI void histAdd4(const int (&bin)[4], const bool (&in)[4]) const {
    const bool same = in[0] && in[1] && in[2] && in[3] && bin[0] == bin[1] &&
                      bin[1] == bin[2] && bin[2] == bin[3];
    const int bin0 = sycl::group_broadcast(sg_, bin[0], 0);
    if (sycl::all_of_group(sg_, same && bin[0] == bin0)) {
      if (lane_ == 0)
        SlmAtomic<int32_t>(hist_[bin0]).fetch_add(4 * SUBGROUP_SIZE);
      return;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
      histAdd(bin[u], in[u]);
  }

  // With the histogram of numBins bins complete: the bin, counted from the
  // top, in which the count reaches need, the count above it and in it.
  // Clears the histogram for the next pass. False if no bin qualifies.
  DLGW_AI bool
  find(int numBins, int need, int& bin, int& above, int& count) const {
    constexpr int kPerThread = kHistBins / kNumThreads;
    if (lid_ == 0) misc_[kSel] = -1;
    int c[kPerThread];
    int sum = 0;
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      c[u] = q < numBins ? hist_[numBins - 1 - q] : 0;
      sum += c[u];
    }
    const int incl = sycl::inclusive_scan_over_group(sg_, sum, sycl::plus<>());
    if (lane_ == SUBGROUP_SIZE - 1) misc_[kSubTotals + sgid_] = incl;
    sycl::group_barrier(item_.get_group());
    const int total = lane_ < kNumSubGroups ? misc_[kSubTotals + lane_] : 0;
    const int prior =
        sycl::exclusive_scan_over_group(sg_, total, sycl::plus<>());
    int a = sycl::group_broadcast(sg_, prior, sgid_) + incl - sum;
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      if (q < numBins && a < need && a + c[u] >= need) {
        misc_[kSel] = numBins - 1 - q;
        misc_[kSelAbove] = a;
        misc_[kSelCount] = c[u];
        misc_[kSmallN] = 0;
      }
      a += c[u];
    }
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
      const int q = lid_ * kPerThread + u;
      if (q < numBins && c[u]) hist_[numBins - 1 - q] = 0;
    }
    sycl::group_barrier(item_.get_group());
    bin = misc_[kSel];
    above = misc_[kSelAbove];
    count = misc_[kSelCount];
    return bin >= 0;
  }

  // Rank the `count` keys of the selected bin (collected as key - base) in
  // subgroup 0: prefix becomes the need-th largest of them.
  DLGW_AI bool rankSmall(int count, int& need, uint32_t& prefix) const {
    if (sgid_ == 0) {
      const uint32_t mine = lane_ < count ? miscU_[kSmall + lane_] : 0u;
      int gt = 0;
      int eq = 0;
      for (int j = 0; j < count; ++j) {
        const uint32_t other = sycl::select_from_group(sg_, mine, j);
        gt += other > mine ? 1 : 0;
        eq += other == mine ? 1 : 0;
      }
      const uint32_t hits =
          ballot(sg_, lane_ < count && gt < need && gt + eq >= need);
      const int first = hits ? static_cast<int>(sycl::ctz(hits)) : 0;
      const int gtFirst = sycl::select_from_group(sg_, gt, first);
      if (lane_ == 0) {
        misc_[kSel] = hits ? first : -1;
        misc_[kSelAbove] = gtFirst;
      }
    }
    sycl::group_barrier(item_.get_group());
    const int sel = misc_[kSel];
    if (sel < 0 || misc_[kSmallN] != count) return false;
    need -= misc_[kSelAbove];
    prefix = miscU_[kSmall + sel];
    return true;
  }

  // The topK-th largest key of a set holding at least topK keys >= base, all
  // below base + 2^bits: key, and need = how many keys == key to take (the
  // other topK - need are > key). pass(digit, collect) adds the digit of every
  // key in the digit's range to the histogram, or with collect appends their
  // key - base to the small set. The histogram is zero on entry and exit.
  template <typename Pass>
  DLGW_AI bool
  select(uint32_t base, int bits, const Pass& pass, uint32_t& key, int& need)
      const {
    uint32_t prefix = 0;
    int done = 0;
    need = topK_;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      if (done < bits) {
        const int width = sycl::min(kRadixBits, bits - done);
        const int shift = bits - done - width;
        pass(Digit(base + prefix, shift, width), false);
        sycl::group_barrier(item_.get_group());
        int bin = 0;
        int above = 0;
        int count = 0;
        if (!find(1 << width, need, bin, above, count)) return false;
        need -= above;
        prefix |= static_cast<uint32_t>(bin) << shift;
        done += width;
        if (done < bits && count <= kSmallSet) {
          pass(Digit(base + prefix, 0, bits - done), true);
          sycl::group_barrier(item_.get_group());
          if (!rankSmall(count, need, prefix)) return false;
          done = bits;
        }
      }
    }
    key = base + prefix;
    return true;
  }

  // The group list: every group whose delegate is > t (fewer than topK). Per
  // subgroup a count pass over its chunks, one atomic, a write pass: an atomic
  // per chunk would put its latency on the loop's dependency chain.
  DLGW_AI void groupList(int numChunks, uint32_t t) const {
    constexpr int kU = kLoadsInFlight;
    constexpr int kStep = kU * kNumSubGroups;
    int mine = 0;
    for (int c0 = sgid_; c0 < numChunks; c0 += kStep) {
      uint32_t d[kU];
      delegatesOf(c0, numChunks, d);
#pragma unroll
      for (int u = 0; u < kU; ++u)
        mine += (c0 + u * kNumSubGroups < numChunks && d[u] > t) ? 1 : 0;
    }
    const int total = sycl::reduce_over_group(sg_, mine, sycl::plus<>());
    int slot = 0;
    if (lane_ == 0 && total)
      slot = SlmAtomic<int32_t>(misc_[kGroupN]).fetch_add(total);
    slot = sycl::group_broadcast(sg_, slot, 0);
    for (int c0 = sgid_; c0 < numChunks; c0 += kStep) {
      uint32_t d[kU];
      delegatesOf(c0, numChunks, d);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = c0 + u * kNumSubGroups;
        const bool above = c < numChunks && d[u] > t;
        const uint32_t mask = ballot(sg_, above);
        const int w = slot + popc(mask & lower_);
        if (above && w < topK_) groups_[w] = c * SUBGROUP_SIZE + lane_;
        slot += popc(mask);
      }
    }
  }

  // This lane's delegates of chunks c0 + u * kNumSubGroups (clamped to the
  // last chunk; the caller masks those).
  DLGW_AI void
  delegatesOf(int c0, int numChunks, uint32_t (&d)[kLoadsInFlight]) const {
#pragma unroll
    for (int u = 0; u < kLoadsInFlight; ++u)
      d[u] = delegates_
          [sycl::min(c0 + u * kNumSubGroups, numChunks - 1) * SUBGROUP_SIZE +
           lane_];
  }

  // More than topK candidates: write those > T and `need` of those == T.
  // Per subgroup a count pass and one atomic each for its share of the == T
  // quota and for its output slots, then the write pass. cand as in
  // candidateThreshold.
  template <typename Cand>
  DLGW_AI void
  emitSelected(int numCands, uint32_t T, int need, const Cand& cand) const {
    int gt = 0;
    int eq = 0;
    for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
      const int i = b + lane_;
      const bool valid = i < numCands;
      int pos = 0;
      const uint32_t key = cand(valid ? i : 0, pos);
      gt += (valid && key > T) ? 1 : 0;
      eq += (valid && key == T) ? 1 : 0;
    }
    const int sgGt = sycl::reduce_over_group(sg_, gt, sycl::plus<>());
    const int sgEq = sycl::reduce_over_group(sg_, eq, sycl::plus<>());
    int eqBase = 0;
    if (lane_ == 0 && sgEq)
      eqBase = SlmAtomic<int32_t>(misc_[kEqTaken]).fetch_add(sgEq);
    eqBase = sycl::group_broadcast(sg_, eqBase, 0);
    const int quota = sycl::clamp(need - eqBase, 0, sgEq);
    int o = 0;
    if (lane_ == 0 && sgGt + quota)
      o = SlmAtomic<int32_t>(misc_[kOutN]).fetch_add(sgGt + quota);
    o = sycl::group_broadcast(sg_, o, 0);
    int eqSeen = 0;
    for (int b = sgid_ * SUBGROUP_SIZE; b < numCands; b += kNumThreads) {
      const int i = b + lane_;
      const bool valid = i < numCands;
      int pos = 0;
      const uint32_t key = cand(valid ? i : 0, pos);
      const bool isEq = valid && key == T;
      const uint32_t eqMask = ballot(sg_, isEq);
      const bool take = (valid && key > T) ||
                        (isEq && eqSeen + popc(eqMask & lower_) < quota);
      const uint32_t takeMask = ballot(sg_, take);
      if (take) out_[o + popc(takeMask & lower_)] = pos - h_;
      o += popc(takeMask);
      eqSeen += popc(eqMask);
    }
  }

  // At most topK candidates, ties included: the first topK positions.
  DLGW_AI void emitPositions(SlmPtr<int32_t> positions) const {
    for (int i = lid_; i < topK_; i += kNumThreads)
      out_[i] = positions[i] - h_;
  }

  const sycl::nd_item<3>& item_;
  const sycl::sub_group sg_;
  const int lid_;
  const int sgid_;
  const int lane_;
  const uint32_t lower_;  // lanes below this one
  const uint32_t* src_;   // the row's float bits from its start
  const int h_;           // stage shift
  const int n_;
  const int topK_;
  const int candCap_;
  int32_t* out_;
  SlmPtr<uint32_t> delegates_;
  SlmPtr<int32_t> groups_;
  SlmPtr<int32_t> hist_;
  SlmPtr<int32_t> misc_;
  SlmPtr<uint32_t> miscU_;
};

// A row of up to the stage cap: copied into SLM once, and every pass reads it
// there.
class StagedRow : public DelegateSelect {
 public:
  DLGW_AI StagedRow(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const ArenaLayout& layout,
      const float* row,
      int n,
      int topK,
      int32_t* out)
      : DelegateSelect(
            item, arena, layout.staged, row, n, topK, layout.candCap, out),
        stage_(reinterpret_cast<SlmPtr<uint32_t>>(arena)),
        cands_(
            reinterpret_cast<SlmPtr<int32_t>>(
                arena + layout.staged.candidates)) {}

  template <int G>
  DLGW_AI bool run() const {
    const int nPad = roundUp(h_ + n_, SUBGROUP_SIZE * G);
    const int numChunks = nPad / (SUBGROUP_SIZE * G);
    const int numDelegates = numChunks * SUBGROUP_SIZE;
    const int delegateWords = roundUp(numDelegates, 128);
    stage(nPad);
    delegates<G>(numChunks, numDelegates, delegateWords);
    const uint32_t maxKey = miscU_[kMaxKey];
    uint32_t t = 0;
    int need = 0;
    if (!delegateThreshold(delegateWords, maxKey, t, need)) return false;

    groupList(numChunks, t);
    sycl::group_barrier(item_.get_group());
    const int numGroups = misc_[kGroupN];
    if (numGroups >= topK_) return false;
    const Threshold th(t);
    candidates<G>(numGroups, th);
    sycl::group_barrier(item_.get_group());
    const int numCands = misc_[kCandN];
    if (numCands > candCap_) return false;

    const auto cand = [&](int i, int& pos) {
      pos = cands_[i];
      return keyOf(stage_[pos]);
    };
    if (numCands > topK_) {
      uint32_t T = 0;
      if (!candidateThreshold(numCands, t, maxKey, cand, T, need)) return false;
      emitSelected(numCands, T, need, cand);
    } else {
      if (numCands < topK_) {
        if (!ties<G>(numDelegates, t, th, numCands)) return false;
        sycl::group_barrier(item_.get_group());
      }
      emitPositions(cands_);
    }
    return true;
  }

 private:
  // Fill the stage up to nPad and clear the select state; returns once every
  // thread can read the stage. The up to kMaxShift floats before the first
  // kCopyAlign-aligned one and the up to 3 after the last whole float4 are
  // loaded one by one, so nothing outside [row, row + n) is read. The row is at
  // least kLineFloats long (minDelegateLen), so it ends after the head.
  DLGW_AI void stage(int nPad) const {
    // Float4s: stage[0, kLineFloats) is the head.
    const int first = h_ ? kLineFloats / 4 : 0;
    const int end = (h_ + n_) >> 2;
    for (int v = first + lid_; v < end; v += kNumThreads) {
      const sycl::vec<uint32_t, 4> x =
          *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src_ + (4 * v - h_));
      stage_[4 * v] = x[0];
      stage_[4 * v + 1] = x[1];
      stage_[4 * v + 2] = x[2];
      stage_[4 * v + 3] = x[3];
    }
    if (lid_ < 4 * first) stage_[lid_] = lid_ < h_ ? kPadBits : src_[lid_ - h_];
    for (int p = 4 * end + lid_; p < nPad; p += kNumThreads)
      stage_[p] = p < h_ + n_ ? src_[p - h_] : kPadBits;
    for (int i = lid_; i < kHistBins; i += kNumThreads)
      hist_[i] = 0;
    if (lid_ < kMiscInts) misc_[lid_] = lid_ == kLowKey ? -1 : 0;
    sycl::group_barrier(item_.get_group());
  }

  // The delegates (key 0 past numDelegates), their maximum and a lower bound
  // of t (RunBound).
  template <int G>
  DLGW_AI void
  delegates(int numChunks, int numDelegates, int delegateWords) const {
    constexpr int kQ = G / 4;
    RunBound bound(numChunks, sgid_, topK_);
    uint32_t rowMax = 0;
    // kU of the subgroup's chunks per iteration, all their loads issued first.
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    for (int c0 = sgid_; c0 < numChunks; c0 += kU * kNumSubGroups) {
      sycl::vec<int32_t, 4> x[kU][kQ];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = sycl::min(c0 + u * kNumSubGroups, numChunks - 1);
#pragma unroll
        for (int q = 0; q < kQ; ++q)
          x[u][q] =
              load4<int32_t>(stage_, c * 8 * G + q * SUBGROUP_SIZE + lane_);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int c = c0 + u * kNumSubGroups;
        if (c < numChunks) {
          const uint32_t d = groupDelegate<kQ>(x[u]);
          delegates_[c * SUBGROUP_SIZE + lane_] = d;
          rowMax = sycl::max(rowMax, d);
          bound.add(d);
        }
      }
    }
    publishDelegates(
        numDelegates,
        delegateWords,
        rowMax,
        RunBound::bounded(numChunks, topK_) ? bound.finish() : 0u);
  }

  // The candidates, every key > t of the listed groups: a count pass, one
  // atomic per subgroup, a write pass. Writes past candCap_ are dropped; the
  // count is kept in full.
  template <int G>
  DLGW_AI void candidates(int numGroups, const Threshold& th) const {
    constexpr int kQ = G / 4;
    // kU listed groups per lane and iteration, their loads issued first.
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    constexpr int kStep = kU * kNumThreads;
    int mine = 0;
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups; b0 += kStep) {
      uint32_t e[kU][G];
      int v[kU];
      groupKeys<G, kU>(b0, numGroups, e, v);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k)
          mine += (valid && th.above(e[u][k])) ? 1 : 0;
      }
    }
    const int total = sycl::reduce_over_group(sg_, mine, sycl::plus<>());
    int slot = 0;
    if (lane_ == 0 && total)
      slot = SlmAtomic<int32_t>(misc_[kCandN]).fetch_add(total);
    slot = sycl::group_broadcast(sg_, slot, 0);
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups; b0 += kStep) {
      uint32_t e[kU][G];
      int v[kU];
      groupKeys<G, kU>(b0, numGroups, e, v);
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
        int cnt = 0;
#pragma unroll
        for (int k = 0; k < G; ++k)
          cnt += (valid && th.above(e[u][k])) ? 1 : 0;
        const int off =
            sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
        int w = slot + off;
#pragma unroll
        for (int k = 0; k < G; ++k) {
          if (valid && th.above(e[u][k])) {
            if (w < candCap_) cands_[w] = stagePos(v[u], k);
            ++w;
          }
        }
        slot += sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      }
    }
  }

  // The float bits of listed groups b0 + u * WG + lane, u < kU, and their
  // first float4 (entries past numGroups > 0 read the last one; the caller
  // masks them).
  template <int G, int kU>
  DLGW_AI void
  groupKeys(int b0, int numGroups, uint32_t (&e)[kU][G], int (&v)[kU]) const {
    constexpr int kQ = G / 4;
#pragma unroll
    for (int u = 0; u < kU; ++u)
      v[u] = groupFloat4<G>(
          groups_[sycl::min(b0 + u * kNumThreads + lane_, numGroups - 1)]);
#pragma unroll
    for (int u = 0; u < kU; ++u) {
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const auto x = load4<uint32_t>(stage_, v[u] + q * SUBGROUP_SIZE);
        e[u][4 * q] = x.x();
        e[u][4 * q + 1] = x.y();
        e[u][4 * q + 2] = x.z();
        e[u][4 * q + 3] = x.w();
      }
    }
  }

  // Fewer than topK candidates: append topK - numCands keys == t from the
  // groups whose delegate is >= t (no other holds one), stopping once the
  // work-group has seen enough. Padding has key 0, so t == 0 would match it:
  // only positions inside the row count.
  template <int G>
  DLGW_AI bool
  ties(int numDelegates, uint32_t t, const Threshold& th, int numCands) const {
    constexpr int kQ = G / 4;
    const int needed = topK_ - numCands;
    int have = 0;
    for (int b = 0; b < numDelegates && have < needed; b += kNumThreads) {
      const int j = sycl::min(b + lid_, numDelegates - 1);
      const bool valid = b + lid_ < numDelegates && delegates_[j] >= t;
      const int v = groupFloat4<G>(j);
      uint32_t e[G];
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const auto x = load4<uint32_t>(stage_, v + q * SUBGROUP_SIZE);
        e[4 * q] = x.x();
        e[4 * q + 1] = x.y();
        e[4 * q + 2] = x.z();
        e[4 * q + 3] = x.w();
      }
      bool tie[G];
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < G; ++u) {
        const int p = stagePos(v, u);
        tie[u] = valid && th.equal(e[u]) && p >= h_ && p < h_ + n_;
        cnt += tie[u] ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      if (total) {
        int base = 0;
        if (lane_ == 0)
          base = SlmAtomic<int32_t>(misc_[kTieN]).fetch_add(total);
        int w = numCands + sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
        for (int u = 0; u < G; ++u) {
          if (tie[u]) {
            if (w < topK_) cands_[w] = stagePos(v, u);
            ++w;
          }
        }
      }
      have += sycl::reduce_over_group(item_.get_group(), cnt, sycl::plus<>());
    }
    return have >= needed;
  }

  SlmPtr<uint32_t> stage_;
  SlmPtr<int32_t> cands_;
};

// A row longer than the stage cap, read from global memory: the delegate
// pass streams it with kGlobalLoads float4 loads per lane in flight, and the
// later passes re-read only the listed groups (and, for ties, the groups whose
// delegate is t), which the first pass has just brought into the caches.
// Candidates are (key, position) pairs in SLM, so the select of T and the
// output do not read the row again. Whole float4s inside the row are single
// 16 B loads; the at most 3 floats before the first one and the at most 3
// after the last one are loaded one by one, and stage positions outside the
// row read as the key-0 pattern, so nothing outside [row, row + n) is read.
//
// G follows the row length as in the staged select (groupSize). A row with
// more candidates than slots (e.g. very tie-heavy rows) falls back to the job.
class GlobalRow : public DelegateSelect {
 public:
  DLGW_AI GlobalRow(
      const sycl::nd_item<3>& item,
      SlmPtr<char> arena,
      const ArenaLayout& layout,
      const float* row,
      int n,
      int topK,
      int32_t* out)
      : DelegateSelect(
            item,
            arena,
            layout.global,
            row,
            n,
            topK,
            layout.globalCandCap,
            out),
        candKeys_(
            reinterpret_cast<SlmPtr<uint32_t>>(
                arena + layout.global.candidates)),
        candPos_(
            reinterpret_cast<SlmPtr<int32_t>>(
                arena + layout.global.positions)) {}

  template <int G>
  DLGW_AI bool run() const {
    const int nPad = roundUp(h_ + n_, SUBGROUP_SIZE * G);
    const int numChunks = nPad / (SUBGROUP_SIZE * G);
    const int numDelegates = numChunks * SUBGROUP_SIZE;
    const int delegateWords = roundUp(numDelegates, 128);
    for (int i = lid_; i < kHistBins; i += kNumThreads)
      hist_[i] = 0;
    if (lid_ < kMiscInts) misc_[lid_] = lid_ == kLowKey ? -1 : 0;
    delegates<G>(numChunks, numDelegates, delegateWords);
    const uint32_t maxKey = miscU_[kMaxKey];
    uint32_t t = 0;
    int need = 0;
    if (!delegateThreshold(delegateWords, maxKey, t, need)) return false;

    groupList(numChunks, t);
    sycl::group_barrier(item_.get_group());
    const int numGroups = misc_[kGroupN];
    if (numGroups >= topK_) return false;
    const Threshold th(t);
    candidates<G>(numGroups, th);
    sycl::group_barrier(item_.get_group());
    const int numCands = misc_[kCandN];
    if (numCands > candCap_) return false;

    const auto cand = [&](int i, int& pos) {
      pos = candPos_[i];
      return candKeys_[i];
    };
    if (numCands > topK_) {
      uint32_t T = 0;
      if (!candidateThreshold(numCands, t, maxKey, cand, T, need)) return false;
      emitSelected(numCands, T, need, cand);
    } else {
      if (numCands < topK_) {
        if (!ties<G>(numDelegates, t, th, numCands)) return false;
        sycl::group_barrier(item_.get_group());
      }
      emitPositions(candPos_);
    }
    return true;
  }

 private:
  // Float4 v of the stage, which lies inside the row.
  DLGW_AI sycl::vec<int32_t, 4> inside4(int v) const {
    return *reinterpret_cast<const sycl::vec<int32_t, 4>*>(src_ + (4 * v - h_));
  }

  // Float4 v of the stage, anywhere.
  DLGW_AI sycl::vec<int32_t, 4> stage4(int v) const {
    const int p = 4 * v - h_;  // row index of its first float
    if (p >= 0 && p + 4 <= n_) return inside4(v);
    const auto at = [&](int i) {
      return i >= 0 && i < n_ ? static_cast<int32_t>(src_[i])
                              : static_cast<int32_t>(kPadBits);
    };
    return sycl::vec<int32_t, 4>(at(p), at(p + 1), at(p + 2), at(p + 3));
  }

  // The G float bits of group j's first float4 v.
  template <int G>
  DLGW_AI void groupBits(int v, uint32_t (&e)[G]) const {
#pragma unroll
    for (int q = 0; q < G / 4; ++q) {
      const auto x = stage4(v + q * SUBGROUP_SIZE);
      e[4 * q] = static_cast<uint32_t>(x.x());
      e[4 * q + 1] = static_cast<uint32_t>(x.y());
      e[4 * q + 2] = static_cast<uint32_t>(x.z());
      e[4 * q + 3] = static_cast<uint32_t>(x.w());
    }
  }

  // The delegates as on the staged path. Chunks [firstInside, endInside) lie
  // inside the row; an iteration whose chunks all do loads them unchecked.
  template <int G>
  DLGW_AI void
  delegates(int numChunks, int numDelegates, int delegateWords) const {
    constexpr int kQ = G / 4;
    constexpr int kU = kGlobalLoads / kQ > 1 ? kGlobalLoads / kQ : 1;
    const int firstInside = h_ ? 1 : 0;
    const int endInside = (h_ + n_) / (SUBGROUP_SIZE * G);
    RunBound bound(numChunks, sgid_, topK_);
    uint32_t rowMax = 0;
    const auto keep = [&](int c, uint32_t d) {
      delegates_[c * SUBGROUP_SIZE + lane_] = d;
      rowMax = sycl::max(rowMax, d);
      bound.add(d);
    };
    for (int c0 = sgid_; c0 < numChunks; c0 += kU * kNumSubGroups) {
      if (c0 >= firstInside && c0 + (kU - 1) * kNumSubGroups < endInside) {
        // One address per lane, the kU * kQ loads at constant offsets.
        const auto* p = reinterpret_cast<const sycl::vec<int32_t, 4>*>(
            src_ + (4 * (c0 * 8 * G + lane_) - h_));
        sycl::vec<int32_t, 4> x[kU][kQ];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
#pragma unroll
          for (int q = 0; q < kQ; ++q)
            x[u][q] = p[u * kNumSubGroups * 8 * G + q * SUBGROUP_SIZE];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u)
          keep(c0 + u * kNumSubGroups, groupDelegate<kQ>(x[u]));
      } else {
        // The first and the last chunks, one at a time.
#pragma unroll
        for (int u = 0; u < kU; ++u) {
          const int c = c0 + u * kNumSubGroups;
          if (c < numChunks) {
            sycl::vec<int32_t, 4> x[kQ];
#pragma unroll
            for (int q = 0; q < kQ; ++q)
              x[q] = stage4(c * 8 * G + q * SUBGROUP_SIZE + lane_);
            keep(c, groupDelegate<kQ>(x));
          }
        }
      }
    }
    // The select state cleared by run() is visible from here on.
    sycl::group_barrier(item_.get_group());
    publishDelegates(
        numDelegates,
        delegateWords,
        rowMax,
        RunBound::bounded(numChunks, topK_) ? bound.finish() : 0u);
  }

  // The candidates, every key > t of the listed groups, as (key, position):
  // one pass, one atomic per subgroup and iteration (a second pass would read
  // the groups from global memory again). Writes past candCap_ are dropped;
  // the count is kept in full.
  template <int G>
  DLGW_AI void candidates(int numGroups, const Threshold& th) const {
    constexpr int kQ = G / 4;
    constexpr int kU = kLoadsInFlight / kQ > 1 ? kLoadsInFlight / kQ : 1;
    for (int b0 = sgid_ * SUBGROUP_SIZE; b0 < numGroups;
         b0 += kU * kNumThreads) {
      uint32_t e[kU][G];
      int v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u)
        v[u] = groupFloat4<G>(
            groups_[sycl::min(b0 + u * kNumThreads + lane_, numGroups - 1)]);
#pragma unroll
      for (int u = 0; u < kU; ++u)
        groupBits<G>(v[u], e[u]);
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k)
          cnt += (valid && th.above(e[u][k])) ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      int base = 0;
      if (lane_ == 0 && total)
        base = SlmAtomic<int32_t>(misc_[kCandN]).fetch_add(total);
      int w = sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const bool valid = b0 + u * kNumThreads + lane_ < numGroups;
#pragma unroll
        for (int k = 0; k < G; ++k) {
          if (valid && th.above(e[u][k])) {
            if (w < candCap_) {
              candKeys_[w] = keyOf(e[u][k]);
              candPos_[w] = stagePos(v[u], k);
            }
            ++w;
          }
        }
      }
    }
  }

  // As StagedRow::ties, reading only the groups whose delegate is >= t.
  template <int G>
  DLGW_AI bool
  ties(int numDelegates, uint32_t t, const Threshold& th, int numCands) const {
    const int needed = topK_ - numCands;
    int have = 0;
    for (int b = 0; b < numDelegates && have < needed; b += kNumThreads) {
      const int j = sycl::min(b + lid_, numDelegates - 1);
      const bool valid = b + lid_ < numDelegates && delegates_[j] >= t;
      const int v = groupFloat4<G>(j);
      uint32_t e[G];
      if (valid) {
        groupBits<G>(v, e);
      } else {
#pragma unroll
        for (int u = 0; u < G; ++u)
          e[u] = kPadBits;
      }
      bool tie[G];
      int cnt = 0;
#pragma unroll
      for (int u = 0; u < G; ++u) {
        const int p = stagePos(v, u);
        tie[u] = valid && th.equal(e[u]) && p >= h_ && p < h_ + n_;
        cnt += tie[u] ? 1 : 0;
      }
      const int off = sycl::exclusive_scan_over_group(sg_, cnt, sycl::plus<>());
      const int total =
          sycl::group_broadcast(sg_, off + cnt, SUBGROUP_SIZE - 1);
      if (total) {
        int base = 0;
        if (lane_ == 0)
          base = SlmAtomic<int32_t>(misc_[kTieN]).fetch_add(total);
        int w = numCands + sycl::group_broadcast(sg_, base, 0) + off;
#pragma unroll
        for (int u = 0; u < G; ++u) {
          if (tie[u]) {
            if (w < topK_) candPos_[w] = stagePos(v, u);
            ++w;
          }
        }
      }
      have += sycl::reduce_over_group(item_.get_group(), cnt, sycl::plus<>());
    }
    return have >= needed;
  }

  SlmPtr<uint32_t> candKeys_;
  SlmPtr<int32_t> candPos_;
};

// What the gate and the arena size need to know about a device.
struct DeviceInfo {
  int slices;         // 0 if the runtime cannot tell
  int coresPerSlice;  // Xe cores (subslices) per slice, 0 if unknown
  int xeCores;        // slices * coresPerSlice
  size_t localMemBytes;
  bool tuned;  // BMG-G31: the only architecture this path is measured on
  size_t maxWorkGroup;  // largest work-group the device runs
};

// Cached per device index (a process may drive several devices).
static DeviceInfo deviceInfo(int deviceIndex, const sycl::device& device) {
  static std::mutex mutex;
  static std::map<int, DeviceInfo> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(deviceIndex);
  if (it != cache.end()) return it->second;
  namespace intel_info = sycl::ext::intel::info::device;
  namespace syclex = sycl::ext::oneapi::experimental;
  DeviceInfo info{
      0,
      0,
      0,
      device.get_info<sycl::info::device::local_mem_size>(),
      device.get_info<syclex::info::device::architecture>() ==
          syclex::architecture::intel_gpu_bmg_g31,
      device.get_info<sycl::info::device::max_work_group_size>()};
  if (device.has(sycl::aspect::ext_intel_gpu_slices) &&
      device.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
    info.slices = static_cast<int>(device.get_info<intel_info::gpu_slices>());
    info.coresPerSlice = static_cast<int>(
        device.get_info<intel_info::gpu_subslices_per_slice>());
    info.xeCores = info.slices * info.coresPerSlice;
  }
  cache.emplace(deviceIndex, info);
  return info;
}

// SLM one work-group may take: its share with kWgPerCore resident, rounded
// down to a power of two, less kSlmReserve.
static size_t arenaBudget(const DeviceInfo& info) {
  const size_t share = info.localMemBytes / kWgPerCore;
  size_t pow2 = 1;
  while (pow2 <= share / 2)
    pow2 *= 2;
  return pow2 > kSlmReserve ? pow2 - kSlmReserve : 0;
}

// Delegate words for staged rows of up to `width` columns: for each group size
// the delegates of its longest row (one per lane and chunk of 32 G floats,
// the stage shift included), padded to 128.
static int64_t delegateWords(int64_t width, int64_t topK) {
  int64_t words = 0;
  for (int64_t g = kGMin; g <= kGMax; g *= 2) {
    // G = g from g * kGroupDiv * topK columns (kGMin: from minDelegateLen) to
    // twice that.
    const int64_t from = g * kGroupDiv * topK;
    const int64_t shortest = g == kGMin ? minDelegateLen(topK) : from;
    const int64_t longest = g == kGMax ? width : std::min(width, 2 * from - 1);
    if (longest >= shortest) {
      const int64_t chunk = SUBGROUP_SIZE * g;
      words = std::max(
          words, (longest + kMaxShift + chunk - 1) / chunk * SUBGROUP_SIZE);
    }
  }
  return (words + 127) / 128 * 128;
}

// The staged select's layout for rows of up to stageCap columns; returns its
// size in bytes.
static size_t stagedLayout(
    int64_t stageCap, int64_t topK, int64_t candCap, SelectOffsets& at) {
  size_t offset =
      alignUp(sizeof(uint32_t) * (stageCap + kStageAlign), size_t{64});
  at.delegates = static_cast<int>(offset);
  offset += alignUp(sizeof(uint32_t) * delegateWords(stageCap, topK), 64);
  at.groups = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * topK, 64);
  at.candidates = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * candCap, 64);
  at.positions = 0;
  at.histogram = static_cast<int>(offset);
  offset += sizeof(int32_t) * kHistBins;
  at.misc = static_cast<int>(offset);
  offset += sizeof(int32_t) * kMiscInts;
  return offset;
}

// The global select's layout for rows of up to `width` columns (delegates from
// offset 0); returns its size in bytes.
static size_t
globalLayout(int64_t width, int64_t topK, int64_t candCap, SelectOffsets& at) {
  const int64_t words = delegateWords(width, topK);
  size_t offset = 0;
  at.delegates = 0;
  offset += alignUp(sizeof(uint32_t) * words, size_t{64});
  at.groups = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * topK, 64);
  at.candidates = static_cast<int>(offset);
  offset += alignUp(sizeof(uint32_t) * candCap, 64);
  at.positions = static_cast<int>(offset);
  offset += alignUp(sizeof(int32_t) * candCap, 64);
  at.histogram = static_cast<int>(offset);
  offset += sizeof(int32_t) * kHistBins;
  at.misc = static_cast<int>(offset);
  offset += sizeof(int32_t) * kMiscInts;
  return offset;
}

// The longest stage, a multiple of kStageAlign within the budget, whose staged
// layout fits the budget; 0 if that is shorter than the shortest staged row.
// Every term of stagedLayout grows with the stage, so a launch's stage cap is
// the smaller of this and its width rounded up. Searched once per (budget,
// topK, slots) and cached: launches differ in width.
static int64_t longestStage(int64_t topK, int64_t slots, size_t budget) {
  static std::mutex mutex;
  static std::map<std::tuple<size_t, int64_t, int64_t>, int64_t> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_tuple(budget, topK, slots);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const int64_t shortest = minDelegateLen(topK);
  int64_t cap = static_cast<int64_t>(budget / sizeof(uint32_t)) / kStageAlign *
                kStageAlign;
  SelectOffsets at{};
  while (cap >= shortest && stagedLayout(cap, topK, slots, at) > budget)
    cap -= kStageAlign;
  if (cap < shortest) cap = 0;
  cache.emplace(key, cap);
  return cap;
}

// The arena of a launch of rows up to `width` (<= kGlobalMaxWidth) columns:
// the staged select takes rows of up to stageCap columns, the longest multiple
// of kStageAlign (at most the width rounded up) whose stage fits the budget,
// 0 if that is shorter than the shortest staged row; the global select takes
// the longer ones if its state for the width fits too (globalCap = width,
// else 0: those rows run the job).
static ArenaLayout arenaLayout(int64_t width, int64_t topK, size_t budget) {
  int64_t slots = std::min<int64_t>(kCandCap, (topK - 1) * kGMax + 1);
  // More slots than SLM: the stage will not fit (and candCap stays an int).
  slots = std::min<int64_t>(std::max(slots, topK), budget / 4 + 1);
  const int64_t shortest = minDelegateLen(topK);
  int64_t cap = std::min<int64_t>(
      (width + kStageAlign - 1) / kStageAlign * kStageAlign,
      longestStage(topK, slots, budget));
  ArenaLayout layout{};
  size_t bytes = 0;
  if (cap >= shortest)
    bytes = stagedLayout(cap, topK, slots, layout.staged);
  else
    cap = 0;
  layout.stageCap = static_cast<int>(cap);
  layout.candCap = static_cast<int>(slots);
  if (width > cap && width >= shortest) {
    // The global select has no stage to size: it takes every candidate slot
    // that still fits (clustered rows - the topK largest in a few groups -
    // carry ~5k candidates, past the 4096 of the staged select).
    int64_t gslots = slots;
    SelectOffsets probe{};
    const size_t base = globalLayout(width, topK, 0, probe);
    if (base < budget) {
      // 8 B a slot (key + position), whole 128s.
      const int64_t fit = static_cast<int64_t>((budget - base) / 8) / 128 * 128;
      gslots = std::max(slots, std::min(kGlobalCandCapMax, fit));
    }
    const size_t globalBytes = globalLayout(width, topK, gslots, layout.global);
    layout.globalCandCap = static_cast<int>(gslots);
    if (globalBytes <= budget) {
      layout.globalCap = static_cast<int>(width);
      bytes = std::max(bytes, globalBytes);
    } else {
      layout.global = SelectOffsets{};
    }
  }
  layout.bytes = std::max(bytes, sharedMemorySize(static_cast<size_t>(topK)));
  return layout;
}

// arenaLayout of a launch, cached: it depends only on the device, the width and
// topK, and decode repeats the same launch every step.
static ArenaLayout cachedLayout(
    int deviceIndex, const DeviceInfo& info, int64_t width, int64_t topK) {
  static std::mutex mutex;
  static std::map<std::tuple<int, int64_t, int64_t>, ArenaLayout> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_tuple(deviceIndex, width, topK);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() >= 1024) cache.clear();
  const ArenaLayout layout = arenaLayout(width, topK, arenaBudget(info));
  cache.emplace(key, layout);
  return layout;
}

// One work-group per row. Decode rows cover [0, len) with len from seqLens as
// in top_k_per_row_decode_kernel; prefill rows [rowStarts[r], rowEnds[r]).
// Output indices are relative to the row's start, -1 past its length.
template <int kThreads, bool kPrefill>
class top_k_per_row_dlgw_kernel {
  static_assert(kThreads == kNumThreads, "kernel threads == module threads");

 public:
  top_k_per_row_dlgw_kernel(
      sycl::local_accessor<char, 1>& arena,
      const ArenaLayout& layout,
      const float* logits,
      const int32_t* seqLens,
      const int32_t* rowStarts,
      const int32_t* rowEnds,
      int32_t* indices,
      int64_t stride0,
      int64_t topK,
      int64_t nextN)
      : arena_(arena),
        layout_(layout),
        logits_(logits),
        seqLens_(seqLens),
        rowStarts_(rowStarts),
        rowEnds_(rowEnds),
        indices_(indices),
        stride0_(stride0),
        topK_(topK),
        nextN_(nextN) {}

  __attribute__((always_inline)) void operator()
      [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] (
          const sycl::nd_item<3>& item) const {
    const int64_t rowIdx = item.get_group(0);
    int rowStart = 0;
    int rowEnd = 0;
    if constexpr (kPrefill) {
      rowStart = rowStarts_[rowIdx];
      rowEnd = rowEnds_[rowIdx];
    } else {
      // Clamp as in CUDA: padded rows with seq_len < next_n have no columns.
      const int32_t seqLen = seqLens_[rowIdx / nextN_];
      rowEnd = sycl::max(0, int(seqLen - nextN_ + (rowIdx % nextN_) + 1));
    }
    const int topK = static_cast<int>(topK_);
    // rowEnd < rowStart is an empty row.
    const int n = sycl::max(rowEnd - rowStart, 0);
    const float* row = logits_ + rowIdx * stride0_;
    int32_t* out = indices_ + rowIdx * topK_;
    const int lid = static_cast<int>(item.get_local_id(0));
    if (n <= topK) {
      for (int i = lid; i < topK; i += kThreads)
        out[i] = i < n ? i : -1;
      return;
    }
    if (n >= minDelegateLen(topK) &&
        (n <= layout_.stageCap || n <= layout_.globalCap)) {
      const auto arena =
          arena_.template get_multi_ptr<sycl::access::decorated::yes>().get();
      bool done = false;
      if (n <= layout_.stageCap) {
        const StagedRow staged(
            item, arena, layout_, row + rowStart, n, topK, out);
        switch (groupSize(n, topK)) {
          case 4:
            done = staged.run<4>();
            break;
          case 8:
            done = staged.run<8>();
            break;
          default:
            done = staged.run<16>();
            break;
        }
      } else {
        const GlobalRow global(
            item, arena, layout_, row + rowStart, n, topK, out);
        switch (groupSize(n, topK)) {
          case 4:
            done = global.run<4>();
            break;
          case 8:
            done = global.run<8>();
            break;
          default:
            done = global.run<16>();
            break;
        }
      }
      if (done) return;
      // The job reuses the arena.
      sycl::group_barrier(item.get_group());
    }
    // The host only launches for stride1 == 1.
    topKPerRowJob<kThreads, kNumBins, false>(
        item,
        arena_,
        /*indices=*/nullptr,
        row,
        rowStart,
        rowStart + n,
        out,
        /*outLogits=*/nullptr,
        /*stride1=*/1,
        topK);
  }

 private:
  sycl::local_accessor<char, 1> arena_;
  ArenaLayout layout_;
  const float* logits_;
  const int32_t* seqLens_;    // decode
  const int32_t* rowStarts_;  // prefill
  const int32_t* rowEnds_;    // prefill
  int32_t* indices_;
  int64_t stride0_;
  int64_t topK_;
  int64_t nextN_;
};

template <bool kPrefill>
static void launch(
    sycl::queue& queue,
    int64_t numRows,
    const ArenaLayout& layout,
    const float* logits,
    const int32_t* seqLens,
    const int32_t* rowStarts,
    const int32_t* rowEnds,
    int32_t* indices,
    int64_t stride0,
    int64_t topK,
    int64_t nextN) {
  sycl::range<3> grid(numRows, 1, 1);
  sycl::range<3> block(kNumThreads, 1, 1);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> arena(sycl::range<1>(layout.bytes), cgh);
    const top_k_per_row_dlgw_kernel<kNumThreads, kPrefill> kernel(
        arena,
        layout,
        logits,
        seqLens,
        rowStarts,
        rowEnds,
        indices,
        stride0,
        topK,
        nextN);
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), kernel);
  });
}

// What both ops need of the call to run it at all: the layout the kernel
// assumes (float32 rows with unit column stride, int32 [numRows, topK]
// contiguous output on the same device) and sizes in its range (rows of up to
// kGlobalMaxWidth columns).
static bool legalCommon(
    const torch::Tensor& logits,
    const torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  if (numRows < 1 || topK < 1 || topK > kMaxTopK || stride1 != 1) return false;
  if (!logits.defined() || !logits.is_xpu() ||
      logits.scalar_type() != at::kFloat || logits.dim() != 2)
    return false;
  if (logits.size(0) != numRows || logits.size(1) < 1 ||
      logits.size(1) > kGlobalMaxWidth || logits.stride(0) != stride0 ||
      stride0 < 0 || logits.stride(1) != 1)
    return false;
  if (!indices.defined() || indices.device() != logits.device() ||
      indices.scalar_type() != at::kInt || indices.dim() != 2 ||
      indices.size(0) != numRows || indices.size(1) != topK ||
      !indices.is_contiguous())
    return false;
  return true;
}

// An int32 vector of numRows entries on the logits' device.
static bool legalRowVector(
    const torch::Tensor& v, const torch::Tensor& logits, int64_t numRows) {
  return v.defined() && v.device() == logits.device() &&
         v.scalar_type() == at::kInt && v.dim() == 1 && v.size(0) == numRows &&
         v.is_contiguous();
}

// Gate, measured on B70 (BMG-G31, 32 Xe cores) against the radix kernels of
// topk_per_row.cpp over decode rows {1..256} x widths {4k..512k} x topK
// {512, 1024, 2048}, prefill q {16..256} x widths {4k..128k}, decode launches
// with per-row lengths (uniform, short, bimodal) and rows of other value
// distributions (clustered topK, ties, narrow, constant, sorted): this path is
// taken only where it was >= ~1.0x on every one of them, with the crossover
// rounded up.
//   rows <= xeCores (one work-group per row: a single wave)
//     and minWidthFor(topK) <= width <= 262144, topK in {512, 1024, 2048}
// Wide launches of more rows than cores are left to the original kernels even
// though full-length rows win there (1.2-1.9x): the gate sees only the padded
// width, and a launch of many *short* rows (seq_len <= 8k in a 128k-wide
// logits tensor) runs the job here at one work-group per core (the arena is
// sized for the widest row, ~100 KB of the 128 KB SLM), where the original
// kernel fits ~5 work-groups per core: 0.6-0.85x. With a single wave the same
// short rows cost the same as there (1.00x).
// 262144 measured 1.2-2.8x; from 327680 on it loses for most shapes (524288
// down to 0.25x). Only BMG-G31 devices (B70) are gated in; set
// VLLM_XPU_TOPK_DLGW=1 to apply the same rule to another device, =0 to turn the
// path off.
// Only the topK values it was measured and tuned at (thresholds, candidate
// slots and the minimum widths are tuned per value).
static bool topKTuned(int64_t topK) {
  return topK == 512 || topK == 1024 || topK == 2048;
}
// The table's smallest minWidthFor.
static constexpr int64_t kMinWidth = 32768;
static constexpr int64_t kMaxWidth = 262144;

// The narrowest launch taken, by topK. Clustered rows (the topK largest in one
// block of the row, e.g. a recent-token bonus) carry ~2x the usual candidates
// here, and the original kernel finishes them in one histogram step: below
// these widths this path loses 0.7-0.9x on them where it wins only 1.05-1.4x
// on smooth rows. At/above them it is >= 0.97x on every row type measured.
// VLLM_XPU_TOPK_DLGW_MIN_WIDTH replaces the table.
static int64_t minWidthOverride() {
  const auto e = ::vllm::xpu::getEnv("VLLM_XPU_TOPK_DLGW_MIN_WIDTH");
  return e.has_value() && !e->empty() ? std::atoll(e->c_str()) : int64_t{0};
}
static int64_t minWidthFor(int64_t topK) {
  if (minWidthOverride() > 0) return minWidthOverride();
  return topK <= 1024 ? 32768 : std::max<int64_t>(45056, 22 * topK);
}

// VLLM_XPU_TOPK_DLGW, read on every call so that it can be toggled in-process:
// "0" off, "1" on for any device, anything else (or unset) the default: only
// BMG-G31 devices. Returns 0, 1 or -1 for the default.
static int envOverride() {
  const auto e = ::vllm::xpu::getEnv("VLLM_XPU_TOPK_DLGW");
  if (!e.has_value()) return -1;
  return *e == "0" ? 0 : *e == "1" ? 1 : -1;
}

// First thing in an entry point: host time spent here is added to every call
// the original kernels keep, so reject on plain integers (the width and topK,
// and at most kPrecheckRows rows, more than any BMG has Xe cores) before any
// tensor or device query.
static constexpr int64_t kPrecheckRows = 64;
static bool
precheck(const torch::Tensor& logits, int64_t numRows, int64_t topK) {
  if (numRows > kPrecheckRows || envOverride() == 0 || !topKTuned(topK) ||
      !logits.defined() || logits.dim() != 2)
    return false;
  const int64_t width = logits.size(1);
  return width >= (minWidthOverride() > 0 ? minWidthOverride() : kMinWidth) &&
         width <= kMaxWidth;
}

static bool gateTakes(
    int64_t numRows, int64_t width, int64_t topK, const DeviceInfo& info) {
  const int env = envOverride();
  if (env == 0 || info.xeCores <= 0) return false;
  if (env < 0 && !info.tuned) return false;
  if (!topKTuned(topK) || numRows > info.xeCores ||
      info.maxWorkGroup < static_cast<size_t>(kNumThreads))
    return false;
  return width >= minWidthFor(topK) && width <= kMaxWidth;
}

}  // namespace dlgw

}  // namespace vllm

bool top_k_per_row_decode_dlgw(
    const torch::Tensor& logits,
    int64_t next_n,
    const torch::Tensor& seqLens,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  namespace dlgw = vllm::dlgw;
  if (!dlgw::precheck(logits, numRows, topK)) return false;
  if (!dlgw::legalCommon(logits, indices, numRows, stride0, stride1, topK))
    return false;
  // Only 1D seqLens: the original kernel reads a 2D (B, next_n) tensor as flat
  // 1D with the real next_n, so a 2D one is left to it to keep both paths
  // agreeing on every row's length.
  if (next_n < 1 || !seqLens.defined() || seqLens.device() != logits.device() ||
      seqLens.scalar_type() != at::kInt || !seqLens.is_contiguous() ||
      seqLens.dim() != 1 || numRows % next_n != 0 ||
      seqLens.size(0) != numRows / next_n)
    return false;

  const at::DeviceGuard device_guard(logits.device());
  const int deviceIndex = logits.device().index();
  auto& queue = c10::xpu::getCurrentXPUStream(deviceIndex).queue();
  const dlgw::DeviceInfo info =
      dlgw::deviceInfo(deviceIndex, queue.get_device());
  const int64_t width = logits.size(1);
  if (!dlgw::gateTakes(numRows, width, topK, info)) return false;
  const dlgw::ArenaLayout layout =
      dlgw::cachedLayout(deviceIndex, info, width, topK);
  if (layout.bytes > dlgw::arenaBudget(info)) return false;

  // A launch the device rejects is left to the original kernels.
  try {
    dlgw::launch<false>(
        queue,
        numRows,
        layout,
        logits.data_ptr<float>(),
        seqLens.data_ptr<int32_t>(),
        /*rowStarts=*/nullptr,
        /*rowEnds=*/nullptr,
        indices.data_ptr<int32_t>(),
        stride0,
        topK,
        next_n);
  } catch (const sycl::exception&) {
    return false;
  }
  return true;
}

bool top_k_per_row_prefill_dlgw(
    const torch::Tensor& logits,
    const torch::Tensor& rowStarts,
    const torch::Tensor& rowEnds,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK) {
  namespace dlgw = vllm::dlgw;
  if (!dlgw::precheck(logits, numRows, topK)) return false;
  if (!dlgw::legalCommon(logits, indices, numRows, stride0, stride1, topK))
    return false;
  if (!dlgw::legalRowVector(rowStarts, logits, numRows) ||
      !dlgw::legalRowVector(rowEnds, logits, numRows))
    return false;

  const at::DeviceGuard device_guard(logits.device());
  const int deviceIndex = logits.device().index();
  auto& queue = c10::xpu::getCurrentXPUStream(deviceIndex).queue();
  const dlgw::DeviceInfo info =
      dlgw::deviceInfo(deviceIndex, queue.get_device());
  const int64_t width = logits.size(1);
  if (!dlgw::gateTakes(numRows, width, topK, info)) return false;
  const dlgw::ArenaLayout layout =
      dlgw::cachedLayout(deviceIndex, info, width, topK);
  if (layout.bytes > dlgw::arenaBudget(info)) return false;

  // A launch the device rejects is left to the original kernels.
  try {
    dlgw::launch<true>(
        queue,
        numRows,
        layout,
        logits.data_ptr<float>(),
        /*seqLens=*/nullptr,
        rowStarts.data_ptr<int32_t>(),
        rowEnds.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        stride0,
        topK,
        /*nextN=*/1);
  } catch (const sycl::exception&) {
    return false;
  }
  return true;
}

}  // namespace topk_dlgw
