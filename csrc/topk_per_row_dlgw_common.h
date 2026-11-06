// Single-group topk_per_row code for the delegate-bound path: constants,
// shared state and the single-group job topKPerRowJob, which runs the rows the
// delegate select does not take (short rows, rows on which it gives up). A
// copy of the job of topk_per_row.cpp with a smaller final-items buffer and a
// longer-row histogram; everything is inside topk_dlgw::vllm so the two never
// share a kernel or a symbol.

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace topk_dlgw {

namespace vllm {

static constexpr int SUBGROUP_SIZE = 32;

// The number of slots for the final pass. This buffer is the largest part of a
// work-group's shared memory, and shared memory is what decides how many
// work-groups a core can keep resident, which is in turn what hides the latency
// of every stage: at topK=128 the six stages of a row cost 8 to 15 us each, all
// six within a factor of two of one another, which is the signature of a
// pipeline that has nothing else to run while it waits rather than of one stage
// being expensive. 512 slots leave 12.6KB per work-group instead of 24.7KB.
// Rows whose threshold bin does not fit take one more radix step, which is only
// correctness-neutral because step 3 writes the last bin's elements without
// using this buffer at all.
static constexpr int kNumFinalItems = 512;

// The number of bins in the histogram.
static constexpr int kNumBins = 2048;

// Rows per work-group in the grouped prefill kernel. One work-group takes this
// many consecutive rows, which in a causal prefill means rows of nearly equal
// length, and splits its threads between them. Two measurements motivate it,
// both on the S=3500 triangle with the split drawn at rowLen > topK:
//
//   * At topK=128 the 3372 rows that need a selection cost 75.2 us of the 75.5,
//     and the pure read and write of the same rows is 23.1. The gap is a per
//     row fixed cost of ~16 ns, five work-group barriers deep, and only ~4 rows
//     are in flight per Xe core to hide it. Sharing one barrier between
//     kRowsPerGroup rows divides that cost and multiplies the rows in flight by
//     the same factor.
//
//   * At topK=2048 the 2048 write-only rows cost 15.1 us alone and the 1452
//   rows
//     that need a selection cost 49.1 alone, but together they cost 66.7: the
//     two add up, so none of the write is hidden behind the selection. Grouped
//     rows write kRowsPerGroup * topK words as one contiguous run.
static constexpr int kRowsPerGroup = 4;

// The reduction path (topKPerRowReduce) keeps this many of the row's elements
// per thread in registers, so it can read the row once. 8 covers every row up
// to 8 * 512 = 4096 elements, which is where DSA prefill rows sit; longer rows
// use the same code with a chunked re-read. Elements each thread keeps in
// registers. kFastMaxRegItems * the block size has to cover the longest row
// this path accepts, and both ends of that trade were measured: 256 threads
// with 16 items each lost 20 us of 71 on the S=3500 triangle, and even the bare
// read and write of level 1 lost 11, because half as many threads on a row
// means half as many loads in flight to hide the latency.
static constexpr int kFastMaxRegItems = 8;

// The reduction path hands the threshold bin's elements to a pairwise ranking,
// which is O(n^2), so it gives up above this many of them and lets the
// multi-step path take the row. The threshold bin holds 44 elements on average
// and 93 at worst over the S=3500 causal triangle; it only overflows this when
// the row's logits are clustered far more tightly than that.
static constexpr int kFastMaxCandidates = 128;

// Above this the accepted and candidate counters no longer share one int in the
// scan that allocates output slots.
static constexpr int kFastMaxTopK = 32768;

// Rows at least this long may hand the pairwise ranking up to half as many
// candidates as the histogram has slots, rather than kFastMaxCandidates, and
// keep them in the histogram: giving up costs them two more reads of a row
// that no longer fits the last level cache, which is more than ranking the
// larger set.
static constexpr int kLongRowMinLen = 16384;

// Rows at least this long read the row as 16-byte vectors.
static constexpr int kVecRowMinLen = 262144;
// The longest row whose pass two reserves candidates once per chunk.
static constexpr int kCandBatchMaxRowLen = 131072;

// The structure to store the final items (for the final pass).
struct FinalItems {
  // Shared memory to store the indices for the final pass.
  int indices[kNumFinalItems];
  // Shared memory to store the logits for the final pass.
  float logits[kNumFinalItems];
};

// The reduce path keeps this many copies of every bin, interleaved, and lane l
// of a subgroup counts into copy l % kHistCopies.
static constexpr int kHistCopies = 4;

// What a row long enough for the large candidate buffer keeps instead. The 32
// lanes of a subgroup then share kLongHistCopies counters per bin rather than
// kHistCopies, so a row whose elements crowd into one fp16 bin serialises half
// as many same-address atomics: Intel Xe has no same-address coalescing for
// subgroup atomics, and on logits quantised to a few fp16 bins (26k of a 49152
// row in one bin) pass one is where the time goes. A shorter row keeps
// kHistCopies, so that it does not pay the wider clear and scan of the
// histogram over a row that is too short to hide them.
#ifndef DLGW_LONG_HIST_COPIES
  #define DLGW_LONG_HIST_COPIES 7
#endif
static constexpr int kLongHistCopies = DLGW_LONG_HIST_COPIES;

struct Histogram {
  // 16-byte aligned so the reduce path can sweep a thread's bins as one vector.
  alignas(16) int data[kNumBins * kLongHistCopies];
  int tmp[SUBGROUP_SIZE + 1];  // used in prefix sum computation
};

// The histogram and the final-pass items used to share one allocation through
// a union. They cannot any more: processBins() now takes an output slot by
// incrementing its own bin's prefix sum, which lives in histo.data, in the same
// loop that fills items. That costs kNumBins * 4 = 8KB more SLM per
// work-group. On this device the work-group limit at 512 threads is thread
// capacity, not the 384KB SLM pool, so it is free.
struct SmemFinal {
  FinalItems items;
  Histogram histo;
};

// Struct to hold all static sized shared memory objects
struct SharedStates {
  // Shared memory to compute the block sort.
  SmemFinal smemFinal;
  // One slot per row of a group, so the rows sharing a work-group in the
  // grouped kernel each get their own. Slot 0 is the one every other path uses.
  // Shared memory to store the threshold bin.
  int smemThresholdBinIdx[kRowsPerGroup];
  // Shared memory counter to register the candidates for the final phase.
  int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  int smemFinalBinSize[kRowsPerGroup];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  int smemFoundTopKValues[kRowsPerGroup];
  // Set by any row of a group that gave up, so the whole work-group can agree
  // to rerun those rows one at a time on the multi-step path.
  int smemGroupBail[1];
  // Output slot counter for the elements processRefineStep() accepts on their
  // fp16 bin alone.
  int smemRefineDstIdx[1];
};

constexpr size_t alignUp(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

constexpr size_t sharedStatesOffset(size_t topK, bool withLogits = false) {
  size_t offset = topK * sizeof(int32_t);
  if (withLogits) {
    offset += topK * sizeof(float);
  }
  return alignUp(offset, alignof(SharedStates));
}

constexpr size_t sharedMemorySize(size_t topK, bool withLogits = false) {
  return sharedStatesOffset(topK, withLogits) + sizeof(SharedStates);
}

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename T>
T atomicAdd(T* address, T val) {
  sycl::atomic_ref<
      T,
      sycl::memory_order::relaxed,
      sycl::memory_scope::device,
      sycl::access::address_space::local_space>
      atom_val(*address);
  return atom_val.fetch_add(val);
}

inline auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

inline auto convert_to_uint16(sycl::half hx) -> uint16_t {
  uint16_t bits = sycl::bit_cast<uint16_t>(hx);
  return (bits & 0x8000) ? bits : ~bits & 0x7fff;
}

template <int step>
static inline uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    sycl::half hx = static_cast<sycl::half>(x);
    uint16_t bits = convert_to_uint16(hx);
    // the highest 11 bits, so total 2048 bins
    return bits >> 5;
  } else {
    uint32_t bits = convert_to_uint32(x);

    if constexpr (step == 1) {
      // the highest 11 bits, so total 2048 bins
      return bits >> 21;
    } else if constexpr (step == 2) {
      // the middle 11 bits, so total 2048 bins
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      // the lowest 10 bits, so total 1024 bins
      return bits & 0x3ff;
    }
  }
}

// Not recursive: a SYCL kernel may not call a recursive function, even one that
// only ever runs at compile time.
static constexpr int constexprLog2(int n) {
  int log = 0;
  while (n > 1) {
    n /= 2;
    ++log;
  }
  return log;
}

// The bin index for the reduction path below. Same monotone half key as
// extractBinIdx<0>(), but only the highest constexprLog2(kNumFastBins) bits, so
// that one bin can be owned by one thread.
template <int kNumFastBins>
static inline uint32_t extractFastBinIdx(float x) {
  static_assert((kNumFastBins & (kNumFastBins - 1)) == 0, "power of two");
  static_assert(kNumFastBins <= (1 << 16));
  sycl::half hx = static_cast<sycl::half>(x);
  return convert_to_uint16(hx) >> (16 - constexprLog2(kNumFastBins));
}

// The kBinsPerThread bins a thread owns are consecutive, so lane l of a
// subgroup addresses dword kBinsPerThread * l and all 32 lanes of a plain
// scalar sweep land on one bank in kBinsPerThread rounds. These move the whole
// set as one 16-byte message instead. The reduce path sweeps its bins three
// times, to clear them, to read the counts and to write the prefix sums back,
// and that traffic was 14.3 us of 74.5 at topK=128.
template <int kN>
static inline void binsLoad(const int* base, int (&out)[kN]) {
  if constexpr (kN % 4 == 0) {
#pragma unroll
    for (int j = 0; j < kN; j += 4) {
      const sycl::vec<int, 4> v =
          *reinterpret_cast<const sycl::vec<int, 4>*>(base + j);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out[j + i] = v[i];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      out[i] = base[i];
    }
  }
}

template <int kN>
static inline void binsStore(int* base, const int (&in)[kN]) {
  if constexpr (kN % 4 == 0) {
#pragma unroll
    for (int j = 0; j < kN; j += 4) {
      sycl::vec<int, 4> v;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] = in[j + i];
      }
      *reinterpret_cast<sycl::vec<int, 4>*>(base + j) = v;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      base[i] = in[i];
    }
  }
}

// Exclusive scan of one value per thread over the whole work-group: a subgroup
// ladder, then one subgroup scans the sg_num subgroup totals. Two barriers,
// ~12 shuffles, no atomics. tmp needs SUBGROUP_SIZE + 1 ints of SLM; the last
// slot carries the block total, which every thread needs to keep a running
// output cursor without a further reduction.
//
// Trading the second barrier for a repeat of the small scan in every subgroup
// was measured and lost 15 us of 71 on the S=3500 triangle: sycl group scans
// and a broadcast from a runtime lane cost more than the barrier they save.
// kThreadsPerRow makes the scan segmented: the carry stops at every multiple of
// kThreadsPerRow, so one call scans kNumThreadsPerBlock / kThreadsPerRow
// independent rows at once and they share both barriers. sg_id is the block
// wide subgroup index, and total comes back as the caller's own row's total.
template <int kNumThreadsPerBlock, int kThreadsPerRow = kNumThreadsPerBlock>
static inline int blockExclusiveScan(
    const sycl::group<3>& group,
    const sycl::sub_group& sg,
    int lane_id,
    int sg_id,
    int* tmp,
    int val,
    int& total) {
  constexpr int sg_num = kNumThreadsPerBlock / SUBGROUP_SIZE;
  constexpr int sg_per_row = kThreadsPerRow / SUBGROUP_SIZE;
  static_assert(
      sg_num <= SUBGROUP_SIZE, "the totals are scanned by one ladder");
  static_assert(sg_per_row >= 1, "a row keeps whole subgroups");
  static_assert(sg_num % sg_per_row == 0, "the rows split the block evenly");

  int inclusive = val;
  for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
    int other = sycl::shift_group_right(sg, inclusive, offset);
    if (lane_id >= offset) {
      inclusive += other;
    }
  }
  if (lane_id == SUBGROUP_SIZE - 1) {
    tmp[sg_id] = inclusive;
  }
  sycl::group_barrier(group);

  if (sg_id == 0) {
    int v = lane_id < sg_num ? tmp[lane_id] : 0;
    int inc = v;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inc, offset);
      if (lane_id >= offset &&
          lane_id / sg_per_row == (lane_id - offset) / sg_per_row) {
        inc += other;
      }
    }
    if (lane_id < sg_num) {
      tmp[lane_id] = inc - v;  // exclusive within the row
      if (lane_id % sg_per_row == sg_per_row - 1) {
        tmp[sg_num + lane_id / sg_per_row] = inc;  // that row's total
      }
    }
  }
  sycl::group_barrier(group);

  total = tmp[sg_num + sg_id / sg_per_row];
  return tmp[sg_id] + inclusive - val;
}

template <int shift>
static inline bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = convert_to_uint32(x);
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename idxT, typename Func>
void vectorized_process(
    size_t thread_rank, size_t num_threads, const T* in, idxT len, Func f) {
  using WideT = vec4_t<float>;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    // in this path, no vectorized load is performed, just process as normal
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

    // This union is to allow to conveniently access individual elements after
    // vectorized load
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(SUBGROUP_SIZE >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, SUBGROUP_SIZE > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // SUBGROUP_SIZE no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <
    int step,
    int kNumThreadsPerBlock,
    int kNumBins,
    int kNumFinalItems,
    bool multipleBlocksPerRow,
    bool mergeBlocks,
    typename SmemFinalType,
    typename SmemOutputType>
bool processHistogramStep(
    const sycl::nd_item<3>& item,
    const int* indices,
    const float* logits,
    int rowEnd,
    uint32_t& logitPattern,
    int& thresholdBinIdx,
    SmemOutputType* smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    SmemFinalType& smemFinal,
    int stride1,
    int rowStart,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);
  auto lane_id = threadIdx_x % SUBGROUP_SIZE;
  auto sg_id = threadIdx_x / SUBGROUP_SIZE;
  auto sg_num = (kNumThreadsPerBlock + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  // Clear the histogram.
  // The histogram size is [kNumBins], and the threads access it in this layout:
#pragma unroll
  for (int idx = threadIdx_x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  sycl::group_barrier(group);

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  // Make sure the histogram is ready.
  sycl::group_barrier(group);

  // How many top-k values the previous steps already committed. It is the base
  // of this step's prefix sums and the first free slot in smemOutput.
  //
  // Step 1 is the exception and must start from zero. Its patternShift is 0, so
  // isPartialMatch() is vacuously true and the step-1 histogram counts the
  // whole row again, not just the elements of step 0's threshold bin. Step 1
  // therefore re-derives the accepted set from scratch, and adding step 0's
  // count on top of a histogram that already contains those elements counts
  // them twice: the threshold bin comes out too early and every element step 0
  // accepted is written to a second slot as well. probe_multistep.py shows the
  // effect (2048 - F unique indices out of topK), and with F large enough the
  // duplicate writes run past smemOutput's topK slots into the shared state
  // behind it. The repo's own worst case, "10LSBits", pins every logit to the
  // same top 22 bits, which forces F == 0 at every step and hides this.
  int lastValue = step == 1 ? 0 : smemFoundTopKValues[0];

  // One block-wide exclusive scan over all kNumBins bins, kBinsPerThread
  // consecutive bins per thread.
  //
  // This replaces a loop of kNumBins / kNumThreadsPerBlock rounds that re-read
  // and re-wrote shared memory every round, serialised a sg_num-element scan of
  // the subgroup totals on lane 0 while the other threads waited at a barrier,
  // and needed a neighbour lookup (plus another barrier) to see the next bin's
  // prefix sum. It gave up an early exit once the threshold bin was found, but
  // the threshold bin is only in an early round for small topK, and the fixed
  // cost of this loop is what dominates the kernel at DSA row lengths: for
  // 3500-element rows it is ~60% of the runtime.
  constexpr int kBinsPerThread = kNumBins / kNumThreadsPerBlock;
  static_assert(kBinsPerThread * kNumThreadsPerBlock == kNumBins);
  // The scan of the subgroup totals below is done by one subgroup in a single
  // ladder, so there must be no more subgroups than lanes.
  static_assert(kNumThreadsPerBlock <= SUBGROUP_SIZE * SUBGROUP_SIZE);
  const int binBase = threadIdx_x * kBinsPerThread;

  // The counts are read twice from SLM rather than held in a register array:
  // keeping kBinsPerThread of them live across the scan pushed the kernel from
  // 64 to 96 GRF per thread, and the occupancy that costs is worth more than
  // the re-read (each thread only ever touches its own bins, so the second read
  // needs no barrier).
  int threadSum = 0;
  for (int j = 0; j < kBinsPerThread; ++j) {
    threadSum += smemFinal.histo.data[binBase + j];
  }

  // Subgroup-level inclusive scan of the per-thread sums.
  int sgInclusive = threadSum;
  for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
    int val = sycl::shift_group_right(sg, sgInclusive, offset);
    if (lane_id >= offset) {
      sgInclusive += val;
    }
  }
  if (lane_id == SUBGROUP_SIZE - 1) {
    smemFinal.histo.tmp[sg_id] = sgInclusive;
  }
  sycl::group_barrier(group);

  // Scan the subgroup totals with the whole first subgroup rather than walking
  // them one at a time on a single lane.
  if (sg_id == 0) {
    int val = lane_id < sg_num ? smemFinal.histo.tmp[lane_id] : 0;
    int inclusive = val;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inclusive, offset);
      if (lane_id >= offset) {
        inclusive += other;
      }
    }
    if (lane_id < sg_num) {
      smemFinal.histo.tmp[lane_id] = inclusive - val;  // exclusive
    }
  }
  sycl::group_barrier(group);

  // Write the prefix sums back and pick the threshold bin. Exactly one bin
  // satisfies prefix < topK <= prefix + count (an empty bin cannot, since its
  // two bounds are equal), so the owning thread can write the shared state with
  // no reduction and no neighbour lookup. smemFoundTopKValues is set here
  // instead of being accumulated by atomics in processBins(): its value is the
  // threshold bin's prefix sum by construction. The read of that same location
  // above is separated from this write by the two barriers of the scan.
  int prefixSum =
      lastValue + smemFinal.histo.tmp[sg_id] + (sgInclusive - threadSum);
  for (int j = 0; j < kBinsPerThread; ++j) {
    const int binCount = smemFinal.histo.data[binBase + j];
    smemFinal.histo.data[binBase + j] = prefixSum;
    const int nextPrefixSum = prefixSum + binCount;
    if (prefixSum < topK && nextPrefixSum >= topK) {
      smemThresholdBinIdx[0] = binBase + j;
      smemFinalBinSize[0] = binCount;
      smemFoundTopKValues[0] = prefixSum;
    }
    prefixSum = nextPrefixSum;
  }

  // Make sure the prefix sums and the threshold bin are in shared memory.
  sycl::group_barrier(group);

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection. Its slot comes from its
        // own bin's prefix sum, so the atomics spread over the bins below the
        // threshold instead of serialising on one counter, which every one of
        // the topK accepted elements in the row used to hit. Step 3 below
        // already allocated its slots this way. Side effect: the output ends up
        // grouped by bin, so it is close to sorted by logit.
        int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        processBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  sycl::group_barrier(group);

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

// vectorized_process() with kDepth vector loads in flight per thread instead of
// one. Its loop body waits on its own load, and a pass that does little per
// element is then bound by the load latency: with one vector per thread a Xe
// core has 32KB in flight, about two thirds of what the bandwidth needs.
template <int kDepth, typename Func>
static inline void vectorizedProcessDeep(
    int thread_rank,
    int num_threads,
    const float* in,
    int len,
    int rotate,
    Func f) {
  const int misalign = reinterpret_cast<uintptr_t>(in) % sizeof(sycl::float4);
  const int skip = sycl::min(
      misalign
          ? static_cast<int>((sizeof(sycl::float4) - misalign) / sizeof(float))
          : 0,
      len);
  const sycl::float4* in4 = reinterpret_cast<const sycl::float4*>(in + skip);
  const int len4 = (len - skip) / 4;
  const int blockLen = kDepth * num_threads;
  const int numBlocks = (len4 + blockLen - 1) / blockLen;
  const int first = numBlocks > 0 ? rotate % numBlocks : 0;
  for (int b = 0; b < numBlocks; ++b) {
    int blk = b + first;
    blk = blk < numBlocks ? blk : blk - numBlocks;
    const int i = blk * blockLen + thread_rank;
    sycl::float4 w[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (i + d * num_threads < len4) {
        w[d] = in4[i + d * num_threads];
      }
    }
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      if (i + d * num_threads < len4) {
        const int real = skip + (i + d * num_threads) * 4;
        f(w[d].x(), real);
        f(w[d].y(), real + 1);
        f(w[d].z(), real + 2);
        f(w[d].w(), real + 3);
      }
    }
  }
  if (thread_rank < skip) {
    f(in[thread_rank], thread_rank);
  }
  const int remain = skip + len4 * 4 + thread_rank;
  if (remain < len) {
    f(in[remain], remain);
  }
}

// Slack, in fp32 key units, left below the first processRefineStep() bucket and
// above the last: one fp16 ulp, for the values that round onto the bin.
static constexpr uint32_t kRefineSlack = 1u << 13;

// Width, as a shift of the fp32 key, of one processRefineStep() bucket. An fp16
// bin of 2^kFastShift halves of one exponent is 2^(kFastShift + 13) consecutive
// fp32 keys, plus the slack at either end, so the pass reaches
// (2^(kFastShift + 13) + 2 * kRefineSlack) >> refineShift buckets: 1088 at
// kLongRefineShift + 2 and 4352 at kLongRefineShift, of the
// kNumBins * kLongHistCopies ints the histogram it borrows holds. A long row
// takes the finer one, so that a row packed into one fp16 bin splits fine
// enough for its threshold bucket to fit the final items; a short row takes the
// coarse one, because clearing the difference costs more than the extra reads
// it saves. The shift is a plain argument, not a template one, so that the two
// choices share one copy of the pass in the kernel.
static constexpr int kLongRefineShift = 6;

// Buckets processRefineStep() splits its fp16 bin into at this shift, rounded
// up to what its scan splits evenly over the threads. Clearing and scanning
// only these, rather than the whole borrowed histogram, is what keeps the finer
// resolution affordable.
template <int kNumThreadsPerBlock, int kNumBins>
static constexpr int refineNumBuckets(int refineShift) {
  constexpr int kFastShift = 16 - constexprLog2(kNumBins);
  const int reached =
      ((1 << (kFastShift + 13)) + 2 * static_cast<int>(kRefineSlack)) >>
      refineShift;
  return ((reached + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock) *
         kNumThreadsPerBlock;
}

// Whether processRefineStep() can split this fp16 bin. The bin shares the top
// 11 bits of the fp16 key, and with them one exponent. Exponent 0 (zero and
// subnormals) and 31 (inf and NaN) are excluded: everything below 2^-24 rounds
// to the zero bin and everything above 65520 to the inf one, so those bins
// cover far more than 2^18 fp32 keys.
template <int kNumFastBins>
static inline bool fastBinIsRefinable(int bin) {
  const uint32_t key16 = static_cast<uint32_t>(bin)
                         << (16 - constexprLog2(kNumFastBins));
  const uint32_t bits16 = (key16 & 0x8000) ? key16 : (~key16 & 0x7fff);
  const uint32_t exp16 = (bits16 >> 10) & 0x1f;
  return exp16 != 0 && exp16 != 0x1f;
}

// The fp32 key of the first bucket of processRefineStep() for this fp16 bin:
// the key of the bin's largest value (its smallest key, since keys descend with
// the value), minus the slack for values that round onto it.
template <int kNumFastBins>
static inline uint32_t refineBase(int bin) {
  const uint32_t key16 = static_cast<uint32_t>(bin)
                         << (16 - constexprLog2(kNumFastBins));
  const uint16_t bits16 =
      static_cast<uint16_t>((key16 & 0x8000) ? key16 : (~key16 & 0x7fff));
  const float first = static_cast<float>(sycl::bit_cast<sycl::half>(bits16));
  return convert_to_uint32(first) - kRefineSlack;
}

// Takes over from the reduction path when it gave up on a threshold bin that
// was too large, and splits that same bin again on its fp32 bits.
//
// This replaces steps 0 and 1 of the chain above for such rows, and both were
// wasted work there. Step 0 rebuilds the reduction path's histogram bit for bit
// (the same top 11 bits of the fp16 key). Step 1 then starts over on the top 11
// bits of the fp32 key, which are sign, 8 exponent bits and only 2 mantissa
// bits, so its bins are 8 fp16 bins wide and its threshold bin is about 8 times
// too large again. At s_kv=49152 the fp16 threshold bin holds ~910 elements, so
// neither fits and the row goes on to step 2: seven reads of the row where
// CUDA, whose final buffer is 2048 slots, makes two.
//
// Instead this keeps what the reduction path already established, the threshold
// fp16 bin fastBin and the `accepted` elements in the bins before it, and
// histograms only that bin's elements, 256 fp32 keys per bucket. Its ~910
// elements then spread over ~1000 buckets, so the new threshold bucket holds a
// handful and the final rank is over those rather than over ~910. Two reads of
// the row, histogram and emit, after the reduction path's one.
//
// Returns false, having written nothing but shared state, if even the threshold
// bucket does not fit kNumFinalItems (a row of mostly exact duplicates). The
// caller then finishes the row with processTieSelect().
template <
    int kNumThreadsPerBlock,
    int kNumBins,
    int kNumFinalItems,
    typename SmemFinalType>
static bool processRefineStep(
    const sycl::nd_item<3>& item,
    const float* logits,
    int rowStart,
    int rowEnd,
    int fastBin,
    int refineShift,
    int accepted,
    int32_t* smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    int* smemRefineDstIdx,
    SmemFinalType& smemFinal,
    int stride1,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);
  auto lane_id = threadIdx_x % SUBGROUP_SIZE;
  auto sg_id = threadIdx_x / SUBGROUP_SIZE;
  auto sg_num = (kNumThreadsPerBlock + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  // Buckets this pass splits fastBin into, out of the ints of the borrowed
  // histogram, which is kLongHistCopies times what the fp16 pass needed. A row
  // packed into one fp16 bin then splits fine enough for its threshold bucket
  // to fit the final items, instead of giving up here and paying the whole
  // fp32 chain (six more reads of the row).
  const int numBuckets =
      refineNumBuckets<kNumThreadsPerBlock, kNumBins>(refineShift);
  static_assert(
      refineNumBuckets<kNumThreadsPerBlock, kNumBins>(kLongRefineShift) <=
          kNumBins * kLongHistCopies,
      "the buckets of the finest shift fit the histogram this borrows");
  const uint32_t base = refineBase<kNumBins>(fastBin);
  // Only called on elements of fastBin, which lie in the first numBuckets
  // buckets. The clamp is there for memory safety only.
  auto bucketOf = [&](float logit) {
    const uint32_t b = (convert_to_uint32(logit) - base) >> refineShift;
    return static_cast<int>(
        sycl::min(b, static_cast<uint32_t>(numBuckets - 1)));
  };

  for (int idx = threadIdx_x; idx < numBuckets; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }
  if (threadIdx_x == 0) {
    smemRefineDstIdx[0] = 0;
  }
  sycl::group_barrier(group);

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (static_cast<int>(extractFastBinIdx<kNumBins>(logit)) == fastBin) {
      atomicAdd(&smemFinal.histo.data[bucketOf(logit)], 1);
    }
  };
  if (stride1 == 1) {
    vectorizedProcessDeep<4>(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        static_cast<int>(item.get_group_linear_id()),
        distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      distributeToBins(logits[idx * stride1], idx);
    }
  }
  sycl::group_barrier(group);

  // The same block-wide exclusive scan as processHistogramStep(), based at the
  // count of elements the fp16 bins before fastBin already accepted.
  // refineNumBuckets() rounds up to a multiple of the threads, so the buckets
  // split evenly.
  const int binsPerThread = numBuckets / kNumThreadsPerBlock;
  static_assert(kNumThreadsPerBlock <= SUBGROUP_SIZE * SUBGROUP_SIZE);
  const int binBase = threadIdx_x * binsPerThread;

  int threadSum = 0;
  for (int j = 0; j < binsPerThread; ++j) {
    threadSum += smemFinal.histo.data[binBase + j];
  }
  int sgInclusive = threadSum;
  for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
    int val = sycl::shift_group_right(sg, sgInclusive, offset);
    if (lane_id >= offset) {
      sgInclusive += val;
    }
  }
  if (lane_id == SUBGROUP_SIZE - 1) {
    smemFinal.histo.tmp[sg_id] = sgInclusive;
  }
  sycl::group_barrier(group);
  if (sg_id == 0) {
    int val = lane_id < sg_num ? smemFinal.histo.tmp[lane_id] : 0;
    int inclusive = val;
    for (int offset = 1; offset < SUBGROUP_SIZE; offset *= 2) {
      int other = sycl::shift_group_right(sg, inclusive, offset);
      if (lane_id >= offset) {
        inclusive += other;
      }
    }
    if (lane_id < sg_num) {
      smemFinal.histo.tmp[lane_id] = inclusive - val;
    }
  }
  sycl::group_barrier(group);

  // accepted < topK <= accepted + |fastBin|, so exactly one bucket brackets
  // topK.
  int prefixSum =
      accepted + smemFinal.histo.tmp[sg_id] + (sgInclusive - threadSum);
  for (int j = 0; j < binsPerThread; ++j) {
    const int binCount = smemFinal.histo.data[binBase + j];
    smemFinal.histo.data[binBase + j] = prefixSum;
    const int nextPrefixSum = prefixSum + binCount;
    if (prefixSum < topK && nextPrefixSum >= topK) {
      smemThresholdBinIdx[0] = binBase + j;
      smemFinalBinSize[0] = binCount;
      smemFoundTopKValues[0] = prefixSum;
    }
    prefixSum = nextPrefixSum;
  }
  sycl::group_barrier(group);

  if (smemFinalBinSize[0] > kNumFinalItems) {
    return false;
  }
  const int thresholdBucket = smemThresholdBinIdx[0];

  // Elements of the bins before fastBin take the first `accepted` slots, in no
  // particular order, from one counter: at most topK atomics per row, against
  // the ~s_kv elements this pass reads. Elements of fastBin take their bucket's
  // slots, and those of the threshold bucket go to the final items.
  auto processBins = [&](float logit, int idx) {
    const int bin = static_cast<int>(extractFastBinIdx<kNumBins>(logit));
    if (bin < fastBin) {
      smemOutput[atomicAdd(&smemRefineDstIdx[0], 1)] = idx;
    } else if (bin == fastBin) {
      const int bucket = bucketOf(logit);
      if (bucket < thresholdBucket) {
        smemOutput[atomicAdd(&smemFinal.histo.data[bucket], 1)] = idx;
      } else if (bucket == thresholdBucket) {
        const int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = idx;
      }
    }
  };
  if (stride1 == 1) {
    vectorizedProcessDeep<4>(
        threadIdx_x,
        kNumThreadsPerBlock,
        logits + rowStart,
        rowEnd - rowStart,
        static_cast<int>(item.get_group_linear_id()),
        processBins);
  } else {
    for (int idx = rowStart + threadIdx_x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      processBins(logits[idx * stride1], idx);
    }
  }
  sycl::group_barrier(group);
  return true;
}

// Digit width of processTieSelect(): 4096 buckets, 8 per thread at 512
// threads, in the ints of the histogram it borrows.
static constexpr int kTieSelectDigitBits = 12;

// Takes over a single-block row whose fp16 threshold bin neither the reduction
// path nor processRefineStep() could split: exact duplicates, or values that
// all round into the zero (or inf) fp16 bin, which processRefineStep() cannot
// split at all. Those rows used to go through steps 1 to 3, seven or more reads
// of the row, each of whose histogram passes funnels the whole tie into one
// SLM address one atomic at a time. At s_q=2730, s_kv=49152, topK=2048 a
// prefill with every logit 0.0 (dummy weights) took 3.0 ms against 0.93 ms on
// spread logits; this takes it to 1.33 ms.
//
// Instead this ranks only fastBin's elements, on their fp32 key relative to the
// smallest one, and only on as many bits as their spread needs:
//
//   * One read writes the elements of the bins before fastBin, fills the rest
//     of the output with fastBin's elements as if they were all tied, and takes
//     their minimum and maximum key. If those are equal, which is the case for
//     a row of ties, that read was the answer.
//   * Otherwise one histogram read per kTieSelectDigitBits digit of the
//     spread, most significant first, each over the elements that share the
//     digits chosen so far, stopping once the chosen bucket is taken whole or
//     the remaining keys are exact ties.
//   * Then one emit read rewrites fastBin's share of the output: the elements
//     ranked strictly above the final bucket and `need` of those in it.
//
// Histogram increments and slot claims are aggregated per subgroup: the lanes
// that share the lowest active lane's bucket add as one, so a tie costs one
// atomic per subgroup instead of 32 to the same address.
template <int kNumThreadsPerBlock, int kNumBins>
static void processTieSelect(
    const sycl::nd_item<3>& item,
    const float* rowBase,
    int rowLen,
    int stride1,
    int fastBin,
    int accepted,
    int* outIndices,
    int* hist,
    int* scanTmp,
    int* selVars,
    int topK) {
  constexpr int kItems = 4;
  constexpr int kNumDigitBuckets = 1 << kTieSelectDigitBits;
  constexpr int kBucketsPerThread = kNumDigitBuckets / kNumThreadsPerBlock;
  static_assert(
      kNumDigitBuckets % kNumThreadsPerBlock == 0,
      "the buckets split evenly over the threads");
  static_assert(
      kNumDigitBuckets <= kNumBins * kLongHistCopies,
      "the buckets fit the histogram this borrows");

  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  const int tid = static_cast<int>(item.get_local_id(0));
  const int lane = tid % SUBGROUP_SIZE;
  const int sgId = tid / SUBGROUP_SIZE;

  int* inCursor = selVars + 3;
  int* tieCursor = selVars + 4;
  if (tid == 0) {
    *inCursor = 0;
    *tieCursor = 0;
  }
  sycl::group_barrier(group);

  // Every thread runs the same number of iterations, so the subgroup
  // operations inside fn see all of their lanes. Element k of a thread's chunk
  // is kNumThreadsPerBlock apart from element k + 1, so each load is one
  // coalesced run across the subgroup.
  constexpr int kPerIteration = kNumThreadsPerBlock * kItems;
  const int numIterations = (rowLen + kPerIteration - 1) / kPerIteration;
  auto walk = [&](auto&& fn) {
    for (int it = 0; it < numIterations; ++it) {
      const int base = it * kPerIteration + tid;
      float v[kItems];
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreadsPerBlock;
        v[k] = j < rowLen ? rowBase[static_cast<int64_t>(j) * stride1] : 0.f;
      }
#pragma unroll
      for (int k = 0; k < kItems; ++k) {
        const int j = base + k * kNumThreadsPerBlock;
        fn(v[k], j, j < rowLen);
      }
    }
  };
  auto inFastBin = [&](float x) {
    return static_cast<int>(extractFastBinIdx<kNumBins>(x)) == fastBin;
  };
  // Subgroup-uniform: counters[b] += 1 for every active lane.
  auto aggregatedInc = [&](int* counters, bool active, int b) {
    if (!sycl::any_of_group(sg, active)) {
      return;
    }
    const int leader = sycl::reduce_over_group(
        sg, active ? lane : SUBGROUP_SIZE, sycl::minimum<int>());
    const int leaderBucket = sycl::select_from_group(sg, b, leader);
    const bool match = active && b == leaderBucket;
    const int n = sycl::reduce_over_group(sg, match ? 1 : 0, sycl::plus<int>());
    if (lane == leader) {
      atomicAdd(&counters[leaderBucket], n);
    } else if (active && !match) {
      atomicAdd(&counters[b], 1);
    }
  };
  // Subgroup-uniform: a slot from *cursor for every active lane. end is the
  // cursor after this subgroup's claim, the same in every lane.
  auto claim = [&](int* cursor, bool active, int& end) {
    const int flag = active ? 1 : 0;
    const int n = sycl::reduce_over_group(sg, flag, sycl::plus<int>());
    if (n == 0) {
      return 0;
    }
    const int rank =
        sycl::exclusive_scan_over_group(sg, flag, sycl::plus<int>());
    int first = 0;
    if (lane == 0) {
      first = atomicAdd(cursor, n);
    }
    first = sycl::group_broadcast(sg, first, 0);
    end = first + n;
    return first + rank;
  };

  sycl::group_barrier(group);

  // Pass one: the elements of the bins before fastBin take slots [0, accepted),
  // and those of fastBin fill [accepted, topK) as if they were all tied, while
  // their minimum and maximum key are taken. If they are tied, which is what
  // brings a row here, that is already the answer; otherwise the passes below
  // rank them and the final one rewrites [accepted, topK).
  const int need0 = topK - accepted;
  uint32_t lo = 0xffffffffu;
  uint32_t hi = 0;
  bool tiesOpen = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid) {
      const int bin = static_cast<int>(extractFastBinIdx<kNumBins>(x));
      in = bin < fastBin;
      if (bin == fastBin) {
        const uint32_t key = convert_to_uint32(x);
        lo = sycl::min(lo, key);
        hi = sycl::max(hi, key);
        tie = tiesOpen;
      }
    }
    int end = 0;
    const int inSlot = claim(inCursor, in, end);
    if (in) {
      outIndices[inSlot] = j;
    }
    end = 0;
    const int tieSlot = claim(tieCursor, tie, end);
    if (tie && tieSlot < need0) {
      outIndices[accepted + tieSlot] = j;
    }
    tiesOpen = tiesOpen && end < need0;
  });
  const uint32_t minKey =
      sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
  const uint32_t range =
      sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) - minKey;
  if (range == 0) {
    sycl::group_barrier(group);
    return;
  }
  // Every claim of pass one is done.
  sycl::group_barrier(group);
  if (tid == 0) {
    *inCursor = accepted;
    *tieCursor = 0;
  }

  // The fastBin elements still in play are those whose key offset, shifted
  // right by `shift`, equals `prefix`; `need` of them complete the top-k. The
  // barrier of the first histogram pass publishes the reset cursors.
  int need = need0;
  int shift = 0;
  uint32_t prefix = 0;
  {
    const int rangeBits = 32 - static_cast<int>(sycl::clz(range));
    int hiShift = 32;
    shift = sycl::max(0, rangeBits - kTieSelectDigitBits);
    while (true) {
      int* myBuckets = hist + tid * kBucketsPerThread;
      int counts[kBucketsPerThread] = {};
      binsStore(myBuckets, counts);
      sycl::group_barrier(group);

      const uint32_t digitMask =
          hiShift == 32 ? static_cast<uint32_t>(kNumDigitBuckets - 1)
                        : (1u << (hiShift - shift)) - 1;
      walk([&](float x, int /* j */, bool valid) {
        bool active = valid && inFastBin(x);
        uint32_t off = 0;
        if (active) {
          off = convert_to_uint32(x) - minKey;
          active = hiShift == 32 || (off >> hiShift) == prefix;
        }
        aggregatedInc(
            hist, active, static_cast<int>((off >> shift) & digitMask));
      });
      sycl::group_barrier(group);

      binsLoad(myBuckets, counts);
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kBucketsPerThread; ++i) {
        sum += counts[i];
      }
      int total;
      int before = blockExclusiveScan<kNumThreadsPerBlock>(
          group, sg, lane, sgId, scanTmp, sum, total);
      // The candidates number at least need, so one bucket brackets it.
      if (before < need && before + sum >= need) {
#pragma unroll
        for (int i = 0; i < kBucketsPerThread; ++i) {
          if (before < need && before + counts[i] >= need) {
            selVars[0] = tid * kBucketsPerThread + i;
            selVars[1] = before;
            selVars[2] = counts[i];
          }
          before += counts[i];
        }
      }
      sycl::group_barrier(group);

      const uint32_t bucket = static_cast<uint32_t>(selVars[0]);
      prefix = hiShift == 32 ? bucket : (prefix << (hiShift - shift)) | bucket;
      need -= selVars[1];
      const int count = selVars[2];
      hiShift = shift;
      if (count == need || shift == 0) {
        break;
      }
      shift = sycl::max(0, shift - kTieSelectDigitBits);
      // selVars is rewritten by the next scan only after the barriers of its
      // histogram pass, which every thread reaches after the reads above.
    }
  }

  // The elements of fastBin below the final bucket take slots
  // [accepted, topK - need), and `need` of the final bucket's take the rest.
  // Once a subgroup sees the tie slots filled it stops claiming them.
  const int tieBase = topK - need;
  tiesOpen = true;
  walk([&](float x, int j, bool valid) {
    bool in = false;
    bool tie = false;
    if (valid && inFastBin(x)) {
      const uint32_t top = (convert_to_uint32(x) - minKey) >> shift;
      in = top < prefix;
      tie = tiesOpen && top == prefix;
    }
    int end = 0;
    const int inSlot = claim(inCursor, in, end);
    if (in) {
      outIndices[inSlot] = j;
    }
    end = 0;
    const int tieSlot = claim(tieCursor, tie, end);
    if (tie && tieSlot < need) {
      outIndices[tieBase + tieSlot] = j;
    }
    tiesOpen = tiesOpen && end < need;
  });
  sycl::group_barrier(group);
}

// One-pass, reduction-based selection. Returns false without touching global
// memory if the row is not one it can finish, and the caller falls back to the
// processHistogramStep() chain above.
//
// What it does differently from that chain, and why:
//
//   * The bins are split evenly over the threads, so clearing the histogram is
//     kBinsPerThread stores, the prefix sum is one value per thread plus a
//     local walk, and the threshold bin needs neither a search nor a broadcast:
//     the thread whose own prefix brackets topK is the one that owns it. 512
//     bins (exactly one per thread) were measured against 2048: the histogram
//     and the scan cost the same either way, but the threshold bin then holds
//     44 elements on average instead of 12 over the S=3500 causal triangle, and
//     the final rank below is quadratic in that count. So the wider histogram
//     wins, since everything after this point scales with that count and 44 is
//     still tiny.
//
//   * Nothing is staged in smemOutput. Slots come from one work-group scan of
//     the per-thread accepted counts, so the ~2048 atomicAdd()s that every
//     accepted element in the row used to aim at a single counter become zero
//     atomics, and the accepted indices go straight to global memory. The
//     output order is unspecified, so no ordering is lost.
//
//   * The row is read once. Each thread keeps its own elements in registers
//     between the histogram and the scatter (kRegResident), so the scatter
//     needs no second load. Rows too long for that re-read in chunks, which is
//     still the two passes the other path always does.
//
//   * The elements sharing the threshold bin are compacted by the same scan and
//     ranked pairwise, exactly as the final pass of the other path does, but
//     over the ~44 of them rather than up to kNumFinalItems.
// kThreadsPerRow is how many of the block's threads work on this row: the whole
// block when one row owns the work-group, or a slice of it when kRowsPerGroup
// rows share one. Everything below is row local except the two barriers of the
// scan and the barriers between stages, which stay block wide and are exactly
// what the rows of a group share. kNumFastBins, kMaxCandidates and the items
// buffer are the per row slices the caller hands over.
template <
    int kNumThreadsPerBlock,
    int kThreadsPerRow,
    int kNumFastBins,
    int kCopies,
    int kMaxRegItems,
    int kMaxCandidates,
    bool kRegResident>
static bool topKPerRowReduce(
    const sycl::nd_item<3>& item,
    const float* rowBase,
    int rowLen,
    int* outIndices,
    int* smemBins,
    int* smemTmp,
    int* itemIndices,
    float* itemLogits,
    int* smemThresholdBinIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    int* smemGroupBail,
    int stride1,
    int topK) {
  // True when this row shares its work-group with others, which changes only
  // how giving up works: see the bail-out below.
  constexpr bool kGrouped = kThreadsPerRow != kNumThreadsPerBlock;
  constexpr int kBinsPerThread = kNumFastBins / kThreadsPerRow;
  static_assert(
      kBinsPerThread * kThreadsPerRow == kNumFastBins,
      "the bins are split evenly over the threads");
  static_assert(
      kMaxCandidates <= kNumFinalItems, "shares the final-items slots");
  static_assert(
      kNumFastBins * kCopies <= kNumBins * kLongHistCopies,
      "the bins and their copies fit the histogram");
  static_assert(
      kThreadsPerRow % SUBGROUP_SIZE == 0, "a row keeps whole subgroups");

  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  // Row local thread id: the rows of a group take contiguous slices of the
  // block, so a subgroup never straddles two rows.
  const int threadIdx_x = item.get_local_id(0) % kThreadsPerRow;
  const int lane_id = threadIdx_x % SUBGROUP_SIZE;
  const int sg_id = threadIdx_x / SUBGROUP_SIZE;
  const int sg_id_block = item.get_local_id(0) / SUBGROUP_SIZE;
  const int binBase = threadIdx_x * kBinsPerThread;
  // Output cursor of the accepted elements, one per row of the group, in scan
  // scratch that blockExclusiveScan() leaves alone (it uses the first
  // kNumThreadsPerBlock / SUBGROUP_SIZE + kRowsPerGroup slots).
  static_assert(
      kNumThreadsPerBlock / SUBGROUP_SIZE + kRowsPerGroup <=
      SUBGROUP_SIZE - 2 * kRowsPerGroup);
  int* acceptCursor = smemTmp + SUBGROUP_SIZE - kRowsPerGroup +
                      item.get_local_id(0) / kThreadsPerRow;
  int* candCursor = acceptCursor - kRowsPerGroup;
  const int histCopy = lane_id % kCopies;

  constexpr int kChunk = kMaxRegItems * kThreadsPerRow;
  constexpr int kNumReg = kRegResident ? kMaxRegItems : 1;
  float regVal[kNumReg];
  // The bin of regVal[i], kept from pass one so pass two classifies with an
  // integer compare instead of extracting the bin from the float again.
  // Elements past the end of the row get a bin one past the last, which no
  // threshold bin can match, so they are rejected without a separate bounds
  // test.
  int regBin[kNumReg];

  // The element this thread keeps in register slot i. A vec4 layout
  // (four consecutive elements per thread, one 16-byte load) was measured here
  // too: it left the read time unchanged, so the row read is bandwidth bound
  // rather than request bound, and the plain strided order is kept because it
  // costs a few instructions less further down.
  auto regOffset = [&](int i) { return threadIdx_x + i * kThreadsPerRow; };

  // How many of this thread's kMaxRegItems slots hold an element of the row.
  // Every loop below stops there instead of running the full kMaxRegItems and
  // masking, because the slots are handed out with a stride of kThreadsPerRow:
  // the 32 lanes of a subgroup agree on this count to within one, so a short
  // row really does leave the loop early instead of paying for kMaxRegItems
  // predicated iterations. At topK=128 the full triangle spends 4096 slot
  // visits on rows of a few hundred elements, which was 26 of the 95 us in the
  // histogram alone.
  const int regItems =
      kRegResident
          ? (rowLen - threadIdx_x + kThreadsPerRow - 1) / kThreadsPerRow
          : kMaxRegItems;

  // The chunked passes read a unit-stride row as 16-byte vectors, two per
  // thread per chunk: the first `head` elements up to 16-byte alignment are
  // taken on their own. With only a few long rows in flight a work-group has
  // to keep its own reads streaming, and a vec4 load moves four times as much
  // per message as the strided scalar ones.
  static_assert(kRegResident || kMaxRegItems % 4 == 0);
  const bool vecLoads =
      !kRegResident && stride1 == 1 && rowLen >= kVecRowMinLen;
  const int head =
      vecLoads
          ? sycl::min(
                static_cast<int>(
                    (16 - reinterpret_cast<uintptr_t>(rowBase) % 16) % 16 / 4),
                rowLen)
          : 0;
  const int bodyLen = rowLen - head;
  // The chunked passes start each row at a chunk picked by its work-group.
  // Rows are laid out at a fixed pitch, often a power of two, and rows that
  // all start their read at the same offset then hit the same memory channels
  // at the same time: at 128 rows of 2^20 that doubled the time of a pass.
  const int numChunks = (bodyLen + kChunk - 1) / kChunk;
  const int firstChunk =
      static_cast<int>(item.get_group_linear_id() % numChunks);
  auto chunkAt = [&](int c) {
    const int k = c + firstChunk;
    return (k < numChunks ? k : k - numChunks) * kChunk;
  };
  auto loadChunk =
      [&](int chunk, float (&val)[kMaxRegItems], int (&off)[kMaxRegItems]) {
        if (vecLoads) {
          const float* body = rowBase + head;
#pragma unroll
          for (int j = 0; j < kMaxRegItems / 4; ++j) {
            const int o = chunk + (threadIdx_x + j * kThreadsPerRow) * 4;
            if (o + 3 < bodyLen) {
              const sycl::float4 w =
                  *reinterpret_cast<const sycl::float4*>(body + o);
              val[4 * j] = w.x();
              val[4 * j + 1] = w.y();
              val[4 * j + 2] = w.z();
              val[4 * j + 3] = w.w();
            } else {
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                val[4 * j + e] = o + e < bodyLen ? body[o + e] : 0.f;
              }
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              off[4 * j + e] = o + e < bodyLen ? head + o + e : rowLen;
            }
          }
        } else if (stride1 == 1) {
#pragma unroll
          for (int i = 0; i < kMaxRegItems; ++i) {
            off[i] = chunk + threadIdx_x + i * kThreadsPerRow;
            val[i] = off[i] < rowLen ? rowBase[off[i]] : 0.f;
            off[i] = off[i] < rowLen ? off[i] : rowLen;
          }
        } else {
#pragma unroll
          for (int i = 0; i < kMaxRegItems; ++i) {
            off[i] = chunk + threadIdx_x + i * kThreadsPerRow;
            val[i] = off[i] < rowLen ? rowBase[off[i] * stride1] : 0.f;
            off[i] = off[i] < rowLen ? off[i] : rowLen;
          }
        }
      };

  {
    int zero[kBinsPerThread * kCopies] = {};
    binsStore<kBinsPerThread * kCopies>(smemBins + binBase * kCopies, zero);
  }
  sycl::group_barrier(group);

  // Pass one over the row: the histogram. These atomics are spread over
  // kNumFastBins addresses, ~one collision per subgroup, and are not the
  // contended ones the scatter below removes.
  //
  // Reading the row before the barrier that publishes the cleared histogram, so
  // that the barrier waits behind loads already in flight, was measured and
  // lost 3 us of 71: the threads then reach their loads at different times and
  // the row reads stop arriving as one front.
  if constexpr (kRegResident) {
    // Both loops run over the whole register set before the next one starts.
    // Folding the atomicAdd() into the fill makes every iteration wait on its
    // own load, so a row costs one memory latency per element instead of one in
    // total; that alone was enough to make this path slower than the two-pass
    // one it replaces. Measured: 38.8 of the 96 us this kernel spent on the
    // S=3500 triangle went into the fill. Reading the row is the largest single
    // cost of this kernel, so the unit stride case gets its own loop. With a
    // runtime stride the address of lane l is base + l * stride1 * 4, which the
    // compiler can only lower to a per lane gather; with the stride known to be
    // one the 32 lanes of a load cover 128 consecutive bytes and it lowers to a
    // linear load instead.
    auto fillFrom = [&](auto row) {
#pragma unroll
      for (int i = 0; i < kMaxRegItems; ++i) {
        if (i >= regItems) {
          break;
        }
        regVal[i] = row[regOffset(i)];
      }
    };
    if (stride1 == 1) {
      fillFrom(rowBase);
    } else {
#pragma unroll
      for (int i = 0; i < kMaxRegItems; ++i) {
        if (i >= regItems) {
          break;
        }
        regVal[i] = rowBase[regOffset(i) * stride1];
      }
    }
#pragma unroll
    for (int i = 0; i < kMaxRegItems; ++i) {
      if (i >= regItems) {
        break;
      }
      regBin[i] = extractFastBinIdx<kNumFastBins>(regVal[i]);
      atomicAdd(&smemBins[regBin[i] * kCopies + histCopy], 1);
    }
  } else {
    // Same two-loop shape as the register-resident case above, one chunk at a
    // time: a chunk is staged in registers before any of its atomics, so a
    // chunk costs one memory latency instead of one per element. This path
    // takes the same care as the register-resident one for the same reason.
    // Walking the row with the atomic folded into the load, here and in pass
    // two, was measured on a variant that ran the whole S=3500 triangle through
    // this path and cost 4.6 us of 75.0 at topK=128, 9.6 of 92.5 at 512
    // and 24.1 of 99.5 at 2048.
    if (threadIdx_x < head) {
      atomicAdd(
          &smemBins
              [extractFastBinIdx<kNumFastBins>(rowBase[threadIdx_x]) * kCopies +
               histCopy],
          1);
    }
    // Clustered logits put most of a row into a few bins, and the lanes of
    // one atomic that hit the same address serialise even across the copies:
    // at 2730 rows of 49152 that doubled this pass. So each subgroup samples
    // one element per lane and picks the window of kHotBins consecutive bins
    // that covers most of the sample, and if that is most of it, counts the
    // elements of the window in registers, kHotBits per bin in one 64-bit
    // word, with one atomic per bin at the end of the row. The counts are
    // exact whatever the sample; it only decides which bins skip the atomics.
    constexpr int kHotBins = 4;
    constexpr int kHotBits = 16;
    constexpr int kHotMinLanes = 16;
    int hotLo = 0;
    bool hotLoop = false;
    {
      const int s = threadIdx_x < rowLen ? threadIdx_x : 0;
      const int sample = static_cast<int>(
          extractFastBinIdx<kNumFastBins>(rowBase[s * stride1]));
      const int lead = sycl::group_broadcast(sg, sample, 0);
      int best = -1;
#pragma unroll
      for (int w = -(kHotBins - 1); w <= kHotBins - 1; ++w) {
        const int lo = sycl::clamp(lead + w, 0, kNumFastBins - kHotBins);
        uint32_t mask;
        sycl::ext::oneapi::group_ballot(
            sg, static_cast<uint32_t>(sample - lo) < kHotBins)
            .extract_bits(mask);
        const int covered = static_cast<int>(sycl::popcount(mask));
        if (covered > best) {
          best = covered;
          hotLo = lo;
        }
      }
      // A lane counts at most rowLen / kThreadsPerRow elements per bin.
      hotLoop = best >= kHotMinLanes &&
                rowLen / kThreadsPerRow < (1 << kHotBits) - kMaxRegItems;
    }
    if (hotLoop) {
      uint64_t hotCount = 0;
      for (int c = 0; c < numChunks; ++c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
                  : -1;
          const uint32_t d = static_cast<uint32_t>(bin - hotLo);
          if (d < kHotBins) {
            hotCount += uint64_t{1} << (d * kHotBits);
          } else if (bin >= 0) {
            atomicAdd(&smemBins[bin * kCopies + histCopy], 1);
          }
        }
      }
#pragma unroll
      for (int k = 0; k < kHotBins; ++k) {
        const int count = static_cast<int>(
            (hotCount >> (k * kHotBits)) & ((uint64_t{1} << kHotBits) - 1));
        const int total = sycl::reduce_over_group(sg, count, sycl::plus<int>());
        if (lane_id == 0 && total > 0) {
          atomicAdd(&smemBins[(hotLo + k) * kCopies], total);
        }
      }
    } else {
      for (int c = 0; c < numChunks; ++c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          if (chunkOff[i] < rowLen) {
            atomicAdd(
                &smemBins
                    [extractFastBinIdx<kNumFastBins>(chunkVal[i]) * kCopies +
                     histCopy],
                1);
          }
        }
      }
    }
  }
  sycl::group_barrier(group);

  // The prefix sum, and with it the threshold bin: the one bin whose exclusive
  // prefix is below topK and whose inclusive prefix is not. It is unique, an
  // empty bin can never be it, and because the prefix is non-decreasing every
  // earlier bin is fully accepted and every later one fully rejected. So that
  // one index classifies the whole row, and each bin's exclusive prefix is
  // where its elements belong in the output; both go back into the histogram
  // slots for pass two to use.
  int threadSum = 0;
  int binCount[kBinsPerThread];
  {
    int raw[kBinsPerThread * kCopies];
    binsLoad<kBinsPerThread * kCopies>(smemBins + binBase * kCopies, raw);
#pragma unroll
    for (int b = 0; b < kBinsPerThread; ++b) {
      binCount[b] = 0;
#pragma unroll
      for (int c = 0; c < kCopies; ++c) {
        binCount[b] += raw[b * kCopies + c];
      }
    }
  }
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    threadSum += binCount[b];
  }
  // The row total the scan hands back is what the multi-step path needs to keep
  // a running output cursor; this path takes its cursor from the threshold
  // bin's own prefix instead, so it is written and never read.
  int rowTotal = 0;
  int prefix = blockExclusiveScan<kNumThreadsPerBlock, kThreadsPerRow>(
      group, sg, lane_id, sg_id_block, smemTmp, threadSum, rowTotal);
#pragma unroll
  for (int b = 0; b < kBinsPerThread; ++b) {
    const int inclusive = prefix + binCount[b];
    if (prefix < topK && topK <= inclusive) {
      smemThresholdBinIdx[0] = binBase + b;
      smemFinalBinSize[0] = binCount[b];
      smemFoundTopKValues[0] = prefix;
      *acceptCursor = 0;
      *candCursor = prefix;
    }
    prefix = inclusive;
  }
  sycl::group_barrier(group);

  // A large candidate set lives in the histogram, which nothing reads after
  // the scan above.
  // The items buffer it replaces then holds the radix select below.
  int* const selHist = itemIndices;
  int* const selVars = reinterpret_cast<int*>(itemLogits);
  const bool bigCandidates =
      !kGrouped && !kRegResident && rowLen >= kLongRowMinLen;
  if (bigCandidates) {
    itemLogits = reinterpret_cast<float*>(smemBins);
    itemIndices = smemBins + kNumFastBins * kCopies / 2;
  }

  // rowLen > topK, so some bin is always the threshold bin. Bail out only when
  // its elements would not fit the compacted candidate buffer: nothing has been
  // written to global memory yet, so the caller can start over.
  //
  // When several rows share the work-group one row cannot return on its own:
  // the rows behind it would wait forever at the barrier below. So the group
  // agrees first, at the cost of one more barrier, and all of its rows start
  // over together. smemGroupBail was cleared by the caller, before the barrier
  // that published the cleared histogram.
  if constexpr (kGrouped) {
    if (smemFinalBinSize[0] > kMaxCandidates && threadIdx_x == 0) {
      smemGroupBail[0] = 1;
    }
    sycl::group_barrier(group);
    if (smemGroupBail[0] != 0) {
      return false;
    }
  } else if (
      smemFinalBinSize[0] >
      (bigCandidates ? kNumFastBins * kCopies / 2 : kMaxCandidates)) {
    return false;
  }

  // Pass two. The slot an element gets is its bin's first output slot plus its
  // rank inside that bin, and the rank comes from bumping the bin's own
  // counter, which is the prefix the scan just left in shared memory. So there
  // is no second scan, no per chunk barrier and no cross lane traffic at all:
  // one shared memory atomic per element, spread over kNumFastBins addresses
  // that hold ~2 elements each. The block wide scan this replaces cost 11.2 us
  // of 84.7 on the S=3500 triangle, against ~1 us for the whole first pass of
  // the same atomics.
  //
  // Elements below the threshold bin go straight to global memory, in
  // descending bin order rather than index order. Elements of the threshold bin
  // land in the candidate buffer instead, at their rank inside the bin, ready
  // for the rank below. Everything above it has nowhere to go.
  const int thresholdBin = smemThresholdBinIdx[0];
  const int candBase = smemFoundTopKValues[0];
  // Skipping a thread whose elements are all rejected was measured here and
  // lost 1.9 us of 74.5. The test is per thread but the hardware is per
  // subgroup: at topK=128 the threshold sits near the row's 87th percentile, so
  // all 64 elements of a subgroup are rejected with probability 0.87^64, and
  // the subgroup walks the slots either way. Only a subgroup-uniform predicate
  // can skip work here.
  //
  // Accepted elements are packed per subgroup: a ballot per register slot gives
  // each lane its place among the subgroup's accepted elements, and one atomic
  // per subgroup reserves the whole run. Taking the slot from the element's
  // own bin instead was one shared memory atomic per element, and on real
  // logits the accepted elements crowd into a few fp16 bins, so the lanes of
  // one atomic hit the same few addresses and serialise; the ranks are
  // consecutive, so the stores coalesce as well.
  const uint32_t lowerLanes = (1u << lane_id) - 1u;
  auto ballot = [&](bool pred) {
    uint32_t bits;
    sycl::ext::oneapi::group_ballot(sg, pred).extract_bits(bits);
    return bits;
  };
  // Candidates are packed the same way. They all share one bin by definition,
  // so a counter bumped per element is the most contended atomic of all.
  auto emitCandidates = [&](bool pred, int off, float logit) {
    const uint32_t mask = ballot(pred);
    if (mask == 0) {
      return;
    }
    int base = 0;
    if (lane_id == 0) {
      base = atomicAdd(candCursor, static_cast<int>(sycl::popcount(mask)));
    }
    base = sycl::group_broadcast(sg, base, 0) - candBase;
    if (pred) {
      const int slot = base + sycl::popcount(mask & lowerLanes);
      itemLogits[slot] = logit;
      itemIndices[slot] = off;
    }
  };
  auto reserve = [&](int count) {
    int base = 0;
    if (lane_id == 0 && count > 0) {
      base = atomicAdd(acceptCursor, count);
    }
    return sycl::group_broadcast(sg, base, 0);
  };
  if constexpr (kRegResident) {
    // The subgroup's lane 0 holds the most slots, so its count bounds the
    // loop for every lane and the ballots see the whole subgroup.
    const int sgItems = sycl::group_broadcast(sg, regItems, 0);
    uint32_t accMask[kMaxRegItems];
    int sgAccepted = 0;
#pragma unroll
    for (int i = 0; i < kMaxRegItems; ++i) {
      if (i >= sgItems) {
        break;
      }
      const bool live = i < regItems;
      accMask[i] = ballot(live && regBin[i] < thresholdBin);
      sgAccepted += sycl::popcount(accMask[i]);
      if (live && regBin[i] == thresholdBin) {
        const int slot = atomicAdd(candCursor, 1) - candBase;
        itemLogits[slot] = regVal[i];
        itemIndices[slot] = regOffset(i);
      }
    }
    int base = reserve(sgAccepted);
#pragma unroll
    for (int i = 0; i < kMaxRegItems; ++i) {
      if (i >= sgItems) {
        break;
      }
      if ((accMask[i] >> lane_id) & 1u) {
        outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
            regOffset(i);
      }
      base += sycl::popcount(accMask[i]);
    }
  } else {
    // The chunk is staged in registers before any of its atomics, for the same
    // reason as in pass one.
    // The head is one more slot, live on the lanes that hold a head element.
    // head is uniform, so every lane takes part in its ballots.
    if (head > 0) {
      const bool live = threadIdx_x < head;
      const float x = live ? rowBase[threadIdx_x] : 0.f;
      const int bin = live
                          ? static_cast<int>(extractFastBinIdx<kNumFastBins>(x))
                          : kNumFastBins;
      const uint32_t mask = ballot(bin < thresholdBin);
      const int base = reserve(sycl::popcount(mask));
      if (bin < thresholdBin) {
        outIndices[base + sycl::popcount(mask & lowerLanes)] = threadIdx_x;
      }
      emitCandidates(bin == thresholdBin, threadIdx_x, x);
    }
    // Backwards, so that it starts on the end of the row, which pass one read
    // last and the last level cache still holds.
    // Rows longer than kCandBatchMaxRowLen keep one candidate atomic per slot:
    // batching them was 1.016 slower at 512 rows of 163840 and 1.069 at
    // 262144 (topK=512), growing with the row, against 1.004 at 131072 and
    // 0.94-0.97 at 49152-98304 (topK=2048). The
    // two loops are whole, as a branch per slot cost the batched loop its gain.
    if (rowLen > kCandBatchMaxRowLen) {
      for (int c = numChunks - 1; c >= 0; --c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kMaxRegItems];
        int sgAccepted = 0;
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
                  : kNumFastBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          emitCandidates(bin == thresholdBin, chunkOff[i], chunkVal[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          if ((accMask[i] >> lane_id) & 1u) {
            outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
                chunkOff[i];
          }
          base += sycl::popcount(accMask[i]);
        }
      }
    } else {
      for (int c = numChunks - 1; c >= 0; --c) {
        const int chunk = chunkAt(c);
        float chunkVal[kMaxRegItems];
        int chunkOff[kMaxRegItems];
        loadChunk(chunk, chunkVal, chunkOff);
        uint32_t accMask[kMaxRegItems];
        uint32_t candMask[kMaxRegItems];
        int sgAccepted = 0;
        int sgCands = 0;
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          // Elements past the end of the row get a bin one past the last, which
          // no threshold bin can match.
          const int bin =
              chunkOff[i] < rowLen
                  ? static_cast<int>(
                        extractFastBinIdx<kNumFastBins>(chunkVal[i]))
                  : kNumFastBins;
          accMask[i] = ballot(bin < thresholdBin);
          sgAccepted += sycl::popcount(accMask[i]);
          candMask[i] = ballot(bin == thresholdBin);
          sgCands += sycl::popcount(candMask[i]);
        }
        int base = reserve(sgAccepted);
#pragma unroll
        for (int i = 0; i < kMaxRegItems; ++i) {
          if ((accMask[i] >> lane_id) & 1u) {
            outIndices[base + sycl::popcount(accMask[i] & lowerLanes)] =
                chunkOff[i];
          }
          base += sycl::popcount(accMask[i]);
        }
        // The candidates of the whole chunk take one atomic, like the accepted
        // ones: on clustered logits most slots hold candidates, and one atomic
        // per slot on the one cursor every subgroup shares serialised them.
        if (sgCands > 0) {
          int cand = 0;
          if (lane_id == 0) {
            cand = atomicAdd(candCursor, sgCands);
          }
          cand = sycl::group_broadcast(sg, cand, 0) - candBase;
#pragma unroll
          for (int i = 0; i < kMaxRegItems; ++i) {
            if ((candMask[i] >> lane_id) & 1u) {
              const int slot = cand + sycl::popcount(candMask[i] & lowerLanes);
              itemLogits[slot] = chunkVal[i];
              itemIndices[slot] = chunkOff[i];
            }
            cand += sycl::popcount(candMask[i]);
          }
        }
      }
    }
  }
  sycl::group_barrier(group);

  // The threshold bin, ranked pairwise on the exact float values. Same
  // comparison and same index tie-break as the other path's final pass, so the
  // set of values selected is identical.
  const int numCandidates = smemFinalBinSize[0];
  const int accepted = smemFoundTopKValues[0];
  const int remainedTopK = topK - accepted;
  // A subgroup per candidate, and the whole subgroup walks the candidate list,
  // so the quadratic work is spread over all the threads instead of over the
  // numCandidates of them that own a candidate. It matters most where the
  // candidate set is widest: at topK=128 the threshold bin covers a decade of
  // the value range and holds tens of elements, and this stage was 10.3 us
  // of 82.
  // A long row's candidate set can hold thousands of elements, and ranking
  // those pairwise is quadratic: at 32 rows of 65536 it doubled the kernel. It
  // is cut down by a radix select instead, kSelBits of the exact float key per
  // pass. The keys are taken as offsets from the smallest one, and the shared
  // fp16 bin keeps those to about 19 bits, so that is two or three passes over
  // the candidates. The ranks within the selected set are not needed: any order
  // of the top-k is a valid output.
  if constexpr (!kGrouped) {
    if (numCandidates > kMaxCandidates) {
      constexpr int kSelBins = kThreadsPerRow;
      constexpr int kSelBits = constexprLog2(kSelBins);
      static_assert(
          kSelBins <= kNumFinalItems, "the histogram fits the items buffer");
      int* lessCursor = selVars + 3;
      int* tieCursor = selVars + 4;
      if (threadIdx_x == 0) {
        *lessCursor = 0;
        *tieCursor = 0;
      }
      uint32_t lo = 0xffffffffu;
      uint32_t hi = 0;
      for (int i = threadIdx_x; i < numCandidates; i += kThreadsPerRow) {
        const uint32_t key = convert_to_uint32(itemLogits[i]);
        lo = sycl::min(lo, key);
        hi = sycl::max(hi, key);
      }
      const uint32_t minKey =
          sycl::reduce_over_group(group, lo, sycl::minimum<uint32_t>());
      const uint32_t range =
          sycl::reduce_over_group(group, hi, sycl::maximum<uint32_t>()) -
          minKey;
      // Publishes the cleared cursors when no pass below runs.
      sycl::group_barrier(group);
      auto offsetOf = [&](int i) -> uint64_t {
        return convert_to_uint32(itemLogits[i]) - minKey;
      };
      // Offset bits below `shift` are still open. `prefix` is the threshold's
      // offset above them, and `need` how many of the elements that share it
      // are still to be taken.
      int shift = range == 0 ? 0 : 32 - static_cast<int>(sycl::clz(range));
      uint64_t prefix = 0;
      int need = remainedTopK;
      while (shift > 0) {
        const int low = sycl::max(shift - kSelBits, 0);
        const uint32_t digitMask = (1u << (shift - low)) - 1u;
        selHist[threadIdx_x] = 0;
        sycl::group_barrier(group);
        for (int i = threadIdx_x; i < numCandidates; i += kThreadsPerRow) {
          const uint64_t off = offsetOf(i);
          if ((off >> shift) == prefix) {
            atomicAdd(&selHist[(off >> low) & digitMask], 1);
          }
        }
        sycl::group_barrier(group);
        const int count = selHist[threadIdx_x];
        int total = 0;
        const int before =
            blockExclusiveScan<kNumThreadsPerBlock, kThreadsPerRow>(
                group, sg, lane_id, sg_id_block, smemTmp, count, total);
        if (before < need && need <= before + count) {
          selVars[0] = threadIdx_x;
          selVars[1] = before;
          selVars[2] = count;
        }
        sycl::group_barrier(group);
        prefix = (prefix << (shift - low)) | static_cast<uint32_t>(selVars[0]);
        need -= selVars[1];
        shift = low;
        // The whole digit is taken: the bits below it cannot split it.
        if (need == selVars[2]) {
          break;
        }
      }
      // Everything below the threshold prefix is in, and `need` of the
      // elements on it. Packed per subgroup like the emits of pass two.
      const int taken = remainedTopK - need;
      for (int i = threadIdx_x; i - lane_id < numCandidates;
           i += kThreadsPerRow) {
        const bool live = i < numCandidates;
        const uint64_t top = live ? offsetOf(i) >> shift : ~uint64_t{0};
        const int index = live ? itemIndices[i] : 0;
        uint32_t mask = ballot(top < prefix);
        if (mask != 0) {
          int base = 0;
          if (lane_id == 0) {
            base =
                atomicAdd(lessCursor, static_cast<int>(sycl::popcount(mask)));
          }
          base = sycl::group_broadcast(sg, base, 0);
          if (top < prefix) {
            outIndices[accepted + base + sycl::popcount(mask & lowerLanes)] =
                index;
          }
        }
        mask = ballot(top == prefix);
        if (mask != 0) {
          int base = 0;
          if (lane_id == 0) {
            base = atomicAdd(tieCursor, static_cast<int>(sycl::popcount(mask)));
          }
          base = sycl::group_broadcast(sg, base, 0);
          const int slot = base + sycl::popcount(mask & lowerLanes);
          if (top == prefix && slot < need) {
            outIndices[accepted + taken + slot] = index;
          }
        }
      }
      return true;
    }
  }
  constexpr int kNumSubgroups = kThreadsPerRow / SUBGROUP_SIZE;
  for (int i = sg_id; i < numCandidates; i += kNumSubgroups) {
    const float logit = itemLogits[i];
    int outIndex = 0;
    for (int j = lane_id; j < numCandidates; j += SUBGROUP_SIZE) {
      const float otherLogit = itemLogits[j];
      if (logit < otherLogit || (logit == otherLogit && i < j)) {
        ++outIndex;
      }
    }
    outIndex = sycl::reduce_over_group(sg, outIndex, sycl::plus<int>());
    if (lane_id == 0 && outIndex < remainedTopK) {
      outIndices[accepted + outIndex] = itemIndices[i];
    }
  }
  return true;
}

template <
    int kNumThreadsPerBlock,
    int kNumBins,
    bool useRadixSort,
    bool multipleBlocksPerRow = false,
    bool mergeBlocks = false>
static void topKPerRowJob(
    const sycl::nd_item<3>& item,
    sycl::local_accessor<char, 1> slm,
    const int* indices,
    const float* logits,
    int rowStart,
    int rowEnd,
    int* outIndices,
    float* outLogits,
    int stride1,
    int topK) {
  sycl::group group = item.get_group();
  auto sg = item.get_sub_group();
  auto threadIdx_x = item.get_local_id(0);

  char* smem_buf =
      slm.template get_multi_ptr<sycl::access::decorated::no>().get();

  size_t sharedStatesOffsetBytes =
      sharedStatesOffset(topK, multipleBlocksPerRow);

  int32_t* smemOutput = reinterpret_cast<int32_t*>(smem_buf);
  SharedStates* sharedStates =
      reinterpret_cast<SharedStates*>(smem_buf + sharedStatesOffsetBytes);

  // Shared memory references
  auto& smemFinal = sharedStates->smemFinal;
  int* smemThresholdBinIdx = sharedStates->smemThresholdBinIdx;
  int* smemFinalDstIdx = sharedStates->smemFinalDstIdx;
  int* smemFinalBinSize = sharedStates->smemFinalBinSize;
  int* smemFoundTopKValues = sharedStates->smemFoundTopKValues;

  // The length of the row. A prefill row that ends before its start is empty.
  int rowLen = sycl::max(rowEnd - rowStart, 0);

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= topK) {
    for (int rowIt = threadIdx_x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx_x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -std::numeric_limits<float>::max();
      }
    }

    return;
  }

  // The threshold fp16 bin the reduction path gave up on, and the number of
  // elements in the bins before it, for processRefineStep(). -1 if it did not
  // run.
  int refineBin = -1;
  int refineAccepted = 0;

  // The reduction path handles the plain single-block case, which is every
  // prefill row and every decode row short enough not to be split. It writes
  // its own output, or returns false having written nothing.
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    if (topK <= kFastMaxTopK) {
      const float* rowBase = logits + static_cast<int64_t>(rowStart) * stride1;
      // Three shapes of the same pass: the register-resident one for a row
      // that fits kFastMaxRegItems slots per thread, and for a longer row the
      // chunked one, with the extra histogram copies once the row is long
      // enough to hide their clear and scan.
      bool done;
      if (rowLen <= kFastMaxRegItems * kNumThreadsPerBlock) {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/true>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      } else if (rowLen >= kLongRowMinLen) {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kLongHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/false>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      } else {
        done = topKPerRowReduce<
            kNumThreadsPerBlock,
            kNumThreadsPerBlock,
            kNumBins,
            kHistCopies,
            kFastMaxRegItems,
            kFastMaxCandidates,
            /*kRegResident=*/false>(
            item,
            rowBase,
            rowLen,
            outIndices,
            smemFinal.histo.data,
            smemFinal.histo.tmp,
            smemFinal.items.indices,
            smemFinal.items.logits,
            smemThresholdBinIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            sharedStates->smemGroupBail,
            stride1,
            topK);
      }
      if (done) {
        return;
      }
      // Everything the reduction path touched is either re-initialised below or
      // cleared by processHistogramStep() before it is read.
      sycl::group_barrier(group);
      // The bin it gave up on is read before anything below rewrites it.
      refineBin = smemThresholdBinIdx[0];
      refineAccepted = smemFoundTopKValues[0];
      // Every thread holds its copy before thread 0 resets them below.
      sycl::group_barrier(group);
    }
  }

  // Initialize values
  if (threadIdx_x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  sycl::group_barrier(group);
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Split the reduction path's threshold bin further if its exponent allows it.
  // Should even its threshold bucket be too large, processTieSelect() finishes
  // the row, so a row the reduction path gave up on never reaches the steps.
  bool refined = false;
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    if (refineBin >= 0 && fastBinIsRefinable<kNumBins>(refineBin)) {
      // Same split as the reduction path above: only a long row is worth the
      // finer buckets, and only a long row can have given up with them.
      refined =
          processRefineStep<kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
              item,
              logits,
              rowStart,
              rowEnd,
              refineBin,
              rowLen >= kLongRowMinLen ? kLongRefineShift
                                       : kLongRefineShift + 2,
              refineAccepted,
              smemOutput,
              smemThresholdBinIdx,
              smemFinalDstIdx,
              smemFinalBinSize,
              smemFoundTopKValues,
              sharedStates->smemRefineDstIdx,
              smemFinal,
              stride1,
              topK);
    }
    // Neither could split the bin: finish the row on the fp32 keys of that bin
    // alone rather than on steps 1 to 3, which re-derive everything from the
    // whole row and funnel a tie into one SLM address per pass.
    if (refineBin >= 0 && !refined) {
      processTieSelect<kNumThreadsPerBlock, kNumBins>(
          item,
          logits + static_cast<int64_t>(rowStart) * stride1,
          rowLen,
          stride1,
          refineBin,
          refineAccepted,
          outIndices,
          smemFinal.histo.data,
          smemFinal.histo.tmp,
          smemFinal.items.indices,
          topK);
      return;
    }
  }

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep = !refined && processHistogramStep<
                                            0,
                                            kNumThreadsPerBlock,
                                            kNumBins,
                                            kNumFinalItems,
                                            multipleBlocksPerRow,
                                            mergeBlocks>(
                                            item,
                                            indices,
                                            logits,
                                            rowEnd,
                                            logitPattern,
                                            thresholdBinIdx,
                                            smemOutput,
                                            smemThresholdBinIdx,
                                            smemFinalDstIdx,
                                            smemFinalBinSize,
                                            smemFoundTopKValues,
                                            smemFinal,
                                            stride1,
                                            rowStart,
                                            topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep = processHistogramStep<
        1,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep = processHistogramStep<
        2,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<
        3,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        item,
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    if constexpr (useRadixSort) {
      static_assert(!useRadixSort, "Radix sort is not implemented yet");
      // TODO
    } else {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx_x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        auto remainedTopK = topK - baseIdx;
        if (outIndex < remainedTopK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    }
    sycl::group_barrier(group);
  }

  // Store to global memory.
  for (int i = threadIdx_x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

}  // namespace vllm

}  // namespace topk_dlgw
