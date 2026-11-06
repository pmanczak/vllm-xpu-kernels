#pragma once

#include <cstdint>

#include <torch/all.h>

namespace topk_dlgw {

// The delegate-bound topk_per_row path for BMG (topk_per_row_dlgw.cpp). Same
// arguments and results as top_k_per_row_decode and top_k_per_row_prefill.
// Returns true if it launched the selection and false if it leaves the call to
// the original kernels: it takes a call only where it is measured to win (the
// gate in topk_per_row_dlgw.cpp). Nothing here throws on bad input: anything
// it does not expect (dtype, device, shape, layout, sizes) is declined, and
// the original checks report it.
bool top_k_per_row_decode_dlgw(
    const torch::Tensor& logits,
    int64_t next_n,
    const torch::Tensor& seqLens,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK);

bool top_k_per_row_prefill_dlgw(
    const torch::Tensor& logits,
    const torch::Tensor& rowStarts,
    const torch::Tensor& rowEnds,
    torch::Tensor& indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK);

}  // namespace topk_dlgw
