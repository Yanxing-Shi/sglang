import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

import logging

logger = logging.getLogger(__name__)


@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    scale_N: tl.constexpr,
    scale_M_pad: tl.constexpr,
    scale_N_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    shuffle: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + \
        tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign, exponents and mantissa fields from FP32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
        0, MXFP4_QUANT_BLOCK_SIZE // 2
    )
    out_offs = (
        out_offs_m[:, None] * stride_x_fp4_m +
        out_offs_n[None, :] * stride_x_fp4_n
    )
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n

    if shuffle:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * scale_N
        )
        bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scale_N)[None, :]
        bs_mask2 = (bs_offs_m < scale_M_pad)[:, None] & (
            bs_offs_n < scale_N_pad)[None, :]
        bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 127)
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask2)
    else:
        bs_offs = bs_offs_m[:, None] * stride_bs_m + \
            bs_offs_n[None, :] * stride_bs_n
        bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < N)[None, :]
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def per_1x32_quant_fp4(
    x: torch.Tensor,
    mxfp4_quant_block_size: int = 32,
    shuffle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into mxfp4 format (a 4-bit floating-point format)
    and returns the quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        shuffle: If True, applies a specific shuffling pattern to the quantization.


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=torch.uint8)
    M, N = x.shape[-2:]

    assert (N // 2) % 2 == 0, "N must be divisible by 4 for mxfp4 quantization"

    # This is fixed by spec for MXFP4. Do not tune this.
    # For performance, perhaps, we should look at passing multiple of 32 column blocks
    # that a triton program can process
    MXFP4_QUANT_BLOCK_SIZE = 32

    scale_M = triton.cdiv(M, 32) * 32
    scale_N_valid = triton.cdiv(N, mxfp4_quant_block_size)
    scale_N = triton.cdiv(scale_N_valid, 8) * 8
    blockscale_e8m0 = torch.empty(
        (
            triton.cdiv(M, 256) * 256,
            scale_N,
        ),
        dtype=torch.uint8,
        device=x.device,
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), scale_N)
    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_q,
        blockscale_e8m0,
        *x.stride(),
        *x_q.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        scale_N=scale_N_valid,
        scale_M_pad=scale_M,
        scale_N_pad=scale_N,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=mxfp4_quant_block_size,
        shuffle=shuffle,
    )

    return x_q, blockscale_e8m0.view(torch.uint8)
