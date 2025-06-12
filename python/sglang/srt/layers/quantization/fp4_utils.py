import torch

from typing import Callable, List, Optional
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp4,
)

from sglang.srt.utils import (
    get_bool_env_var,
    is_hip,
)

_is_hip = is_hip()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import gemm_a4w4_blockscale_CK


def dispatch_w4a4_block_fp4_linear() -> Callable:
    if _use_aiter:
        return aiter_w8a8_block_fp4_linear
    else:
        raise NotImplementedError(
            "SGLANG_USE_AITER is not set, and no fallback implementation is provided."
        )


def aiter_w4a4_block_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp4(
        input_2d, block_size[1], column_major_scales=False
    )
    output = gemm_a4w4_blockscale_CK(
        q_input, weight, x_scale, weight_scale, dtype=input.dtype
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)
