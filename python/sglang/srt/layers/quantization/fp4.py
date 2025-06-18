# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp4_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    dispatch_w4a4_block_fp4_linear,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    log_info_on_rank0,
    print_warning_once,
    set_weight_attrs,
)

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Fp4Config(QuantizationConfig):
    """Config class for MXFP4."""

    def __init__(
        self,
        is_checkpoint_fp4_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
    ) -> None:
        self.is_checkpoint_fp4_serialized = is_checkpoint_fp4_serialized
        if is_checkpoint_fp4_serialized:
            log_info_on_rank0(logger, "Detected fp4 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp4_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp4-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp4Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp4_serialized = "fp4" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(
            config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp4_serialized=is_checkpoint_fp4_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp4LinearMethod(LinearMethodBase):
    """Linear method for MXFP4.
    Supports loading MXFP4 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp4Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        self.w4a4_block_fp4_linear = dispatch_w4a4_block_fp4_linear()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp4_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp4, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp4_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(
                        len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(
                    data=torch.empty(
                        len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                weight_scale, requires_grad=False
            )
            return

        layer.weight = torch.nn.Parameter(
            layer.weight.data, requires_grad=False)

        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp4_serialized:
            # per-tensor quantization
            qweight, weight_scale = input_to_float8(layer.weight)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp4, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(
                    layer.input_scale.data, requires_grad=False
                )

            # Dequant -> Quant with max scale so we can run per tensor.
            weight = layer.weight
            weight_scale = layer.weight_scale

            weight_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=weight_scale,
                logical_widths=layer.logical_widths,
            )

            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.w4a4_block_fp4_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )
