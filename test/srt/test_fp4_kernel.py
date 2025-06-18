import unittest

import torch

from sglang.srt.layers.quantization.fp4_kernel import (
    per_1x32_quant_fp4
)
from sglang.srt.layers.quantization.fp4_utils import (
    fp32_to_mxfp4
)
from sglang.test.test_utils import CustomTestCase


class TestMXFP4Base(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 256
        cls.N = 1024
        cls.K = 512
        cls.quant_block_size = 128

    @staticmethod
    def _make_A(M, K, quant_block_size):
        # Generate a random tensor A and its quantized version
        A = torch.rand(M, K, dtype=torch.float32, device="cuda")

        # compute scale
        amax = torch.max(
            torch.abs(A.unfold(1, quant_block_size, quant_block_size)), dim=2)[0]
        scale = torch.floor(torch.log2(amax)) - 2
        scale = torch.clamp(scale, min=-127, max=127)
        quant_scale = torch.pow(2, -scale)

        # Compute quantized tensor A
        quant_A = fp32_to_mxfp4(A * quant_scale).reshape(M, K)
        A = A.reshape(M, K)

        return A, quant_A, scale


class TestPerTokenGroupQuantFP4(TestMXFP4Base):
    def test_per_token_group_quant_fp4(self):
        if torch.cuda.get_device_capability()[0] < 10:
            return
        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, mxfp4_quant_block_size=self.quant_block_size
        )
        A_quant, scale = per_1x32_quant_fp4(
            x=A, mxfp4_quant_block_size=self.quant_block_size, shuffle=False
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


if __name__ == "__main__":
    unittest.main()
