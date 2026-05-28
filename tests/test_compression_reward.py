import unittest

import torch

from metrics import JpegCompressibilityMetric
from rl.compression import (
    jpeg_bpp_for_sample,
    jpeg_bytes_for_sample,
    sample_to_uint8_pil,
)
from rl.reward import JpegCompressibilityReward


class CompressionRewardTest(unittest.TestCase):
    def test_grayscale_tensor_becomes_l_mode_jpeg(self):
        sample = torch.zeros(1, 28, 28)
        image = sample_to_uint8_pil(sample)

        self.assertEqual(image.mode, "L")
        self.assertGreater(jpeg_bytes_for_sample(sample, quality=75), 0)

    def test_constant_image_more_compressible_than_noise(self):
        constant = torch.zeros(1, 28, 28)
        gen = torch.Generator().manual_seed(123)
        noise = torch.rand(1, 28, 28, generator=gen) * 2 - 1

        self.assertLess(
            jpeg_bpp_for_sample(constant, quality=75),
            jpeg_bpp_for_sample(noise, quality=75),
        )

    def test_reward_sign_prefers_lower_bpp(self):
        constant = torch.zeros(1, 1, 28, 28)
        gen = torch.Generator().manual_seed(123)
        noise = torch.rand(1, 1, 28, 28, generator=gen) * 2 - 1
        batch = torch.cat([constant, noise], dim=0)
        reward = JpegCompressibilityReward(quality=75)

        actual = reward(batch, torch.tensor([0, 0]))

        self.assertGreater(actual[0].item(), actual[1].item())

    def test_jpeg_settings_are_stable(self):
        gen = torch.Generator().manual_seed(456)
        sample = torch.rand(1, 28, 28, generator=gen) * 2 - 1

        first = jpeg_bytes_for_sample(
            sample, quality=75, optimize=False, progressive=False
        )
        second = jpeg_bytes_for_sample(
            sample, quality=75, optimize=False, progressive=False
        )

        self.assertEqual(first, second)

    def test_metric_reports_bpp_and_raw_bytes(self):
        samples = torch.stack([torch.zeros(1, 28, 28), torch.ones(1, 28, 28)])
        metric = JpegCompressibilityMetric(include_png=True, include_diagnostics=True)

        result = metric(samples)

        self.assertEqual(result["name"], "JPEGCompressibility")
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(len(result["jpeg_bytes"]), 2)
        self.assertIn("jpeg_bpp_mean", result)
        self.assertIn("png_bpp_mean", result)
        self.assertIn("total_variation_mean", result)
        self.assertIn("duplicate_fraction", result)


if __name__ == "__main__":
    unittest.main()
