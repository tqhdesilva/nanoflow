import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

import config as _config  # noqa: F401
import vae as vae_module
from image_transforms import build_cache_transform
from vae import VAEWrapper


class FakeLatentDist:
    def __init__(self, z):
        self._z = z
        self.mean = z + 10

    def mode(self):
        return self._z

    def sample(self):
        return self._z + 1


class FakeAutoencoderKL(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=0.5)
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.from_pretrained_calls.append(kwargs)
        return cls()

    def encode(self, x):
        z = torch.nn.functional.avg_pool2d(x.mean(dim=1, keepdim=True), 8)
        z = z.repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=FakeLatentDist(z))

    def decode(self, z):
        y = torch.nn.functional.interpolate(z[:, :3], size=(256, 256), mode="nearest")
        return SimpleNamespace(sample=y)


class VAETest(unittest.TestCase):
    def test_cache_transform_rgb_scale_and_shape(self):
        image = Image.new("L", (128, 96), color=128)
        transform = build_cache_transform(image_size=256, crop="resize")
        x = transform(image)

        self.assertEqual(tuple(x.shape), (3, 256, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertGreaterEqual(float(x.min()), -1.0)
        self.assertLessEqual(float(x.max()), 1.0)
        torch.testing.assert_close(x[0], x[1])
        torch.testing.assert_close(x[1], x[2])

    def test_invalid_cache_crop_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_cache_transform(crop="bad")

    def test_diffusers_wrapper_encodes_and_decodes(self):
        FakeAutoencoderKL.from_pretrained_calls = []
        with mock.patch.object(
            vae_module, "_import_autoencoder_kl", return_value=FakeAutoencoderKL
        ):
            wrapper = VAEWrapper(
                model_id="fake/vae",
                latent_shape=[4, 32, 32],
                torch_dtype="float32",
                device="cpu",
            )
            x = torch.randn(2, 3, 256, 256).clamp(-1, 1)
            z = wrapper.encode(x)
            y = wrapper.decode(z)

        self.assertEqual(tuple(z.shape), (2, 4, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 3, 256, 256))
        self.assertEqual(wrapper.scaling_factor, 0.5)
        self.assertEqual(
            FakeAutoencoderKL.from_pretrained_calls[0]["pretrained_model_name_or_path"],
            "fake/vae",
        )

    def test_sample_posterior_is_optional(self):
        with mock.patch.object(
            vae_module, "_import_autoencoder_kl", return_value=FakeAutoencoderKL
        ):
            wrapper = VAEWrapper(
                model_id="fake/vae",
                latent_shape=[4, 32, 32],
                sample_posterior=True,
            )
            x = torch.zeros(1, 3, 256, 256)
            sampled = wrapper.encode(x)
            mode = wrapper.encode(x, sample_posterior=False)

        self.assertGreater(float((sampled - mode).abs().mean()), 0.0)

    def test_hydra_vae_and_transform_config_materialize(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["vae=sd_vae_ft_ema", "vae_transform=imagenet256_resize"],
            )

        self.assertEqual(cfg.vae._target_, "vae.VAEWrapper")
        self.assertEqual(cfg.vae.model_id, "stabilityai/sd-vae-ft-ema")
        self.assertEqual(list(cfg.vae.latent_shape), [4, 32, 32])
        self.assertEqual(
            cfg.vae_transform._target_, "image_transforms.build_cache_transform"
        )
        self.assertEqual(cfg.vae_transform.crop, "resize")

    def test_transform_is_separate_from_wrapper(self):
        with mock.patch.object(
            vae_module, "_import_autoencoder_kl", return_value=FakeAutoencoderKL
        ):
            wrapper = VAEWrapper(model_id="fake/vae")
        transform = build_cache_transform(image_size=64, crop="resize")
        image = Image.new("RGB", (32, 32), color=(10, 20, 30))
        x = transform(image)

        self.assertFalse(hasattr(wrapper, "cache_transform"))
        self.assertEqual(tuple(x.shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
