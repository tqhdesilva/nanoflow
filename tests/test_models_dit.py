import os
import unittest

import torch

import config as _config  # noqa: F401
from config import LossMode
from models_dit import (
    ClassCondDeferredMaskingDiT,
    ClassCondDiT,
    DenseFFN,
    DiTBackbone,
    DiTBlock,
    PatchMixer,
    RandomTokenMasker,
    RoPE2D,
    SelfAttention,
    apply_2d_rope,
    build_2d_patch_coords,
)


def _make_model(
    *,
    in_ch=4,
    latent_size=32,
    patch_size=2,
    hidden_size=32,
    num_heads=4,
    num_classes=1000,
):
    block = DiTBlock(
        hidden_size=hidden_size,
        attention=SelfAttention(hidden_size=hidden_size, num_heads=num_heads),
        ffn=DenseFFN(hidden_size=hidden_size, mlp_width=hidden_size * 2),
    )
    backbone = DiTBackbone(hidden_size=hidden_size, blocks=[block])
    return ClassCondDiT(
        in_ch=in_ch,
        latent_size=latent_size,
        patch_size=patch_size,
        num_classes=num_classes,
        backbone=backbone,
        time_dim=hidden_size,
        class_dim=hidden_size,
    )


class SpyPatchMixer(PatchMixer):
    def __init__(self, hidden_size):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = False
        self.blocks = torch.nn.ModuleList()
        self.last_num_tokens = None

    def forward(self, x, cond, coords=None, pos_embedding=None):
        self.last_num_tokens = x.size(1)
        return x


class SpyBackbone(DiTBackbone):
    def __init__(self, hidden_size):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = False
        self.blocks = torch.nn.ModuleList()
        self.last_num_tokens = None
        self.last_coords = None

    def forward(self, x, cond, coords=None, pos_embedding=None):
        self.last_num_tokens = x.size(1)
        self.last_coords = coords.detach().clone() if coords is not None else None
        return x


def _make_deferred_model(
    *,
    in_ch=4,
    latent_size=32,
    patch_size=2,
    hidden_size=32,
    num_classes=1000,
    masker=None,
    patch_mixer="spy",
):
    if patch_mixer == "spy":
        patch_mixer = SpyPatchMixer(hidden_size)
    backbone = SpyBackbone(hidden_size)
    model = ClassCondDeferredMaskingDiT(
        in_ch=in_ch,
        latent_size=latent_size,
        patch_size=patch_size,
        num_classes=num_classes,
        patch_mixer=patch_mixer,
        masker=masker,
        backbone=backbone,
        time_dim=hidden_size,
        class_dim=hidden_size,
    )
    return model


class DiTModelTest(unittest.TestCase):
    def test_patchify_imagenet_latent_shape(self):
        model = _make_model(hidden_size=32)
        x = torch.randn(2, 4, 32, 32)
        tokens = model.patchify(x)
        self.assertEqual(tokens.shape, (2, 256, 32))

    def test_unpatchify_imagenet_latent_shape(self):
        model = _make_model(hidden_size=32)
        patches = torch.randn(2, 256, 16)
        x = model.unpatchify(patches)
        self.assertEqual(x.shape, (2, 4, 32, 32))

    def test_row_major_patch_order(self):
        model = _make_model(
            in_ch=1,
            latent_size=4,
            patch_size=2,
            hidden_size=4,
            num_heads=1,
            num_classes=2,
        )
        with torch.no_grad():
            model.patch_embed.weight.zero_()
            model.patch_embed.bias.zero_()
            model.patch_embed.weight[0, 0, 0, 0] = 1.0
        x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        tokens = model.patchify(x)
        self.assertEqual(tokens[0, :, 0].tolist(), [0.0, 2.0, 8.0, 10.0])

        patches = torch.tensor(
            [
                [
                    [1.0, 2.0, 5.0, 6.0],
                    [3.0, 4.0, 7.0, 8.0],
                    [9.0, 10.0, 13.0, 14.0],
                    [11.0, 12.0, 15.0, 16.0],
                ]
            ]
        )
        image = model.unpatchify(patches)
        expected = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
        self.assertTrue(torch.equal(image, expected))

    def test_rope_coordinates_and_axes(self):
        coords = build_2d_patch_coords(2, 3)
        expected = torch.tensor(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(coords, expected))

        q = torch.ones(1, 3, 1, 8)
        k = torch.ones(1, 3, 1, 8)
        coords = torch.tensor([[0, 0], [1, 0], [0, 1]])
        q_rot, k_rot = apply_2d_rope(q, k, coords, base=10.0)
        self.assertTrue(torch.allclose(q_rot[0, 0], q[0, 0]))
        self.assertTrue(torch.allclose(k_rot[0, 0], k[0, 0]))
        self.assertFalse(torch.allclose(q_rot[0, 1, :, :4], q[0, 1, :, :4]))
        self.assertTrue(torch.allclose(q_rot[0, 1, :, 4:], q[0, 1, :, 4:]))
        self.assertTrue(torch.allclose(q_rot[0, 2, :, :4], q[0, 2, :, :4]))
        self.assertFalse(torch.allclose(q_rot[0, 2, :, 4:], q[0, 2, :, 4:]))

    def test_apply_2d_rope_accepts_per_sample_coordinates(self):
        q = torch.ones(2, 2, 1, 8)
        k = torch.ones(2, 2, 1, 8)
        coords = torch.tensor(
            [
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
            ]
        )
        q_rot, k_rot = apply_2d_rope(q, k, coords, base=10.0)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertFalse(torch.allclose(q_rot[0, 1, :, :4], q[0, 1, :, :4]))
        self.assertTrue(torch.allclose(q_rot[0, 1, :, 4:], q[0, 1, :, 4:]))
        self.assertTrue(torch.allclose(q_rot[1, 1, :, :4], q[1, 1, :, :4]))
        self.assertFalse(torch.allclose(q_rot[1, 1, :, 4:], q[1, 1, :, 4:]))

    def test_random_token_masker_keeps_expected_count_and_coords(self):
        tokens = torch.randn(2, 256, 8)
        coords = build_2d_patch_coords(16, 16)
        masker = RandomTokenMasker(mask_ratio=0.75)
        out = masker(tokens, coords)

        self.assertEqual(out.tokens.shape, (2, 64, 8))
        self.assertEqual(out.coords.shape, (2, 64, 2))
        self.assertEqual(out.keep_indices.shape, (2, 64))
        self.assertEqual(out.keep_mask.shape, (2, 256))
        self.assertTrue(torch.all(out.keep_mask.sum(dim=1) == 64))
        gathered = coords.to(out.keep_indices.device)[out.keep_indices]
        self.assertTrue(torch.equal(out.coords.cpu(), gathered.cpu()))

    def test_deferred_model_masks_backbone_tokens_only_in_training(self):
        model = _make_deferred_model(masker=RandomTokenMasker(mask_ratio=0.75))
        model.train()
        x = torch.randn(2, 4, 32, 32)
        t = torch.rand(2)
        y = model(x, t, torch.tensor([1, model.null_token]))

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(model.patch_mixer.last_num_tokens, 256)
        self.assertEqual(model.backbone.last_num_tokens, 64)
        self.assertEqual(model.backbone.last_coords.shape, (2, 64, 2))

        model.eval()
        y = model(x, t, torch.tensor([1, model.null_token]))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(model.backbone.last_num_tokens, 256)
        self.assertEqual(model.backbone.last_coords.shape, (256, 2))

    def test_deferred_model_without_masker_keeps_all_tokens_in_training(self):
        model = _make_deferred_model(masker=None)
        model.train()
        x = torch.randn(2, 4, 32, 32)
        t = torch.rand(2)
        y = model(x, t, torch.tensor([1, 2]))

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(model.backbone.last_num_tokens, 256)
        self.assertEqual(model.backbone.last_coords.shape, (256, 2))

    def test_deferred_model_allows_no_patch_mixer(self):
        model = _make_deferred_model(
            hidden_size=32,
            masker=RandomTokenMasker(mask_ratio=0.75),
            patch_mixer=None,
        )
        model.train()
        x = torch.randn(2, 4, 32, 32)
        y = model(x, torch.rand(2), torch.tensor([1, 2]))

        self.assertEqual(y.shape, x.shape)
        self.assertIsNone(model.patch_mixer)
        self.assertEqual(model.backbone.last_num_tokens, 64)

    def test_deferred_aux_loss_mask_is_image_compatible(self):
        model = _make_deferred_model(
            in_ch=1,
            latent_size=4,
            patch_size=2,
            hidden_size=8,
            num_classes=2,
            masker=RandomTokenMasker(mask_ratio=0.5),
        )
        model.train()
        x = torch.randn(1, 1, 4, 4)
        out = model(x, torch.rand(1), torch.tensor([1]), return_aux=True)

        self.assertEqual(set(out.keys()), {"pred", "loss_mask"})
        self.assertEqual(out["pred"].shape, x.shape)
        self.assertEqual(out["loss_mask"].shape, (1, 1, 4, 4))
        self.assertEqual(out["loss_mask"].sum().item(), 8.0)
        self.assertTrue(torch.all((out["loss_mask"] == 0) | (out["loss_mask"] == 1)))

        model.eval()
        out = model(x, torch.rand(1), torch.tensor([1]), return_aux=True)
        self.assertTrue(
            torch.equal(out["loss_mask"], torch.ones_like(out["loss_mask"]))
        )

    def test_deferred_masked_checkpoint_loads_without_masker(self):
        masked = _make_deferred_model(masker=RandomTokenMasker(mask_ratio=0.75))
        unmasked = _make_deferred_model(masker=None)
        missing, unexpected = unmasked.load_state_dict(masked.state_dict())
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_deferred_masked_loss_is_finite_with_null_labels(self):
        model = _make_deferred_model(masker=RandomTokenMasker(mask_ratio=0.75))
        model.train()
        x = torch.randn(3, 4, 32, 32)
        labels = torch.tensor([0, model.null_token, 999])
        out = model(x, torch.rand(3), labels, return_aux=True)
        target = torch.randn_like(out["pred"])
        mask = out["loss_mask"].expand_as(out["pred"])
        loss = ((out["pred"] - target).pow(2) * mask).sum() / mask.sum()
        self.assertTrue(torch.isfinite(loss))

    def test_forward_shape_with_normal_and_null_labels(self):
        model = _make_model(
            in_ch=4,
            latent_size=32,
            patch_size=2,
            hidden_size=32,
            num_heads=4,
            num_classes=1000,
        )
        x = torch.randn(2, 4, 32, 32)
        t = torch.rand(2)
        labels = torch.tensor([7, model.null_token])
        y = model(x, t, labels)
        self.assertEqual(y.shape, x.shape)
        y_null = model(x, t)
        self.assertEqual(y_null.shape, x.shape)

    def test_checkpoint_state_dict_loads(self):
        model = _make_model(hidden_size=32)
        fresh = _make_model(hidden_size=32)
        missing, unexpected = fresh.load_state_dict(model.state_dict())
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_dense_ffn_default_activation_is_gelu_tanh(self):
        ffn = DenseFFN(hidden_size=8)
        self.assertIs(ffn.activation, ffn.net[1])
        self.assertIsInstance(ffn.activation, torch.nn.GELU)
        self.assertEqual(ffn.activation.approximate, "tanh")

    def test_dense_ffn_activation_is_configurable(self):
        ffn = DenseFFN(hidden_size=8, activation=torch.nn.SiLU())
        self.assertIs(ffn.activation, ffn.net[1])
        self.assertIsInstance(ffn.activation, torch.nn.SiLU)

    def test_dense_ffn_rejects_uninstantiated_activation_config(self):
        with self.assertRaises(TypeError):
            DenseFFN(hidden_size=8, activation={"_target_": "torch.nn.SiLU"})

    def test_hydra_dit_smoke_config_materializes(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["experiment=imagenet256_latent_dit_smoke"],
            )

        self.assertEqual(cfg.dataset._target_, "datasets.ImageNetLatentMMapDataset")
        self.assertEqual(cfg.model._target_, "models_dit.ClassCondDiT")
        self.assertTrue(cfg.model._recursive_)
        self.assertEqual(cfg.model.pos_embedding._target_, "models_dit.RoPE2D")
        self.assertEqual(cfg.model.backbone._target_, "models_dit.DiTBackbone")
        self.assertEqual(
            cfg.model.backbone.use_gradient_checkpointing,
            cfg.model.use_gradient_checkpointing,
        )
        self.assertGreater(len(cfg.model.backbone.blocks), 0)
        for block in cfg.model.backbone.blocks:
            self.assertEqual(block._target_, "models_dit.DiTBlock")
            self.assertEqual(block.hidden_size, cfg.model.backbone.hidden_size)
            self.assertEqual(block.attention._target_, "models_dit.SelfAttention")
            self.assertEqual(block.attention.hidden_size, block.hidden_size)
            self.assertFalse(hasattr(block.attention, "pos_embedding"))
            self.assertEqual(block.ffn._target_, "models_dit.DenseFFN")
            self.assertEqual(block.ffn.hidden_size, block.hidden_size)
            self.assertEqual(block.ffn.activation._target_, "torch.nn.GELU")
            self.assertEqual(block.ffn.activation.approximate, "tanh")

        model = instantiate(cfg.model)
        self.assertIsInstance(model, ClassCondDiT)
        self.assertIsInstance(model.backbone, DiTBackbone)
        self.assertEqual(len(model.backbone.blocks), len(cfg.model.backbone.blocks))
        self.assertIsInstance(model.pos_embedding, RoPE2D)
        for block in model.backbone.blocks:
            self.assertIsInstance(block, DiTBlock)
            self.assertIsInstance(block.attention, SelfAttention)
            self.assertFalse(hasattr(block.attention, "pos_embedding"))
            self.assertIsInstance(block.ffn, DenseFFN)
            self.assertIsInstance(block.ffn.activation, torch.nn.GELU)
            self.assertEqual(block.ffn.activation.approximate, "tanh")
            self.assertEqual(block.hidden_size, model.backbone.hidden_size)
            self.assertEqual(model.pos_embedding.base, cfg.model.pos_embedding.base)

        self.assertEqual(cfg.model.in_ch, cfg.dataset.latent_shape[0])
        self.assertEqual(cfg.model.latent_size, cfg.dataset.latent_shape[-1])
        self.assertEqual(cfg.dataset.latent_shape, cfg.sample_logger.latent_shape)
        self.assertEqual(cfg.dataset.latent_shape, cfg.inference.sampler.latent_shape)
        self.assertEqual(cfg.model.num_classes, cfg.dataset.num_classes)
        self.assertEqual(cfg.model.num_classes, cfg.inference.class_sampler.num_classes)
        self.assertIsNotNone(cfg.training.p_uncond)

    def test_hydra_deferred_masking_smoke_config_materializes(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["experiment=imagenet256_latent_dit_masked_smoke"],
            )

        self.assertEqual(cfg.model._target_, "models_dit.ClassCondDeferredMaskingDiT")
        self.assertTrue(cfg.model._recursive_)
        self.assertEqual(cfg.model.patch_mixer._target_, "models_dit.PatchMixer")
        self.assertEqual(cfg.model.masker._target_, "models_dit.RandomTokenMasker")
        self.assertEqual(cfg.model.masker.mask_ratio, 0.75)
        self.assertEqual(cfg.training.loss_mode, LossMode.masked_mse)

        model = instantiate(cfg.model)
        self.assertIsInstance(model, ClassCondDeferredMaskingDiT)
        self.assertIsInstance(model.patch_mixer, PatchMixer)
        self.assertIsInstance(model.masker, RandomTokenMasker)
        self.assertIsInstance(model.backbone, DiTBackbone)

        x = torch.randn(2, *cfg.dataset.latent_shape)
        labels = torch.tensor([0, model.null_token])
        model.train()
        out = model(x, torch.rand(2), labels, return_aux=True)
        self.assertEqual(out["pred"].shape, x.shape)
        self.assertEqual(out["loss_mask"].shape, (2, 1, 32, 32))

        from inference import guided_euler_sample

        model.eval()
        samples = guided_euler_sample(
            model, x, num_steps=2, cond=labels, guidance_scale=2.0
        )
        self.assertEqual(samples.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
