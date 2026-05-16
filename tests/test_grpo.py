import copy
import unittest

import torch
import torch.nn as nn

from config import SamplerConfig
from rl.grpo import compute_group_advantage, gaussian_kl_mu, grpo_loss
from rl.sde_sampler import recompute_logprobs, sde_rollout


class TinyClassCondPolicy(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.null_token = num_classes
        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.class_bias = nn.Embedding(num_classes + 1, 1)

    def forward(self, x, t, cond=None):
        if cond is None:
            cond = torch.full((x.size(0),), self.null_token, device=x.device)
        time_bias = t.view(-1, 1, 1, 1)
        class_bias = self.class_bias(cond).view(-1, 1, 1, 1)
        return self.conv(x) + time_bias + class_bias


class GRPOTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.device = torch.device("cpu")
        self.sampler = SamplerConfig(
            T_rollout=4,
            sigma_a=0.3,
            t_min=0.1,
            t_max=0.9,
            guidance_scale=1.25,
        )
        self.policy = TinyClassCondPolicy().to(self.device)
        self.prompts = torch.tensor([0, 0, 1, 1], device=self.device)

    def rollout(self):
        return sde_rollout(
            self.policy,
            self.prompts,
            self.sampler,
            latent_shape=(1, 4, 4),
            device=self.device,
        )

    def test_rollout_shapes_and_finite_logprobs(self):
        traj = self.rollout()

        self.assertEqual(traj.xs.shape, (5, 4, 1, 4, 4))
        self.assertEqual(traj.actions.shape, (4, 4, 1, 4, 4))
        self.assertEqual(traj.log_probs_old.shape, (4, 4))
        self.assertEqual(traj.means.shape, (4, 4, 1, 4, 4))
        self.assertEqual(traj.stds.shape, (4,))
        self.assertTrue(torch.isfinite(traj.log_probs_old).all())

    def test_recompute_logprobs_matches_unchanged_policy(self):
        traj = self.rollout()
        logprobs, mus = recompute_logprobs(
            self.policy, traj, self.sampler, no_grad=False
        )

        torch.testing.assert_close(logprobs, traj.log_probs_old)
        torch.testing.assert_close(mus, traj.means)

    def test_kl_is_zero_against_copied_policy(self):
        traj = self.rollout()
        ref_policy = copy.deepcopy(self.policy)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

        _, new_mus = recompute_logprobs(self.policy, traj, self.sampler)
        _, ref_mus = recompute_logprobs(ref_policy, traj, self.sampler, no_grad=True)
        kl = gaussian_kl_mu(new_mus, ref_mus, traj.stds)

        self.assertLessEqual(kl.item(), 1e-12)

    def test_policy_gradient_flows_and_reference_stays_frozen(self):
        traj = self.rollout()
        ref_policy = copy.deepcopy(self.policy)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

        rewards = torch.tensor([0.0, 2.0, 3.0, 1.0], device=self.device)
        advantage = compute_group_advantage(rewards, G=2)
        new_logprobs, new_mus = recompute_logprobs(
            self.policy, traj, self.sampler, no_grad=False
        )
        _, ref_mus = recompute_logprobs(
            ref_policy, traj, self.sampler, no_grad=True
        )
        self.assertFalse(ref_mus.requires_grad)

        loss, _ = grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=traj.log_probs_old,
            new_mus=new_mus,
            ref_mus=ref_mus,
            sigma_step=traj.stds,
            advantage=advantage,
            clip_eps=0.2,
            kl_beta=0.04,
        )
        loss.backward()

        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in self.policy.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)
        self.assertTrue(all(p.grad is None for p in ref_policy.parameters()))

    def test_group_advantage_is_normalized_within_each_group(self):
        rewards = torch.tensor([1.0, 2.0, 2.0, 4.0, -1.0, 1.0])
        advantage = compute_group_advantage(rewards, G=2).view(-1, 2)

        torch.testing.assert_close(advantage.mean(dim=1), torch.zeros(3))
        torch.testing.assert_close(advantage.std(dim=1), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
