import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from rl.reward import FixedClassReward, TargetClassReward


class FakeClassifier(nn.Module):
    def forward(self, x):
        base = torch.tensor(
            [
                [0.0, 2.0, -1.0],
                [3.0, 1.0, -2.0],
            ],
            device=x.device,
        )
        return base[: x.size(0)]


class RewardTest(unittest.TestCase):
    def test_target_class_reward_uses_prompt_labels(self):
        with patch("rl.reward.load_classifier", return_value=FakeClassifier()):
            reward = TargetClassReward("unused.pt", device="cpu")

        x = torch.zeros(2, 1, 4, 4)
        prompts = torch.tensor([1, 0])
        actual = reward(x, prompts)
        expected = torch.log_softmax(FakeClassifier()(x), dim=-1)[
            torch.arange(2), prompts
        ]

        torch.testing.assert_close(actual, expected)

    def test_fixed_class_reward_ignores_prompt_labels(self):
        with patch("rl.reward.load_classifier", return_value=FakeClassifier()):
            reward = FixedClassReward("unused.pt", target_class=1, device="cpu")

        x = torch.zeros(2, 1, 4, 4)
        prompts = torch.tensor([0, 2])
        actual = reward(x, prompts)
        expected = torch.log_softmax(FakeClassifier()(x), dim=-1)[:, 1]

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
