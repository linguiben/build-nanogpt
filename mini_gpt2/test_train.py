"""训练性质测试；MPS 可用时额外验证真实词表大小下的前向与反向数值。"""

import unittest

import torch

from train import GPT, GPTConfig, get_batch, evaluate


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT(GPTConfig(block_size=8, vocab_size=32, n_embd=16, n_head=2))

    def test_batch_shift_and_repeatability(self):
        tokens = torch.arange(100)
        args = (tokens, 4, 8)
        x, y = get_batch(*args, torch.Generator().manual_seed(7), 'cpu')
        repeated, _ = get_batch(*args, torch.Generator().manual_seed(7), 'cpu')
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(x.dtype, torch.long)
        torch.testing.assert_close(x, repeated)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        torch.testing.assert_close(y, x + 1)

    def test_causality(self):
        self.model.eval()
        x = torch.randint(32, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 32
        with torch.no_grad():
            before, _ = self.model(x)
            after, _ = self.model(changed)
        self.assertEqual(before.shape, (2, 8, 32))
        torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0, atol=1e-6)

    def test_weight_tying_and_parameter_count(self):
        self.assertIs(self.model.lm_head.weight, self.model.transformer.wte.weight)
        C, V, T, L = 16, 32, 8, 2
        expected = V*C + T*C + L*(12*C*C + 13*C) + 2*C
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)

    def test_backward_and_optimizer_learn_fixed_batch(self):
        x = torch.randint(32, (2, 8))
        y = torch.randint(32, (2, 8))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = self.model(x, y)[1].item()
        for _ in range(5):
            optimizer.zero_grad()
            loss = self.model(x, y)[1]
            loss.backward()
            self.assertTrue(torch.isfinite(self.model.transformer.wte.weight.grad).all())
            optimizer.step()
        self.assertLess(self.model(x, y)[1].item(), initial)

    def test_evaluation_preserves_weights_and_existing_gradients(self):
        x = torch.randint(32, (2, 8))
        y = torch.randint(32, (2, 8))
        self.model(x, y)[1].backward()
        weights = [p.detach().clone() for p in self.model.parameters()]
        gradients = [p.grad.clone() for p in self.model.parameters()]
        first = evaluate(self.model, [(x, y)])
        self.assertEqual(evaluate(self.model, [(x, y)]), first)
        self.assertFalse(self.model.training)
        for p, weight, gradient in zip(self.model.parameters(), weights, gradients):
            torch.testing.assert_close(p, weight, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)


@unittest.skipUnless(torch.backends.mps.is_available(), '当前环境不可使用 MPS')
class MPSParityTests(unittest.TestCase):
    def test_repeated_backward_matches_cpu_with_real_vocab_size(self):
        # 不能仅用 32 个 token 的玩具词表：本机异常出现在真实大小的词表输出层。
        torch.manual_seed(1337)
        cpu = GPT(GPTConfig())
        mps = GPT(GPTConfig()).to('mps')
        mps.load_state_dict(cpu.state_dict())
        x = torch.arange(256).reshape(4, 64) % 100
        y = (x + 1) % 100
        cpu_logits, cpu_loss = cpu(x, y)
        cpu_loss.backward()
        reference = {name: p.grad.clone() for name, p in cpu.named_parameters()}
        # 不更新权重而重复反向，捕捉先前“相同输入下梯度交替异常”的问题。
        for _ in range(4):
            mps.zero_grad(set_to_none=True)
            logits, loss = mps(x.to('mps'), y.to('mps'))
            torch.testing.assert_close(logits.cpu(), cpu_logits, rtol=1e-4, atol=1e-5)
            loss.backward()
            for name, p in mps.named_parameters():
                torch.testing.assert_close(p.grad.cpu(), reference[name], rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
