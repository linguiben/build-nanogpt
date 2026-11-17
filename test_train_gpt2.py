"""根目录教学训练脚本的回归测试：python -m unittest test_train_gpt2 -v。"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from train_gpt2 import DataLoaderLite, GPT, GPTConfig, TrainConfig, get_lr, parse_config


class CommandLineTests(unittest.TestCase):
    def test_help_does_not_start_training(self):
        # --help 必须只说明用法，不初始化模型、读取 FineWeb 或启动训练。
        result = subprocess.run(
            [sys.executable, "train_gpt2.py", "--help"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            env=os.environ.copy(), timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--preset", result.stdout)
        self.assertNotIn("using device:", result.stdout)


class DataTests(unittest.TestCase):
    def test_next_token_labels_reset_and_wrap(self):
        loader = DataLoaderLite(2, 3, 0, 1, 'train', tokens=torch.arange(13))
        x, y = loader.next_batch()
        self.assertEqual(x.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(y.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(loader.next_batch()[0].tolist(), [[6, 7, 8], [9, 10, 11]])
        torch.testing.assert_close(loader.next_batch()[0], x) # 读完回绕
        loader.reset()
        torch.testing.assert_close(loader.next_batch()[0], x) # 验证重置后仍是相同文本

    def test_distributed_ranks_keep_distinct_batches_and_wrap_together(self):
        loaders = [DataLoaderLite(1, 2, rank, 2, 'train', tokens=torch.arange(9)) for rank in range(2)]
        self.assertEqual([loader.next_batch()[0].tolist() for loader in loaders], [[[0, 1]], [[2, 3]]])
        self.assertEqual([loader.next_batch()[0].tolist() for loader in loaders], [[[4, 5]], [[6, 7]]])
        self.assertEqual([loader.next_batch()[0].tolist() for loader in loaders], [[[0, 1]], [[2, 3]]])

    def test_fineweb_shards_advance_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / 'train_000.npy', np.arange(5, dtype=np.uint16))
            np.save(Path(folder) / 'train_001.npy', np.arange(10, 15, dtype=np.uint16))
            loader = DataLoaderLite(1, 4, 0, 1, 'train', data_root=folder, master_process=False)
            self.assertEqual(loader.next_batch()[1].tolist(), [[1, 2, 3, 4]])
            self.assertEqual(loader.next_batch()[1].tolist(), [[11, 12, 13, 14]])
            loader.reset()
            self.assertEqual(loader.next_batch()[0].tolist(), [[0, 1, 2, 3]])


class TrainingTests(unittest.TestCase):
    def test_short_run_learning_rate_is_finite_and_reaches_floor(self):
        cfg, model_cfg, _ = parse_config(['--preset', 'smoke', '--steps', '1', '--seq-len', '8'])
        self.assertEqual(model_cfg.block_size, 8)
        self.assertGreater(get_lr(0, cfg), 0)
        self.assertAlmostEqual(get_lr(1, cfg), cfg.max_lr * 0.1)
        cfg = TrainConfig(max_steps=10, warmup_steps=2, max_lr=0.001)
        self.assertAlmostEqual(get_lr(0, cfg), 0.0005)
        self.assertAlmostEqual(get_lr(1, cfg), 0.001)
        self.assertAlmostEqual(get_lr(10, cfg), 0.0001)

    def test_model_is_causal_and_learns_with_accumulated_gradients(self):
        torch.manual_seed(42)
        model = GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16))
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        x = torch.randint(32, (2, 8))
        y = torch.randint(32, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 32
        with torch.no_grad():
            before, _ = model(x)
            after, _ = model(changed)
        self.assertEqual(before.shape, (2, 8, 32))
        torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0, atol=1e-6)

        # 两个 micro-batch 的平均梯度，应等于合并 batch 的梯度。
        model(x, y)[1].backward()
        combined = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        for i in range(2):
            (model(x[i:i+1], y[i:i+1])[1] / 2).backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, combined[name], rtol=1e-4, atol=1e-6)

        optimizer = model.configure_optimizers(0.1, 0.01, 'cpu', master_process=False)
        initial = model(x, y)[1].item()
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), initial)


@unittest.skipUnless(torch.backends.mps.is_available(), '当前环境不可使用 MPS')
class MPSParityTests(unittest.TestCase):
    def test_repeated_backward_matches_cpu_with_real_vocab_size(self):
        # 小词表无法复现本机输出层反向问题；保留 50304 词表，模型使用 smoke 尺寸。
        torch.manual_seed(1337)
        cfg = GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=128)
        cpu_model = GPT(cfg)
        mps_model = GPT(cfg).to('mps')
        mps_model.load_state_dict(cpu_model.state_dict())
        x = torch.randint(50257, (4, 64))
        y = torch.randint(50257, (4, 64))
        cpu_logits, cpu_loss = cpu_model(x, y)
        cpu_loss.backward()
        expected = {name: p.grad.clone() for name, p in cpu_model.named_parameters()}
        for repeat in range(4):
            mps_model.zero_grad()
            logits, loss = mps_model(x.to('mps'), y.to('mps'))
            torch.testing.assert_close(logits.cpu(), cpu_logits, rtol=1e-4, atol=1e-5)
            loss.backward()
            for name, p in mps_model.named_parameters():
                with self.subTest(repeat=repeat, parameter=name):
                    torch.testing.assert_close(p.grad.cpu(), expected[name], rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
