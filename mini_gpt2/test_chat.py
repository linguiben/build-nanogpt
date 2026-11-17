"""交互测试已训练的 Mini GPT-2，输入一个前缀并生成 50 个新 token。

从仓库根目录运行：.venv/bin/python mini_gpt2/test_chat.py
默认使用 MPS；也可添加 --device cpu、--device cuda 或 --device auto。
输入 /quit、按 Ctrl-D 或 Ctrl-C 退出。每次输入独立续写，不保留历史对话。
只加载 out/model.pt 进行推理，不重新训练或覆盖检查点。
"""

import argparse
import os
from pathlib import Path

import tiktoken
import torch

from train import GPT, GPTConfig, select_device


def main():
    parser = argparse.ArgumentParser(description='Mini GPT-2 交互续写测试')
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu', 'auto'], default='mps')
    args = parser.parse_args()
    device = select_device(args.device)
    folder = Path(__file__).resolve().parent
    checkpoint_path = folder / 'out' / 'model.pt'
    if not checkpoint_path.is_file():
        parser.error('找不到 mini_gpt2/out/model.pt，请先运行 train.py 训练模型')

    # 根据检查点中的配置还原网络，再加载参数；eval 与 inference_mode 分别控制
    # 模块的推理行为和梯度记录，不会更新权重。
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(folder / 'out' / 'tokenizer_cache'))
    enc = tiktoken.get_encoding('gpt2')
    rng = torch.Generator().manual_seed(42)
    print(f'设备：{device}；已加载训练 {checkpoint["step"]} 步的模型。', flush=True)
    print('输入英文前缀进行续写；输入 /quit 退出。每次输入独立，不保留对话历史。')

    while True:
        try:
            prompt = input('你：')
        except EOFError:
            break
        if prompt.strip() == '/quit':
            break
        if not prompt.strip():
            continue

        prompt_ids = enc.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            for _ in range(50):
                # 只保留模型支持的上下文窗口；输出分数形状为 [1, T, V]。
                logits, _ = model(x[:, -model.config.block_size:])
                # 取最后一个位置并排除填充 ID，temperature=0.8 控制随机性。
                scores = logits[:, -1, :enc.n_vocab] / 0.8
                values, ids = torch.topk(scores, k=50, dim=-1)
                probs = torch.softmax(values, dim=-1)
                # 抽样使用独立的 CPU 随机数生成器，模型前向仍在 device 执行。
                choice = torch.multinomial(probs.cpu(), 1, generator=rng).to(device)
                next_id = torch.gather(ids, -1, choice)
                x = torch.cat((x, next_id), dim=1)

        # 只解码新生成部分，提示词已经在输入处显示。
        continuation = x[0, len(prompt_ids):].tolist()
        print('模型续写：', enc.decode(continuation), '\n', flush=True)
    print('\n已退出。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已退出。')
