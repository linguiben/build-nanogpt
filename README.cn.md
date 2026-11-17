# 仓库文件
build-nanogpt/  
├── [train_gpt2.py](./train_gpt2.py)       ← 最重要  
├── [fineweb.py](./fineweb.py)          ← 数据准备  
├── [hellaswag.py](./hellaswag.py)        ← benchmark，用于测试模型的推理能力  
└── [input.txt](./input.txt)           ← 小数据测试

# build nanoGPT

[English](README.md) | 中文

本仓库保存了对 [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) 的从零复现实现。提交历史被刻意保持为逐步且清晰，方便你通过查看 git 提交记录来一步步理解实现过程。作者还配有一段在 YouTube 上的[讲解视频](https://youtu.be/l8pRSuU81PU)，演示并解释每一次提交中的关键点。

我们基本上从一个空文件开始，逐步复现出与 GPT-2（124M）相当的模型。如果你愿意投入更多时间和预算，代码也可以用于复现 GPT-3 级别的模型。虽然 GPT-2（124M）在原始训练时可能耗时较久（2019 年，大约 5 年前），但现在复现实验大致需要 ~1 小时与 ~$10（视所用硬件与云价格而定）。如果本地没有足够的 GPU，推荐使用云 GPU（例如 Lambda）。

注意：GPT-2 与 GPT-3 本质上是简单的语言模型，训练语料来自互联网文本，它们的行为就是“生成互联网文本”。因此本仓库与视频并不涵盖 Chat 风格的微调（SFT）或对话系统——那属于之后的内容。就目前而言，经过 10B tokens 训练的 124M 模型可能会输出类似下面的文字：

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

而在 40B tokens 之后，输出可能变成：

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

好啦。视频发布后，这里也会作为 FAQ 与勘误位置，欢迎大家提出问题与修正。

有关讨论请使用仓库的 Discussions（讨论）页，或加入作者的 Zero To Hero Discord（频道 **#nanoGPT**）：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 视频

[Let's reproduce GPT-2 (124M) YouTube lecture](https://youtu.be/l8pRSuU81PU)

## 勘误

- 有一次小的清理遗漏了 `register_buffer` 的 bias，在切换到 flash attention 后需要删除该缓冲区，已在近期的 PR 中修复。
- 早期版本的 PyTorch 在从 `uint16` 转换到 `long` 时可能会出问题。因此在 `load_tokens` 中，我们使用 `npt = npt.astype(np.int32)` 先用 numpy 把 `uint16` 转为 `int32`，再转换为 torch tensor，然后再转为 `long`。
- `torch.autocast` 接受一个名为 `device_type` 的参数，我曾尝试直接传入完整 `device`（例如 `cuda:3`），但 PyTorch 需要的是设备类型字符串（如 `cuda`），因此要把 `cuda:3` 之类的值裁剪为 `cuda`。目前 `mps`（Apple Silicon）可能会被错误解释为 CPU 的 `device_type`，具体行为视 PyTorch 版本而异。
- `model.require_backward_grad_sync` 实际上会影响前向与反向过程。把它提到更靠前的位置以确保也在前向中生效。

## 适合生产的仓库推荐

以下仓库实现更贴近生产级别的跑法，推荐参考：

- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## 常见问题（FAQ）

(如有更多问题请在 Discussions 提问)

## 许可证

MIT
