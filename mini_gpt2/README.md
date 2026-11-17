# Mini GPT-2 教学构建目录
- [How Large Language Models Work](https://www.marovi.ai/How_Large_Language_Models_Work/zh)

## 当前阶段

本目录用于逐步构建一个约 700 万参数的 GPT-2 风格小模型，以便在本机快速验证训练、评估、检查点和文本生成的完整流程。

`train.py` 已实现完整训练闭环，并保留分节中文讲解和伪代码。默认从随机权重开始，使用仓库根目录的 `input.txt`，训练 100 步。

## 文件结构

```text
mini_gpt2/
├── DESIGN.md          # 完整目标、模型规格、运行流程和验收标准
├── SCAFFOLD_PLAN.md   # 本次教学脚手架的执行计划
├── README.md          # 目录说明和推荐学习顺序
├── train.py           # 模型、数据、训练、评估、重载和生成；附中文讲解
├── test_train.py      # 数据对齐、因果性、权重共享与反向传播测试
└── .gitignore         # 忽略生成的输出和 Python 本地缓存
```

运行训练后生成：

```text
mini_gpt2/out/model.pt
mini_gpt2/out/metrics.json
```

`model.pt` 保存模型权重、配置、训练步数和损失；`metrics.json` 记录设备、逐步 loss、前后评估、生成样例和通过状态。重复运行会更新这两个文件。检查点支持推理重载，不包含精确断点续训所需的优化器状态。

## 代码阅读顺序

`train.py` 按学习依赖关系分成八个部分：

1. **配置与命令行参数**：集中描述模型规模、训练参数和可覆盖选项。
2. **运行设备选择**：按 CUDA、MPS、CPU 的顺序选择可用设备。
3. **数据准备**：读取根目录 `input.txt`，使用 GPT-2 tokenizer，并构造下一 token 预测 batch。
4. **GPT-2 模型组件**：实现因果注意力、MLP、Transformer Block 和完整模型。
5. **训练与评估**：记录初始损失，执行优化循环，再比较最终损失。
6. **检查点校验**：保存 `model.pt`，重新加载并比较 logits。
7. **文本生成**：使用固定提示词和随机种子验证自回归推理链路。
8. **主流程**：按固定顺序串联上述阶段并输出 PASS/FAIL。

## 计划采用的默认规模

| 项目 | 默认值 |
|---|---:|
| Transformer 层数 | 2 |
| 注意力头数 | 2 |
| 嵌入宽度 | 128 |
| 上下文长度 | 64 tokens |
| Batch size | 4 |
| 优化步数 | 100 |
| 词表大小 | 50,304 |
| 参数量 | 约 700 万 |

这些参数用于工程冒烟测试，不用于复现 OpenAI GPT-2 124M 的最终能力。

## 验证信号

默认运行只有同时满足以下条件才报告通过，并返回退出码 0；失败时返回 1：

- 初始和最终损失均为有限数值；
- 固定训练探针的最终损失低于初始损失；
- 最终验证损失低于初始验证损失；
- 检查点重新加载前后的 logits 最大绝对误差不超过 `1e-5`；
- 文本生成达到要求长度，且所有 token ID 均在词表范围内；
- `mini_gpt2/out/model.pt` 存在并可重新读取。

仅训练 100 步后的文本可能仍然不连贯，因此损失下降和检查点等价性才是冒烟测试的主要判断依据。HellaSwag 对这个微型短训练模型没有参考意义。

## 运行命令

从仓库根目录使用 MPS 训练（FP32，默认学习率 `1e-3`）：

```bash
.venv/bin/python -u mini_gpt2/train.py --device mps
```

若指定的 MPS 不可用，脚本直接报错；使用 `--device auto` 则按 CUDA → MPS → CPU 选择。分词器默认缓存在 `mini_gpt2/out/tokenizer_cache/`（可通过 `TIKTOKEN_CACHE_DIR` 覆盖），首次缺少缓存时可能需要下载，无需下载预训练 GPT-2 权重。生成时仅把 50 个候选的抽样放到 CPU，以使用独立随机数生成器，模型前向与反向仍在所选设备上运行。

可用 `--steps`、`--batch-size`、`--seq-len`（1–64）、`--lr` 调整训练；步骤过少时 loss 下降检查可能不通过。

运行小尺寸 CPU 单元测试：

```bash
.venv/bin/python -m unittest discover -s mini_gpt2 -p test_train.py -v
```

## 启动聊天测试
```bash
.venv/bin/python mini_gpt2/test_chat.py
```
完整设计见 [DESIGN.md](./DESIGN.md)。
