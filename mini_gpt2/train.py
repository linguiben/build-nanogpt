"""Mini GPT-2：可运行训练脚本，以及中文讲解与伪代码。

阅读目标：跟踪一段文本如何变成 token、预测分数和损失，再通过梯度更新模型参数。
建议先读第 3 节的数据例子，再读第 4 节的 forward，最后串起第 5 节的一次训练步。
各节注释说明算法意图，相邻的 Python 代码实现对应步骤。
从仓库根目录运行：.venv/bin/python mini_gpt2/train.py --device mps
完整目标和验收标准见同目录的 DESIGN.md；完整规模的实现可对照 ../train_gpt2.py。

模型训练六部曲：
  1. 建模
  2. 准备数据
  3. 模型初始化，准备训练
  4. 正向传播计算loss
  5. 反向传播更新权重
  6. 验证与评估
其中单步训练（4+5）是核心循环: 
  get_batch → forward → loss → backward → optimizer.step。
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
# MPS 兼容设置：本机 PyTorch 2.13 在大词表 Linear 的反向传播中出现过错误梯度，
# 即使前向 loss 正常也不能证明训练正确。这里与 GPT.forward 中的显式二维 mm 配合，
# 使用经过 CPU/MPS 梯度对照验证的 Metal 矩阵内核。仅设置此变量、仍调用 Linear 不够。
# 必须在首次 MPS 矩阵运算前设置；CPU/CUDA 路径不受影响。
# setdefault 尊重外部设置；若外部显式设为 0，就不会启用这里验证过的内核选择。
# https://docs.pytorch.org/docs/stable/mps_environment_variables.html
os.environ.setdefault('PYTORCH_MPS_PREFER_METAL', '1')
import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# 1. 配置与命令行参数：先明确模型要处理什么尺寸的数据
# =============================================================================
# 原始 GPT-2 124M 使用 L=12、H=12、C=768、最大上下文长度=1024。
# 本模型约 700 万参数，用较小规模学习相同的核心计算过程。
#
# 1.1 统一符号，后面所有 shape 都按这个约定阅读。
# B = batch size，一次并行处理的序列数；默认 4。
# T = 当前序列长度（上下文长度）；训练默认 64，生成时会逐步增长。
# V = 模型输出的词表大小；这里为 50,304。
# C = 每个 token 的隐藏表示维度；默认 128。
# L = Transformer 层数；默认 2。
# H = 注意力头数；默认 2；每头维度 D = C/H = 64。
# token ID 是整数编号；隐藏表示则是每个 token 对应的 C 个可学习浮点特征。

@dataclass
class GPTConfig:
    block_size: int = 64 # 最大上下文长度
    vocab_size: int = 50304 # GPT-2 tokenizer 实际有 50,257 个 ID；填充到 50,304 个 ID，便于在 GPU 上按 64 的倍数分配内存。这里并没有增加新的文字 token。生成时需要屏蔽额外输出位置，才能安全解码。
    n_layer: int = 2 # Transformer 层数
    n_head: int = 2 # 注意力头数
    n_embd: int = 128 # 隐藏维度

# 1.2 设置小模型参数，并允许未来通过命令行覆盖部分训练设置。
# 伪代码：
#   config = {层数: 2, 头数: 2, 隐藏维度: 128, 最大上下文长度: 64, 词表: 50304}
#   options = 读取命令行(步数=1000, batch_size=4, 序列长度=64, 学习率, 设备=auto)
#   检查 C 能被 H 整除，且实际 T 不超过最大上下文长度

def parse_args():
    parser = argparse.ArgumentParser(description="Mini GPT-2 本地训练与验证")
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--steps', type=int, default=1000) # 训练步数，每步表示循环一次 optimizer 更新；每步消耗 B*T 个 token。
    parser.add_argument('--batch-size', type=int, default=4) # 一次训练的 batch size
    parser.add_argument('--seq-len', type=int, default=64) # 序列长度，决定位置嵌入表长度
    parser.add_argument('--lr', type=float, default=1e-3) # 学习率
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.seq_len <= 64:
        parser.error('steps 和 batch-size 必须为正数，seq-len 必须在 1..64 之间')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr 必须为有限正数')
    return args


# =============================================================================
# 2. 运行设备选择：让模型和输入在同一个计算设备上
# =============================================================================
# 2.1 按 cuda > mps > cpu 的顺序选择可用设备，或使用用户指定的设备。
#
# 2.2 决定数值精度，初始化模型后将其移动到 device。
#   MPS / CPU 使用 float32
#   CUDA 在支持时可以对前向计算启用 bfloat16 autocast
# token ID 与标签保持整数类型，不应随模型一起转换为浮点类型。
# 设备与精度是两件事：选择 MPS 不代表必须开启混合精度。

def select_device(requested):
    available = {
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu': True,
    }
    if requested == 'auto':
        return next(name for name in ('cuda', 'mps', 'cpu') if available[name])
    if not available[requested]:
        raise RuntimeError(f'指定的设备 {requested} 不可用；本次不会回退到其他设备')
    return requested

def synchronize(device):
    # GPU 运算异步提交，计时前后等待完成才能得到实际耗时。
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


# =============================================================================
# 3. 数据准备：用文本本身构造“预测下一个 token”的监督信号
# =============================================================================
# 3.1 定位并读取已有的 ../input.txt，不依赖启动脚本时的当前工作目录。
#   repo_root = 根据本文件的位置找到仓库根目录
#   text = 读取 repo_root / "input.txt"
#
# 3.2 编码为整数 token 流。
#   tokenizer = 获取 GPT-2 的 tiktoken 编码
#   tokens = tokenizer.encode(text)
#   确认 tokenizer.decode(tokens) == text
# token 可能是词、词的一部分、标点或空格；移动一个 token 不等于移动一个字符。
# tokenizer 是已有的编码规则，本次训练更新的是模型权重，不重新训练 tokenizer。
#
# 3.3 先切分，再分别采样，避免训练窗口跨入验证数据。
#   split = floor(len(tokens) * 0.9)
#   train_tokens = tokens[:split]
#   val_tokens = tokens[split:]
#   检查两个分区都至少包含 T+1 个 token
# 验证集用于衡量未参与梯度更新的文本上的预测效果。
#
# 3.4 构造 get_batch(split)：从相应分区采样 B 个长度为 T+1 的窗口。
#   stream = 根据 split 选择训练或验证 token 流
#   对 b 从 0 到 B-1：
#       start = 随机整数，满足 0 <= start <= len(stream)-T-1
#       x[b] = stream[start : start+T]
#       y[b] = stream[start+1 : start+T+1]
#   将 x、y 作为整数张量移动到 device；二者形状都为 [B,T]
#
# 例：B=1，T=4，窗口中的 ID 为 [10,20,30,40,50]。
#   输入 x = [[10,20,30,40]]
#   目标 y = [[20,30,40,50]]
#   位置 0：看到 10，预测 20
#   位置 1：看到 10,20，预测 30
#   位置 2：看到 10,20,30，预测 40
#   位置 3：看到 10,20,30,40，预测 50
# 标签由文本偏移得到，不需要人工标注；未来位置通过因果掩码对当前预测隐藏。
#
# 3.5 固定用于前后对比的训练探针和验证 batch。
#   train_probe = 固定训练窗口
#   val_probe = 固定验证窗口，可包含多个 batch
# 评估不消耗训练采样器的随机状态，避免改变“下一步训练会抽到什么”。

def get_batch(tokens, batch_size, seq_len, rng, device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) < seq_len + 1:
        raise ValueError('数据分区至少需要 seq_len+1 个 token')
    starts = torch.randint(len(tokens) - seq_len, (batch_size,), generator=rng)
    x = torch.stack([tokens[s:s + seq_len] for s in starts.tolist()])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts.tolist()])
    return x.to(device), y.to(device)


# =============================================================================
# 4. GPT-2 模型组件：从 token ID 到每个候选 token 的分数
# =============================================================================
# Tokenizer 的词表和编码规则通常在训练前就已经确定（GPT-2 的 BPE），例如 "Hello" -> 15496 (Token ID)。
# 4.1 构建参数，采用随机初始化，从零开始学习。
#   token_embedding = 可学习矩阵 [V,C] # V 是词表大小 50304，C 是隐藏维度 128，每一个 token 会映射到一个 C 维的向量。
#   position_embedding = 可学习矩阵 [最大上下文长度,C] # T 是支持的最大上下文长度 64, C 是隐藏维度 128
#   blocks = L 个 Transformer Block
#   final_layer_norm = 最后一层归一化
#   lm_head = 从 C 维映射到 V 维的无偏置线性层
#   lm_head.weight 与 token_embedding.weight 绑定为同一份参数
# 权重共享降低参数量；输入查表和输出预测会共同更新这份权重。
# LayerNorm 的缩放通常初始化为 1、偏置为 0，线性层偏置初始化为 0。
# 线性层与嵌入权重采用标准差 0.02 的正态初始化；每条残差分支的输出投影
# 再乘以 (2*L)^(-0.5) 缩放其初始化标准差，控制多层相加时的数值尺度。
#
# 4.2 Embedding：ID 查表，并加入位置信息。
#   positions = [0,1,...,T-1]
#   tok = token_embedding[x]                    # [B,T,C]
#   pos = position_embedding[positions]         # [T,C]
#   hidden = tok + pos                         # 广播相加 → [B,T,C]
# 仅有 token embedding 无法表达顺序；位置向量使同一 token 在不同位置具有不同表示。
#
# 4.3 因果多头自注意力：每个位置从已知前缀提取信息。
#   qkv = Linear(C → 3*C)(normalized_hidden)    # [B,T,3*C]
#   q,k,v = 沿最后一维分成三份                  # 各为 [B,T,C]
#   拆分头维并交换轴                           # 各为 [B,H,T,D]
#   scores = q @ transpose(k, 最后两维) / sqrt(D)  # [B,H,T,T]
#   对所有 j > i，将 scores[...,i,j] 设为负无穷
#   attention = softmax(scores, 沿最后一维)     # 每个查询位置对可见位置的权重
#   context = attention @ v                    # [B,H,T,D]
#   合并各头                                   # [B,T,C]
#   attention_output = Linear(C → C)(context) # [B,T,C]
# Q 表示当前位置要查询什么，K 表示各位置如何被匹配，v 提供被汇总的信息。
# 注意这里小写 v 是 value 张量，与表示词表大小的大写 V 不同。
#
# T=4 时，因果掩码允许访问的列为：
#   查询位置 0 → 位置 0
#   查询位置 1 → 位置 0,1
#   查询位置 2 → 位置 0,1,2
#   查询位置 3 → 位置 0,1,2,3
# scores 轴上的行是查询位置 i，列是被查询位置 j；只屏蔽严格的未来 j>i。
# 实际实现可用 scaled_dot_product_attention(..., is_causal=True)，
# 无需手动保存完整注意力矩阵；上面的展开用于理解计算含义。
#
# 4.4 MLP：对每个位置的特征做非线性变换，不在这里混合不同位置。
#   expanded = Linear(C → 4*C)(normalized_hidden)  # [B,T,4*C]
#   activated = GELU(expanded)
#   mlp_output = Linear(4*C → C)(activated)         # [B,T,C]
#
# 4.5 拼成 GPT-2 的 pre-norm Transformer Block。
#   hidden = hidden + Attention(LayerNorm_1(hidden))
#   hidden = hidden + MLP(LayerNorm_2(hidden))
# 两次相加都是残差连接，输入输出维度均为 [B,T,C]。
# Attention 交流不同位置的信息，MLP 处理每个位置的特征，LayerNorm 归一化最后一维。
#
# 4.6 forward(x)：串起全部组件。
#   hidden = token_embedding[x] + position_embedding[positions]
#   对每个 block：hidden = block(hidden)
#   hidden = final_layer_norm(hidden)
#   logits = hidden @ transpose(token_embedding.weight)  # [B,T,V]
#   返回 (logits, loss)；未提供 targets 时 loss=None
# logits 是未归一化的预测分数：每条序列、每个位置，都有 V 个候选的分数。
# 默认 batch 的形状变化：[4,64] → [4,64,128] → [4,64,50304]。


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.residual_projection = True

    # 4.3 因果多头自注意力：每个位置从已知前缀提取信息。
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q, k, v = [t.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v)]
        # is_causal=True 保证位置 t 只能看到 0..t 的输入。
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.residual_projection = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # 初始化完成后绑定，输入与输出共同更新同一个 Parameter。
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(module, 'residual_projection', False):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError('输入序列超过模型上下文长度')
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # [B,T,C]：每个位置准备好用于预测的隐藏表示。
        if x.device.type == 'mps':
            # lm_head 无 bias，因此数学定义就是 X @ Wᵀ，其中 W 的形状是 [V,C]。
            # [B,T,C] → [B*T,C]，再乘 [C,V] → [B*T,V]，最后还原 [B,T,V]。
            # 显式二维 mm 配合文件开头的 Metal 选项，避开本机 Linear 反向数值问题。
            # reshape 仅合并 batch/位置维，不混合位置，也不会解除因果掩码。
            logits = torch.mm(x.reshape(B * T, self.config.n_embd), self.lm_head.weight.t())
            logits = logits.reshape(B, T, self.config.vocab_size)
        else:
            logits = self.lm_head(x)
        # 两条路径共用原有 lm_head.weight；没有新建/复制 Parameter，权重共享仍然成立。
        # 不使用 detach 或 no_grad，梯度可从输出层继续传回全部 Transformer 层。
        loss = None
        if targets is not None:
            # targets 已由 get_batch 取后一个 token 对齐，不要在此再次移动标签。
            # 交叉熵接收原始分数，并在内部计算 log_softmax；默认对 B*T 个位置取平均。
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# =============================================================================
# 5. 训练与评估：一次训练步怎样改变模型？
# =============================================================================
# 5.1 交叉熵将预测分数与正确的下一 token ID 对比。
#   logits, _ = model(x)                     # 返回二元组；此时第二项为 None
#   loss = F.cross_entropy(
#       logits.reshape(B*T,V),                # 每一行是一道 V 选 1 的预测题
#       y.reshape(B*T)                        # 每道题的正确 token ID
#   )
# 上面展开了数学计算；实际训练用 logits, loss = model(x, y)，forward 已计算相同的 loss。
# model(x) 会经过 nn.Module 的调用机制，不需要直接调用 model.forward(x)。
# cross_entropy 内部包含稳定的 log_softmax，不要先对 logits 做 softmax 再传入。
# 第 (b,t) 个位置的损失 = -log(模型分给正确 token y[b,t] 的概率)。
# 最终对 B*T 个位置取平均；B=4、T=64 时，一个 batch 提供 256 个预测目标。
# 若预测完全均匀，loss=ln(V)，这里约为 10.83；随机初始化时不一定恰好等于它。
#
# 为什么训练能同时计算所有位置？
#   已有完整真实文本，所有目标 y 都已知；用因果掩码限制每个位置能看到的输入，
#   就能在一次前向中并行计算 T 个下一 token 预测。训练时不必先采样一个 token
#   再继续算后面的位置，也不需要把 logits 转成离散 token 才能计算 loss。
#
# 5.2 准备 AdamW 优化器：只优化模型中需要梯度的参数。
#   parameters = list(model.parameters())    # 注意括号：取出 Parameter，不是传入方法对象
#   矩阵组 = [p for p in parameters if p.dim() >= 2]，weight_decay=0.1
#   向量组 = [p for p in parameters if p.dim() < 2]，weight_decay=0.0
#   optimizer = torch.optim.AdamW(上述两组, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
# 本模型所有参数均需要梯度；共享的 embedding/lm_head 参数只被枚举和更新一次。
# AdamW 为参数维护梯度一阶、二阶矩估计，以此决定更新量；学习率控制更新尺度。
#
# 5.3 训练前评估：给学习效果建立基准。
#   initial_train_loss = evaluate(model, train_probe)
#   initial_val_loss = evaluate(model, val_probe)
#   evaluate 的步骤：切换 eval 模式 → 关闭梯度记录 → 前向计算平均 loss
# eval() 改变 dropout 等层的行为，no_grad() 关闭梯度记录，两者作用不同。
# 验证阶段既不 backward()，也不 optimizer.step()。
#
# 5.4 训练循环：下面一轮对应一次 optimizer 更新。
#   对 step 从 1 到 args.steps（包含末尾；当前默认 1000 次更新）：
#       model.train()                          # 切换回训练模式
#       x,y = get_batch(train)                 # 抽取本步训练窗口
#       optimizer.zero_grad(set_to_none=True)  # 清除上一步的梯度，不是清空权重
#       logits, loss = model(x, y)             # 用当前权重计算分数和平均交叉熵
#       检查 loss 为有限数值；记录 loss
#       loss.backward()                        # 链式法则计算 d(loss)/d(参数)
#       clip_grad_norm_(model.parameters(),1.0) # 裁剪过大的总梯度范数
#       optimizer.step()                       # AdamW 根据梯度实际更新参数
#       定期输出 step、loss、耗时、tokens/秒
#
# backward() 只计算并累加 .grad；optimizer.step() 才改变权重。
# 新权重用于下一个 batch。单步 loss 可能上升，尤其更换 batch 后，不要求逐步单调下降。
# 测量 GPU 耗时时应在计时边界同步设备，否则异步提交时间不等于计算完成时间。
#
# 5.5 梯度累积扩展：用多个小 batch 实现更大的有效 batch，常用于显存不足时。
# 这是扩展原理，当前 mini 训练循环没有实现 K>1 的累积；当前每个 batch 更新一次。
#   optimizer.zero_grad()
#   对 micro_step 从 0 到 K-1：
#       x,y = get_batch(train)
#       _, loss = model(x, y)                 # 返回二元组，不能把 model(x) 直接交给 CE
#       (loss / K).backward()                 # 每次累加到同一组 .grad
#   clip_grad_norm_(model.parameters(),1.0)    # 对累积后的梯度裁剪一次
#   optimizer.step()                          # 累积 K 次，只更新一次
# 相同大小的 micro-batch 下，有效 token batch = B*T*K；除以 K 恢复平均尺度。
# 不要在 micro-batch 之间清空梯度，也不要每个 micro-batch 都更新参数。
# 对照 ../train_gpt2.py 的 grad_accum_steps；本地基础测试可先采用 K=1。
#
# 5.6 训练后评估：使用与训练前相同的探针和同样的评估模式。
#   final_train_loss = evaluate(model, train_probe)
#   final_val_loss = evaluate(model, val_probe)
# 固定训练探针下降说明拟合有所改善；验证损失下降提供对未训练文本的泛化证据。
# train loss 下降但 val loss 上升可能提示过拟合；小探针也存在采样误差。
# 此 smoke test 以两类固定探针 loss 均下降为通过条件，不能据此宣称复现 124M 能力。


@torch.no_grad() # 仍执行前向计算，但不构建供 backward 使用的计算图。
def evaluate(model, batches):
    """在固定、等大小的 batch 上求平均 loss，不更新权重或清空已有梯度。

    本函数结束后保留 eval 模式；主训练循环在每轮开头显式调用 model.train()。
    这里每个 batch 的 B、T 相同，所以 batch loss 的平均值等于所有 token loss 的平均值。
    若以后支持变长或带 padding 的 batch，则应按有效 token 数加权平均。
    """
    model.eval()
    return sum(model(x, y)[1].item() for x, y in batches) / len(batches)


# =============================================================================
# 6. 检查点保存与重新加载校验：权重能否完整保存并继续用于推理？
# =============================================================================
# 6.1 保存训练结果，只有实际运行训练程序时才创建输出目录与文件。
#   checkpoint = {
#       model: 模型 state_dict,
#       config: 构建模型所需的基础类型配置,
#       step: 已完成的优化步数,
#       losses: 初始与最终评估结果,
#       seed: 实验随机种子
#   }
#   保存 checkpoint 到 mini_gpt2/out/model.pt
# state_dict 保存权重和相关 buffer；配置描述架构，两者配合才能重建模型。
# 若要精确恢复中断的训练，还需要 optimizer 状态、当前随机数状态、数据位置等；
# 单独记录初始 seed 不能还原训练中途的状态，本次设计重点验证推理重载。
#
# 6.2 比较保存前后的输出，而不是仅检查文件是否存在。
#   model.eval()
#   关闭梯度记录：reference_logits = model(固定验证输入)
#   checkpoint = 加载刚才保存的文件
#   restored_model = 根据 checkpoint.config 新建模型并移动到相同 device
#   restored_model.load_state_dict(checkpoint.model)
#   restored_model.eval()
#   关闭梯度记录：restored_logits = restored_model(同一验证输入)
#   max_error = max(abs(reference_logits-restored_logits))
#   检查 max_error <= 1e-5
# 比较时统一设备、精度、输入和 eval 模式，避免 dropout 或精度不同干扰判断。


# 保存和重载的实际调用在第 8 节，紧接训练后评估，便于跟踪整个流程。


# =============================================================================
# 7. 自回归文本生成：利用学到的分布逐 token 续写
# =============================================================================
# 7.1 准备前缀和生成随机种子，并进入 eval/no_grad 模式。
#   generated = tokenizer.encode(固定提示词)，增加 batch 维，移动到 device
#
# 7.2 反复计算“当前前缀之后的下一个 token”。
#   重复预定的新增 token 数次：
#       context = generated 的最后最多 64 个 token  # 避免超出位置嵌入上限
#       logits = model(context)                   # [B,当前T,V]
#       next_logits = logits[:,-1,:]              # 只取最后位置 → [B,V]
#       屏蔽 next_logits 中 ID >= tokenizer.n_vocab 的填充位置
#       probs = softmax(next_logits, 沿词表维)
#       top_probs,top_ids = topk(probs,k=50)
#       candidate_index = multinomial(top_probs,每行抽取1个候选)
#       next_id = 按 candidate_index 从 top_ids 中取回真实 token ID
#       generated = 在序列维拼接 generated 和 next_id
#   检查 token ID 有效且生成长度符合预期，再解码输出
#
# 生成时新 token 尚未知，所以必须先预测一个，再把它加入下一轮的上下文。
# 本伪代码每轮重新计算前缀；KV cache 加速可在理解完整流程后另行学习。
# 100 步后文本仍可能重复或不连贯，生成环节主要检查推理链路。


@torch.no_grad()
def generate(model, tokenizer, device, seed, new_tokens=40):
    model.eval()
    # 抽样在 CPU 上用独立 generator，避免重置训练或模型初始化的随机状态。
    rng = torch.Generator().manual_seed(seed)
    prompt = tokenizer.encode("Hello, I'm a language model,")
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    for _ in range(new_tokens):
        logits, _ = model(x[:, -model.config.block_size:])
        scores = logits[:, -1, :tokenizer.n_vocab]  # 排除不可解码的填充 ID
        probs, ids = torch.topk(F.softmax(scores, dim=-1), 50, dim=-1)
        choice = torch.multinomial(probs.cpu(), 1, generator=rng).to(device)
        x = torch.cat((x, torch.gather(ids, -1, choice)), dim=1)
    tokens = x[0].tolist()
    valid = len(tokens) == len(prompt) + new_tokens and all(
        0 <= t < tokenizer.n_vocab for t in tokens)
    return tokenizer.decode(tokens), valid


# =============================================================================
# 8. 主流程与最终结果：把每一步连接成可验证的闭环
# =============================================================================
#   配置参数 → 选择设备与种子
#   → 读取文本、分词、切分数据、固定评估探针
#   → 随机初始化模型、绑定输入输出权重、建立优化器
#   → 初始评估
#   → 训练 100 步：batch → forward → loss → backward → optimizer.step
#   → 最终评估
#   → 保存检查点、构建新模型重载、比较 logits
#   → 自回归生成、解码短文本
#   → 输出检查结果
#
# 默认通过条件：
#   [ ] 初始和最终 loss 都为有限数值
#   [ ] final_train_loss < initial_train_loss
#   [ ] final_val_loss < initial_val_loss
#   [ ] 保存前后 logits 最大绝对误差 <= 1e-5
#   [ ] 生成 token ID 有效，且达到要求的长度
#   [ ] mini_gpt2/out/model.pt 存在并可读取
# 全部满足输出 PASS；否则报告失败原因并返回非零退出状态。
def main():
    args = parse_args()
    device = select_device(args.device) # 选择设备
    seed = 1337 # 固定随机种子，便于复现
    torch.manual_seed(seed)
    config = GPTConfig()
    print(f'device: {device} | dtype: float32 | torch: {torch.__version__}', flush=True)
    print(f'config: {asdict(config)}', flush=True)

    folder = Path(__file__).resolve().parent
    # 缓存在本目录持久保存，避免系统临时目录清理后重新下载；尊重用户已有的设置。
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(folder / 'out' / 'tokenizer_cache'))
    tokenizer = tiktoken.get_encoding('gpt2')
    text = (folder.parent / 'input.txt').read_text(encoding='utf-8')
    encoded = tokenizer.encode(text)
    if tokenizer.decode(encoded) != text:
        raise RuntimeError('tokenizer 编码/解码校验失败')
    tokens = torch.tensor(encoded, dtype=torch.long)
    split = int(len(tokens) * 0.9)
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    train_rng = torch.Generator().manual_seed(seed + 1)
    probe_rng = torch.Generator().manual_seed(seed + 2)
    train_probe = [get_batch(train_tokens, args.batch_size, args.seq_len, probe_rng, device)
                   for _ in range(4)]
    val_probe = [get_batch(val_tokens, args.batch_size, args.seq_len, probe_rng, device)
                 for _ in range(4)]
    model = GPT(config).to(device)
    # 对应 5.1：初始化模型权重
    parameters = list(model.parameters())  # 权重共享的 Parameter 只枚举一次
    # 对应 5.2：矩阵衰减用于正则化；bias/LayerNorm 的一维缩放参数不做衰减。
    # AdamW 还维护一阶/二阶动量，因此训练内存占用不只有模型权重这么大。
    optimizer = torch.optim.AdamW([
        {'params': [p for p in parameters if p.dim() >= 2], 'weight_decay': 0.1},
        {'params': [p for p in parameters if p.dim() < 2], 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    print(f'parameters: {sum(p.numel() for p in parameters):,} | '
          f'train tokens: {len(train_tokens):,} | val tokens: {len(val_tokens):,}', flush=True)
    # 对应 5.3：固定探针建立训练前基准，不消耗 train_rng 的随机状态。
    initial_train = evaluate(model, train_probe)
    initial_val = evaluate(model, val_probe)
    if not all(math.isfinite(v) for v in (initial_train, initial_val)):
        raise RuntimeError('初始评估 loss 非有限数值')
    print(f'initial train loss: {initial_train:.4f} | val loss: {initial_val:.4f}', flush=True)

    history = []
    synchronize(device)
    started = time.perf_counter()
    # 5.4：训练循环，逐步更新模型权重。
    for step in range(1, args.steps + 1):
        # ① 进入训练模式，启用 dropout 等层的训练行为
        model.train()
        # 取训练数据，y 是 x 的偏移一位的目标
        x, y = get_batch(train_tokens, args.batch_size, args.seq_len, train_rng, device)
        # ② 清除上轮 .grad；set_to_none=True 将梯度置为 None，不会清空模型权重，
        # 也不会清空 AdamW 内部的动量。下一次 backward 会重新创建梯度张量。
        optimizer.zero_grad(set_to_none=True)
        # ③ 前向传播与 loss 计算（交叉熵 cross_entropy）
        # model(x, y) 会经过 nn.Module 的调用机制，不需要直接调用 model.forward(x, y)。
        # _ 是 logits，这里表示暂时不需要保存 logits
        _, loss = model(x, y) 
        # item() 提取 Python 数字用于日志和检查，不改变原 loss 张量的计算图。
        # 记录的是本次更新之前的 loss；更新后是否改善由后续固定探针评估衡量。
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f'step {step}: loss 非有限数值')
        # ④ 反向传播，用链式法则计算梯度并写入 Parameter.grad；这一步还没有改变任何权重。
        loss.backward()
        # ⑤ 把全体参数的梯度视为一个向量，范数超过 1 时按同一比例缩小。
        # 返回的 norm 是裁剪前的范数；日志显示 >1 不表示裁剪失效。
        # 非有限值检查只捕捉 NaN/Inf；有限但错误的梯度需靠 CPU/MPS 对照测试发现。
        norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True) 
        # ⑥ AdamW 用梯度、历史动量、学习率及衰减项更新参数，更新模型权重。
        optimizer.step()
        history.append({'step': step, 'loss': value})
        if step == 1 or step % 10 == 0 or step == args.steps: # 每 10 步输出一次日志
            synchronize(device)
            elapsed = time.perf_counter() - started
            rate = step * args.batch_size * args.seq_len / elapsed
            print(f'step {step:3d}/{args.steps} | loss {value:.4f} | '
                  f'grad norm {norm.item():.4f} | {elapsed:.1f}s | {rate:.0f} tok/s', flush=True)
    synchronize(device) # 等待所有异步计算完成，确保计时准确
    train_seconds = time.perf_counter() - started
    # 对应 5.6：用相同探针重新评估，避免把不同 batch 的难度差异误认为学习效果。
    # 5.6.1：在训练集上评估模型
    final_train = evaluate(model, train_probe)
    final_val = evaluate(model, val_probe)
    print(f'final train loss: {final_train:.4f} | val loss: {final_val:.4f}', flush=True)

    # 保存模型和配置，便于后续重载验证。
    out = folder / 'out'
    out.mkdir(exist_ok=True)
    checkpoint_path = out / 'model.pt'
    with torch.no_grad():
        reference = model(val_probe[0][0])[0]
    torch.save({
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': asdict(config), 'step': args.steps, 'seed': seed,
        'val_loss': final_val, 'train_loss': final_train,
    }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    restored = GPT(GPTConfig(**checkpoint['config'])).to(device)
    restored.load_state_dict(checkpoint['model'])
    restored.eval()
    with torch.no_grad():
        max_error = (reference - restored(val_probe[0][0])[0]).abs().max().item()
    # 验证保存的模型能否继续生成文本，且生成的 token ID 有效。
    sample, generation_valid = generate(restored, tokenizer, device, seed + 3)
    checks = {
        'finite_losses': all(math.isfinite(v) for v in
                             (initial_train, initial_val, final_train, final_val)),
        'train_loss_decreased': final_train < initial_train,
        'val_loss_decreased': final_val < initial_val,
        'checkpoint_logits_match': math.isfinite(max_error) and max_error <= 1e-5,
        'generation_valid': generation_valid,
        'checkpoint_readable': checkpoint_path.is_file() and checkpoint['step'] == args.steps,
    }
    # 5.6.2：计算并保存训练和验证的指标
    metrics = {
        'device': device, 'dtype': 'float32', 'config': asdict(config),
        'options': vars(args), 'seed': seed, 'parameters': sum(p.numel() for p in parameters),
        'initial_train_loss': initial_train, 'final_train_loss': final_train,
        'initial_val_loss': initial_val, 'final_val_loss': final_val,
        'train_seconds': train_seconds, 'reload_max_abs_error': max_error,
        'checks': checks, 'sample': sample, 'history': history,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n',
                                    encoding='utf-8')
    print(f'checkpoint: {checkpoint_path}\nreload max error: {max_error:.8g}', flush=True)
    print(f'sample:\n{sample}', flush=True)
    for name, passed in checks.items():
        print(f'{"PASS" if passed else "FAIL"}: {name}', flush=True)
    passed = all(checks.values())
    print('SMOKE TEST: ' + ('PASS' if passed else 'FAIL'), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
