"""从 Transformer 理论到 GPT-2 训练：按下面的路线阅读本文件。

一、先读 GPTConfig / TrainConfig：区分模型结构与训练规模。
二、再读 main 中的数据准备、DataLoaderLite：文本 → token ID → (x, y)。
三、沿 GPT.forward → Block → Attention / MLP 追踪张量形状。
四、回到主循环：前向预测 → 交叉熵 → backward → AdamW 更新参数。
五、最后读验证、生成和保存；DDP、预训练权重导入可在第二遍阅读。

符号：B=一次前向传播的序列数，T=每条序列的 token 数，C=隐藏维度，
      H=注意力头数，D=C/H，V=词表大小，L=Transformer Block 层数。
主链路：x[B,T] → embedding[B,T,C] → L 个 Block → logits[B,T,V]。
GPT-2 是仅含解码器的因果语言模型；这里没有编码器或交叉注意力。

从项目根目录运行（需安装 torch/numpy/tiktoken；完整评测另需 transformers）：
  python train_gpt2.py --preset smoke --device mps
  python train_gpt2.py --preset smoke --device mps --steps 200 --batch-size 2
  python train_gpt2.py --preset full   # 原始 124M 规模，需预先准备 FineWeb 分片
  torchrun --standalone --nproc_per_node=8 train_gpt2.py --preset full

smoke 从随机权重开始训练小模型，不会下载 GPT-2 模型或 FineWeb。
首次调用 tiktoken 可能下载 GPT-2 分词器；可用 TIKTOKEN_CACHE_DIR 指定已有缓存。
输出默认写入 out/train_gpt2_smoke/；完整训练仍使用 log/。
100 步只能验证训练流程，不代表复现 GPT-2 的能力，也不是对话微调。

配置对照：              full               smoke
  L / H / C            12 / 12 / 768       2 / 2 / 128
  B / T                64 / 1024           4 / 64
  每次更新的 token 数   524288              B*T*进程数（单设备 256）
  更新次数              19073               100
  语料                  FineWeb 分片        本地 input.txt
  HellaSwag             开启                关闭

学习实验：先跑默认 smoke，再只修改一个参数观察结果。
  --steps 200：增加更新次数；--batch-size 2：减少单次前向占用的内存。
  --total-batch-size 1024：默认 B=4,T=64、单设备时累积 4 次梯度再更新。
  --n-layer 4 --n-head 4 --n-embd 256：增加模型容量（同时增加内存与计算量）。
调整模型结构后应重新训练；不要把不同结构的 checkpoint 混用。
"""

import argparse
import os
import math
import time
import inspect
from contextlib import nullcontext
from dataclasses import asdict, dataclass
# 本机 PyTorch 2.13 的 MPS 词表输出层反向传播出现数值差异；配合下面显式 mm，
# 使用 Metal 矩阵内核后通过 CPU/MPS 梯度对照。CUDA/CPU 路径不受此选项影响。
# 需在第一次 MPS 矩阵运算前设置；用户已设置的环境变量优先。
# 官方说明：https://docs.pytorch.org/docs/stable/mps_environment_variables.html
os.environ.setdefault('PYTORCH_MPS_PREFER_METAL', '1')
import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------------------------------------------------
# 模块 1：模型的积木。学习时先看 Block，再展开 Attention 和 MLP。

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 把理论中的 W_Q、W_K、W_V 合成一个 Linear；输出最后一维是 3C。
        # PyTorch Linear 的 weight 形状为 [out_features, in_features]。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 各头拼接后，再乘理论中的 W_O；它不是词表预测层。
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # 仅记录头数和隐藏维度；本实现没有加入 dropout。
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # 每个 token 已经是一个 C 维向量，不再是整数 ID。
        # ① [B,T,C] → [B,T,3C] → 三个 [B,T,C]。
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # ② 拆成 H 个头：[B,T,C] → [B,T,H,D] → [B,H,T,D]。
        # 124M 配置：C=768，H=12，所以每头 D=64。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # ③ 等价于 softmax(QKᵀ / sqrt(D) + 因果掩码)V。
        # is_causal=True 禁止位置 t 看到 t+1 之后的 token，避免偷看训练答案。
        # PyTorch 按设备选择内核；调用 SDPA 并不意味着所有设备都使用 FlashAttention。
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # ④ 拼回各头；transpose 后先 contiguous，再 view，确保内存布局可重塑。
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """逐位置前馈网络：[B,T,C] → [B,T,4C] → [B,T,C]。

    所有位置共享同一组参数；位置之间的信息交换由注意力完成。
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """一个 Pre-LN Transformer 层：归一化在子层之前，残差连接在子层之外。"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # 先利用上下文；残差路径帮助梯度传播。
        x = x + self.mlp(self.ln_2(x))  # 再处理每个位置的特征；两步均保持 [B,T,C]。
        return x

@dataclass
class GPTConfig:
    """模型结构：决定参数形状；从检查点恢复时必须与保存的结构一致。"""
    block_size: int = 1024 # 位置嵌入表长度，即模型最多能看多少个 token
    vocab_size: int = 50257 # 50,000 次 BPE 合并 + 256 个 byte token + 1 个特殊 token
    n_layer: int = 12 # L：串联多少个 Transformer Block
    n_head: int = 12 # H：注意力头数，必须整除 n_embd
    n_embd: int = 768 # C：每个 token 的隐藏向量长度


# 模块 2：训练配置。修改这里或命令行参数，就能调整演练规模。
# 模型大小主要由 L、C 决定；显存/统一内存还受 B、T、词表输出和优化器状态影响。
@dataclass
class TrainConfig:
    B: int = 64
    T: int = 1024
    total_batch_size: int | None = 524288 # 每次 optimizer.step() 对应的总 token 数
    max_steps: int = 19073 # 参数更新次数；不是 epoch 数
    warmup_steps: int = 715
    max_lr: float = 6e-4
    eval_interval: int = 250
    val_loss_steps: int = 20
    checkpoint_interval: int = 5000
    data_root: str = "edu_fineweb10B"
    text_file: str | None = None # 有文本路径时，改用本地小语料，前 90% 训练、后 10% 验证
    log_dir: str = "log"
    eval_hellaswag: bool = True
    use_compile: bool = False # 初学时关闭，便于逐行理解；MPS 演练也保持关闭


def parse_config(argv=None):
    """先选一组默认值，再覆盖少量学习时经常调整的参数。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=['full', 'smoke'], default='full')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--steps', type=int, help='优化器更新次数；缩短时也会缩短预热')
    parser.add_argument('--batch-size', type=int, help='B：一个 micro-batch 中的序列数')
    parser.add_argument('--seq-len', type=int, help='T：序列长度，同时决定位置嵌入表长度')
    parser.add_argument('--total-batch-size', type=int, help='每次更新的总 token 数，须整除 B*T*进程数')
    parser.add_argument('--n-layer', type=int)
    parser.add_argument('--n-head', type=int)
    parser.add_argument('--n-embd', type=int)
    parser.add_argument('--log-dir', help='日志和权重输出目录；重跑时同名文件会覆盖')
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    # 50304 是为计算对齐而补齐的词表容量；实际 GPT-2 tokenizer 仍只有 50257 个 ID。
    model_cfg = GPTConfig(vocab_size=50304)
    if args.preset == 'smoke':
        cfg = TrainConfig(B=4, T=64, total_batch_size=None, max_steps=100,
                          warmup_steps=10, max_lr=1e-3, eval_interval=25,
                          val_loss_steps=4, checkpoint_interval=100,
                          text_file='input.txt', log_dir='out/train_gpt2_smoke',
                          eval_hellaswag=False)
        model_cfg = GPTConfig(block_size=64, vocab_size=50304, n_layer=2, n_head=2, n_embd=128)
    for arg_name, field in [('steps', 'max_steps'), ('batch_size', 'B'), ('seq_len', 'T'),
                            ('total_batch_size', 'total_batch_size'), ('log_dir', 'log_dir')]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, field, value)
    for field in ['n_layer', 'n_head', 'n_embd']:
        value = getattr(args, field)
        if value is not None:
            setattr(model_cfg, field, value)
    model_cfg.block_size = cfg.T
    if min(cfg.B, cfg.T, cfg.max_steps, model_cfg.n_layer, model_cfg.n_head, model_cfg.n_embd) <= 0:
        parser.error('batch、序列长度、训练步数、层数、头数和隐藏维度必须为正数')
    if model_cfg.n_embd % model_cfg.n_head != 0:
        parser.error('n_embd 必须能被 n_head 整除')
    if cfg.total_batch_size is not None and cfg.total_batch_size <= 0:
        parser.error('total_batch_size 必须为正数')
    # 例如只演练 5 步时，不能继续使用完整训练的 715 步预热。
    cfg.warmup_steps = min(cfg.warmup_steps, cfg.max_steps - 1)
    return cfg, model_cfg, args.device


# 模块 3：把积木装配成 GPT，并定义 next-token prediction 的训练目标。

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token ID 查表 [V,C]
            wpe = nn.Embedding(config.block_size, config.n_embd), # 可学习的位置表 [block_size,C]
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # L 层各自有独立参数
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 输入嵌入与输出分类器共享同一个 [V,C] 参数，不是复制两份相同初值。
        # 这样可减少参数量；反向传播时两条使用路径的梯度都汇入这个参数。
        self.transformer.wte.weight = self.lm_head.weight

        # 随机初始化≠加载预训练权重。每层 residual 输出使用较小标准差。
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 每个 Block 有注意力和 MLP 两条残差支路，共 2L 条。
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # ① 输入 idx[B,T] 是整数 token ID；每一行是一条独立的上下文。
        # 训练时用真实前缀（teacher forcing）并行预测全部 T 个位置；生成时才逐 token 循环。
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # ② token 含义 + token 位置；pos_emb[T,C] 自动广播到 B 条序列。
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # ③ 同一张量顺序经过 L 层，每层都保持 [B,T,C]。
        for block in self.transformer.h:
            x = block(x)
        # ④ 每个位置预测下一个 token：输出未归一化分数 logits，而非概率。
        x = self.transformer.ln_f(x)
        if x.device.type == 'mps':
            # 与 lm_head(x) 相同的 X @ Wᵀ；显式二维 mm 才会使用上面的 Metal 选项。
            # 这是设备兼容处理，不改变权重共享、模型结构或训练目标。
            logits = torch.mm(x.reshape(B * T, -1), self.lm_head.weight.t())
            logits = logits.view(B, T, self.config.vocab_size)
        else:
            logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # ⑤ targets 已由数据加载器取后一个 token 对齐，此处不要再错位。
            # 合并 B、T，把 B*T 个位置作为分类样本；cross_entropy 内含 log_softmax。
            # loss 是所有位置的平均负对数似然，越低表示给正确下一个词的概率越高。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """选读：加载 Hugging Face 已训练权重，不属于本脚本默认的从零训练路径。

        HF 的 Conv1D 权重存储方向与 nn.Linear 相反，下面四类矩阵需转置。
        小模型的 L/H/C 与 GPT-2 不一致，不能直接装入 124M 的权重。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        # AdamW 保存参数的一阶/二阶动量；训练占用的内存不只有权重文件那么大。
        # 矩阵参数做权重衰减；bias 和 LayerNorm 的一维参数不做。
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
# 模块 4：准备监督信号。语料只需文本，标签就是每个位置的“下一个 token”。
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # 磁盘 uint16 节约空间；转为整数索引供 Embedding/CE 使用。
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    """按顺序取连续 token；既支持 FineWeb .npy 分片，也支持已编码的小文本。

    例如 B=2,T=3，buf=[10,11,12,13,14,15,16]：
    x=[[10,11,12],[13,14,15]]，y=[[11,12,13],[14,15,16]]。
    x[:,t] 的输出负责预测 y[:,t]；因果掩码保证不能直接看到答案。
    DDP 各进程从同一批数据的不同区段取样，避免重复训练同一段。
    """
    def __init__(self, B, T, process_rank, num_processes, split,
                 data_root='edu_fineweb10B', tokens=None, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        self.tokens = tokens
        self.shards = None
        if tokens is None:
            shards = sorted(s for s in os.listdir(data_root) if split in s and s.endswith('.npy'))
            self.shards = [os.path.join(data_root, s) for s in shards]
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # 每次验证都从固定位置开始，便于比较训练前后的 loss。
        self.current_shard = 0
        if self.shards is not None:
            self.tokens = load_tokens(self.shards[self.current_shard])
        self._check_size()
        self.current_position = self.B * self.T * self.process_rank

    def _check_size(self):
        if len(self.tokens) < self.B * self.T * self.num_processes + 1:
            raise ValueError('数据太短：每个 split/分片至少需要 B*T*进程数+1 个 token')

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # 输入，丢掉最后一个 token
        y = (buf[1:]).view(B, T) # 标签，丢掉第一个 token：始终比输入靠后一个位置
        # 前进一个所有进程共同消耗的数据块；不要把 micro-batch 与 optimizer step 混淆。
        self.current_position += B * T * self.num_processes
        # 各 rank 用同一组起点判断边界；尾部不够一整组时一起换分片/从头再读。
        next_group_start = self.current_position - B * T * self.process_rank
        if next_group_start + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.shards is not None:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self._check_size()
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# 模块 5（选读）：HellaSwag 是四选一评测，不参与训练，不执行 backward。
# 比较四个候选续写的平均负对数似然，只统计 mask=1 的续写区域。
# 此处传入完整文本，所以需要现场右移；与已经给出 (x,y) 的训练入口不同。

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# 模块 6：训练基础设施。先理解单设备分支，再阅读 DDP 多卡分支。
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def precision_context(device_type):
    # CUDA 保留原来的 BF16 混合精度；MPS/CPU 使用 FP32，方便小规模学习和排查。
    # MPS 不是 CPU，不能通过把 device_type 写成 "cpu" 来给 MPS 开启 autocast。
    if device_type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def get_lr(it, cfg):
    # 先线性预热，再余弦衰减到峰值的 10%；it 从 0 开始计数。
    if it < cfg.warmup_steps:
        return cfg.max_lr * (it + 1) / cfg.warmup_steps
    min_lr = cfg.max_lr * 0.1
    if it >= cfg.max_steps:
        return min_lr
    decay_ratio = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (cfg.max_lr - min_lr)


def main():
    cfg, model_cfg, requested_device = parse_config()

    # 步骤 1：选设备、设随机种子。torchrun 会设置 RANK/LOCAL_RANK/WORLD_SIZE。
    # rank=进程编号；每个 CUDA 进程负责一张 GPU，并通过 DDP 同步梯度。
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP 分支需要 CUDA"
        if requested_device not in {'auto', 'cuda'}:
            raise ValueError('DDP 分支只支持 --device auto/cuda')
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        device = requested_device
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('当前 Python 环境不可使用 MPS，请检查 PyTorch/运行环境或使用 --device cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('当前 Python 环境不可使用 CUDA')

    master_process = ddp_rank == 0
    device_type = device.split(':')[0]
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # 步骤 2：区分 micro-batch 和一次参数更新所使用的有效 batch。
    # 总 token 数 = B * T * DDP 进程数 * 梯度累积次数。
    # full 单卡：64*1024*1*8 = 524288；smoke 单卡：4*64*1*1 = 256。
    # 若只缩小 B、T 却保留 524288，累积次数反而会增多，单步工作量仍然很大。
    B, T = cfg.B, cfg.T
    tokens_per_micro_step = B * T * ddp_world_size
    if cfg.total_batch_size is None:
        cfg.total_batch_size = tokens_per_micro_step
    if cfg.total_batch_size % tokens_per_micro_step != 0:
        raise ValueError('total_batch_size 必须能被 B*T*DDP进程数整除')
    grad_accum_steps = cfg.total_batch_size // tokens_per_micro_step
    if master_process:
        print(f"using device: {device}")
        print(f"model config: {model_cfg}")
        print(f"training config: {cfg}")
        print(f"total desired batch size: {cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # 步骤 3：文本只编码一次。分词器是固定工具，不会随着模型一起训练。
    # 前后切分避免训练/验证直接使用相同位置；每次验证重置游标，比较同一批文本。
    enc = tiktoken.get_encoding('gpt2')
    train_tokens = val_tokens = None
    if cfg.text_file is not None:
        with open(cfg.text_file, 'r', encoding='utf-8') as f:
            tokens = torch.tensor(enc.encode(f.read()), dtype=torch.long)
        split_at = int(0.9 * len(tokens))
        train_tokens, val_tokens = tokens[:split_at], tokens[split_at:]
        if master_process:
            print(f"tokens: train={len(train_tokens):,}, val={len(val_tokens):,}")
    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'train',
                                 cfg.data_root, train_tokens, master_process)
    val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'val',
                               cfg.data_root, val_tokens, master_process)

    # 步骤 4：创建随机初始化模型和优化器。默认不会调用 from_pretrained。
    torch.set_float32_matmul_precision('high')
    raw_model = GPT(model_cfg).to(device)
    model = raw_model
    if master_process:
        print(f"parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    # compile 是可选的运行优化，与 Transformer 数学定义无关；本文件中启用后跳过评测/采样。
    if cfg.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    optimizer = raw_model.configure_optimizers(0.1, cfg.max_lr, device_type, master_process)

    # 仅完整训练按需导入 HellaSwag；smoke 不会因此下载或遍历 10,042 道评测题。
    if cfg.eval_hellaswag and not cfg.use_compile:
        from hellaswag import render_example, iterate_examples

    log_file = os.path.join(cfg.log_dir, 'log.txt')
    if master_process:
        os.makedirs(cfg.log_dir, exist_ok=True)
        with open(log_file, 'w') as f:
            pass

    # 步骤 5：先测随机模型，再更新参数，定期验证，最后保存“完成全部更新”的权重。
    # step 表示已完成的 optimizer.step 次数；最后一轮只验证/保存，不再做第 max_steps+1 次更新。
    for step in range(cfg.max_steps + 1):
        last_step = step == cfg.max_steps
        should_eval = step % cfg.eval_interval == 0 or last_step

        # 5a. 验证集只衡量泛化，不参与学习：eval() 与 no_grad() 含义不同。
        # eval() 切换 Dropout 等层的行为（本实现没有 Dropout）；no_grad() 关闭梯度记录。
        # 均匀猜测时 loss 约为 ln(50304)=10.83；观察固定验证集是否降低比只看训练 loss 更可靠。
        if should_eval:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.zeros((), device=device)
                for _ in range(cfg.val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with precision_context(device_type):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / cfg.val_loss_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:5d} | validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % cfg.checkpoint_interval == 0 or last_step):
                    checkpoint_path = os.path.join(cfg.log_dir, f'model_{step:05d}.pt')
                    # state_dict 是参数名→张量的映射；config 用普通字典保存，便于安全加载。
                    # 此检查点供推理/检查权重；严格断点续训还需优化器、RNG、数据游标等状态。
                    torch.save({
                        'model': raw_model.state_dict(),
                        'config': asdict(raw_model.config),
                        'train_config': asdict(cfg),
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }, checkpoint_path)
                    print(f"saved checkpoint: {checkpoint_path}")

        # 5b. 选读：完整 HellaSwag 评测；与验证 loss 一样，不执行 backward。
        if should_eval and cfg.eval_hellaswag and not cfg.use_compile:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples('val')):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with precision_context(device_type):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                counts = torch.tensor([num_total, num_correct_norm], dtype=torch.long, device=device)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                num_total, num_correct_norm = counts.tolist()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # 5c. 自回归生成：只取最后位置的预测，采一个 token，拼回输入，再预测下一个。
        if step > 0 and should_eval and not cfg.use_compile:
            model.eval()
            num_return_sequences, max_length = 4, 32
            prompt = enc.encode("Hello, I'm a language model,")
            xgen = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0).repeat(num_return_sequences, 1)
            # 使用独立 CPU 随机数生成器：兼容 MPS，也避免采样影响训练的 RNG。
            sample_rng = torch.Generator(device='cpu').manual_seed(42 + ddp_rank)
            with torch.no_grad():
                while xgen.size(1) < max_length:
                    with precision_context(device_type):
                        logits, _ = model(xgen[:, -model_cfg.block_size:])
                    # 50304 中多出的占位 ID 没有 tokenizer 对应文本，生成时必须排除。
                    probs = F.softmax(logits[:, -1, :enc.n_vocab].float(), dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs.cpu(), 1, generator=sample_rng).to(device)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                print(f"rank {ddp_rank} sample {i}: {enc.decode(xgen[i].tolist())}")

        if last_step:
            break

        # 步骤 6：核心 训练循环负责更新权重。一次 step 的伪代码：
        # 清空旧梯度 → 重复[取 batch、前向、loss、反向累积] → 梯度裁剪 → 调学习率 → 更新权重。
        t0 = time.time() # 从这里计时，验证/生成时间不计入训练吞吐量。
        model.train()
        optimizer.zero_grad() # backward 默认累加梯度，所以每次参数更新前必须清空。
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # 只在最后一个 micro-batch 同步梯度，省去中间不必要的通信。
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with precision_context(device_type):
                _, loss = model(x, y)
            # backward 累加梯度；除以累积次数，使最终梯度对应 batch 平均值而非和。
            loss = loss / grad_accum_steps
            loss_accum += loss.detach() # 日志只记录数值，不保留整张计算图。
            loss.backward() # 链式法则计算 d(loss)/d(parameter)，此时权重还没有改变。
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if not torch.isfinite(loss_accum).item():
            raise RuntimeError(f'训练 loss 非有限值，step={step}')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        # norm 是裁剪前的总梯度范数，不是“权重大小”；裁剪用于限制一次更新的梯度幅度。
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step() # AdamW 使用梯度与动量真正修改模型参数。.
        if device_type == 'cuda':
            torch.cuda.synchronize()
        elif device_type == 'mps':
            torch.mps.synchronize() # GPU 异步执行，计时前等待计算完成。
        dt = time.time() - t0
        tokens_per_sec = cfg.total_batch_size / dt
        if master_process:
            print(f"step {step + 1:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, 'a') as f:
                f.write(f"{step + 1} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()


# import 本文件只定义模型/配置，便于实验和测试；直接运行脚本才启动训练。
if __name__ == '__main__':
    main()
