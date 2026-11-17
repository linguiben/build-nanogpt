最快的方式是分两步：先在当前 Mac 上做 smoke test；真正从零复现 GPT‑2 124M，换到 Linux + NVIDIA 多卡机器。
当前环境是 Apple Silicon，Python 3.12.8，但 torch/numpy/tiktoken/transformers/datasets 都尚未安装。

## 1. 安装依赖
```bash
cd /Users/jupiter/19.git-repo/build-nanogpt

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install torch numpy tiktoken transformers datasets tqdm requests
```
*验证安装：*
```bash
python - <<'PY'
import torch, transformers, datasets, tiktoken
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
PY
```

## 2. 几分钟验证环境和 GPT‑2 基线
```bash
# 评估预训练 GPT‑2 在 HellaSwag 数据集上的表现
python hellaswag.py -m gpt2 -d mps
```

## 学习材料
> 1. [学习材料-来自：雪梅AI学习日记](https://getgaoding.feishu.cn/docx/Dw5OddGbNo3XmxxYAhachld1nKb)
> 2. [Karpathy大神三个半小时LLM入门课程【雪梅AI学习日记】](https://getgaoding.feishu.cn/docx/NllPdSue1oS0BGxgjG1c0fnWn9b)
> 3. [OpenAI GPT‑2: 1.5B release](https://openai.com/index/gpt-2-1-5b-release/)
> 4. [Andrej Karpathy讲座：深入探讨ChatGPT类大语言模型](https://www.bilibili.com/video/BV1WnNHeqEFK/?spm_id_from=888.80997.embed_other.whitelist&bvid=BV1WnNHeqEFK&vd_source=4635a9015e41611a3328c0bf4258ebf5)
> 5. [Karpathy Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
> 6. [Kaggle Datastes Chinese Population](https://www.kaggle.com/datasets?search=Chinese+Population)
> 7. [How Large Language Models Work](https://www.marovi.ai/How_Large_Language_Models_Work/zh)

[1]: https://github.com/linguiben/build-nanogpt
[2]: https://github.com/linguiben/gpt-2/tree/master/src
