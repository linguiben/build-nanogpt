# Mini GPT-2 Smoke Test Design

Date: 2026-08-15

## Goal

Add a self-contained, small-scale GPT-2 training path under `mini_gpt2/` that verifies the complete local training workflow in roughly 2–5 minutes. Existing project files remain unchanged.

This is an engineering smoke test, not a statistical reproduction of OpenAI GPT-2 124M. Success means that GPT-2-style training, evaluation, checkpointing, checkpoint reload, and autoregressive generation all work locally.

## Files and boundaries

All new implementation and generated artifacts stay under `mini_gpt2/`:

- `mini_gpt2/train.py`: model, data loader, training loop, evaluation, checkpoint reload check, and generation.
- `mini_gpt2/README.md`: exact commands, expected output, and interpretation.
- `mini_gpt2/.gitignore`: ignores the generated `out/` directory.
- `mini_gpt2/out/model.pt`: generated checkpoint; not committed.

The script reads the existing repository-level `input.txt` without modifying or copying it. No existing source or configuration file will be edited.

## Model

The model retains the defining GPT-2 architecture:

- decoder-only Transformer;
- learned token and positional embeddings;
- pre-layer-normalized residual blocks;
- causal multi-head self-attention;
- GELU MLP with 4x expansion;
- tied token-embedding and language-model-head weights;
- GPT-2 tokenizer and padded vocabulary size of 50,304.

Smoke-test defaults:

- 2 layers;
- 2 attention heads;
- embedding width 128;
- context length 64;
- batch size 4;
- 100 optimizer steps.

This is approximately 7 million parameters. CLI flags may override steps, batch size, sequence length, learning rate, and device without editing the file.

## Data flow

`input.txt` is encoded with `tiktoken`'s GPT-2 encoding. The token stream is split deterministically into 90% training and 10% validation data. Each batch samples contiguous next-token prediction windows using a seeded PyTorch generator.

The script resolves `input.txt` relative to the repository, so it works when launched either from the repository root or from `mini_gpt2/`.

## Runtime behavior

Device selection order is CUDA, MPS, then CPU when `--device auto` is used. CUDA may use bfloat16 autocast when supported; MPS and CPU use float32 to avoid backend-specific autocast issues.

The run performs these stages:

1. Print environment, device, model configuration, parameter count, and token counts.
2. Evaluate deterministic initial train and validation losses.
3. Train for the configured number of steps and periodically print loss, elapsed time, and token throughput.
4. Evaluate final train and validation losses.
5. Save `mini_gpt2/out/model.pt` with model weights, configuration, optimizer step, losses, and seed.
6. Load the checkpoint into a fresh model and compare logits on a fixed validation batch.
7. Generate a short sample from a fixed prompt and seed.
8. Print a final PASS/FAIL summary and return a nonzero exit status on failure.

## Validation criteria

The default smoke test passes only when all of the following hold:

- initial and final losses are finite;
- final fixed-probe training loss is lower than initial fixed-probe training loss;
- final validation loss is lower than initial validation loss;
- checkpoint reload logits match the pre-save logits within a maximum absolute difference of `1e-5`;
- generated token IDs remain in vocabulary range and generation reaches the requested length;
- the checkpoint exists and is readable.

The README will explain that generated prose may remain poor after only 100 steps and that HellaSwag is not meaningful for this tiny model. Loss reduction and checkpoint equivalence are the authoritative smoke-test signals.

## Error handling

The script reports clear errors for a missing `input.txt`, insufficient tokens, an unavailable explicitly requested device, non-finite loss, checkpoint reload mismatch, or failed validation criteria. It does not silently fall back when the user explicitly requests CUDA or MPS.

## Verification commands

Primary command from the repository root:

```bash
.venv/bin/python mini_gpt2/train.py
```

Fast CPU-compatible diagnostic override:

```bash
.venv/bin/python mini_gpt2/train.py --device cpu --steps 10
```

Implementation verification will include syntax compilation, CLI help, a short diagnostic run, and the default smoke test on the best device available to the execution environment.
