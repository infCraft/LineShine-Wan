# Training throughput knobs

Single-bucket baseline: 1.3B Wan T2V, 256×256 / 49 frames → ~3328 tokens/sample on
8×A800-80GB (Ampere, no FP8). Sequence is short, so attention is not the bottleneck;
the levers are the data pipeline, per-GPU batch (MFU), and `torch.compile`.

Measured: baseline 4.5 s/step (eff batch 64, MFU ~20%) → ~2.1 s/step (MFU ~35%) with
the defaults below — ~2.1×, with the effective batch (64) unchanged.

## Knobs in `slurm/train_8gpu_w5.sbatch` (env-overridable)

| env | default | meaning |
|-----|---------|---------|
| `BATCH_SIZE` | 2 | per-GPU micro-batch. `eff batch = BATCH_SIZE * GRAD_ACCUM_STEPS * 8`. |
| `GRAD_ACCUM_STEPS` | 4 | keep `eff batch = 64` if you change `BATCH_SIZE`. |
| `NUM_WORKERS` | 6 | DataLoader workers per rank (cpus-per-gpu=8). |
| `PREFETCH_FACTOR` | 4 | batches prefetched per worker. |
| `COMPILE` | 1 | `torch.compile` the model (fixed-shape single-bucket only). `0` to disable. |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported to reduce fragmentation.

## Memory ceiling (no gradient checkpointing)

Per-GPU peak (compute-only): bs1 40GB, bs2 55GB, bs3 69GB, bs4 84GB. **bs4 OOMs under
DDP.** Stay at bs2 (safe) — MFU plateaus there anyway. `--grad-checkpointing` is a no-op
(WanModel has no `gradient_checkpointing_enable`); add `torch.utils.checkpoint` if you
need it for higher resolution.

## torch.compile + checkpoints

`torch.compile` is applied before DDP. `src/train/ckpt.py: unwrap_model()` strips both the
DDP `.module` and compile `._orig_mod` wrappers so saved state_dict keys stay plain (no
`_orig_mod.` prefix) and remain compatible with the sampler and resume. Fixed shapes mean
no recompiles; multi-bucket / variable resolution will recompile per shape — re-evaluate then.

## Profiling

`sbatch slurm/profile_speed.sbatch` (single GPU) reports compute-only / data-only /
end-to-end throughput and an approximate MFU across a batch-size sweep. Add
`COMPILE_FLAG=--compile` to include the compiled compute path.

## Muon optimizer (B-class, experimental)

`src/train/train.py` now accepts `--optimizer muon`; the default remains `--optimizer adamw`.
Muon mode uses one combined optimizer so checkpointing, scheduler stepping, zero-grad, and
LR logging keep the same single-optimizer interface. Hidden 2D attention/FFN weight matrices
go to Muon, while norms, biases, embeddings, head, modulation, conv, projection, and all 1D
parameters stay on AdamW. Tune Muon with `--muon-lr` (default `0.02`) and `--muon-momentum`
(default `0.95`); auxiliary AdamW uses the normal `--lr` and `--weight-decay`. Treat this as
fixed-shape single-bucket only until multi-shape throughput and compile behavior are retested.
