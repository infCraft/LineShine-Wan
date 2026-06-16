# LineShine

LineShine 是一个面向 Wan2.1 T2V-1.3B 的预训练 pipeline 仓库。这里保存代码、脚本、测试和文档，不保存原始数据、权重、cache、run、report 产物。

## 仓库结构

```text
code/
  src/          # 数据处理、训练、采样
  slurm/        # 集群作业脚本
  tests/        # 回归测试
  third_party/  # Wan2.1 子模块
  configs/      # 路径和环境模板
  docs/         # 运行和数据约定
  README.md
  requirements.txt
```

## 数据和权重

训练代码不直接读取 GitHub 上的原始视频。标准约定是把运行工作区放在 `LINESHINE_ROOT`，仓库放在 `LINESHINE_ROOT/code`，外部数据放在工作区旁边的目录中：

```bash
export LINESHINE_ROOT=/mnt/beegfs/home/huang_z/lineshine
export LINESHINE_CODE_ROOT=$LINESHINE_ROOT/code
export LINESHINE_SHARED_OPENVID_DIR=/mnt/beegfs/home/yezy/openvid
```

读数规则如下：

- `src/data/build_manifest.py` 读取 `LINESHINE_ROOT/data/openvid/meta/OpenVid-1M.csv` 和 `phil329/OpenVid-1M-mapping`，并只读扫描 `LINESHINE_SHARED_OPENVID_DIR`。
- `src/data/shared_openvid.py` 只读扫描外部 OpenVid 目录，冻结 split 并抽取 smoke 样本。
- `src/data/preprocess_cache.py` 只读取已经抽取好的 `local_path`，把视频转成 `LINESHINE_ROOT/cache/{train,val,smoke}`。
- `src/data/cache_prompts.py`、`src/train/train.py`、`src/sample/sample.py` 只读取 `LINESHINE_ROOT/cache/prompts`、`LINESHINE_ROOT/cache/*` 和 `LINESHINE_ROOT/weights/wan2.1_t2v_1.3b`。
- 训练和采样不再碰原始 OpenVid part。

## 快速开始

1. 初始化子模块。

```bash
git submodule update --init --recursive
```

2. 载入路径模板并激活环境。

```bash
source configs/env.example.sh
source "$CONDA_SH"
conda activate "$CONDA_ENV"
```

3. 运行测试。

```bash
PYTHONPATH=. pytest -q tests/test_ckpt.py tests/test_batch_adapter.py tests/test_flow_target.py tests/test_shared_openvid.py
```

4. 跑训练 smoke。

```bash
sbatch slurm/train_smoke.sbatch
```

默认 smoke 使用 1 卡 synthetic tiny-model，不依赖真实数据；如果你已经准备好真实 cache，可以改成：

```bash
SMOKE_MODE=real sbatch slurm/train_smoke.sbatch
```

## 说明

- `data/`、`cache/`、`runs/`、`reports/`、`weights/` 不进 Git。
- `third_party/Wan2.1` 是子模块，保留其上游结构。
- 当前仓库定位是组内私有工程仓库，不包含公开发布所需的额外脱敏工作。

更多维护规则见 `docs/project_management.md`。W5 30k 正式预训练使用 `slurm/submit_w5_chain.sh` 提交链式 8 卡作业；该脚本只读取 `LINESHINE_ROOT/cache/train` 和 `LINESHINE_ROOT/cache/prompts/empty.safetensors`，不会访问或提交原始数据。
