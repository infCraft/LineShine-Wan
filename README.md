# LineShine Training：在鲲鹏 CPU 上训练 Wan2.1

这是一套面向鲲鹏 72F5 的 **Wan2.1 T2V-1.3B 全参数训练**实现。同一个[训练器](src/train/shared_train.py)支持原图文生图（T2I）和原视频文生视频（T2V）。输入是原始媒体和文本，不是缓存的 latent 或文本向量。每四个 HMPI rank 组成一组：组内一个 rank 用冻结的 VAE 和 UMT5 在线编码，四个 rank 共享 DiT 参数、分担优化器更新。[数据读取与编码](src/data/online_training.py)和[并行实现](src/train/shared_parallel.py)可独立阅读。

本仓库是源码发布，不包含权重、数据、定制 PyTorch、预编译算子或厂商工具链。**仅克隆仓库并安装 PyPI 依赖不足以启动训练。**

## 1. 适用平台

| 部件 | 已使用的配置 |
| --- | --- |
| 计算节点 | Linux aarch64，Kunpeng 72F5（CPU part `0xd22`），每节点 16 个 DDR NUMA 域，每域 38 核；训练 rank 使用 36 个计算核和 1 个通信核 |
| 作业与通信 | Donau `dsub --mpi hmpi`、HPCKit HMPI `mpirun`、UCX、`numactl`；仓库、环境、数据和运行目录须在所有节点同路径可见 |
| 参考运行时 | Python 3.11、启用 MPI 的 PyTorch `2.9.1+shine.hmpi.kblas2521.beta0fix1`、KBLAS 25.2.1 SME multi；使用修补版以避免 BF16 GEMM 在 `beta=0` 时读旧输出 |
| 编译与系统工具 | BiSheng C++（历史原生构建脚本使用 HPCKit 25.1.0.SPC001）、KBLAS 开发头、支持 SVE2/BF16 的 GCC、libnuma 开发包、GNU make、Ninja、Bash 4+、GNU coreutils、`readelf`、`ldd`、FFmpeg |

不要在登录节点加载专用 Torch 或编译原生库。`WAN21_NATIVE_OPS=0` 关闭可选模型算子，不会关闭 HMPI、共享归约或节点拓扑检查，也不会让训练器成为通用笔记本后端。

## 2. 配置运行时与权重

以下命令从仓库根目录执行；`/path/to/...` 和队列名须换成本站点的实际值。先准备经本站点验证的 HMPI PyTorch 环境，再在其中离线安装 [Python 依赖](requirements.txt)。不要用普通 PyPI Torch 替换专用构建。

```bash
export WAN21_ENV_ACTIVATE=/path/to/runtime/activate.sh
export WAN21_QUEUE=your_queue
export WAN21_ENCODER_WEIGHTS=/path/to/Wan2.1-T2V-1.3B
export WAN21_BLAS_CHECK="$PWD/kunpeng/checks/check-wan21-links.sh"
# dsub 不在提交节点 PATH 中时：export WAN21_DSUB=/path/to/dsub
```

`WAN21_ENV_ACTIVATE` 在作业内被 `source`。它应激活环境并导出 `WAN21_PYTHON`（Python 可执行文件的绝对路径）、`VIRTUAL_ENV`、HMPI/GCC/KBLAS 的 `PATH` 和动态库路径。默认 [KBLAS 检查](kunpeng/checks/check-wan21-links.sh)还要求激活脚本导出：

| 变量 | 含义 |
| --- | --- |
| `HMPI_ROOT`、`HPCKIT_GCC_ROOT`、`KBLAS_SME_MULTI_DIR` | 分别用于验证 MPI、`libgomp` 和 `libkblas.so.25.2.1` 的真实加载位置 |
| `WAN21_WHEEL_ROOT`、`WAN21_WHEEL_CHECKSUMS` | 可信 wheel 校验清单中相对路径的基目录，以及清单本身的绝对路径 |
| `WAN21_TORCH_PREFIX` | Torch 的预期安装前缀；默认是 `VIRTUAL_ENV` |

检查器会验证 wheel、ELF 链接、GEMM 符号和 FP32/BF16 矩阵结果，失败即停止。可用本站点等价的检查脚本替换，不要用空脚本跳过。若使用已验证的 FP32/OpenBLAS 对照环境，须同时切换激活脚本、`WAN21_EXPECT_TORCH_VERSION`、`WAN21_EXPECT_BLAS=openblas`、`WAN21_TRAIN_PRECISION=fp32` 和[对应检查器](kunpeng/checks/check-openblas-links.sh)；该检查器还需 `WAN21_OPENBLAS_PREFIX`、`OPENBLAS_LIBRARY_REALNAME`、`OPENBLAS_LIBRARY_SHA256`。不使用专用原生模型算子时显式设置 `WAN21_NATIVE_OPS=0`。

编码权重目录须包含 `Wan2.1_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth` 及完整的 `google/umt5-xxl/` tokenizer 目录。计算节点离线加载；这些文件请自行取得。`scratch` 默认随机初始化 DiT，**不是**自动加载预训练 DiT；只希望载入初始模型时设置 `WAN21_INIT_CHECKPOINT`，不要与 `resume` 混用。

## 3. 从源码构建原生算子

在已获分配的 **72F5 计算节点**上，从共享仓库根目录运行：

```bash
export WAN21_BISHENG_ROOT=/path/to/compiler/bisheng
export WAN21_KBLAS_INCLUDE=/path/to/kblas/include  # 目录内有 kblas.h
./kunpeng/build_native.sh
```

[构建脚本](kunpeng/build_native.sh)使用节点本地 `/tmp` 构建，在四个库均生成后安装 SME GEMM、Torch Linear 桥及 attention 前后向库到仓库的默认加载位置，并打印 SHA256。Linear 桥使用当前 PyTorch 扩展环境的 C++ 编译器，须支持 `-march=armv9-a+sme+bf16+sve`；GEMM 和 attention 使用显式配置的 BiSheng。源码内已展开正式 GEMM 变体，不需要额外下载补丁。SVE 算子和共享归约器由训练入口编译。生成的 `.so` 不应提交。

## 4. 准备原始数据

每个 JSONL 一行一个样本，`path` 相对 **JSONL 所在目录**（也可用绝对路径），`caption` 非空。下列各一行只展示格式，不是可跑满默认 batch 的数据集：

图像 `images.jsonl`：

```json
{"type":"image","path":"images/a.jpg","caption":"窗边桌上的一杯茶"}
```

视频 `videos.jsonl`：

```json
{"type":"video","path":"clips/a.mp4","caption":"镜头缓慢移向窗边的茶杯","clip_start_sec":0.0}
```

T2I 也接受含 `catalog.json` 的原图索引目录。T2V 默认按 16 fps 取 49 帧并缩放、中心裁剪为 256×256，视频需能提供所需帧数。使用自备 JSONL 不必设置 `WAN21_VIDEO_ROOT`；只有通过 `video` 文件名定位时才需要提供原视频根目录。T2V 参数 `default` 只会读取显式设置的 `WAN21_VIDEO_MANIFEST`。**同一个数据桶内至少要有 `rank 总数 × 每 rank batch` 个样本**；按下表默认配置，T2I 至少 64 张，T2V 至少 16 个切片。`WAN21_CACHE_ROOT` 不是在线训练输入。

## 5. 提交首次训练

```bash
export DATA=/path/to/manifests RUNS=/path/to/runs
./kunpeng/submit_wan21_t2i_multinode.sh "$DATA/images.jsonl" "$RUNS/image-01" 40 scratch
./kunpeng/submit_wan21_t2v_multinode.sh "$DATA/videos.jsonl" "$RUNS/video-01" 40 scratch
```

两个命令分别提交独立作业，选择需要的一个。`scratch` 的目标目录必须尚不存在。

| 默认项 | T2I | T2V |
| --- | --- | --- |
| 节点 × 每节点 rank | 4 × 16 | 4 × 4 |
| 每 rank batch / 梯度累积 | 1 / 1 | 1 / 1 |
| 全局 batch | 64 | 16 |
| 每 rank 调度申请 | 38 CPU / 27 GiB | 38 CPU / 108 GiB |

用 `WAN21_NNODES`、`WAN21_PACK_RPN`（4/8/12/16）、`WAN21_TRAIN_BATCH_SIZE`、`WAN21_TRAIN_GRAD_ACCUM` 调整规模。默认 DiT 使用 BF16 autocast，参数、梯度、AdamW 状态、VAE 和 UMT5 均为 FP32。`WAN21_STORED_PRECISION=bf16` 与 `WAN21_TEXT_PRECISION=bf16` 分别改变训练状态和文本编码精度。BF16 状态配 16 rank/节点时默认分层归约；49 帧、256×256、每 rank batch=1 的 BF16 原生视频路径默认整节点申请 608 CPU / 400 GiB，以访问 HBM。不要对该路径附加只允许 DDR 的 NUMA 约束。

## 6. 恢复训练、查看日志

```bash
./kunpeng/submit_wan21_t2i_multinode.sh "$DATA/images.jsonl" "$RUNS/image-01" 80 resume
```

`80` 是累计目标，不是再跑 80 步。默认每 200 步和末步保存，恢复选择有 `COMPLETED` 且通过校验的 checkpoint。保持原数据、rank 布局、存储/文本精度和学习率日程不变。默认 `WAN21_TRAIN_LR_TOTAL_STEPS=2000`；超过 2000 步的任务应在**首次启动**前设置目标日程。T2V 同样使用原运行目录并传 `resume`。

- `logs/train_<jobid>.tee.log`：可用 `tail -f` 实时跟踪初始化、在线编码、各阶段和错误。
- `metrics.jsonl`：每步汇总；`step_seconds` 包含在线编码、前后向、通信和优化，不含初始化和保存。`rank_timing.jsonl` 可进一步区分 VAE、UMT5 和训练开销。
- `checkpoints/step_*/`：DiT 权重、四个 owner 的优化器状态、训练器与采样状态、清单和完成标记。只信任自己或可信来源的训练 checkpoint。

## 验证范围

本地验证正式提交参数，不需要安装 Torch 或连接 Donau：

```bash
python3 -m unittest discover -s tests -p test_online_submission.py -v
```

已有 72F5 训练记录（2026-11-16）验证过完整模型原图 2 步、5 帧视频 1 步、49 帧视频 1 步，以及双节点 32 rank 梯度归约/累积和微型模型断点恢复。**这些记录使用当时的预编译原生库**。此次纯源码发布仅做本地静态检查和模拟提交，没有重新验证从源码构建后在 72F5 的数值、速度、长期收敛或完整模型多节点恢复。

在匹配平台复验时，[实机回归入口](kunpeng/run_wan21_shared_checks.sh)须在 Donau 分配内执行：`WAN21_CHECK_MODE=core` 需 2 节点×16 rank；`media` 或 `media49` 需 1 个可访问 HBM 的整节点，用显式绝对路径 `WAN21_CHECK_VIDEO_DATA` 提供视频清单，`media` 还需 `WAN21_CHECK_IMAGE_DATA`。这些测试不随本地验收自动提交。

项目自有代码采用 [Apache-2.0](LICENSE)。Wan2.1 来源、适配范围和第三方声明见 [NOTICE.md](NOTICE.md)。
