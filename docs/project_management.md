# 项目管理规范

## 仓库边界

本仓库只保存可审查、可复现的代码、脚本、测试和文档。以下内容不进入 Git：

- 原始数据、抽取后视频、metadata 运行产物。
- VAE/T5/DiT 权重、checkpoint、导出权重。
- cache shards、runs、reports、TensorBoard 日志。
- `*.pt`、`*.pth`、`*.safetensors`、`*.tar`、`*.mp4`、`*.zip` 等大文件产物。

运行数据统一放在 `LINESHINE_ROOT` 下，仓库本体放在 `LINESHINE_CODE_ROOT`。

## 路径规范

训练和数据处理脚本不得新增个人硬编码路径作为默认执行路径。优先使用：

```bash
export LINESHINE_ROOT=/mnt/beegfs/home/huang_z/lineshine
export LINESHINE_CODE_ROOT=$LINESHINE_ROOT/code
export LINESHINE_SHARED_OPENVID_DIR=/mnt/beegfs/home/yezy/openvid
```

脚本需要读取数据、权重、cache 或 run 目录时，应通过环境变量或显式 CLI 参数传入。GitHub 仓库不包含真实数据，因此文档和 Slurm 脚本必须写清楚需要哪些外部路径。

## 改动流程

1. 保持根目录结构清晰：`src/`、`tests/`、`slurm/`、`configs/`、`docs/`、`third_party/Wan2.1/`。
2. 修改训练、数据、采样或 Slurm 行为时，同步更新 `README.md` 或 `docs/` 中对应说明。
3. 对行为变化补最小测试；至少运行相关 pytest。无法运行 GPU 验证时，在提交说明或 `STATUS.md` 中记录原因。
4. 远端 `$ROOT/code` 每次完成阶段性改动后保持 `git status` 干净并提交。

## 训练作业规范

- GPU 程序只能在 Slurm 作业内运行，登录节点只做文件、metadata 和作业提交。
- 正式训练首次启动必须使用干净 `RUN_DIR` 和 `--from-scratch`。
- 续跑必须显式使用 `--auto-resume --no-from-scratch` 或 `--resume <checkpoint>`，不得依赖隐式恢复。
- 当前 DDP 训练循环禁用内联 validation，正式训练脚本保持 `--val-every 0`；validation 和采样作为独立作业或后续同步验证实现。
- 长时间训练脚本必须把 stdout/stderr 同步写入可追踪日志，W5 脚本会在 `RUN_DIR/logs/` 下用 `tee` 保存每段日志。
