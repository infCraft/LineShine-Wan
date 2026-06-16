# 仓库布局

## 保留目录

- `src/data/`：manifest、shared OpenVid、cache、verify。
- `src/train/`：训练核心、flow、checkpoint、metrics、dataset adapter。
- `src/sample/`：采样脚本。
- `slurm/`：集群作业脚本。
- `tests/`：回归测试。
- `third_party/Wan2.1/`：上游子模块。

## 生成物

以下内容都属于运行产物，不进 Git：

- `data/`
- `cache/`
- `runs/`
- `reports/`
- `weights/`
- `*.pt`、`*.pth`、`*.safetensors`、`*.tar`、`*.mp4`

## 维护原则

- 不改训练和数据逻辑，只改仓库组织和外部路径约定。
- 训练代码默认从环境变量读取路径；硬编码个人目录只保留在历史记录里，不保留在默认执行路径里。
