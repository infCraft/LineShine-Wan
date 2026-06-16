# 环境与启动

## 目录约定

工作区采用以下布局：

```text
/mnt/beegfs/home/huang_z/lineshine
  code/
  data/
  cache/
  runs/
  reports/
  weights/
```

仓库本体放在 `code/`，其他目录都属于外部运行数据，不提交 Git。

## 环境变量

先加载路径模板：

```bash
source configs/env.example.sh
```

常用变量：

- `LINESHINE_ROOT`：工作区根目录。
- `LINESHINE_CODE_ROOT`：代码仓库目录。
- `LINESHINE_SHARED_OPENVID_DIR`：外部 OpenVid 完整 part 目录，只读使用。
- `HF_ENDPOINT`：HuggingFace 下载镜像。
- `CONDA_SH`、`CONDA_ENV`：Slurm 和交互式环境的 conda 入口。

## 安装顺序

1. 更新子模块。
2. 激活 `lineshine-wan`。
3. 安装 `requirements.txt` 中的 Python 依赖。
4. 确认 `third_party/Wan2.1` 里的模型代码可 import。

## 验证

先跑单测，再跑 smoke：

```bash
PYTHONPATH=. pytest -q tests/test_ckpt.py tests/test_batch_adapter.py tests/test_flow_target.py tests/test_shared_openvid.py
sbatch slurm/train_smoke.sbatch
```
