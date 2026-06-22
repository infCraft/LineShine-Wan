# 数据读取约定

## 原始数据

- OpenVid 原始视频 part 不进 Git。
- `src/data/build_manifest.py` 和 `src/data/shared_openvid.py` 只读扫描 `LINESHINE_SHARED_OPENVID_DIR`。
- `src/data/shared_openvid.py` 的抽取结果会写出带 `local_path` 的 manifest。

## Cache

- `src/data/preprocess_cache.py` 只吃抽取后的 manifest 和 `local_path`。
- 训练侧只读 `LINESHINE_ROOT/cache/train`、`LINESHINE_ROOT/cache/val`、`LINESHINE_ROOT/cache/prompts`。
- `src/data/audit_cache_bucket.py` 只读扫描已有 WebDataset cache，核对固定 bucket 的 latent/token/视频采样规格，并可把全合格 shard 链接到新的 clean bucket 目录。
- `src/data/stage1_expand.py` 基于当前已完成 shared OpenVid part 生成额外 stage1 manifest，默认按 sample_id 排除已经处理过的旧 split。
- 训练和采样不会再访问原始视频 part。

## 权重

- Wan2.1 的 VAE、T5 和相关 tokenizer 放在 `LINESHINE_ROOT/weights/wan2.1_t2v_1.3b`。
- `third_party/Wan2.1` 只作为代码子模块，不把上游权重塞进仓库。

## 命令输入输出

- `build_manifest.py`：CSV + mapping + 外部 part inventory -> manifest。
- `shared_openvid.py freeze/extract`：shared part inventory + manifest -> frozen split / extracted manifest。
- `preprocess_cache.py`：extracted manifest -> WebDataset shards。
- `audit_cache_bucket.py`：cache shards -> audit report / clean sample manifest / bad sample manifest / clean shard links。
- `stage1_expand.py`：filtered manifest + current shared inventory + exclude manifests -> extra stage1 manifest。
- `train.py`：cache shards -> checkpoint / metrics / run dir。
- `sample.py`：checkpoint + prompt cache -> mp4。
