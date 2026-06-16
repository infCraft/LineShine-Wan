# 数据读取约定

## 原始数据

- OpenVid 原始视频 part 不进 Git。
- `src/data/build_manifest.py` 和 `src/data/shared_openvid.py` 只读扫描 `LINESHINE_SHARED_OPENVID_DIR`。
- `src/data/shared_openvid.py` 的抽取结果会写出带 `local_path` 的 manifest。

## Cache

- `src/data/preprocess_cache.py` 只吃抽取后的 manifest 和 `local_path`。
- 训练侧只读 `LINESHINE_ROOT/cache/train`、`LINESHINE_ROOT/cache/val`、`LINESHINE_ROOT/cache/prompts`。
- 训练和采样不会再访问原始视频 part。

## 权重

- Wan2.1 的 VAE、T5 和相关 tokenizer 放在 `LINESHINE_ROOT/weights/wan2.1_t2v_1.3b`。
- `third_party/Wan2.1` 只作为代码子模块，不把上游权重塞进仓库。

## 命令输入输出

- `build_manifest.py`：CSV + mapping + 外部 part inventory -> manifest。
- `shared_openvid.py freeze/extract`：shared part inventory + manifest -> frozen split / extracted manifest。
- `preprocess_cache.py`：extracted manifest -> WebDataset shards。
- `train.py`：cache shards -> checkpoint / metrics / run dir。
- `sample.py`：checkpoint + prompt cache -> mp4。
