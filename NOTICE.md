# 许可与来源

LineShine Training 自有源码按 [Apache License 2.0](LICENSE) 提供。

[内置 Wan2.1 训练子集](third_party/Wan2.1/wan)来自 **The Alibaba Wan Team Authors**，上游为 `Wan-Video/Wan2.1`，公共基线提交 `9737cba9c1c3c4d04b33fcad41c111989865d315`。保留上游[许可证](third_party/Wan2.1/LICENSE.txt)和源码署名。本仓库包含实际用于训练的修改，而不是指向未公开提交的子模块：

- `attention.py` 和 `model.py` 接入 CPU 注意力、RoPE、重算及固定形状原生算子；`t5.py` 调整文本编码器的加载，`vae.py` 包含 CPU 适配。
- 发布目录只保留 1.3B 配置和训练依赖，三个包的 `__init__.py` 不再导入未随附的推理管线或其他模型。

自研 GEMM 的源码由正式配方 `pool_adaptive_reuse8_half0_long47` 展开，并应用按节点内 MPI rank 绑定的修正。原始源码包 SHA256：`3a97de3f03e478d3033db58ebe3037472ffb47f262c0688e186d60386266a13b`。本仓库分发最终 C++ 源码，不分发历史二进制或实验补丁合集。

模型权重、训练数据、定制 PyTorch wheel、HPCKit、HMPI、KBLAS 及其他外部软件不随本仓库分发，使用者须分别取得它们并遵守其各自条款。
