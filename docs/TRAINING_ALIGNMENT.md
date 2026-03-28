# 训练管线与本部署工程的关系（写论文时可整节引用）

## 1. 你当前权重属于哪一类？

路径中的 `model_type` 为 **`ImageBaseClusterDistancePlusGatingModel`** 时：

- 网络前向为 **`model(image)`**，**不接收** `clinic_feature`、**不接收** `manual_features`。
- 模型类定义在 **`PDCC`** 工程的 `lib/model/CMEL.py`（本项目的 `pdcc_root` 指向该目录）。
- 这与 **`CMEL/main.py` + `Dataset_Slice_text_manual`** 默认使用的 **`ClinicalImageBaseClusterDistancePlusGatingModel`**（`model(images, clinic_feature)`）**不是同一条管线**。  
  论文中应分开写：**多模态临床–图像模型** vs **纯图像 MoE 聚类门控模型**。

## 2. 为何 `config.json` 里仍出现 slice_path、临床 CSV？

`Multi_Modal_MoE` 保存的 `config.json` 往往沿用**通用训练脚本**的参数字段（如 `slice_path`、`manual_csv_path`）。  
只要 **`model_type` 为 `ImageBaseClusterDistancePlusGatingModel`**，实际前向仍由 **PDCC 中该类的 `forward(self, image)`** 决定，**与临床表无耦合**。  
（若你当时改动了 `main.py` 用别的数据集喂该模型，以你**真实训练代码**为准，并在论文中说明数据来源与预处理。）

## 3. 原图 PNG + 分割 PNG 如何与训练对齐？

本系统支持两种**张量构造**（与界面中 `input_mode` 一致）：

| 模式 | 含义 | 适用说明 |
|------|------|----------|
| `masked` | 灰度原图 × 掩膜（归一化到 [0,1] 后逐像素相乘） | 符合「仅保留病灶区域」的叙述；若训练时未使用该方式，需再训练或做离线对齐实验。 |
| `full` | 仅整幅灰度图，忽略掩膜对乘积的贡献 | 与 **PDCC `main.py` 中 `model(batch['image'])`** 行为一致（整幅 CEUS 灰度图）。 |

**建议**：用验证集若干样本，对比 `infer_cli.py` 输出与训练时 `validate` 的概率是否一致，再固定 `input_mode` 写进论文。

## 4. 空间尺寸、通道数与归一化

- 训练 `config.json` 中 **`img_size: 224`** 时，本项目通过 `training_config_json` + `sync_image_size_from_training_json: true` **自动将 `image_size` 设为 224**，与参数表一致。
- **首层卷积权重形状**决定 `in_chan`：若加载报错 `encoder.conv_1_3x3.weight` 为 **`[64, 3, 3, 3]`**，须设 **`in_chan: 3`**，预处理走 **RGB**；若为 **`[64, 1, 3, 3]`**，设 **`in_chan: 1`**，走灰度。
- **`use_cmel_normalize: true`** 时，对 RGB 使用与 **`Dataset_Slice_text_manual`** 相同的 **`Normalize((0.1591,...),(0.2593,...))`**。若训练来自 **PDCC 单通道 CEUS** 且未做该归一化，请设 **`in_chan: 1`**、**`use_cmel_normalize: false`** 并与验证集对数。

## 5. 验证清单（答辩 / 论文复现）

1. `pdcc_root`、`checkpoint_path` 与 `training_config_json` 路径正确。  
2. `conda activate nnUNet`（或等价环境）下可 `import torch`，且能加载权重。  
3. 固定 `input_mode`，与离线验证集对比 AUC/概率误差在可接受范围。  
4. 界面文案标明：**辅助分析，非诊断**。
