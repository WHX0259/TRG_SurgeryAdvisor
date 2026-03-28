# TRG Surgery Advisor（毕业论文辅助演示系统）

基于已训练 `ImageBaseClusterDistancePlusGatingModel` 权重的本地 Web 演示：**原图 PNG + 分割掩膜 PNG** → **目标区域高亮** + **预测类别**（界面与 JSON 不展示概率数值，非临床诊断）。

## 快速开始

### 方式 A：使用已有 Conda 环境（推荐，例如 `nnUNet` 里已带 PyTorch）

```bash
conda activate nnUNet
cd /data16t/huixuan/code/TRG_SurgeryAdvisor
# 仅安装本仓库额外依赖（不要再装一遍 torch，除非你想固定版本）
pip install gradio PyYAML opencv-python-headless
cp config.example.yaml config.yaml
# 编辑 config.yaml：pdcc_root、checkpoint_path、类别名称等
python app_gradio.py
```

也可一次性安装（可能微调当前环境里 torch 版本）：`pip install -r requirements.txt`。

### 方式 B：新建 venv（需自行先安装 PyTorch）

```bash
cd /data16t/huixuan/code/TRG_SurgeryAdvisor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
python app_gradio.py
```

浏览器访问 `http://127.0.0.1:7860`（或通过环境变量 `TRG_HOST`、`TRG_PORT` 修改）。指定配置：`TRG_CONFIG=/绝对路径/config.yaml python app_gradio.py`。

### 命令行推理（论文复现 / 批处理）

```bash
conda activate nnUNet
cd /data16t/huixuan/code/TRG_SurgeryAdvisor
python infer_cli.py --orig /path/to/orig.png --mask /path/to/mask.png --out ./outputs
# 可选：--mode masked|full  --config /path/to/config.yaml
```

### 与训练脚本、CMEL 数据集说明（必读）

- **`docs/TRAINING_ALIGNMENT.md`**：说明 `ImageBaseClusterDistancePlusGatingModel` 与 `CMEL/main.py` 临床多模态管线的区别、`masked`/`full`、224 与归一化。
- **`docs/ARCHITECTURE.md`**：分层架构与数据流（可插入论文「系统设计」）。

## 项目结构

| 路径 | 说明 |
|------|------|
| `app_gradio.py` | Gradio 入口，上传图像与展示结果 |
| `src/preprocess.py` | 灰度、对齐、Pad/Resize、可选原图×掩膜 |
| `src/model_loader.py` | 注入 PDCC 路径、构建模型、加载权重 |
| `src/predict.py` | `softmax` 推理与阈值文案 |
| `src/visualize.py` | 原图上的目标区域半透明高亮 |
| `config.yaml` | 默认配置（已填你的权重与训练 `config.json` 路径） |
| `config.example.yaml` | 配置模板 |
| `infer_cli.py` | 命令行单次推理，输出高亮图 + JSON |
| `src/pipeline.py` | Gradio 与 CLI 共用的推理管线 |
| `src/config_merge.py` | 从训练 `config.json` 同步 `img_size` |
| `docs/ARCHITECTURE.md` | **架构说明与数据流（可直接摘入论文）** |
| `docs/TRAINING_ALIGNMENT.md` | **训练管线对齐与 CMEL/ImageBase 辨析** |
| `export_onnx.py` | 导出推理子图 ONNX，可选 INT8 动态量化、`--verify` 数值对齐 |
| `src/inference_wrapper.py` | 可导出的推理子图（无队列副作用） |
| `src/runtime.py` / `src/onnx_backend.py` | `backend: torch` / `onnx` 切换 |
| `requirements-onnx-runtime.txt` | **无 PyTorch** 的推理依赖（ONNX 模式） |
| `docs/PACKAGING.md` | **可执行文件打包、CPU 部署、剪枝说明** |
| `packaging/trg_app.spec` | PyInstaller 示例 spec |

## ONNX 与纯 CPU 推理（原 Gradio / CLI 功能保留）

1. 在 **有 PyTorch + PDCC** 的环境导出：
   ```bash
   pip install onnx onnxruntime
   python export_onnx.py --verify --quantize --out-dir ./onnx_export
   ```
2. 在 `config.yaml` 中设置 `backend: onnx` 与 `onnx_model_path`（见文件内注释示例）。
3. 推理机仅安装：`pip install -r requirements-onnx-runtime.txt`，再 `python app_gradio.py` 或 `python infer_cli.py ...`。

**剪枝**：结构化剪枝需重新训练；部署侧「轻量化」推荐 ONNX **动态 INT8 量化**（`--quantize`）。详见 `docs/PACKAGING.md`。

## 制作此类软件的步骤（论文「开发过程」可用）

1. **需求与范围**：输入/输出形态、是否与训练管线严格一致、是否仅演示或需合规审计。
2. **环境冻结**：记录 `torch`、CUDA、PDCC 提交哈希、权重文件路径与 `config.json`。
3. **推理最小闭环**：命令行脚本单张图跑通，数值与 `validate` 一致。
4. **预处理契约**：文档化 `image_size`、灰度化方式、掩膜二值化规则、`masked` vs `full`。
5. **服务封装**：Gradio / FastAPI 只做 I/O，核心逻辑保持在 `src/` 便于测试。
6. **可视化**：与模型输入解耦，避免把「展示图」误当作网络输入。
7. **部署**：CPU/GPU、Docker；若需无 Torch 环境则规划 ONNX 导出与对齐测试。
8. **声明与验证**：免责声明、固定测试集回归、可选前瞻性临床验证计划。

## 无 GPU、无 Torch 能否使用？

- **无 GPU**：可以。`backend: torch` 时用 CPU 版 PyTorch；或改用 **`backend: onnx` + onnxruntime（默认 CPU EP）**。
- **无 Torch**：先在有 PyTorch 的机器运行 `export_onnx.py` 生成 `.onnx`，再在目标机用 **`requirements-onnx-runtime.txt`** + `backend: onnx`。详见 `docs/PACKAGING.md`。

## 依赖仓库

- 模型定义：`config.yaml` 中的 `pdcc_root`（默认 `/data16t/huixuan/code/PDCC`）下的 `lib/model/CMEL.py` 中 `ImageBaseClusterDistancePlusGatingModel`。
