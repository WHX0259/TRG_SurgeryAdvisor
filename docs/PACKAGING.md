# 打包可执行文件与 ONNX / CPU 部署

## 1. 能力说明

| 方式 | 说明 |
|------|------|
| **PyTorch（默认）** | `backend: torch`，功能与原先一致，可 GPU。 |
| **ONNX + onnxruntime** | `backend: onnx`，**纯 CPU** 可跑；无需安装 torch（见 `requirements-onnx-runtime.txt`）。 |
| **INT8 量化** | `export_onnx.py --quantize` 生成动态量化权重，体积与 CPU 延迟通常优于 FP32 ONNX。 |
| **剪枝** | 结构化剪枝会改变网络拓扑，需 **重新训练** 才能得到合法权重；本仓库不提供「剪后直接部署」的捷径。 |

## 2. 导出 ONNX（在装有 PyTorch + PDCC 的机器上执行）

```bash
conda activate nnUNet
cd /data16t/huixuan/code/TRG_SurgeryAdvisor
pip install onnx onnxruntime
python export_onnx.py --verify --quantize --out-dir ./onnx_export
```

- 导出的是 **推理子图**（`src/inference_wrapper.py`）：与完整 `forward` 相比 **去掉队列 / KMeans / 动量更新**；**融合分类 logits** 与 eval 下主路径一致，`--verify` 可打印 softmax 空间最大误差。
- 输出：`onnx_export/trg_imagebase_infer_fp32.onnx`，量化：`trg_imagebase_infer_int8.onnx`。

在 `config.yaml` 中设置：

```yaml
backend: onnx
onnx_model_path: "/data16t/huixuan/code/TRG_SurgeryAdvisor/onnx_export/trg_imagebase_infer_int8.onnx"
onnx_providers: ["CPUExecutionProvider"]
```

然后：

```bash
pip install -r requirements-onnx-runtime.txt
python app_gradio.py
```

## 3. PyInstaller 可执行文件（实验性）

Gradio + 依赖体积大，打包前请 **在目标环境实测**。

```bash
cd /data16t/huixuan/code/TRG_SurgeryAdvisor
pip install pyinstaller
# 必须在项目根目录执行（pathex 使用 cwd）
pyinstaller packaging/trg_app.spec
```

生成物在 `dist/TRG_SurgeryAdvisor/`。请将同目录下的 `config.yaml`、`onnx_export/*.onnx`（若用 ONNX）与可执行文件一并分发，或通过环境变量 `TRG_CONFIG` 指向配置。

**Torch 后端打包**：会把 PyTorch 打进目录，体积常达 **数 GB**；论文/演示更推荐 **ONNX + onnxruntime** 再打包。

## 4. CPU 上没有 torch 可以吗？

- **可以**，前提是已完成 ONNX 导出，且运行环境使用 `backend: onnx` 与 `requirements-onnx-runtime.txt`。
- **导出步骤**仍需要一次具备 PyTorch 的环境（例如本机 `nnUNet`）。
