# DRAEM 项目层入口

状态：入口与轻量测试已实现；未启动正式训练，正式协议和种子仍需冻结。

## 数据目录

训练与推理统一读取 MVTec 风格目录：

```text
<root>/<category>/train/good/*
<root>/<category>/test/good/*
<root>/<category>/test/<defect-or-bad>/*
<root>/<category>/ground_truth/<defect-or-bad>/*
```

MVTec AD 的异常掩码可使用 `<stem>_mask.png`；VisA 官方 1-class 整理结果使用 `<stem>.png`。扩展名按大小写不敏感方式识别，因此 VisA 的 `.JPG` 可直接训练和推理。输入保持作者代码的 OpenCV BGR 通道顺序。评测掩码使用最近邻缩放并二值化。

## 只检查输入

以下命令只核验类别、正常训练图和 DTD 数量，不分配模型，也不开始训练：

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle --validate-only
```

## 训练

训练入口默认值来自作者 README：256 像素、batch size 8、700 epochs、学习率 `1e-4`。默认设备为 CPU，正式 GPU 启动前仍需执行项目实验协议规定的资源、进程、锁和数据清单门禁。

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset mvtec_ad --category bottle --seed 42 --device cuda:0 `
  --output-dir .\runs\draem\mvtec_ad\bottle\seed42
```

`--max-steps` 只用于 smoke/debug，运行配置会记录 `debug_limited`，不能作为性能结果。每个训练目录包含 `run_config.json`、`run_state.json`、逐步损失 JSONL、两个 tensor-only checkpoint 及 SHA-256 清单。

## 推理与统一评测

```powershell
& .\.venv\Scripts\python.exe .\scripts\infer_draem.py `
  --dataset mvtec_ad --category bottle --device cuda:0 `
  --checkpoint-dir .\runs\draem\mvtec_ad\bottle\seed42\checkpoints `
  --output-dir .\runs\draem\mvtec_ad\bottle\seed42_eval
```

推理逐样本保存未量化、未逐图归一化的 `float32` 异常图和实际参与评测的二值掩码。`predictions/index.jsonl` 记录样本 ID、原图/GT 引用、图像级分数和文件 SHA-256。指标计算会重新加载这些保存文件，因此指标和后续可视化可以使用同一份异常图。结果目录还包含 checkpoint SHA-256、完整运行配置、状态和 `metrics.json`。
