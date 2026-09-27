# DRAEM 项目层入口

状态：入口与轻量测试已实现；未启动正式训练。开发协议和三组种子已冻结，候选方法及正式训练配方仍待登记。

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

默认输入模式 `author_full_train` 保留作者式复现行为，会读取类别下全部 `train/good` 和全部 DTD。它不能作为冻结开发协议的运行。以下命令只做作者式输入检查，不分配模型，也不开始训练：

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle --validate-only
```

开发运行必须显式选择 `visa_train_dev_dtd_v1` 并使用一个预注册种子：

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle --seed 1887537248 `
  --input-protocol visa_train_dev_dtd_v1 --validate-only
```

该模式先核验 `configs/development_protocol.json` 中固定的根清单 SHA-256，再核验全部子清单的集合、行数和 SHA-256。随后仅选择当前类别中 `dataset=visa`、类别匹配、`split=train` 且路径位于 `<category>/train/good` 的条目，以及 DTD 中 `dataset=dtd`、类别与路径匹配、`split=train` 的条目。每个入选文件均复核大小和 SHA-256；根路径、字段、注册种子、内容分组或任一哈希不一致都会在 Dataset 和模型构造前失败。

真实只读核验证据保存在 `artifacts/validation/draem_manifest_inputs_candle.json`。该证据登记 720 张 candle 训练原图、4,512 张 DTD-train 来源，并明确记录 `formal_test_content_accessed=false`、`ground_truth_content_accessed=false` 和 `training_started=false`。

## 冻结合成开发集

`scripts/evaluate_draem_synthetic_dev.py` 是唯一正式的 DRAEM 合成开发入口。它有 `--validate-only`、`--render-audit-count N` 和 `--evaluate` 三种互斥模式。前两种不加载模型；最后一种要求重建与分割 checkpoint，只执行推理。

入口只接受 schema-v3 冻结 recipe，并从 salt 重推完整清单。它只读取清单中的 VisA-dev 和 DTD-dev，逐文件校验大小与 SHA-256。完整评测同时包含未增强正常 dev 样本，因此会输出固定阈值 `0.5` 下的正常图像/像素误报率；所有开发输出固定标记为 `synthetic-development-only` 和 `claim_eligible=false`。

## 训练

训练入口默认值来自作者 README：256 像素、batch size 8、700 epochs、学习率 `1e-4`。默认设备为 CPU，正式 GPU 启动前仍需执行项目实验协议规定的资源、进程、锁和数据清单门禁。

开发训练命令必须保留 `--input-protocol visa_train_dev_dtd_v1`；省略该参数会回到作者式全训练输入，运行配置会明确标记 `manifest_backed=false` 和 `development_protocol_eligible=false`。

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle --seed 1887537248 --device cuda:0 `
  --input-protocol visa_train_dev_dtd_v1 --run-purpose development_full `
  --output-dir .\runs\draem\visa\candle\seed1887537248
```

`development_full` 固定为作者 700-epoch 配方，并且不依赖尚未冻结的正式方法表；它始终是 claim-ineligible。作者式完整复现使用 `--run-purpose author_reproduction`。`--max-steps` 只用于 `--run-purpose debug_limited`，不能作为性能结果。每个训练目录包含 `run_config.json`、`run_state.json`、逐步损失 JSONL、两个 tensor-only checkpoint 及 SHA-256 清单。

## 推理与统一评测

```powershell
& .\.venv\Scripts\python.exe .\scripts\infer_draem.py `
  --dataset mvtec_ad --category bottle --device cuda:0 `
  --checkpoint-dir .\runs\draem\mvtec_ad\bottle\seed1887537248\checkpoints `
  --output-dir .\runs\draem\mvtec_ad\bottle\seed1887537248_eval `
  --seed 1887537248 --run-purpose author_reproduction
```

推理必须显式声明 `--run-purpose`；可选 `--execution-scope` 只做一致性检查。`formal_full` 的 train-only 输入已接入 `formal_training_input_bundle_v1`，checkpoint、推理和聚合会重验当前文件集合与内容哈希；当前正式协议、方法表和方法配方仍未冻结，所以真实正式运行仍会在构造 test dataset 之前拒绝。逐样本保存未量化、未逐图归一化的 `float32` 异常图和实际参与评测的二值掩码。`predictions/index.jsonl` 记录样本 ID、原图/GT 引用、从落盘异常图重算的 `raw_map_max_score`、DRAEM 21×21 池化所得 `method_native_image_score` 和文件 SHA-256。主表 `image_auroc` 只使用前者，`image_auroc_method_native` 单列后者。指标计算会重新加载这些保存文件，因此指标和后续可视化使用同一份异常图。完成态还绑定运行配置、指标和预测索引 SHA-256。

方法与正式协议冻结并建立 F0 测试清单后，`formal_full` 还必须提供 `--formal-test-inventory-dir`。入口会重验完整 bundle 和源文件，再只从认证清单提取当前类别的显式样本；Dataset 不会重新发现测试目录。清单 ID、inventory/receipt SHA-256、类别条目数及逐样本源文件 SHA-256 会写入运行配置、指标、完成态和预测索引，聚合时再次验证。例如：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_draem.py `
  --dataset mvtec_ad --category bottle --device cuda:0 `
  --checkpoint-dir <正式checkpoint目录> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\mvtec_ad_v1 `
  --output-dir <空的正式输出目录> `
  --seed <注册seed> --run-purpose formal_full
```
