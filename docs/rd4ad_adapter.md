# RD4AD 项目层适配与兼容审计

状态：**推理适配、可复现训练入口、教师权重门禁和 train-only 校验已完成；没有真实训练，也没有在 MVTec AD 或 VisA 上运行性能评测。** 本文只证明冻结源码可在当前环境构造，并证明训练/推理接口和输入门禁闭环，不能作为基线复现结果。

## 冻结对象

- 作者仓库：`hq-deng/RD4AD`
- commit：`6554076872c65f8784f6ece8cfb39ce77e1aee12`
- 许可证：MIT
- 第三方目录：`third_party/RD4AD`，适配前后 `git status --porcelain` 均为空；运行前还会对 5 个冻结源码文件执行 SHA-256 fail-closed 校验
- 当前环境：Windows、Python 3.11.16、PyTorch 2.7.0+cu128、torchvision 0.22.0、NumPy 1.26.4

源码文件哈希和 smoke 结果见 `manifests/rd4ad_adapter_compatibility.json`。

## 已识别的兼容与口径问题

| 作者实现位置 | 问题 | 项目层处理 |
| --- | --- | --- |
| `dataset.py:46-57` | `os.listdir` 加 `*.png` 只覆盖 MVTec PNG，不能读取 VisA 的 JPEG；排序和 mask 配对也依赖目录返回 | `rd4ad_data.py` 复用经过验证的统一 manifest，大小写无关地识别 PNG/JPEG/TIFF，并分别解析 MVTec 的 `_mask` 与 VisA 同 stem mask |
| `dataset.py:12-22` | resize 插值与新版 torchvision 的 antialias 行为没有显式冻结，GT 也沿用默认插值 | RGB 显式固定 bilinear + antialias；统一评测 GT 显式用 nearest，二值化后保存 |
| `main.py:80`、`test.py:119` | `pretrained=True` 会隐式访问网络并写入默认 Torch Hub cache | vendor 仅以 `pretrained=False` 构造；教师 ImageNet 权重必须以显式本地路径、SHA-256 和 tensor-only loader 注入 |
| `test.py:125,155,247` | `torch.load` 未指定 `map_location`，也未明确安全加载策略 | 项目层固定 `map_location="cpu", weights_only=True`，验证映射、字符串键及 tensor 值，再 strict load |
| `test.py:39` | 每层异常图硬编码取 `[0, 0]`，实际只支持 batch=1 | `rd4ad_anomaly_maps` 对整个 NCHW batch 计算，逐尺度相加，且检查尺度、shape 和有限值 |
| `test.py:36-38,84` | 作者图为各尺度 `1-cos`、`align_corners=True` 双线性插值、求和后 Gaussian sigma 4 | 逐项保持并冻结；输出是 Gaussian 后、未做逐图 min-max 的 `float32` raw map |
| `test.py:93` | 作者图像分数是平滑异常图 raw max | 作为 `method_native_image_score` 独立保留；共同 `image_auroc` 仍从已落盘 raw map 重算最大值。两者当前数值相同，但语义字段不合并 |
| `test.py:356` | `np.bool` 在 NumPy 1.24 后移除；作者 AUPRO 还是近似阈值与逐异常图均值口径 | 不调用作者 metric；统一从落盘 raw maps/GT 重新计算 I-AUROC、P-AUROC、P-AP、AUPRO@0.30 |
| `main.py:77-78` | Windows spawn、多 worker 和入口保护没有形成可复现实验契约 | CLI 有 `if __name__ == "__main__"`；默认 `workers=0`，正式 smoke 后才可冻结其他 worker 数 |

## 数据与预测契约

`Rd4adNormalTrainDataset` 只读取 `CATEGORY/train/good`。`Rd4adTestDataset` 同时支持：

- MVTec AD：`CATEGORY/test/DEFECT/*.png` 与 `ground_truth/DEFECT/*_mask.png`
- VisA 1-class：`CATEGORY/test/{good,bad}/*.{JPG,jpg,png}` 与 `ground_truth/bad/<same-stem>.png`

输入始终转 RGB，resize/center crop 至 256×256，并使用 ImageNet mean/std。预测契约如下：

1. 冻结教师输出三个 feature scales；bottleneck 和 decoder 产生相同 shape 的三个反向特征。
2. 每尺度计算 `1 - cosine_similarity(teacher, student)`。
3. 以 bilinear、`align_corners=True` 映射到评测平面后求和。
4. 每张图独立做 Gaussian sigma 4；不做逐图 min-max、量化或阈值化。
5. raw map 与 nearest-neighbor GT 分别以 `.npy` 保存并记录 SHA-256；索引保存 `sample_id`、label、defect type、源文件引用、`raw_map_max_score` 和 `method_native_image_score`。
6. 指标函数重新读取这些落盘数组，通过项目统一 `evaluate_category` 计算共同指标，并把作者分数单独写入 `image_auroc_method_native`；指标和可视化共享同一份 raw map。

## 可复现训练入口

`scripts/train_rd4ad.py` 调用项目层 `rd4ad_train.py`，vendor 保持只读。正式模式固定作者 WideResNet50-2 教师、`BN_layer` 和反向 decoder，Adam 的学习率为 `0.005`、betas 为 `(0.5, 0.999)`，训练 200 epochs，损失为三个尺度逐图展平后的余弦距离之和。训练期间不构造 test dataset，不按测试指标挑 epoch，只保存最终 `{bn, decoder}` tensor state dict。

教师权重必须通过 `--teacher-checkpoint` 指向项目内本地文件，并通过 `--teacher-sha256` 提供完整 64 位 SHA-256。共享 `wrn50_weights.py` 先以 `weights_only=True` 检查 tensor-only state dict，再核对完整 keys、shape、dtype 与有限值。官方 V1 文件中的 55 个 `float16` tensor 只允许显式扩为目标模型的 `float32`，其他 dtype 差异一律拒绝；转换完成后再 strict load。任何哈希、数据、源码、loss、梯度或写盘异常都会令 run state 进入 `failed`，并移除不完整的最终 checkpoint。

每个 run 保存：

- `run_config.json`：完整参数、冻结 vendor 哈希、教师权重来源与 strict-load 报告、normal-only 输入内容摘要；
- `run_state.json`：状态、epoch、step、错误或最终 checkpoint SHA-256；
- `train_steps.jsonl`：逐 step 的 loss、梯度范数和学习率；
- `checkpoints/rd4ad_final.pt` 与 `checkpoints/manifest.json`：最终 tensor-only 权重、文件大小和 SHA-256。

`--max-steps` 只允许与 `--execution-scope debug_limited` 同时使用，产物状态为 `debug_completed` 且不具备性能声明资格。`full` 模式禁止 `max_steps`，固定 256 像素、200 epochs 和作者学习率；batch 只允许按资源 smoke 顺序在 16、8、4 中选择并冻结，8 或 4 必须披露为显存兼容差异。

项目内官方 ImageNet V1 文件已通过共享门禁：大小 138,223,492 bytes，完整 SHA-256 为 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`，320 个 tensor 的 strict load 通过，manifest 为 `formal_use_allowed=true`。机械校验命令为：

```powershell
.\scripts\Set-ProjectEnvironment.ps1
.\.venv\Scripts\python.exe .\scripts\train_rd4ad.py `
  --dataset visa --category candle `
  --teacher-checkpoint .\artifacts\weights\wide_resnet50_2-95faca4d.pth `
  --teacher-sha256 <完整64位SHA-256> `
  --seed 1887537248 --validate-only
```

真正的短跑只能明确标成调试：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_rd4ad.py `
  --dataset visa --category candle `
  --teacher-checkpoint .\artifacts\weights\wide_resnet50_2-95faca4d.pth `
  --teacher-sha256 <完整64位SHA-256> `
  --seed 1887537248 --run-purpose debug_limited --max-steps 1
```

完整作者复现必须显式使用 `--run-purpose author_reproduction`，仍不具备正式声明资格。`formal_full` 还会验证冻结方法表、精确训练/推理配方、三组注册 seed、train-only 正式输入 bundle 和 schema-v2 checkpoint manifest；当前正式方法与配方尚未冻结，所以真实正式运行继续 fail-closed。RD4AD 的 `formal_full` 已能重验完整类别 `train/good`，但独立的 manifest-backed train/dev 适配尚未实现，所以 `development_full` 当前不可用。

正式推理还必须显式提供对应数据集的 `--formal-test-inventory-dir`。入口先重验完整 F0 bundle 和当前源文件，再从清单选择当前类别的全部显式样本；`Rd4adTestDataset` 不会重新遍历测试目录。运行配置、指标、完成态和每个预测索引行均绑定 inventory ID、inventory/receipt SHA-256，索引还逐样本绑定源图/掩码 SHA-256。聚合阶段会重验这些证据和样本一一对应关系。冻结后命令形状为：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_rd4ad.py `
  --dataset visa --category candle --dataset-root .\data\visa\1cls `
  --teacher-checkpoint .\artifacts\weights\wide_resnet50_2-95faca4d.pth `
  --student-checkpoint <正式RD4AD checkpoint> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\visa_v1 `
  --output-dir <空的正式输出目录> `
  --seed <注册seed> --run-purpose formal_full --device cuda:0
```

## 已完成验证

- 训练入口另有 5 项专项测试：`max_steps`/正式 batch 门禁、作者 loss 数值、normal-only validate、单 step CPU 小替身闭环、非有限 loss 失败且无最终 checkpoint。没有使用正式 test 图像。
- 原有 7 项 RD4AD 推理专项测试覆盖 MVTec/VisA 合成目录、JPEG 与两种 mask 命名、batch anomaly map、shape 拒绝、安全 checkpoint、vendor import、raw-map 落盘后统一评测。
- 集成后的项目全量测试为 72 passed；12 条 warning 均来自既有 DRAEM/imgaug 测试中的 Pillow `ImageOps` 弃用提示。
- 真实 VisA `candle/train/good` 的 `--validate-only` 通过：900 张 normal-only 图片、105,056,660 bytes，输入内容摘要 `cae1f03f...b7ad1d5`；教师完整 SHA 和 strict load 通过，`training_started=false`、`test_split_accessed=false`。证据保存于 `artifacts/validation/rd4ad_train_validate_visa_candle.json`，文件 SHA-256 为 `b88dcf256e672ebed56b98d39c03dc5f769f0e3eeb7060f1492a840e28145dd6`。
- 真实 MVTec AD `bottle/train/good` 的相同校验也通过：209 张 normal-only 图片、111,806,381 bytes，输入内容摘要 `aa76e6d9...ecf03284`，同样记录 `training_started=false`、`test_split_accessed=false`。证据为 `artifacts/validation/rd4ad_train_validate_mvtec_bottle.json`，SHA-256 `00995210dcc4f54d9599f517257d1e0903bf07ea6774530e1bbcab03dd88122c`。
- 同一权重已实际 strict-load 到冻结 vendor encoder；missing、unexpected、shape、不安全 dtype 和非有限 tensor 均为 0。该 smoke 没有构造 optimizer、没有访问数据集、没有训练，证据为 `artifacts/validation/rd4ad_vendor_teacher_load.json`，SHA-256 `2339855d463beafcc7e7d52db8bf632d36902cd5297353029a9925076173277b`。
- 无权重 CPU smoke：1×3×64×64 输入在 PyTorch 2.7.0 上完成作者 WideResNet50-2 → bottleneck → reverse decoder 前向；三层 teacher/student shape 分别为 `[1,256,16,16]`、`[1,512,8,8]`、`[1,1024,4,4]`；输出 `[1,64,64]` 且全部有限。
- 参数只用于结构核对：encoder 68,883,240；bottleneck 67,277,824；decoder 24,917,504。该 smoke 未加载任何权重或数据集图片。

## 正式复现前仍需通过的门槛

1. 每类别训练产生的 `{bn, decoder}` checkpoint 必须记录 seed、epoch、数据 manifest 与 SHA-256。
2. 在一个 MVTec 类和一个 VisA 类先做不计入主表的 8 GiB 显存 smoke，实测并冻结 batch、workers、峰值显存与耗时。
3. 正式结果按预注册 seed 和完整类别集合运行，200 epochs 后只评最终 checkpoint；不得按测试集挑 epoch。

推理 CLI 已就绪；真实正式推理仍需等待方法表、配方、源码与 F0 清单全部冻结：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_rd4ad.py --help
```
