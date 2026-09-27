# VisA 官方 1-class 数据适配

## 固定来源

- 数据与划分来源：Amazon Science `spot-diff` 官方仓库。
- 本地只读快照：`third_party/spot-diff`，commit `2a692ab575001cbde74d402d897a7286086c6199`。
- 代码许可：Apache-2.0；数据许可：CC BY 4.0。交付数据派生物时须保留归属说明。
- 正式实验只采用 `split_csv/1cls.csv`，不自行重抽样，也不把测试异常用于训练或调参。

源代码和数据集许可证不同。`LICENSE` 适用于工具代码，`LICENSE-DATASET` 适用于 VisA 数据。

## 固定划分

官方 1-class CSV 有 10,821 行：9,621 个正常样本和 1,200 个异常测试样本。12 个类别各有 100 个异常测试样本；训练集中只有正常样本。逐类别数量记录在 `manifests/visa_source_manifest.json`，并由 `src/industrial_anomaly_lab/visa.py` 执行校验。

`utils/prepare_data.py` 的行为是：

1. 按官方 CSV 复制图像；
2. 把 `normal` 映射为 `good`，把 `anomaly` 映射为 `bad`；
3. 把原始多类掩码二值化为 0/255；
4. 生成与 MVTec AD 相同的 `train/test/ground_truth` 目录结构。

该官方脚本在导入时直接解析命令行参数，且直接写目标目录。项目不修改它，而是由 `scripts/Prepare-VisA.ps1` 在临时目录中运行，并在发布到 `data/visa/1cls` 前检查完整性。

## 准备命令

归档完成下载、哈希核验并解压到 `data/visa/raw/VisA` 后，在项目根目录运行：

```powershell
.\scripts\Prepare-VisA.ps1
```

若解压根目录不同，可显式传入，但路径必须位于项目目录内：

```powershell
.\scripts\Prepare-VisA.ps1 -RawRoot "data/visa/raw/<actual-root>"
```

脚本会固定检查 vendor commit、CSV SHA-256 和官方准备脚本 SHA-256。已有 `data/visa/1cls` 时脚本会停止，避免静默覆盖。成功后生成 `manifests/visa_prepared.json`。

仅审计固定划分可运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_visa.py
```

完整目录审计会核对：

- 10,821 张图像和 1,200 张掩码与 CSV 一一对应；
- 不缺文件、不多文件、无空文件；
- 所有异常掩码只含 0/255 且至少有一个异常像素；
- 12 个类别的训练正常、测试正常、测试异常数量精确匹配官方划分。

## 训练和评估接口

训练/推理层把 `data/visa/1cls/<category>` 当作 MVTec 风格类别根目录即可。正常测试图没有实体掩码，加载时应构造全零掩码；异常图掩码路径为 `ground_truth/bad/<stem>.png`。评价按类别计算后做 12 类宏平均，并保留逐类结果。VisA 的正式结果不得与 MVTec AD 的 15 类均值混成一个 27 类均值。

为避免测试集泄漏，模型选择和超参数选择只能使用预先冻结的训练侧开发协议；官方 test 只用于冻结配置后的最终报告。
