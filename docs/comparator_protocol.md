# 强对照方法冻结协议 v1

状态：方法与作者源码已冻结；项目层适配、PatchCore 可哈希 memory-bank 拆分运行器、FAISS/timm 运行依赖和共用的官方 WideResNet50-2 V1 权重已核验。尚未通过 8 GiB GPU 资源 smoke，也未在真实数据上拟合、推理或产生性能结果。

## 选择结论

正式对照保留三条互补路线：DRAEM 是重建/判别机制基线；**PatchCore** 是特征记忆强对照；**RD4AD** 是表征蒸馏强对照。后两者均为 CVPR 2022 作者开源实现，分别使用 Apache-2.0 与 MIT 许可证。它们不会替代主改进对象，也不能用作者论文值代替本项目复现值。

| 对照 | 技术位置 | 冻结作者实现 | 初始作者兼容配置 | 8 GB 约束 |
| --- | --- | --- | --- | --- |
| PatchCore | ImageNet 中层 patch 特征、coreset 记忆库、最近邻评分 | `amazon-science/patchcore-inspection@fcaa92f` | WideResNet50，layer2+layer3，resize 256 / crop 224，embedding 1024→1024，patch 3，1-NN，10% approximate greedy coreset，seed 0 | batch 2；特征网络用 GPU，FAISS 优先 CPU；coreset 设备在正式运行前以 smoke 实测冻结。作者仓库中的 LFS 模型仍只保留指针；项目使用单独核验的 torchvision WRN50-V1 本地权重。 |
| RD4AD | 冻结教师编码器、可训练 one-class bottleneck 与反向解码器，多层余弦差异 | `hq-deng/RD4AD@6554076` | WideResNet50-2，256×256，200 epochs，Adam lr 0.005 / betas 0.5,0.999，作者 seed 111，异常图为三层差异求和后 Gaussian sigma 4 | 先测试作者 batch 16；若超显存，按 8、4 递减并在正式协议中固定。批量变化会影响 BN，不能称为完全作者等价；不得用 AMP/缩短轮数冒充正式复现。 |

## MVTec AD 与 VisA 的公平口径

- 两个数据集都按 one-class、逐类别训练；只用正常训练图。MVTec AD 全 15 类和 VisA 官方 1-class 全 12 类分别宏平均，不增加第三个数据集。
- 两个作者仓库原生都只提供 MVTec 数据入口。VisA 必须通过项目适配层读取已冻结的 `data/visa/1cls` 清单；不改第三方源码的类别语义，不把目录重排当成新数据版本。
- 保留每种方法的作者预处理作为方法配置，并统一把最终浮点异常图映射到 `industrial_anomaly_raw_map_max_256_v2` 的 256×256 评测平面。GT 始终最近邻缩放；预测图只做一次双线性重采样，不逐图 min-max。
- 统一读取同一份样本清单并保存 `sample_id`、数据集/类别、标签、GT 引用、原始异常图、`raw_map_max_score`、`method_native_image_score`、配置与源码 commit。主表 I-AUROC、P-AUROC、P-AP、AUPRO@0.30 全部由项目评测器基于落盘 raw map 重算；作者脚本结果仅放在 native/作者口径核对列。
- PatchCore 的作者图像分数是 patch 最近邻分数的作者聚合；RD4AD 的作者图像分数是平滑后异常图 raw max；DRAEM 的作者分数是 21×21 平均池化后最大值。三者只进入 `image_auroc_method_native`，共同 `image_auroc` 一律使用落盘 raw map 最大值。
- 不使用测试集决定 coreset 比例、层、平滑 sigma、epoch、batch 或 checkpoint。RD4AD 作者代码每 10 epochs 评测测试集的做法在正式协议中禁用；固定 200 epochs 最终 checkpoint 后一次性评测。

## 资源与复现实验门槛

每个方法先在一个 MVTec 类和一个 VisA 类执行不计入正式结果的 smoke：记录 GPU 峰值、主机 RAM、耗时、memory-bank/权重大小和预测文件契约。只有峰值显存低于 8 GiB 且无 OOM，才能冻结正式 batch/设备分工。资源调整须同时用于两个数据集，不按类别选择有利配置。

PatchCore 的 coreset 随机投影和抽样受 seed 影响，因此按项目至少 3 个预注册 seed 完整运行。RD4AD 同样至少 3 seed。逐类、逐 seed 原始结果全量保留，不能用作者单 seed 或最好 seed 作为主比较。

## 尚未解决的边界

1. 两个官方仓库均无原生 VisA loader；RD4AD 与 PatchCore 项目适配器均已完成合成数据契约测试。PatchCore 绕开作者 pickle/隐式 backbone 加载，使用 `.npy + FAISS + hash-gated JSON` 拆分拟合和推理，细节见 `docs/patchcore_adapter.md`。
2. 8 GB 显存结论目前只是受约束配置设计，尚无峰值实测；PatchCore 的 coreset sampler 设备与 RD4AD batch 必须经 smoke 冻结。
3. RD4AD 的现代 Windows/PyTorch 项目层适配、显式本地教师权重 strict-load、训练入口和无权重 CPU 前向已验证，见 `docs/rd4ad_adapter.md`；GPU 资源和完整训练仍未验证。
4. PatchCore 官方运行器会对结果集合做 min-max，RD4AD 作者 AUPRO 是逐异常图平均；两者都不能作为本项目统一指标。正式结论只来自保存的未归一化异常图和已交叉验证的项目评测器。
5. PatchCore 与 RD4AD 共用的 ImageNet WRN50-V1 已从 torchvision 官方 HTTPS URL 取得，完整 SHA-256、dtype 扩展和 strict-load 证据见 `manifests/wide_resnet50_2_imagenet1k_v1.json`；运行仍须按该哈希 fail-closed。

源码与许可证证据见 `manifests/comparator_sources.json`；第三方目录保持干净，正式实现通过项目 wrapper 调用。
