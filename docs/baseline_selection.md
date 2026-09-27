# 基线选择记录

记录日期：2026-09-27。范围：另选公开项目开展工业异常检测研究。

## 选择标准

依次检查来源和许可证、完整训练/推理链、数据及权重可得性、社区复现反馈、工程维护、与待研究问题的匹配、资源成本。Stars 和 forks 只说明关注度；他人 issue 中的成绩不是本项目验证结果。

## 初步决定

先复现 DRAEM 作者实现。理由是其重建与判别结构适合研究异常定位、背景误报和评分改进，提供正式论文、MIT 代码和发布权重入口。Anomalib 是较成熟的工程参考，但引入其实现前必须登记与作者配置的差异。

这个选择尚不意味着已验证能超过 DRAEM。若研究目标改为多尺度特征蒸馏，可在测试结果揭示前选择 RD4AD，并保存切换理由。较强参照保留 PatchCore，最终成绩不只与单一较早方法比较。

| 项目 | 定位 | 当前可核查事实 | 需要处理的事项 |
| --- | --- | --- | --- |
| DRAEM | 首个复现候选 | 作者代码，MIT；GitHub API 快照为 294 stars / 71 forks | 旧 Linux Conda 环境；DTD 外部纹理；完整训练成本未测量 |
| RD4AD | 条件备选 | 作者代码，MIT；254 stars / 37 forks | 旧环境；评测与可视化特征层不一致；PRO 汇总实现需要审计 |
| Anomalib | 工程参考 | Apache-2.0；6201 stars / 995 forks；API 返回最新正式发布为 `lib/v2.6.2` | 库实现不能直接视为作者实现；须固定版本并核对损失、优化器、预处理及评测 |
| PatchCore | 较强参照 | 作者代码，Apache-2.0；1401 stars / 257 forks | 保留合理官方配置，冻结 backbone/图像尺寸/coreset 和评分规则 |

快照见 `manifests/repository_snapshot.json`。API 字段 `open_issues_count` 包含未关闭 PR，不能直接当作未解决问题数量。远端 `pushed_at` 也不等于默认分支最后一次有效代码更新。

## DRAEM 本地源码审计

仓库提交：`2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`。

- `requirements.txt` 是带 Linux 构建号的 Conda 导出，包含 Python 3.8、PyTorch 1.8.0 / CUDA 10.2；不是可直接在 Windows 上执行的 pip requirements。
- README 训练示例为 700 epochs、batch 8、lr 1e-4、256 像素；发布权重示例名称含 `800`。两者需要区分，不能用 700 轮示例推断发布权重训练协议。
- `train_DRAEM.py` 固定 16 个 DataLoader workers，缺少显式多种子运行入口；本机资源适配和确定性控制需在项目适配层中实现，并记录差异。
- 图像级分数来自异常概率图的 21×21 平均池化后最大值；像素级指标使用未池化的异常概率图。图像级聚合本来就可以不同，但需要完整记录。
- 原测试脚本计算 AUROC/AP，未提供本项目统一 AUPRO 评测；后续需实现或引入经过验证的统一评测器。
- 数据与权重下载脚本包含旧入口和直接安装命令。本轮仅检查脚本，未执行；实际下载前重新验证来源和内容。

## 社区反馈的用途

- DRAEM issue #7 中有人报告发布权重的逐类结果与论文不一致；本项目尚未复现该报告。它提示必须记录逐类结果，不能仅核对总体均值。
- RD4AD issue #22 讨论论文损失表述和代码展平余弦损失的差异；修正实现之前先明确要复现“作者代码”还是“重新解释论文”。
- 已检查的 RD4AD 官方评测使用多层相加，visualization 只使用最后一层。若选该方法，评测和最终展示必须改为共享原始预测文件，并保留原仓库核对结果。

## 直接来源

- DRAEM：https://github.com/VitjanZ/DRAEM
- DRAEM 复现讨论：https://github.com/VitjanZ/DRAEM/issues/7
- RD4AD：https://github.com/hq-deng/RD4AD
- RD4AD 损失讨论：https://github.com/hq-deng/RD4AD/issues/22
- RD4AD 评测代码：https://github.com/hq-deng/RD4AD/blob/main/test.py
- Anomalib：https://github.com/open-edge-platform/anomalib
- Anomalib 发布：https://github.com/open-edge-platform/anomalib/releases/tag/lib/v2.6.2
- PatchCore：https://github.com/amazon-science/patchcore-inspection
- MVTec AD：https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- DTD：https://www.robots.ox.ac.uk/~vgg/data/dtd/

下载后的代码与数据分别保留各自许可证及使用范围。数据来自真实公开基准；合成异常仅按公开训练/开发协议使用，不替代真实测试集。
