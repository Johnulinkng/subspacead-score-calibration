# 资源恢复与低占用运行记录

更新日期：2026-10-09。项目主目录 `E:\3k`。此前“启动前必须有 8 GiB 空闲主机内存”是未经过容量实测的保守估计，不是 DRAEM 算法要求。现在 DRAEM S0 使用实测峰值加余量；科研指标、数据、作者网络、batch、轮数和种子要求保持原有口径。

## 当前门禁

正式或完整开发运行由 `scripts/Invoke-MonitoredExperiment.ps1` 先执行
`scripts/Test-TrainingPreflight.ps1`。DRAEM S0 使用 `configs/resource_budget.json` 指向的实测 profile。脚本检查当前 RAM、提交空间、专用显存、项目盘、外部计算进程和项目锁。Codex 桌面 GPU 进程只在系统查询确认其为安装包内界面程序时排除；其他计算进程仍检查冲突。WSL 状态是观测信息，不单独阻止资源充足的运行。门禁失败不会创建训练子进程或 checkpoint。

2026-10-07 的历史检查为：可用内存 6.759 GiB、项目盘可用空间 214.054 GiB、GPU 显存空闲 8018 MiB、无计算进程、无 WSL、无训练锁，当时因固定 8 GiB 估计拒绝启动。旧证据保存于
`artifacts/preflight/preflight_20261007T151053011Z.json`。

## 本机实测与预算

RTX 3070 Ti Laptop（8 GiB 专用显存），完整作者网络、FP32、256 px、batch=8，合成输入，两次前向/反向/Adam 分配。诊断学习率为 0，不读取数据集，不保存模型，不产生性能结果。

| 执行路径 | 结果 | GPU 张量峰值 | GPU 缓存预留峰值 | 主机工作集峰值 |
|---|---|---:|---:|---:|
| 原始路径 | 前向时触发 CUDA 分配上限 | 6.09 GiB | 6.12 GiB | 1.26 GiB |
| 激活重计算 | 两步通过 | 5.27 GiB | 5.92 GiB | 1.67 GiB |

原始失败说明它不能在本机保留显存余量的配置中运行，不能把 6.09 GiB 当作完整训练峰值。诊断设置显存分配上限，避免 Windows 将专用显存不足转换成共享主机内存压力。

通过证据：`artifacts/resource_profiles/draem_fp32_256_batch8_checkpointed_20261009.json`。原始失败证据：同目录 `draem_fp32_256_batch8_original_failed_20261009.json`。完整日志在 `runs/monitored/draem-capacity-20261009-fp32-b8` 和 `runs/monitored/draem-capacity-20261009-fp32-b8-final`，失败不覆盖、不改成成功。

- 空闲物理内存预算 = 2 GiB 系统余量 + 0.5 GiB 数据加载/保存余量 + 1.25 × 实测工作集峰值，本次 **4.587 GiB**。
- 可用提交空间另查，同样加余量，本次约 **14.471 GiB**。Windows 的进程提交量包含 CUDA/WDDM 分配；9.58 GiB 提交峰值不等于 9.58 GiB 物理主机内存。不会自动修改系统页文件。
- 专用空闲显存至少为缓存预留峰值加 0.75 GiB，即 **6830 MiB**；项目盘继续保留 40 GiB。
- profile 验证源码 SHA、依赖锁、Torch 版本和显卡型号；源码或执行形状变化时重新测量。

## 已采用的低占用设置

- DRAEM、RD4AD 和 PatchCore 的项目层 DataLoader 默认 `workers=0`。这避免 Windows 多进程 worker 复制父进程数据集和预取队列。
- DRAEM 保留原有的 CUDA 自动 pinned-memory 行为，同时支持在资源 smoke 中显式传入 `--no-pin-memory`；最终解析值写入 `run_config.json`，避免运行口径不可追溯。
- S0 启用 `--activation-checkpointing`，在项目层重计算作者网络的 29 个卷积/BN 块，不改作者源码、参数键名、batch、图像尺寸或精度。重计算后恢复 BN 运行缓冲区，避免统计更新两遍。CPU fixture 核对输出、梯度、Adam 后参数和 BN 状态；完整作者宽度的 GPU 容量诊断已通过。这是运行资源优化，不算论文方法创新。
- PatchCore 的 FAISS 查询 worker 在正式 smoke 前不盲目提高；设备分工和 worker 数必须通过受监控 smoke 冻结。
- 正式训练仍保留完整作者轮数、统一输入尺寸和预注册随机种子。降低 batch、轮数或关闭数据增强只能标为 debug/smoke，不能进入正式结果。

## 公开资料依据

PyTorch 官方 DataLoader 文档说明，多 worker 的总内存开销会随 worker 数量和父进程大小增长，并且 Windows 使用 spawn，启动行为不同于 Unix。项目因此默认使用 `num_workers=0`，同时保留显式配置并在 run_config 中记录。参见：
<https://docs.pytorch.org/docs/stable/data.html>

激活重计算以额外计算换取显存：[PyTorch checkpoint](https://docs.pytorch.org/docs/2.7/checkpoint.html)。峰值使用 [CUDA 内存统计](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.max_memory_allocated.html)。本地一致性检查与容量测量不替代真实数据的训练或性能证据。

## 恢复顺序

1. 使用同一测量 profile 执行新的 preflight；S0 wrapper 和等待器都读取同一预算，不再独立等待 8 GiB。
2. 用 VisA candle 真实 train-only S0 确认数据增强、日志、模型保存和实际峰值。合成两步不算真实 S0 或长时间稳定性证明。
3. 按真实 S0 峰值冻结完整运行资源配方，基线、改进和消融采用相同执行策略；其他方法单独测量自己的预算。
4. 然后按完整轮数、类别和三种子开发矩阵推进，不将未经实测的 8 GiB 重新当作必要下限。

真实 S0 完成后执行 `scripts/freeze_draem_resource_profile.py --monitored-run <S0监控目录> --output artifacts/resource_profiles/draem_real_s0_confirmed.json`。脚本重验运行配置、两步完成状态、源文件、监控/日志和 checkpoint SHA，取合成与真实 S0 峰值的较大值。完整运行向 `Invoke-MonitoredExperiment.ps1` 传入这个已确认 profile，继续使用动态预算；不会回退成必须空闲 8 GiB。完整命令必须显式保持 256、batch=8、workers=0、no-pin-memory、cuda:0 和相同重计算策略，同时仍须通过实验协议/配方门禁。

2026-10-09 用户明确要求“今天启动推进”，覆盖此前等待另一研究项目完成的会话顺序。当天真实 VisA candle 两步训练通过，随后于北京时间 09:40 启动完整 700 轮开发基线，队列包含 12 类与 3 个固定种子。旧等待器的暂停记录保留为历史证据。启动路径与状态见 `training_started_20261009.md`。

真实短测还发现：保留上一批中间张量与未限制的 CUDA 缓存会使第二步预留峰值达到 7.61 GiB。训练循环现在在日志落盘后释放已用完的张量，并把缓存上限设为 6.5 GiB、当前空闲显存减 0.75 GiB、显卡总量的 90% 三者中较小者，实际值记录在 `run_config.json`。重复真实短测的预留峰值为 5.92 GiB，两步各损失分量与原短测相同；此项观察不构成模型性能或 GPU 逐位数值一致性结论。全部 14 个数据划分与合成配方子清单 SHA 保持不变，源文件 provenance 已显式重冻。

任何只为绕过门禁而减少数据、轮数、类别或随机种子的运行都不进入正式矩阵，也不用于支持改进结论。
