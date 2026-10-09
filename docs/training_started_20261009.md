# 2026-10-09 训练启动记录

北京时间 09:40:53，完整 DRAEM 开发基线在 `E:\3k` 后台启动。用户当天明确要求“今天启动推进”，此前跨项目串行等待顺序由这条新指令覆盖。

## 已执行的真实训练

VisA candle 两步 train-only S0 使用 720 张冻结正常训练图与 4,512 张 DTD-train 纹理。GPU 完成前向、反向与 Adam 参数更新，并保存 reconstructive/discriminative 两份模型及哈希清单。未打开正式 test/GT。

第一次 S0 成功，但显存缓存峰值 7.61 GiB，不适合保留运行余量。释放完成批次的中间张量并限制 CUDA 缓存后，第二次真实 S0 的预留峰值降至 5.92 GiB，张量峰值 5.40 GiB；两步所有损失分量与第一次一致。初始证据与重复证据分别保留，不覆盖。

- 初始短测：`runs/monitored/draem-s0-visa-candle-seed1887537248-20261009T093335`。
- 确认短测：`runs/monitored/draem-s0-visa-candle-seed1887537248-20261009-bounded-cache`。
- 完整运行资源 profile：`artifacts/resource_profiles/draem_real_s0_confirmed_20261009.json`。
- 同一数据划分的源码重冻记录：`artifacts/validation/resource_refreeze_launch_20261009/receipt.json`。

## 当前完整开发训练

采用作者网络、256 分辨率、batch 8、FP32、Adam 学习率 1e-4、700 轮、final checkpoint only；workers=0、pin_memory=false、BatchNorm 安全的激活重计算。CUDA 缓存上限为 6.5 GiB、启动时空闲显存减 0.75 GiB、总专用显存的 90% 三者较小值。实际值和完整配方记录在每次运行的 `run_config.json`。

队列登记 VisA 12 类 × 3 个固定种子，共 36 组完整基线训练。先执行种子 1887537248 的全部类别，再执行 1337854654 和 1609762728。每组启动前都重查资源与冻结输入；失败保留证据并停止队列，后续任务不重复覆盖。

- 队列状态：`runs/queues/draem-s1-20261009T014053Z/state.json`。
- 后台启动记录：`runs/launchers/draem-s1-20261009T014053Z/launch.json`，启动 PID 5396。
- 首组训练：`runs/monitored/draem-s1-20261009T014053Z-candle-seed1887537248`，启动时实际 Python PID 19572。
- 实际参数更新进度：上述目录的 `model/train_metrics.jsonl`；`run_state.json` 在初始化和结束/失败时更新，不能用其初始 step=0 判断训练停滞。
- 运行资源：上述目录的 `resource_samples.jsonl`；训练进程和桌面/WSL 的占用同时记录。

09:42 的检查已观测到第 28 步，GPU 缓存峰值保持 5.92 GiB。PID 和进度均是当天快照；后续核查应同时核对命令、路径、锁 token、日志增长与当前进程。

最初一次后台启动用了 Windows PowerShell 5.1，因当前会话模块路径下找不到 Get-FileHash，在训练创建前退出。证据保留于 `runs/launchers/draem-s1-20261009T014022Z`。改用当前已验证的 PowerShell 7 后实际训练正常启动。

本阶段是完整开发训练。后续仍需合成开发评测、候选设计、对比与消融，最后按冻结方案完成 MVTec AD 和 VisA 正式评测；启动成功与训练损失不能替代性能提升结论。
