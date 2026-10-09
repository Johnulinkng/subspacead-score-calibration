# 训练启动门禁

每次 GPU 训练或推理启动前运行：

```powershell
.\scripts\Test-TrainingPreflight.ps1
```

脚本检查并把带时间戳的 JSON 保存到 `artifacts/preflight`。无实测 profile 时的 8 GiB 是保守估计，不是算法最低要求。DRAEM S0 使用 `configs/resource_budget.json` 指向的 `draem_fp32_256_batch8_bounded_cache_20261009.json`；完整开发训练使用同目录的 `draem_real_s0_confirmed_20261009.json`。按工作集、提交空间和显存峰值加余量计算阈值。它检查当前 RAM、提交空间、项目盘、GPU、外部计算进程和项目锁；WSL 列为观测信息。任一检查失败退出码为 2。profile 失败、不完整、执行形状不匹配或源码漂移时拒绝使用。

`-CapacityDiagnostic` 只表示固定合成诊断所需的 bootstrap 条件，不授予训练资格；报告 `ready_for_training_launch=false`。受监控启动器在该模式只允许运行固定诊断脚本，两步、学习率 0，最长 180 秒。合成 profile 只用于真实数据 S0 准入。完整运行使用 `freeze_draem_resource_profile.py` 生成的 `draem_real_s0_capacity_profile`，重验 S0 配置/完成状态/源文件/监控 SHA，同时通过注册配方门禁；不会永远回退成固定的 8 GiB。

该快照只对记录时刻有效。训练入口启动前应再次执行，不能用历史报告替代当前状态。debug smoke 也保留报告，并在运行配置中继续标记为不可用于性能结论。
