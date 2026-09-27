# 训练启动门禁

每次 GPU 训练或推理启动前运行：

```powershell
.\scripts\Test-TrainingPreflight.ps1
```

脚本只读检查并把带时间戳的 JSON 保存到 `artifacts/preflight`。门禁包括可用物理内存、E 盘空间、GPU 状态、GPU 计算进程、正在运行的 WSL 发行版，以及 `runs/.training.lock.json` 对应进程。任一检查失败时退出码为 2；不得把失败门禁后的运行记为正式实验。

该快照只对记录时刻有效。训练入口启动前应再次执行，不能用历史报告替代当前状态。debug smoke 也保留报告，并在运行配置中继续标记为不可用于性能结论。
