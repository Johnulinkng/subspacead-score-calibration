# MVTec AD 安全解压与机械审计

状态：正式归档已完成安全解压与机械复核。`manifests/mvtec_prepared.json` 记录 15 类、3,629 张正常训练图、467 张正常测试图、1,258 张异常测试图及 1,258 张一一配对掩码；未解码或查看正式测试图像。

## 执行命令

归档下载完整后，从项目根目录运行：

```powershell
& .\scripts\Set-ProjectEnvironment.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_mvtec_ad.py
```

默认输入、输出与证据文件分别是：

- `data/archives/mvtec_anomaly_detection.tar.xz`
- `data/mvtec_ad`
- `manifests/mvtec_prepared.json`

脚本从 `manifests/mvtec_source_manifest.json` 读取官方归档的预期大小与 SHA-256。目标目录或正式 manifest 已存在时，默认模式会拒绝覆盖。若数据树已经安全安装而 manifest 需要重建，可显式使用 `--audit-only`；该模式仍要求归档大小和 SHA-256 完全匹配。

## 解压安全门

正式文件进入 `data/mvtec_ad` 前依次经过：

1. 归档大小必须为登记值，随后完整计算并核对 SHA-256。
2. 所有 tar member 只能是普通文件或目录；拒绝软链接、硬链接、设备、FIFO 等特殊项。
3. 拒绝绝对路径、`..`、反斜杠、Windows drive/ADS 冒号、保留设备名、末尾点或空格、大小写冲突、重复路径和“穿过普通文件”的路径。
4. member 数上限 100,000，解压文件字节上限 20 GiB；同时在目标盘检查保守可用空间。
5. 内容先流式写入 `data` 下随机命名的隔离 staging；每个文件核对声明字节数。任何失败都会只清理经过父目录和前缀复核的 staging。
6. staging 完成全部结构审计后，才在同一卷内原子移动为 `data/mvtec_ad`；既有目标永不覆盖。

归档可以把 15 类直接放在根目录，也可以包含一个共同 wrapper。最终数据根统一为 `data/mvtec_ad/<category>`。

## 15 类结构与配对审计

审计要求类别集合精确等于项目冻结的 15 类。每类必须满足：

- 类别根目录恰有 `train`、`test`、`ground_truth` 三个子目录；
- `train` 只能含 `good`，且至少一张非空 PNG；
- `test` 必须含 `good` 和至少一种异常类型；
- `ground_truth` 的异常类型集合必须与 `test` 的异常类型集合精确相等；
- 每张异常测试图必须有唯一 `<test-stem>_mask.png`，不能缺失，也不能有孤立 mask；
- 每个图像目录只能含非空 PNG，不允许嵌套目录或其他后缀。

manifest 记录逐类 `train/good`、正常/异常 test、各异常类型、mask 数量，以及全数据总计。另对“相对路径 + 文件大小”生成 metadata-tree SHA-256，便于发现目录漂移；同时记录来源 manifest、准备脚本和审计模块自身的 SHA-256。该摘要不读取图像像素。

## 内容访问边界

这个流程会读取压缩归档以计算 SHA-256，并把普通文件字节流复制到隔离目录；它只使用 tar metadata、文件路径、文件大小和文件名做掩码配对。代码不调用 PIL、OpenCV 或其他图像解码器，不展示正式 test 图像，不运行模型，不计算性能指标，也不训练。

成功后 `manifests/mvtec_prepared.json` 会明确记录这些边界。未出现该文件、或其中任一 hash/结构门不通过时，MVTec AD 仍视为不可用于实验。
