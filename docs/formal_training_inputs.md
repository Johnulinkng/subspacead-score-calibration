# 正式训练输入 bundle

`manifests/formal_training_inputs/v1/manifest.json` 是 MVTec AD、VisA 正式训练输入与 DRAEM 所用 DTD 辅助纹理的内容寻址清单。它只允许以下目录：

- MVTec AD：各注册类别的 `train/good`；
- VisA：各注册类别的 `train/good`；
- DTD：`data/dtd/images` 下的全部 47 类纹理，仅供 DRAEM 合成异常。

构建器不会枚举或打开任何 `test`、`ground_truth` 内容。根清单将每个子清单的行数、字节数和 SHA-256，与三个数据来源 manifest 的 SHA-256、builder/validator 源码 provenance 一并绑定。每个文件行保存项目根目录相对的 POSIX 路径、size 和内容 SHA-256；不保存机器绝对路径，因此项目整体复制后仍可重验。

构建命令：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe scripts\build_formal_training_inputs.py
```

输出目录必须不存在、为空，或恰好包含上一版完整 bundle；只有第三种情况下构建器才允许原位重建，任何额外文件、目录或链接都会拒绝。当前 v1 根清单 SHA-256 为 `8c40abce70270309f9b3f8f9135aa559c07b4849a97957722b2e479e01836e8a`，并已预冻结于 `configs/project.json`。它覆盖 12,288 张 normal training images（MVTec AD 3,629；VisA 8,659）和 5,640 张 DTD 图像。

正式训练时，resolver 首先要求根路径和 SHA-256 与项目配置的预冻结值完全相同。DRAEM 随后重新枚举所选类别的 `train/good` 和全量 DTD，逐项比较清单并重算所有文件哈希；PatchCore 与 RD4AD 只重验所选类别的 `train/good`。缺失、多余、字节变化、重新构建但未注册的新根、类别错配、路径逃逸、symlink 或 Windows reparse point 都会终止运行。开发协议 `visa_train_dev_dtd_v1` 使用自己的冻结清单，不经过这一正式 bundle，因而没有被改变。

DRAEM、PatchCore 与 RD4AD 的 schema-v2 checkpoint/fitted-state manifest 在 `formal_full` 时保存 `formal_training_inputs`。`validate_formal_checkpoint_manifest()` 在正式推理和结果聚合阶段会根据当前项目根目录再次核验根清单、子清单、来源 manifest、validator provenance 和当前训练源文件。这只解除原先“正式训练输入不可重验”的硬阻断；当前正式方法表、候选方法、消融和配方仍为 pending，因此真实 `formal_full` 仍然 fail-closed，不能据此启动正式训练或生成性能声明。
