# 正式测试清单

`formal_test_inventory` 只在 F0 正式评估入口创建。它不是开发集工具，也不能在候选模块、超参数或种子仍可能变化时运行。

构建器首先检查以下条件：

- `configs/project.json` 中正式评估协议已经冻结；
- `configs/preregistered_experiments.json` 中方法列表、DRAEM 主基线、候选方法以及每个方法的训练和推理配方均已冻结；
- 数据集只允许 `mvtec_ad` 或 `visa`，类别集合与项目登记和准备清单完全一致；
- 数据集、输出目录、配置和清单均位于项目目录内。

全部条件满足后，构建器枚举该数据集的完整官方测试集，为每张输入图和异常掩码记录 SHA-256，并按预先冻结的原始掩码面积比例生成 `small`、`medium`、`large` 分层。阈值在 [formal_test_inventory_protocol.json](../configs/formal_test_inventory_protocol.json) 中固定：面积不超过 1% 为 small，超过 1% 且不超过 10% 为 medium，超过 10% 为 large。正常图固定为 `normal`。这个分层只服务于预注册可视化抽样，不修改指标使用的掩码。

输出目录包含：

- `inventory.json`：可视化和各方法推理共同使用的完整样本身份、相对路径、标签、缺陷类型、面积分层和源文件哈希；
- `receipt.json`：绑定项目配置、正式登记、分层协议、数据准备清单和 inventory 本身的哈希，并明确记录正式 test/GT 已在 F0 被访问，但模型预测尚未访问。

正式冻结前运行构建命令会失败，且不会生成清单。冻结后的命令形式为：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --dataset mvtec_ad `
  --dataset-root .\data\mvtec_ad `
  --output-dir .\manifests\formal_evaluation\mvtec_ad_v1
```

正式推理前应重验当前文件集合和内容：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --output-dir .\manifests\formal_evaluation\mvtec_ad_v1 `
  --validate-only
```

## 正式推理绑定

DRAEM、PatchCore 和 RD4AD 的 `formal_full` 推理必须显式传入同一数据集的 bundle 目录，例如：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_draem.py `
  --dataset mvtec_ad --category bottle `
  --checkpoint-dir <冻结的DRAEM checkpoint目录> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\mvtec_ad_v1 `
  --run-purpose formal_full --seed <注册seed> --device cuda:0
```

入口会先重建并验证整个 bundle，再从已认证的 `inventory.json` 选择当前类别的完整条目。模型 Dataset 接收这份显式样本列表，不会再次遍历 `test` 目录来另建样本集合。`debug_limited` 和 `author_reproduction` 继续使用原有目录发现逻辑；`development_full` 仍禁止进入真实 test 推理入口。

正式运行保存 `formal_test_inventory_category_selection_v1` 证据，其中包含项目相对 bundle 路径、inventory ID、inventory/receipt SHA-256、数据集、类别、全量条目数和类别条目数。相同证据会写入 `run_config.json`、`metrics.json`、完成态 `run_state.json` 和 `formal_provenance`。逐样本预测索引还绑定源图/源掩码 SHA-256 以及 inventory/receipt SHA-256，并要求样本顺序、ID、标签、缺陷类型和相对路径与清单逐项相同。

聚合器不会只相信运行时的“已验证”字段。它会再次验证完整 bundle 与当前源文件、比对配置/指标/完成态三处证据，并逐行核对预测索引。缺少 bundle、数据集或类别错误、类别不全、路径逃逸、清单/receipt 篡改、源图或掩码漂移都会失败关闭。

当前方法表和候选方法尚未冻结，因此项目中不会提前生成 MVTec AD 或 VisA 的真实正式测试清单，也不会提前解码 test/GT 内容。
