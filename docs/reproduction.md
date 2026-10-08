# 无监督工业异常检测项目复现说明

## 当前证据边界

本项目只把 **MVTec AD 15 类**和 **VisA 12 类**作为正式验证数据集。DTD R1.0.1 仅为 DRAEM 合成异常提供纹理源，不是第三个验证数据集。DDAD、MDG-AD 以及此前的两阶段异常图融合不属于本项目。

截至本说明落盘时，环境、两个数据集的结构与内容清单、VisA 合成开发协议、统一指标实现、DRAEM/PatchCore/RD4AD 项目层入口、正式结果聚合器和正式可视化链已建立。**尚未完成真实基线训练、候选方法训练、正式 MVTec AD/VisA 推理或性能比较，也没有可交付的正式指标与正式可视化。** `configs/preregistered_experiments.json` 中候选方法和消融仍未冻结，`configs/project.json` 中正式训练配方与正式评测协议也仍标记为未冻结。因此，当前仓库不能被描述成已经复现基线、已经超过基线或已经满足论文效果标准。

`configs/delivery_spec.json` 是最终四项交付的顶层显式文件白名单。正式 figure manifest 是第二层、带 payload SHA-256 的 PNG 精确白名单；这样机器失败案例可在 F0 规则执行后落盘，而无需修改已绑定正式运行的 Git commit。当前运行 `--validate-only` 会失败关闭，并且不会创建 ZIP；这是正确行为，不是待绕过的错误。只有候选、消融、正式矩阵、对比表和两套可视化都完成且注册声明通过后，打包器才允许生成最终包。

## 目录和本地写入规则

项目根目录固定为 `E:\3k`。每个 PowerShell 会话先执行：

```powershell
Set-Location E:\3k
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
```

脚本会把 `TEMP`、`TMP`、uv、pip、Torch、Matplotlib、Numba、Hugging Face 和 Python bytecode 缓存重定向到项目目录。后续 Python 命令统一使用：

```powershell
.\.venv\Scripts\python.exe
```

不要把环境、数据、权重、临时文件或训练输出写到 C 盘。项目中的 `data/`、`artifacts/weights/`、`runs/`、`.cache/`、`.runtime/`、`.venv/` 和 `tmp/` 不进入交付 ZIP。

## 环境建立与核验

依赖版本由 `pyproject.toml` 和 `uv.lock` 固定。已有项目环境的只读核验命令为：

```powershell
.\scripts\Test-Environment.ps1
.\.venv\Scripts\python.exe -m pytest -q
```

需要重新建立环境时使用项目脚本：

```powershell
.\scripts\Sync-Environment.ps1
.\scripts\Test-Environment.ps1
```

`Sync-Environment.ps1` 可能需要从已配置的软件源取得锁定依赖；离线复现者应预先准备相同 wheel 缓存。第三方源码提交由 `.gitmodules` 和交付清单共同记录。已有 checkout 可离线核对：

```powershell
git submodule status
git -C third_party/DRAEM status --porcelain
git -C third_party/PatchCore status --porcelain
git -C third_party/RD4AD status --porcelain
```

交付包不会携带第三方仓库全文，而是在 `DELIVERY-MANIFEST.json` 中记录 DRAEM、PatchCore 和 RD4AD 的精确 commit、源码清单 SHA-256 与 clean 状态。

## 数据准备与核验

数据集和权重因体积与授权边界不进入交付包。复现者应把数据放在 `configs/project.json` 登记的位置，并用项目清单核验。

MVTec AD 已准备目录的只读结构、计数和元数据哈希核验：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_mvtec_ad.py `
  --audit-only `
  --source-manifest .\manifests\mvtec_source_manifest.json `
  --output-root .\data\mvtec_ad `
  --json-output .\artifacts\validation\mvtec_audit_reproduction.json
```

VisA 官方 1-class 布局核验：

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_visa.py `
  --split-file .\data\visa\split_csv\1cls.csv `
  --raw-root .\data\visa\raw `
  --prepared-root .\data\visa\1cls `
  --json-output .\artifacts\validation\visa_audit_reproduction.json
```

上述输出属于本地验证证据，不是正式效果。MVTec 的 `--audit-only` 不解码图像；VisA 审计会按适配器要求核对整理结果。数据来源、归档哈希和现有计数分别记录在 `docs/data_acquisition.md`、`manifests/mvtec_source_manifest.json`、`manifests/mvtec_prepared.json`、`manifests/visa_source_manifest.json` 与 `manifests/visa_prepared.json`。

## 开发协议和测试隔离

开发阶段只允许使用 VisA `train/good` 与内容隔离后的 DTD 纹理。重新构建开发清单的命令为：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_development_manifests.py `
  --config .\configs\development_protocol.json `
  --output-root .\manifests\development\visa_v1
```

该命令的输出目录必须按工具约束为空或由明确的重建流程接管。重新生成会改变根清单哈希；不能自动把新哈希写成已接受版本。必须先审计新清单，再显式更新 `configs/development_protocol.json` 的冻结哈希并提交该变化。

无需模型的 candle 输入校验和确定性渲染审计：

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_draem_synthetic_dev.py `
  --category candle --seed 1887537248 --validate-only `
  --output-dir .\artifacts\validation\synthetic_dev_candle_validate_reproduction

.\.venv\Scripts\python.exe .\scripts\evaluate_draem_synthetic_dev.py `
  --category candle --seed 1887537248 --render-audit-count 2 `
  --output-dir .\artifacts\validation\synthetic_dev_candle_render_reproduction
```

这些命令不读取真实 `test/ground_truth`，不加载模型，不计算性能。完整开发评测必须使用 `--evaluate` 和项目训练产生的两个 DRAEM checkpoint；所得结果固定标记为 `synthetic-development-only` 与 `claim_eligible=false`，不能写入正式对比表。

正式训练输入清单只遍历 MVTec AD/VisA 的 `train/good` 与 DTD：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_formal_training_inputs.py `
  --output-dir .\manifests\formal_training_inputs\v1
```

现有根清单已经生成。需要重建时先保留旧版证据，使用空输出目录，核对文件集合和根 SHA，再冻结新版本；不要以覆盖后的文件冒充原注册版本。

## 每次 GPU 运行前的资源门禁

任何训练或推理启动前都重新运行：

```powershell
.\scripts\Test-TrainingPreflight.ps1
```

只有退出码为 0 的**当前快照**才允许启动对应 GPU 进程。退出码 2 表示当前资源预算或训练锁未满足。8 GiB 是历史未实测的保守主机预算；DRAEM S0 已改为实测峰值加余量，约需 4.59 GiB 空闲主机内存，同时另查提交空间和专用显存。使用 `Invoke-DraemS0Smoke.ps1` 自动携带 profile，并启用保持 BN 状态的激活重计算。合成两步已通过，真实 S0 和长训练仍未运行；详见 `docs/resource_recovery.md`。

实际的长时训练和推理统一通过 `scripts/Invoke-MonitoredExperiment.ps1` 启动。该外层入口会自行执行同一 preflight、以原子方式持有 `runs/.training.lock.json`、隐藏启动项目 Python、记录命令/SHA，并周期记录主机 RAM/提交空间、子进程工作集和 GPU 状态。它不提供门禁绕过。下面各方法章节中的 Python 参数说明的是模型内层 CLI；真正运行时把这些参数逐项放进 `-ScriptArgument`。例如 DRAEM 开发运行的完整启动形状为：

```powershell
& .\scripts\Invoke-MonitoredExperiment.ps1 `
  -OutputDirectory .\runs\monitored\draem-visa-candle-seed1887537248 `
  -ScriptPath .\scripts\train_draem.py `
  -RunPurpose development_full `
  -ScriptArgument @(
    '--dataset', 'visa', '--category', 'candle',
    '--seed', '1887537248', '--run-purpose', 'development_full',
    '--input-protocol', 'visa_train_dev_dtd_v1',
    '--development-protocol-config', 'configs/development_protocol.json',
    '--image-size', '256', '--batch-size', '8', '--epochs', '700',
    '--learning-rate', '0.0001',
    '--output-dir', 'runs/monitored/draem-visa-candle-seed1887537248/model'
  )
```

每次使用新的空目录。launcher 的 `claim_eligible` 不替代方法运行自己的正式 provenance；资源监控通过也不代表结果可以进入正式主表。`-SelfTest` 仅运行固定的两秒 CPU 子进程并标记 `claim_eligible=false`，不能用于验证模型效果。

## DRAEM 运行边界

只检查真实 train-only 输入而不建模或训练：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle `
  --seed 1887537248 --validate-only
```

短步连通性检查必须显式不可声明：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle `
  --seed 1887537248 --run-purpose debug_limited `
  --max-steps 1 --output-dir .\runs\draem\debug\visa_candle_seed1887537248
```

开发阶段的完整 DRAEM 运行只能使用冻结的 VisA/DTD 开发协议：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_draem.py `
  --dataset visa --category candle `
  --seed 1887537248 --run-purpose development_full `
  --input-protocol visa_train_dev_dtd_v1 `
  --development-protocol-config .\configs\development_protocol.json `
  --image-size 256 --batch-size 8 --epochs 700 --learning-rate 0.0001 `
  --output-dir .\runs\draem\development\visa\candle\seed_1887537248
```

这只是单类别命令示例；S1/S3/S4 要按 `configs/preregistered_experiments.json` 的 12 类与三个注册 seed 完整执行。不能把单类、单 seed 或中间 epoch 当成开发结论。

`author_reproduction` 复现作者配方，但仍是 `author_reproduction_only`；它不进入本项目正式矩阵。`formal_full` 必须在方法表和精确配方冻结后，显式绑定 `manifests/formal_training_inputs/v1/manifest.json`，并只使用最终 checkpoint。当前正式冻结门禁尚未满足，因此本文不提供一个会被现有代码拒绝的“可直接运行正式训练”命令。

## PatchCore 运行边界

只读训练目录、本地 WideResNet50-2 权重与作者源码检查：

```powershell
.\.venv\Scripts\python.exe .\scripts\fit_patchcore.py `
  --dataset visa --category candle `
  --seed 1887537248 --execution-scope debug_limited `
  --run-purpose debug_limited --max-steps 1 `
  --validate-only
```

`debug_limited` 拟合状态不能用于 `full` 推理。`full` 禁止 `max_steps`，固定 256→224→256 几何，并要求注册 seed；它仍然只表示单次运行完整，不表示外层 15/12 类、三个 seed 的正式矩阵已经完成。

PatchCore 现在与 DRAEM/RD4AD 一样显式区分 `debug_limited`、`development_full`、`author_reproduction` 与 `formal_full`，并写入共同的 `run_purpose`、`claim_scope`、`formal_matrix_eligible` 和 authority。`formal_full` 拟合会绑定 train-only 正式输入、冻结训练 recipe、clean Git 源码身份和安全 fitted-state manifest；正式推理会重新验证 memory bank、FAISS index、本地 backbone、fitted-state manifest、推理 recipe 与源码身份。该合同已经实现并通过合成测试，但尚未真实拟合 memory bank 或产生性能结果；当前候选/配方未冻结且项目工作树不干净，所以正式入口仍会失败关闭。不得手工补字段把早期 `full` 或 debug 结果升级成正式结果。

## RD4AD 运行边界

只读 train-only 与本地教师权重校验：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_rd4ad.py `
  --dataset visa --category candle `
  --teacher-checkpoint .\artifacts\weights\wide_resnet50_2-95faca4d.pth `
  --teacher-sha256 95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8 `
  --seed 1887537248 --validate-only
```

短步运行只能使用 `--run-purpose debug_limited --max-steps 1`。作者式完整运行要显式使用 `author_reproduction`，不具备正式声明资格。RD4AD 当前没有 `development_full`；它作为强对照只在正式冻结后使用 `formal_full`，并绑定 train-only 正式输入清单。正式训练固定 200 epochs、Adam 学习率 0.005、betas `(0.5, 0.999)`、最终 checkpoint；资源 smoke 决定并冻结 16/8/4 中可行的 batch，不能根据精度挑 batch。

## 正式冻结、推理与统一指标

进入 F0 前必须同时完成：

1. 在 `configs/preregistered_experiments.json` 中写入候选、各消融和强对照的精确方法名与配方，把 `formal_method_registry.status` 设为 `frozen_before_formal_evaluation`；
2. 把基线、候选、消融、PatchCore、RD4AD 的训练/推理 recipe 和资源配置固定下来；
3. 核对 PatchCore 正式 fitted-state 与 provenance 合同仍通过冻结配置测试；
4. 把 `configs/project.json` 的 `formal_training_recipe_frozen` 与 `formal_protocol_frozen` 设为 `true`；
5. 在任何正式推理前冻结两数据集的正式测试清单与可视化样本选择；
6. 提交上述源码、配置、清单和本复现文档，使打包清单可绑定精确 Git commit。

正式轮次必须保持以下顺序，避免在看到结果后改动已登记源码：先冻结并提交方法表、训练/推理 recipe、`configs/delivery_spec.json` 和两套可视化选择配置；再生成并提交两套 F0 inventory/receipt 与对应 `visualization_selection.json`；随后在同一 clean commit 上完成正式训练和推理。完整矩阵结束后，先运行聚合器并让它复验源码身份与全部正式运行，再按 F0 已冻结规则解析失败案例、渲染 PNG，最后执行交付包 `--validate-only`。失败解析清单、figure manifest、PNG 和最终表格属于结果产物，以文件 SHA-256 绑定，不得为了把它们写入 tracked spec 而修改正式运行所绑定的 commit。

正式启动还要通过 `configs/formal_source_provenance.json` 定义的 `clean_git_source_provenance_v1` 门禁。它要求项目仓库与 DRAEM、PatchCore、RD4AD 三个 vendor 仓库都位于登记 commit 且工作树干净，并绑定各仓库 HEAD/tree 以及项目配置、开发协议、正式登记、测试清单协议和正式训练输入根清单。训练 checkpoint 或 PatchCore fitted state 必须保存这份证据，推理与聚合重新采集并逐项比对。未提交源码、配置漂移或 vendor 漂移不能通过复制 checkpoint 或修改结果 JSON 绕过。当前工作树尚在开发且有未提交文件，所以该门禁现在应失败。

方法、配方和源码全部提交冻结后，在任何正式模型推理前，为两个数据集分别建立 F0 测试身份清单：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --dataset mvtec_ad `
  --dataset-root .\data\mvtec_ad `
  --output-dir .\manifests\formal_evaluation\mvtec_ad_v1

.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --dataset visa `
  --dataset-root .\data\visa\1cls `
  --output-dir .\manifests\formal_evaluation\visa_v1
```

该步骤是本轮第一次被允许读取完整 test/GT 的 F0 边界。每个输出目录包含 `inventory.json` 和 `receipt.json`，绑定所有输入/掩码 SHA-256、项目配置、方法登记、准备清单以及 `configs/formal_test_inventory_protocol.json` 的面积分层阈值。构建后、正式推理前用原目录重新核验：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --output-dir .\manifests\formal_evaluation\mvtec_ad_v1 --validate-only

.\.venv\Scripts\python.exe .\scripts\build_formal_test_inventory.py `
  --output-dir .\manifests\formal_evaluation\visa_v1 --validate-only
```

三种方法的 `formal_full` 推理都必须把对应 bundle 作为显式输入。入口会重验完整清单和源文件，然后从清单构造当前类别的显式样本列表；模型数据适配器不再独立发现 `test` 目录。下面只表示冻结后的命令形状，当前 pending 方法表不能直接执行：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_draem.py `
  --dataset mvtec_ad --category bottle `
  --checkpoint-dir <正式DRAEM checkpoint目录> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\mvtec_ad_v1 `
  --output-dir <空的正式输出目录> --seed <注册seed> `
  --run-purpose formal_full --device cuda:0
```

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_patchcore.py `
  --dataset mvtec_ad --category bottle `
  --fitted-model-dir <正式PatchCore fitted-state目录> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\mvtec_ad_v1 `
  --output-dir <空的正式输出目录> --seed <注册seed> `
  --run-purpose formal_full --execution-scope full --device cuda:0
```

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_rd4ad.py `
  --dataset visa --category candle --dataset-root .\data\visa\1cls `
  --teacher-checkpoint .\artifacts\weights\wide_resnet50_2-95faca4d.pth `
  --student-checkpoint <正式RD4AD checkpoint> `
  --formal-test-inventory-dir .\manifests\formal_evaluation\visa_v1 `
  --output-dir <空的正式输出目录> --seed <注册seed> `
  --run-purpose formal_full --execution-scope full --device cuda:0
```

`run_config.json`、`metrics.json`、完成态 `run_state.json` 和 `formal_provenance` 会保存相同的 inventory ID、bundle 相对路径、inventory/receipt SHA-256 与类别条目数。`predictions/index.jsonl` 逐样本绑定源图/掩码 SHA-256。聚合器重新验证 bundle、源文件和逐行对应关系；缺失或篡改这些字段的旧结果不能升级为正式结果。

最终交付校验不会重新打开真实 test/GT 文件，但会从已冻结 `inventory.json` 取得该类别的完整有序身份集合，逐行核对 prediction index 的 sample ID、标签、缺陷类型、源路径与源 SHA，并要求 index 行数、`metrics.num_images`、`run_state.num_images` 都等于 inventory 类别计数。它还会把每个正式运行中 checkpoint 门禁保存的 clean-Git source evidence 与当前项目 HEAD/tree、三套 vendor HEAD/tree 和全部 authority 文件字节重新对齐；所有正式运行必须来自同一个 source identity。F0 selection 配置和 selection manifest 必须由 Git 跟踪且属于该同一项目 commit，因此不能在正式运行后补写选择清单再打包。

当前正式方法表未冻结，所以仓库中故意不存在 `manifests/formal_evaluation` 的真实清单。`configs/delivery_spec.json` 已为两套 F0 `inventory.json`/`receipt.json` 预留精确路径；文件不存在时打包门禁会失败。不得提前生成占位内容或伪造该目录，F0 生成后由打包器核对 receipt、authority 哈希、完整类别计数及每个正式 run 的 inventory 绑定，但不会在打包阶段重新打开真实 test/GT 内容。

共同指标协议是 `industrial_anomaly_raw_map_max_256_v2`，指标文件必须为 schema v2。I-AUROC 的共同口径使用已落盘 raw map 的最大值；方法原生图像分数只进入单独的 `image_auroc_method_native`。P-AUROC、P-AP、AUPRO@0.30 和可视化均从同一批 SHA-256 绑定的 raw map 与 evaluation mask 得到。任何通用旧 `image_score`、旧 schema、逐图 min-max 后指标或另算一套可视化异常图都会被拒绝。

每个正式运行必须同时满足：

```text
run_state.status = completed
execution_scope = full
run_purpose = formal_full
claim_scope = formal_matrix_candidate
formal_matrix_eligible = true
```

完成注册方法 × MVTec AD 15 类 × VisA 12 类 × 三个 seed 的全部运行后，按冻结顺序聚合。下面是参数形状，`<冻结方法列表>` 必须逐项替换成注册文件里的完整有序列表，不能照抄占位符：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\aggregate_results.py `
  --runs-root .\runs\formal `
  --output-dir .\artifacts\final_results `
  --methods <冻结方法列表> `
  --seeds 1887537248 1337854654 1609762728 `
  --baseline DRAEM --candidate <冻结候选方法名> `
  --bootstrap-samples 10000 --bootstrap-seed 20260927
```

聚合器要求输出目录为空，并生成 `per_category_seed.csv`、`macro_by_seed.csv`、`method_summary.csv` 与 `aggregation_manifest.json`。最终交付门禁还会逐运行复核 schema、正式状态和文件哈希，再把 `per_category_seed.csv` 与原始 `metrics.json` 逐行对齐并重算两张宏平均表。

## 正式可视化

可视化流程和 JSON schema 详见 `docs/visualization_delivery.md`。必须先于 F0 推理冻结 result-independent 样本与机器失败排序规则；推理后只能由冻结规则生成 outcome-dependent failure cohort。渲染命令读取同一正式预测目录的 raw map，不改写数组：

```powershell
.\.venv\Scripts\python.exe .\scripts\render_visualization_comparisons.py `
  --selection-manifest <冻结的dataset选择清单> `
  --dataset-root <对应数据集根目录> `
  --prediction-dir <基线正式预测目录> `
  --prediction-dir <候选正式预测目录> `
  --failure-resolution <机器失败解析清单> `
  --output-dir <artifacts/formal_visualizations下对应数据集目录>
```

MVTec AD 与 VisA 都必须生成 `figure_manifest.json` 和 PNG。最终打包器要求两套图都包含预注册样本与机器选择失败案例，包含基线和候选，且图 manifest、正式 run、指标、prediction index 与 PNG 的 SHA-256 相互一致。

打包校验还会证明固定样本属于对应 F0 inventory，failure candidate pool 正好覆盖 inventory 中全部异常样本，机器选出的失败案例属于该冻结池且使用同一冻结规则；图中的 dataset、seed、sample ID 及 `prediction_runs` 必须精确覆盖 `method_order × 数据集全部类别`，不得缺方法、缺类别、重复或混入另一数据集/seed。

## 最终四项交付与确定性打包

四项交付是：

1. 可运行的调优项目包，包含基线、候选、消融与强对照实现、配置和清单；
2. 基线/候选/消融/强对照的完整正式对比表；
3. MVTec AD 与 VisA 的同口径正式可视化；
4. 本复现说明。

正式结果尚未完成时只运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_delivery_package.py --validate-only
```

当前预期退出码为 2、`status=rejected`、`archive_created=false`，且不会写出 ZIP。候选、消融、方法实现文件及固定输出路径必须与正式方法表一起在 F0 前写入 `configs/delivery_spec.json` 并提交，不能在正式运行后修改该文件，否则源码 commit 门禁会使本轮结果失效。F0 后 figure manifest 按冻结规则枚举每张 PNG 及 SHA-256，打包器把它作为第二层显式白名单；无需回写 tracked spec。正式矩阵和可视化完成后先再次执行 `--validate-only`，只有通过后才构建：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_delivery_package.py `
  --output .\artifacts\delivery\industrial-anomaly-lab-final.zip
```

构建器固定成员排序、时间戳和权限，生成 `DELIVERY-MANIFEST.json`，记录每个文件的大小/SHA-256、payload tree SHA-256、项目 commit、第三方 commit 和全部正式 run 证据；随后重新打开 ZIP，执行 CRC、成员集合与逐文件 SHA-256 检查。独立复核现有包：

独立复核还要求四类交付各至少包含一个 manifest 记录、顶层 `present=true`，payload 成员按 UTF-8 POSIX 路径排序、`DELIVERY-MANIFEST.json` 位于最后且所有成员使用 deflate。即使有人同步改写内部清单与 tree SHA，删掉某类交付、乱序或改为 stored 压缩也会被拒绝。

```powershell
.\.venv\Scripts\python.exe .\scripts\build_delivery_package.py `
  --verify-archive .\artifacts\delivery\industrial-anomaly-lab-final.zip
```

打包器拒绝绝对路径、`..` 逃逸、符号链接/Windows reparse point、数据集、权重、缓存、临时验证产物、模型二进制以及旧或非正式结果。不能通过复制、重命名或修改 manifest 把 debug、development、author reproduction 或不完整矩阵变成正式交付。
