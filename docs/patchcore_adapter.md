# PatchCore 强对照运行与证据边界

项目固定使用 `amazon-science/patchcore-inspection@fcaa92f124fb1ad74a7acf56726decd4b27cbcad`。第三方目录保持原样，项目适配层负责本地权重注入、训练/测试输入约束、安全 fitted-state、统一指标和正式来源校验。

## 运行用途不是 `full` 的别名

每次 fit 和 infer 都必须显式给出 `run_purpose`：

| `run_purpose` | `execution_scope` | claim scope | 能否进入正式矩阵 |
|---|---|---|---|
| `debug_limited` | `debug_limited` | `debug_only` | 否 |
| `development_full` | `full` | `development_only` | 否；正式 test 推理入口拒绝该用途 |
| `author_reproduction` | `full` | `author_reproduction_only` | 否 |
| `formal_full` | `full` | `formal_matrix_candidate` | 只有全部正式门槛通过后才可以 |

因此，旧的 `execution_scope=full` 结果只是完整运行，不能自动进入 F0 聚合。正式结果必须同时在 `run_config.json` 和 `run_state.json` 中满足 `run_purpose=formal_full`、`claim_scope=formal_matrix_candidate`、`formal_matrix_eligible=true`，并通过登记、输入、源码、模型和测试清单的重验。

## 固定算法配置

- WideResNet50-2 ImageNet1K V1，本地文件完整 SHA-256 为 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`；不调用第三方下载接口。
- 特征层为 `layer2`、`layer3`，嵌入维度为 1024/1024，patch size 为 3。
- 10% approximate greedy coreset，1-NN，CPU FAISS squared-L2 index。
- 输入先缩放到 256，再中心裁剪至 224；作者输出的 224×224 异常图只进行一次双线性上采样至统一的 256×256 评估平面。
- 不做逐图 min-max。PatchCore 作者 Gaussian sigma 4 已属于作者输出语义。
- 通用 I-AUROC 使用持久化 raw map 的最大值；作者 patch-score 聚合只记录为 `image_auroc_method_native`。P-AUROC、P-AP 与 AUPRO@0.30 均从同一份持久化 raw map 和评估 mask 重算。

## 正式 fit

`formal_full` fit 不扫描数据目录决定训练样本。它只使用 `configs/project.json` 固定 SHA 的 `manifests/formal_training_inputs/v1/manifest.json`，解析当前 dataset/category 的完整 `train/good` 路径并逐文件复核大小和 SHA-256。PatchCore 不消费 DTD。目录中存在但未登记的图片不会进入 fit。

正式训练启动前还必须同时满足：

- `configs/preregistered_experiments.json` 的方法登记和 PatchCore training recipe 已在 F0 前冻结；
- `configs/project.json` 的正式评估协议已冻结，seed、dataset、category 均与登记一致；
- `configs/formal_source_provenance.json` 所登记的项目仓库和第三方仓库均为 clean Git worktree，HEAD/tree 与证据一致；
- 资源 preflight 通过，并由受监控启动器持有训练锁。

目前方法登记和正式 recipe 仍是 pending，因此真实 `formal_full` 会 fail closed，不能据此启动训练。

正式 fitted-state 使用 schema v2，文件仅包括：

- `model/memory_bank.npy`：有限值二维 `float32` memory bank；
- `model/nnscorer_search_index.faiss`：可通过 `reconstruct_n()` 与 memory bank 逐值核对的 FAISS index；
- `model/manifest.json`：绑定以上两项、本地 WRN 权重、训练输入证据、训练 recipe、正式 authority、Git/source provenance、vendor commit/clean 状态和每个文件的大小/SHA-256。

项目不使用作者 pickle serializer，也不允许隐式 backbone 下载。任何写入失败都会删除上述最终 fitted artifacts，失败目录只保留状态和诊断，不留下看似可用的模型。

验证输入但不 fit 的示例：

```powershell
Set-Location E:\3k
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\fit_patchcore.py `
  --dataset visa --category candle --seed 1887537248 `
  --execution-scope debug_limited --run-purpose debug_limited `
  --max-steps 1 --validate-only
```

正式命令只能在登记和资源门槛完成后运行，配方示意如下：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\fit_patchcore.py `
  --dataset visa --category candle --seed 1887537248 `
  --execution-scope full --run-purpose formal_full `
  --device cuda:0 --coreset-device cpu `
  --output-dir E:\3k\runs\patchcore\visa\candle\fit_seed1887537248
```

## 正式 infer

正式推理在分配模型和读取测试样本前执行以下校验：

1. inference recipe 与冻结登记逐字段一致；
2. fitted-state manifest 为 schema v2，identity、scope 和 claim 字段匹配；
3. memory bank、FAISS index、WRN 权重、formal train input 和 source provenance 仍能通过当前文件/仓库重验；
4. 必须提供对应数据集的 F0 `formal_test_inventory` bundle。

测试数据集由 `resolve_formal_test_inventory_category()` 返回的显式 `TestSample` 列表构造，不再次调用目录发现。每个预测行绑定 source image/mask SHA、inventory ID、inventory SHA 和 receipt SHA；`run_config.json`、`metrics.json`、`run_state.json` 保存相同的 category evidence。聚合器会再次验证 inventory 和预测行一一对应。

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
.\.venv\Scripts\python.exe .\scripts\infer_patchcore.py `
  --dataset visa --category candle --seed 1887537248 `
  --execution-scope full --run-purpose formal_full `
  --device cuda:0 `
  --fitted-model-dir E:\3k\runs\patchcore\visa\candle\fit_seed1887537248\model `
  --formal-test-inventory-dir E:\3k\manifests\formal_evaluation\visa_v1 `
  --output-dir E:\3k\runs\patchcore\visa\candle\infer_seed1887537248
```

成功的 inference `run_state.json` 会绑定 `run_config.json`、`metrics.json` 和 `predictions/index.jsonl` 的 SHA-256。聚合器拒绝旧 schema、debug、普通 full、文件篡改、路径逃逸、训练输入漂移、源码漂移、测试清单漂移以及不完整类别结果。

## 当前证据边界

项目已通过纯合成 fixture 的适配层、safe-state roundtrip、哈希篡改、路径逃逸、旧 schema、pending formal gate、失败清理、显式正式 test inventory 和统一导出协议测试。已有 CPU/FAISS smoke 只证明运行路径可用，不是性能结果。尚未运行真实 PatchCore fit、正式 test inference 或性能比较，不能声称优于 DRAEM、RD4AD 或其他方法。
