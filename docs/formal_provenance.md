# 运行目的与正式结果 provenance 门禁

`execution_scope=full` 只说明运行没有步数截断，不能证明它属于正式矩阵。所有训练和推理必须另外记录 `run_purpose` 与由代码确定的 `claim_scope`：

| `run_purpose` | `execution_scope` | `claim_scope` | 可进入 F0 正式矩阵 |
| --- | --- | --- | --- |
| `debug_limited` | `debug_limited` | `debug_only` | 否 |
| `development_full` | `full` | `development_only` | 否 |
| `author_reproduction` | `full` | `author_reproduction_only` | 否 |
| `formal_full` | `full` | `formal_matrix_candidate` | 仅全部 provenance 门禁通过后 |

## 当前边界

当前 `configs/project.json` 的正式评测协议尚未冻结，`configs/preregistered_experiments.json` 的正式方法表、候选方法、消融和逐方法配方也尚未冻结。因此任何 `formal_full` 训练、推理或聚合都会主动拒绝；这表示 F0 尚未获得授权，不表示开发训练被阻塞。

正式训练输入的独立 blocker 已由 `formal_training_input_bundle_v1` 解除：DRAEM 正式训练绑定所选类别 `train/good` 与全量 DTD，PatchCore 和 RD4AD 绑定所选类别 `train/good`；schema-v2 checkpoint/fitted-state 保存项目相对路径、文件 size/SHA-256 和 bundle SHA，推理与聚合会重新验证当前文件集合和字节。详见 `docs/formal_training_inputs.md`。这一完成项不改变正式方法表、候选方法、消融和逐方法配方仍为 pending 的状态。

DRAEM 的 `development_full` 独立读取冻结的 VisA train/dev + DTD 协议，只接受三组注册 seed、256 像素、batch 8、700 epochs 和学习率 `1e-4`，并始终标记为 `development_only`。DRAEM 的 S1/S3 可在正式方法表仍为 pending 时完成。RD4AD 尚无对应的 manifest-backed train/dev 适配，所以当前拒绝 `development_full`；`author_reproduction` 仍可运行，但不能进入正式矩阵。

## F0 冻结时必须登记的配方

正式方法表需设为 `frozen_before_formal_evaluation`，并为每个方法加入以下结构；`training` 和 `inference` 必须与代码生成的完整 canonical recipe 精确相等：

```json
{
  "formal_method_registry": {
    "status": "frozen_before_formal_evaluation",
    "methods": ["DRAEM", "frozen_candidate", "PatchCore", "RD4AD"],
    "recipes": {
      "DRAEM": {
        "recipe_id": "draem-f0-v1",
        "training": {},
        "inference": {}
      }
    }
  }
}
```

正式训练还要求项目配置与注册文件中的 seed 精确一致，数量至少为三；当前实现要求 `minimum_seeds` 与注册 seed 数相等。方法、数据集、类别、seed、phase、配方或正式训练输入证据任一不匹配都会失败。当前仍需先冻结正式方法表和逐方法配方。

## checkpoint 与推理绑定

正式推理只接受 schema-v2 训练 manifest。manifest 必须记录并通过以下检查：

- `method`、`dataset`、`category`、`seed`、`run_purpose=formal_full`、`execution_scope=full` 和 `claim_scope=formal_matrix_candidate`；
- `formal_training_inputs` 的根/子清单 SHA、来源 manifest、builder/validator provenance，以及当前 train-only 文件集合和逐文件内容哈希重验；
- `checkpoint_origin=project_training_run`、`final_checkpoint_only=true`、`test_selected=false`；
- 完整训练配方及其当前注册 authority；
- 每个 checkpoint role 的文件名、字节数与 SHA-256；
- RD4AD 的本地教师权重 SHA-256；
- PatchCore fitted-state 的 memory bank、FAISS index 与本地 WRN 权重；
- 每种正式方法都必须保存 `clean_git_source_provenance_v1`：项目 HEAD/tree 及 DRAEM、PatchCore、RD4AD 三个 vendor 的登记 commit 和干净状态。推理与聚合会重新采集并逐字段比较，不能只信任训练时写入的布尔值。

正式推理还必须使用 F0 入口生成的完整 `formal_test_inventory`。DRAEM、PatchCore、RD4AD 均从该清单取得显式样本，不重新扫描 test 目录；预测索引逐行绑定 inventory/receipt、源图和源掩码 SHA。调试或作者复现仍可使用原有目录发现，但不能进入正式矩阵。

正式推理输出会把 inference recipe、当前 authority 和 checkpoint 验证报告写入 `run_config.json`。完成态另行绑定 `run_config.json`、`metrics.json` 和 `predictions/index.jsonl` 的 SHA-256。

## 聚合门禁

聚合器只接受 `formal_full`、`formal_matrix_candidate` 且 `formal_matrix_eligible=true` 的完成态。它不会信任历史的“已验证”布尔值，而会重新读取当前项目配置与注册文件，复验训练 manifest、checkpoint 字节、外部教师哈希和推理配方，再核对完成态绑定的运行配置、指标和预测索引哈希。旧格式运行、官方预训练权重的直接评测、作者复现、开发运行以及手工标记为 `full` 的结果都会被拒绝。

这些检查用于阻止协议混淆和产物误配；它们不是抗恶意篡改的数字签名。需要对外归档时，仍应对整个交付包生成独立 SHA-256 清单并保留只读副本。
