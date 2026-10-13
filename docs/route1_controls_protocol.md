# 路线 1：同骨干对照与资源归因协议

登记日期：2026-10-14。机器协议见 `configs/route1_controls_20261014.json`，本文件与其在完整运行启动前完成一致化；此前仅完成normal-only pilot，完整对照尚未执行。本文件规定新增对照的有效范围，不代表实验已完成。机器可读配置与实际运行记录必须逐项对应本文件；首次完整拟合与测试预测前冻结它们的 SHA-256，后续不得根据测试结果放宽阈值。

## 目标与已有证据边界

沿用 DINOv2-S 正常样本子空间检测，回答三个问题：原版两遍 PCA 与当前单遍统计的检测结果是否保持一致；各自真实拟合耗时与内存是多少；单遍统计、按需读取图片和不返回 attention 张量分别贡献多少执行收益。

公开来源固定为 `CLendering/SubspaceAD@ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`（Apache-2.0）。原版 `PCAModel.fit()` 已是两遍流式 PCA，不能将“采用流式 PCA”作为新增算法。原版 `main.py:feature_generator_full()` 每遍先读取全部训练图片为 PIL RGB 列表；当前实现单遍 float64 Chan 统计，并逐张加载。当前 extractor 同时省去返回 attention；这是第三个已存在的执行变化，需要固定或单独测量。

原版 `aggregate_image_score()` 已有 `mtop1p`；top1p 分数不是本项目新增组件。本轮没有加入新的检测组件，因此不进行新算法组件消融，也不把 PaDiM 对照收益归因给 PCA 实现。检测精度保持与资源收益均按实际结果报告，包括无收益和回退。

旧运行 `runs/quick/subspace-dinos-20261012`、PaDiM 基线和 `artifacts/deliveries/quick-padim-20261012/交付文件夹` 作为只读证据。不得覆盖其模型、预测、协议、完成记录、清单或交付包；新增材料保存到独立目录。测试集早已接触，本轮属于固定设置复现和执行对照，`scientific_claim_eligible=false`。

## 固定检测设置

- 同一个本地 DINOv2-small 权重：HF revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`，checkpoint SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`。同时核验 config 和 preprocessor_config。
- RGB、HF slow processor、整图 bicubic 448，无中心裁剪；float32 eager attention；hidden states `[-4,-5]` 原始 patch token 均值，去 CLS，无 L2 归一化；每图 1024×384 token。
- 全部 12,288 张正常训练图、27 类；每类独立 PCA，float64、Npatch−1 分母、解释方差 0.99、eps 1e−6、无 whitening、无中途截断秩、无训练增强。
- float32 平方重建残差；OpenCV 32→448 双线性、3×3 sigma=4 GaussianBlur、448→256 双线性；保存 raw float32 map，不逐图归一化。
- 读取旧 PaDiM 冻结的 train manifest 和 test inventory；全 3,887 测试图；GT nearest 256。主表 I-AUROC 取保存 raw map 的最大值；报告 P-AUROC、P-AP、AUPRO@0.30，15/12 类分别等权宏平均。top1p 只作已有公开评分规则的辅助列。

## 全 27 类精度对照

每类实际拟合三个控制：`official_cpu_two_pass`、`numpy_cpu_two_pass`、`current_cpu_one_pass`。第一个实际调用未修改 vendor 的 `PCAModel(ev=.99, whiten=False, eps=1e-6).fit(...)`，实例明确设置 `device=cpu`，不修改 vendor 模块源码；报告“原版 PCA，CPU 执行适配”，不能称为作者默认 GPU 整套运行。第二个用 NumPy float64 两遍均值/协方差，拆开 Torch与NumPy统计后端的影响。第三个为当前 `StreamingPCA` 单遍统计。三个控制均在 CPU 消费完全相同的 float32 特征、顺序和分块；两个two-pass控制必须计数为两遍和2×N张特征消费，当前必须为一遍和N张。

允许每类提取一次训练/测试 token，落盘 memmap，三个 PCA 复用；不能把缓存读取时间当作未缓存端到端速度。缓存按 image ID、原图 SHA-256、提取器源码、权重、环境、后端、shape、dtype 和 token 文件 SHA-256 绑定。按类别处理可把单类 token 存储控制在约 2 GiB；全量同时保留约需 24 GiB，启动前按真实空间决定保留策略。

每类三个实际模型均用既有 `SubspaceModel.reconstruction_scores()` 评分同一批测试token，其float32重建算术已经通过与vendor `calculate_anomaly_scores(..., method='reconstruction')` 的对照测试。当前16项engine测试包括这项评分交叉验证；这证明统一评分适配器的实现，不代表每张测试图实际调用vendor评分函数。允许多个“只改读取方式”的精度表行引用同一组模型/预测，但必须标明引用和无重复拟合，不能伪造独立运行。旧GPU模型/旧maps是单列历史诊断，新旧推理后端不同须披露，不能仅复制旧指标代替新控制模型预测。

预先规定一致性检查，以机器配置 `model_tolerances` 为准：token shape/dtype/图像集合完全一致，三个 PCA rank 相同；float64 mean `atol=1e-9, rtol=1e-10`；eigenvalues与子空间projector均为 `atol=1e-9, rtol=1e-9`。比较projector，避免特征向量符号与退化子空间旋转造成误判。重建score与raw map均用 `atol=1e-4, rtol=3e-5`；四项检测指标的绝对差阈值均为 `1e-5`（比例值，等于0.001个百分点）。每类保存实际最大差、相对差、rank、阈值判断和所有指标差，阈值失败不删类、不放宽阈值、不自动宣称精度等价。

本轮CPU特征与旧GPU运行另逐图比较，用于历史诊断，不把跨后端误差混入三个共享CPU特征控制的等价判定。全27类×3控制×max/top1p为162行指标；实际拟合81个PCA模型，3,887图各有3份预测，共11,661张raw map及3,887份GT。max是共同主表图像评分，top1p是既有公开聚合规则辅助列。不能从测试结果调层、rank、图像评分、平滑或类别选择。

## 资源对照与逐项归因

资源基准固定取 MVTec AD `bottle` 和 VisA `cashew`，分别为旧训练清单的前 64 张正常图；两个数据集各一个类别，样本固定与检测成绩无关。每张原图均按原尺寸 RGB 解码；禁止为 eager 组先缩小图像。实测第一张原图为 bottle 900×900、cashew 1274×1176；64 张 PIL 4 bytes/pixel 的粗略容量分别约 198 MiB、366 MiB，此数字只是预估，不是峰值测量。整类 cashew 450 张约 2.51 GiB 图片容量，当前共享 RAM 下不能直接假定全量 eager 可安全启动。

每种资源基准在 fresh process 中重复三次，`batch_size=1, workers=0, cpu_threads=4`，统一 BLAS、PyTorch、OpenCV 线程；CPU固定正常样本热身后再计时，本轮不调用CUDA。三次是资源计时重复，不是三个训练随机种子。该 PCA 配方无抽样、增强、随机投影、可训练参数；固定种子与顺序，避免把确定性重放包装为多种子精度稳定性。

| 变体 | PCA | 图片读取 | attention 返回 | 直接归因 |
|---|---|---|---|---|
| R00 | NumPy 两遍 | eager PIL 列表 | 固定关闭 | 执行对照起点 |
| R10 | 当前单遍 | eager PIL 列表 | 固定关闭 | 单遍统计 |
| R01 | NumPy 两遍 | lazy 每张读取 | 固定关闭 | 按需读取 |
| R11 | 当前单遍 | lazy 每张读取 | 固定关闭 | 两项合并 |

同时记录 R00→R10、R00→R01、R10→R11、R01→R11 的配对差和交互；不能只给最好的一对。资源 worker 的两遍采用与原版均值/协方差目标一致的 NumPy float64 CPU 实现，明确标为 `project_numpy_two_pass`；全27类精度另外实际调用 vendor PCAModel，不把资源worker冒充vendor整套运行。eager 的图片列表应在每次 generator 调用内部按作者方式构造，保证两遍实际解码两次，释放上一遍后才开始下一遍。每遍 generator 调用、解码次数、forward 次数和 token 数全部计数，并封存每遍token内容摘要。

另在固定 one-pass/lazy/CPU 后端下做 attention 返回开/关的三次成对比较，使用同一模型与作者相同隐藏层路径；先用第一张正常图检查token一致，再完成64张fit。关闭 attention 的 2×2 是净归因；attention开关对照只证明单遍lazy实现中的边际影响，不能与两遍eager结果直接相减归因。共2类×6模式×3重复=36个fresh process，attention_off与one_lazy为独立重复路径且均保留。若只执行有限 normal-only attention forward，可以报告每次 forward 峰值与耗时，不能将其推算成完整拟合速度。

资源分成两个真实实验层：①缓存 token 上同 CPU 后端 PCA 内核比较，包含真实两遍读取/一遍读取；②未缓存原图开始的提取＋PCA 拟合比较，包含 eager 解码、lazy 解码、forward 和统计。分别报告模型加载、热身、正式拟合、序列化时间；“端到端”须把所声称阶段全部纳入。64 张基准只证明该固定规模的训练阶段资源效果，不是 12,288 张完整训练、测试推理或全类加速证明。

记录 wall time、实际子进程 peak working set/RSS、private bytes/commit、真实 NVML 进程显存（可用时）、torch peak allocated/reserved、GPU全局最小free、系统RAM/commit最小free、PID、父PID、后端、硬件、背景负载和退出码。torch allocator 显存不含全部 CUDA context；不能替代进程显存。每次 fresh process 的全进程峰值及阶段峰值分别标明，三次原始值、均值、样本标准差、中位数全部保留。按轮交错变体顺序，保存其他任务负载变化；共享机器的测量不得声称独占性能。

## 资源准入与并发

用户已授权在允许条件下充分并发。先刷新真实进程命令行、GPU/VRAM、可用物理 RAM、commit、磁盘、WSL 与项目锁；不能杀停其他项目任务，也不能从 UI idle 推断资源空闲。旧 `subspace_dinos_20261012.json` 的 4 GiB GPU admission/2 GiB reserve 保持原样；旧 runner 不满足其门禁时不能启动。

本轮运行决策为CPU：全27类共享特征提取和所有PCA对照均使用CPU，`cpu_threads=4`。其他项目GPU任务继续运行。本轮不改旧GPU门禁、不使用新GPU准入放宽路径。先执行normal-only小规模pilot和固定64张资源试验，再按实际RAM峰值评估并发额度。

新CPU runner初始可用RAM≥3.25 GiB、可用commit≥3.25 GiB、可用磁盘≥10 GiB；运行中可用RAM保留≥1.5 GiB、commit保留≥2 GiB、自身working set≤2.5 GiB。实际按类缓存空间和指标临时空间额外核查磁盘容量。资源监控至少每0.5秒一次，Windows PSAPI记录实际自身working set/private bytes/commit与系统余量；阶段采样峰值注明可能低于短时真实峰值，全进程OS峰值另外报告。

RAM/commit低于保留余量、自己working set越限或监控失败时，停止本runner并保留失败receipt；不能停止其他项目任务。只并发已测得能同时保留上述余量的独立阶段，不因GPU忙碌强制CPU任务串行，也不能用一次快照保证整轮可用。所有GPU字段填 `not_used`，不能填0伪装有GPU测量。CPU/GPU相对旧map的精度差只能作为历史诊断，不能将原版GPU PCA与当前CPU PCA耗时差直接归因于单遍算法。

## 完整性与验收

首次运行封存新runner、launcher/监控器、报告/审核脚本、协议、配置、`subspace_features.py`、`quick_metrics.py`、其导入的 `metrics.py`、执行适配helpers全部源码哈希；同时封存完整 vendor source hashes、许可证、权重文件、uv.lock、pyproject.toml、实际运行依赖版本、硬件与驱动。旧原始模型/预测/清单哈希验证在新增对照前后各执行一次。不可只封存顶层runner而遗漏参与评分的helper。

每类完成记录绑定 train/test manifest、缓存身份、三个模型、全部raw maps、GT与指标；恢复执行先重验源码、环境和依赖输入，不能把半成品当完成。若token缓存删除，保留其清单/哈希、验证记录、实际拟合/预测产物与重建命令。报告构建后独立重读raw maps重算指标，结果表不能只信任运行器写下的数字。

36条资源worker回执逐条绑定本轮protocol SHA-256、实际worker及统计模块源码SHA-256、CPU后端、两个固定类别的旧训练清单前64条路径与原图hash。每种模式验证generator/解码/forward/token消费次数；two-pass每遍特征内容摘要必须一致，各模式及各重复的同类别特征内容必须一致，attention开关不能改变token。只有成功且输入内容与后端完全一致的同编号重复才计算配对减幅，失败、阻断、缺测原样保留。全27类运行中共享进程峰值只用于监控，不能分摊成每方法内存收益。

54张可视化源图保持旧清单每类第一张正常和第一张异常的机械选择，并绑定旧图像hash；新GT逐图绑定旧交付已认证256×256 GT，主runner另用原始源mask的hash和nearest256重建复核。离线验收需要同时复核新GT与旧GT一致，不能只核验新GT自己写下的hash。

交付包含：全27类同骨干精度表和数据集宏平均、三次资源原始值/汇总、2×2及attention归因表、固定样本成对可视化、命令与环境、来源/许可证、源码和结果哈希、复核脚本、未通过项及限制、给同事的可直接转发中文说明。若精度等价且资源有收益，表述为“基于公开SubspaceAD的低资源执行优化”；检测收益只报告已测实际差值，不因更换骨干或超过PaDiM就宣称新算法超过原版。
