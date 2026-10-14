"""把项目说明整理成容易读的文稿，先写草稿，不改交付文件。"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

EXT = "优化实验/同骨干Subspace对照-20261014"
STORY = [
    "这个项目做的是无监督工业异常检测：先让模型看正常产品，再判断新图片里有没有异常、异常在哪里。我希望结果靠得住，也希望它能在资源有限的电脑上跑起来。",
    "前面先试了协方差收缩。固定收缩和 OAS 都没有带来整体改善，所以没有沿着这条路继续加参数。后来采用 DINOv2-S 提取特征，再用 SubspaceAD 建立正常样本的子空间，两个数据集的平均指标都有了提高。",
    "这次我主要处理运行过程：把两遍统计合成单遍，图片用到时再读取，并单独检查是否需要保留额外的注意力输出。SubspaceAD 原本就采用流式统计，我做的是在这套方法上减少重复计算和内存占用。",
    "为了判断效果到底来自哪里，我固定了骨干、数据和评分方式，对照原版两遍实现、NumPy 两遍实现和当前单遍实现。MVTec AD 的15类和 VisA 的12类都跑完了，共用到12,288张正常训练图、3,887张测试图。三种实现的检测与定位指标基本一致。",
]
RESOURCE_NOTE = "资源对照选了 bottle 和 cashew，各用固定的64张正常图，每种设置重复三次。这里比较两遍统计加提前加载与单遍统计加按需读取，降幅按各组三次测量的中位数计算。这些比例只代表这两个子集；电脑上还有其他任务，不能直接当成整套数据的加速比例。"
RESULT_NOTE = "这次最明确的收获，是在保持检测结果基本一致的前提下，把同一套方法跑得更省。精度提高主要发生在前面的方法选择阶段，本轮验证的是运行成本。"
SOURCE_NOTE = "特征模型采用 DINOv2-S，子空间方法采用 SubspaceAD；数据来自 MVTec AD 和 VisA。我的工作放在执行优化、对照实验和统一复算上。方法、权重和数据的出处及许可证继续保留。"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def write(stage: Path, relative: str, text: str) -> None:
    p = stage / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def paras(items: list[str]) -> str:
    return "".join("<p>" + html.escape(x) + "</p>" for x in items)


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "<table><thead><tr>" + "".join("<th>" + html.escape(x) + "</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--package-root", type=Path, required=True)
    a.add_argument("--stage-root", type=Path, required=True)
    args = a.parse_args()
    p, stage = args.package_root.resolve(), args.stage_root.resolve()
    if stage.is_relative_to(p) or not p.is_dir():
        raise ValueError("草稿需要放在交付文件夹外")
    q = p / EXT
    measured = json.loads(read(q / "验收记录/整合数值摘要.json"))
    names = {"mvtec_ad": "MVTec AD", "visa": "VisA"}
    accuracy_rows = []
    accuracy_lines = []
    for row in measured["primary_max"]:
        d = names[row["dataset"]]
        for field, label in (("reference_metrics", "原版两遍"), ("method_metrics", "当前单遍")):
            v = row[field]
            accuracy_rows.append([d, label, *[f"{100*v[k]:.4f}%" for k in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")]])
        accuracy_lines.append(f"{d}：图像AUROC {100*row['method_metrics']['image_auroc']:.4f}%，像素AUROC {100*row['method_metrics']['pixel_auroc']:.4f}%，像素AP {100*row['method_metrics']['pixel_ap']:.4f}%，AUPRO {100*row['method_metrics']['aupro_0_30']:.4f}%。原版两遍和当前单遍在这些位数上相同。")
    resource_rows = []
    resource_lines = []
    for row in measured["representative_resource_comparisons"]:
        b, n, delta = row["reference_medians"], row["method_medians"], row["reductions_percent_of_medians"]
        resource_rows.append([names[row["dataset"]] + "/" + row["category"], f"{b['fit_seconds']:.3f} → {n['fit_seconds']:.3f}", f"{delta['fit_seconds']:.2f}%", f"{b['process_peak_working_set_bytes']/2**20:.2f} → {n['process_peak_working_set_bytes']/2**20:.2f}", f"{delta['process_peak_working_set_bytes']:.2f}%"])
        resource_lines.append(f"{names[row['dataset']]}/{row['category']}：特征提取加拟合耗时由 {b['fit_seconds']:.3f} 秒降到 {n['fit_seconds']:.3f} 秒，减少 {delta['fit_seconds']:.2f}%；进程内存峰值由 {b['process_peak_working_set_bytes']/2**20:.2f} MiB 降到 {n['process_peak_working_set_bytes']/2**20:.2f} MiB，减少 {delta['process_peak_working_set_bytes']:.2f}%。")
    accuracy = table(["数据集", "实现", "图像AUROC（max）", "像素AUROC", "像素AP", "AUPRO"], accuracy_rows)
    resource = table(["固定子集", "提取加拟合耗时（秒）", "耗时减少", "进程内存峰值（MiB）", "内存减少"], resource_rows)
    css = "body{font:16px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.8;color:#222}table{border-collapse:collapse;width:100%;display:block;overflow:auto;margin:18px 0;font-size:14px}th,td{border:1px solid #d8dde3;padding:8px;white-space:nowrap}th{background:#f3f5f6}a{color:#0754a0}details{margin:24px 0}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}figure{margin:26px 0}h1{font-size:28px}h2{font-size:21px}"
    def page(title: str, body: str) -> str:
        return "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>" + html.escape(title) + "</title><style>" + css + "</style></head><body>" + body + "</body></html>"

    root_old = read(p / "项目总览.html")
    old_tables = re.findall(r"<table\b.*?</table>", root_old, re.S)
    old_links = list(dict.fromkeys(re.findall(r"<a\b[^>]*>.*?</a>", root_old, re.S)))
    root_body = "<h1>把工业异常检测做得更省一些</h1>" + paras(STORY) + "<h2>检测结果保持住了</h2>" + accuracy
    root_body += "<p>这里使用异常图的最大值作为图像评分。前一阶段还报告过最高1%像素均值，两种评分方式分开比较，完整表中都保留。</p><h2>运行开销降了多少</h2>" + paras([RESOURCE_NOTE]) + resource + paras([RESULT_NOTE])
    root_body += f"<p>异常图和逐项对照见<a href='{EXT}/report.html'>本轮报告</a>。想复算可以看<a href='{EXT}/复现说明.txt'>复现说明</a>，代码的整体思路在<a href='code/代码说明.txt'>代码说明</a>。</p>"
    root_body += "<details><summary>前面几轮实验的结果</summary><p>早期固定收缩和OAS没有整体改善；换用DINOv2-S与SubspaceAD后，平均指标提高，但 transistor 的AUPRO下降19.2112个百分点。评分方式调整时，24类上升、3类下降。这些结果都保留，方便判断这条路线是怎么选下来的。</p>" + "".join(old_tables) + "<p>" + " · ".join(old_links) + "</p></details>"
    root_body += paras([SOURCE_NOTE, "全部27类已完成指标重算，固定两张原图的六张CPU异常图也已核对一致。没有新增检测算法组件，三次资源重复用于观察运行波动。论文层面的新算法或多随机种子稳定性结论，不由这组结果推出。DRAEM训练继续保持暂停。"])
    root_body += "<p>出处集中在<a href='report/来源说明.md'>来源说明</a>，技术口径和完整记录在各轮报告末尾。</p>"
    write(stage, "项目总览.html", page("工业异常检测项目", root_body))

    full_q = read(q / "report.html")
    start = full_q.index("<h2>数据集宏平均</h2>")
    gallery = full_q.index("<h2>54 组原冻结样本可视化</h2>")
    details = full_q[start:gallery]
    replacements = {
        "减少比例=(参考值−当前值)/参考值；负值表示变差。共享进程的内存对照留空。完整进程时间与加载权重、预热后的特征提取加拟合时间分列；重复测量是运行波动测量，不是随机种子稳定性实验。": "降幅按参考值与当前值的差计算，负值就是变慢或占用更多。含加载、预热的进程内计时单独列出，共享进程的内存不分摊到某个方法。三次重复反映运行波动。",
        "交互节省=(两遍eager−单遍lazy)−(两遍eager−单遍eager)−(两遍eager−两遍lazy)。正值表示合并条件观测到的节省超过两个单项节省之和；负值表示不足。共享机器运行波动也会进入该值，三次重复不能据此证明统计显著性。": "交互项比较两项改动合在一起时的节省，与各自单独节省之和。正值表示合并时更多，负值表示更少；共享电脑的运行波动也会影响它。精确公式和逐次数据仍在对应表里。",
    }
    for before, after in replacements.items():
        details = details.replace(before, after)
    figures = re.findall(r"<figure\b.*?</figure>", full_q[gallery:], re.S)
    if len(figures) != 54:
        raise ValueError("固定样本数量不符")
    q_body = "<h1>这次把重点放在运行成本上</h1>" + paras([STORY[2], STORY[3], RESULT_NOTE])
    q_body += "<h2>先确认结果没有变</h2>" + accuracy + "<p>NumPy两遍对照也一致。完整小数保留在后面的表格里，细小浮点差别不算检测精度提升。</p>"
    q_body += "<h2>再看时间和内存</h2>" + paras([RESOURCE_NOTE]) + resource
    q_body += "<p>单遍统计主要省去重复的特征提取；按需读图主要降低图片同时驻留的内存。它们并非在每次测量中都更快，所以我也保留了各项单独比较、输出开关和变慢的记录。</p>"
    q_body += "<h2>异常图看起来怎样</h2><p>每类固定一张正常图和一张异常图，三种方法使用同一色标。色标按该样本原版异常图的1%与99%分位数设置，只影响显示，评分仍用原值。</p>" + "".join(figures[:2])
    q_body += "<details><summary>其余52组对比图</summary>" + "".join(figures[2:]) + "</details>"
    q_body += "<details><summary>完整指标和逐项对照</summary>" + details + "</details>"
    q_body += "<p>27类的162条指标已完整重算，固定两张原图的六张CPU异常图也与保存结果一致。资源试验最终完成36次，另有6次因内存不足中断的尝试，记录仍保留。</p><p>具体参数、恢复过程和计时定义放在<a href='技术说明.txt'>技术说明</a>，复算命令见<a href='复现说明.txt'>复现说明</a>。</p>"
    q_body += paras([SOURCE_NOTE]) + "<p>方法和适配记录见<a href='公开来源/来源与贡献.json'>来源记录</a>。预训练权重没有反向训练；这一轮改的是执行过程。</p>"
    if re.findall(r"<table\b.*?</table>", details, re.S) != re.findall(r"<table\b.*?</table>", full_q[start:gallery], re.S):
        raise ValueError("原表格发生变化")
    write(stage, EXT + "/report.html", page("同骨干对照：精度与运行成本", q_body))

    verify = 'python -B code/scripts/verify_readable_delivery.py --package-root . --metrics-categories 27 --inference-samples 2 --verify-master --work-dir ../同骨干复算临时文件 --receipt ../同骨干复算回执.json'
    write(stage, "先读我.txt", "工业异常检测项目\n\n先打开 项目总览.html。项目从没有改善的协方差试验，走到DINOv2-S与SubspaceAD，再做到保持结果的运行优化，经过和结果都放在里面。\n\nMVTec AD的15类和VisA的12类已经跑完。本轮指标重算和两张原图的CPU推理核对通过。代码有保留，先读 code/代码说明.txt 会比较容易。\n\n想自己核对，使用已有Python 3.11及对应依赖，在本目录运行：\n" + verify + "\n\n它使用本地权重、模型和保存的预测，不训练、不联网。完整重新拟合还需要官方完整数据及原项目工作区；具体步骤见 " + EXT + "/复现说明.txt。\n\n早期结果、未改善的尝试和真实出处也都保留。当前文稿从报告进入，原文和整理记录另存，避免把不同阶段混在一起。")
    write(stage, "实验说明.txt", "这个项目是怎么往下做的\n\n" + "\n\n".join(STORY) + "\n\n" + "\n".join(accuracy_lines) + "\n\n" + RESOURCE_NOTE + "\n" + "\n".join(resource_lines) + "\n\n" + RESULT_NOTE + "\n\n前一阶段在相同top1%图像评分下，MVTec AD的AUROC由93.1983%到97.3258%，VisA由84.3934%到93.1362%。这反映方法和骨干选择的差别；本轮没有把这部分提升算到单遍统计头上。transistor的AUPRO下降19.2112个百分点，完整下降项仍保留。\n\n所有结果按类别等权平均。这里是已访问测试集上的探索性比较，三次资源计时不是多随机种子精度实验，也没有新增检测算法组件。参数和验收细节见 " + EXT + "/技术说明.txt，逐类表和异常图见该目录report.html。\n\n" + SOURCE_NOTE)
    write(stage, EXT + "/给同事的说明.txt", "这次我还是围绕无监督工业异常检测来做。前面试过协方差收缩，整体没改善，所以后来选了DINOv2-S提取特征、SubspaceAD做正常子空间建模的路线。\n\n现在主要想解决的是运行开销：正常图片的特征重复提取挺费时间，提前把很多图放进内存也比较占资源。我把统计改成单遍，图片用到时再读，并把输出开关单独做了对照。\n\n为免把骨干差别算成优化效果，我固定了模型、数据和评分方式。MVTec AD的15类、VisA的12类都跑完，原版两遍、NumPy两遍和单遍实现的指标基本一致。\n\n在bottle和cashew各64张正常图、每组重复三次的测量里，提取加拟合耗时分别减少38.47%和47.46%，进程内存峰值分别减少20.35%和31.39%。这部分只说这两个固定子集，逐次变慢的情况也留着。\n\n我觉得比较有价值的是，把一套已有方法在资源有限的环境里跑通，还能把每项改动的效果分开验证。指标、图像、代码和复算方法都放在一起，方便后面继续做。")
    write(stage, EXT + "/同骨干对照报告.txt", "同样的检测结果，能不能跑得更省\n\n" + "\n\n".join(STORY[2:]) + "\n\n" + "\n".join(accuracy_lines) + "\n\n" + RESOURCE_NOTE + "\n" + "\n".join(resource_lines) + "\n\n单遍统计、图片加载方式和注意力输出分别做了对照。按需加载主要省内存，速度并非每次都改善；完整逐次数据和下降项保留在report.html。进程内存峰值指工作集，提交内存另列，两者不混用。\n\n全27类还记录了81次PCA拟合耗时，独立资源试验完成36次。另外6次因可用内存不足中断的尝试单列保留。指标已复算，两张原图对应的六张CPU异常图也核对一致。\n\n" + RESULT_NOTE + "\n\n" + SOURCE_NOTE + "\n\n这轮没有新增检测算法组件，也没有用重复计时声称多随机种子稳定性。完整参数和计时定义见技术说明.txt，复算命令见复现说明.txt；逐类指标、所有下降项和54组固定对比图见report.html。")
    write(stage, EXT + "/复现说明.txt", "怎么自己核对这份结果\n\n先准备Python 3.11及交付代码中的固定依赖。下面的命令从交付文件夹根目录运行，用本地模型和权重重算全部27类四项指标，再对固定两张原图做三种方法的CPU推理：\n\n" + verify + "\n\n输出和临时文件放在交付文件夹外；再次运行请换一个新的回执文件名。它不下载数据，不重新拟合PCA，也不进行梯度训练。文稿整理后的入口只增加文档版本核对，指标与推理仍调用原来的核验代码，原脚本和旧回执保留。\n\n如果只想检查文件是否齐全，可运行：\npowershell -NoProfile -ExecutionPolicy Bypass -File .\\Verify.ps1 -PythonPath E:\\3k\\.venv\\Scripts\\python.exe -InferenceSamples 0\n\n完整重新拟合需要原项目工作区、全部官方数据，以及配置引用的早期输入记录。交付中保留了模型、全部配对预测、本地权重和54张固定源图，没有包含全部训练、测试原图。数据准备好后，在项目工作区执行，输出选择新的目录：\n\nSet-Location -LiteralPath 'E:\\3k'\n& .\\.venv\\Scripts\\python.exe -B scripts/run_route1_controls.py --config configs/route1_controls_20261014.json --output-dir runs/quick/route1-reproduction-new\n& .\\.venv\\Scripts\\python.exe -B scripts/run_route1_resource_matrix.py --config configs/route1_controls_20261014.json --output-dir runs/quick/route1-resource-reproduction-new\n\n遇到内存不足时，用保留的恢复脚本继续，别直接覆盖旧结果。参数、数据范围、计时区间和恢复约定见技术说明.txt，早期实验各自的运行步骤见对应报告。")
    tech = "参数和读表时需要知道的事\n\n模型使用DINOv2-S，输入448，全图缩放、不裁剪；取隐藏层[-4,-5]的均值，得到32×32个384维局部特征。三种PCA均在CPU上用相同float32输入和float64统计，协方差分母为N-1，保留99%解释方差，不白化、不做L2归一化。原版用Torch，两遍控制和单遍实现用NumPy，用来分开后端与遍数的影响。\n\n异常分数为平方重建残差。异常图先插值到448，再做3×3、sigma=4的平滑，最后到256×256评测平面；标注最近邻缩放。正常图背景进入像素评测，AUPRO积分范围为0至0.30，各区域等权，最终按类别等权平均。主图像评分取最大值，辅助评分取最高655像素的均值；两种图像评分共享原始异常图，定位指标相同。\n\n本轮有27类、81个PCA模型、3,887张测试图的11,661张配对异常图和162条指标。单遍对官方两遍的配对指标最大绝对差为1.1540756683636744e-9（比例），两数据集宏平均最大变化约9.0337349e-9个百分点，可视为浮点差异。完整四指标复算及两张原图的六张CPU异常图核对误差均为0，旧验收回执原样保留。测试集已经访问过，不能据此称为盲测或新算法稳定性证明。\n\n资源试验固定bottle和cashew各前64张正常图，2种统计遍数×2种读图方式，并另做注意力输出开关，每种3次独立进程，共36次。资源试验中的两遍组是NumPy控制，精度对照才另实际调用官方PCA。每遍实际解码和前向次数、输入内容摘要均保存。3次是运行重复，不是训练随机种子；共享电脑上还有背景负载。\n\nfit_seconds记特征提取加拟合，含解码、DINO前向、PCA和特征分解，不含模型加载与预热。total_process_seconds是含加载、预热的进程内计时：从main解析参数和准备输出目录之后开始，到监控结束之后、最终回执写入打印及系统退出之前结束，不是操作系统创建到退出的全生命周期。全27类的fit_only只记缓存特征上的读取、内容校验、统计和分解，不含DINO前向。\n\n正文资源降幅用各组3次中位数之比，逐次配对降幅的中位数在原表另列。例如bottle的两种口径不同，不能替换使用。负降幅就是回退。单遍和读图的交互项按（两遍提前加载-单遍按需读取）-（两遍提前加载-单遍提前加载）-（两遍提前加载-两遍按需读取）计算，只描述观测，不做显著性推断。\n\n内存工作集、私有内存、提交内存分列。全类运行共用进程的峰值仅作监控，不分摊到方法；81条共享进程拟合计时没有独立内存降幅。6次内存不足的资源试验中断记录仍保留，不混入最后成功的36次。GPU本轮没有使用，相应字段保留未使用而非测量值0。\n\n正常特征逐图写成float32的.npy块，并逐块读取；冻结协议曾写memmap，实际未使用整类常驻映射。内存中断后只复用有原图、特征、生产上下文及运行环境校验的正常特征，测试图始终重新前向，三个PCA仍实际拟合。恢复时normal_extraction_seconds含缓存读取，不能当成全新前向速度；独立36次资源测量不使用这种缓存时间来计算收益。已完成的临时缓存会清理。\n\n完整运行外部监控实际每2秒一次，独立资源进程每0.5秒一次；冻结协议中笼统的0.5秒不能当成主运行实际间隔。阶段采样可能漏过短时峰值，系统进程峰值另保留。启动资源门槛及运行保留余量按原配置执行，其他项目进程未被停止。\n\n运行环境变化时不复用旧缓存。PyTorch关于浮点数值的说明见 https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html 。完整恢复记录、冻结协议和原始配置保留，其字段名和历史措辞不随这次文稿整理改写。\n"
    write(stage, EXT + "/技术说明.txt", tech)
    write(stage, "code/代码说明.txt", "先沿着一张图片看代码\n\n训练阶段只读正常图片。DINOv2-S把图片转成局部特征，PCA用这些特征估计正常变化所在的子空间。测试时看特征偏离这个子空间有多远，把平方重建残差还原成异常图，再计算图像和像素指标。模型、权重和方法的来源没有改名。\n\n这次主要改的是正常图的处理过程。单遍统计合并均值和中心二阶统计量，按需加载让图片用完就释放；额外的注意力输出单独做了开关对照。原版两遍、NumPy两遍、当前单遍使用相同输入，便于判断差异来自哪里。原版本身已有流式统计。\n\n想从实现看起，可以先读同骨干对照目录code/src/industrial_anomaly_lab中的subspace_features.py和subspace_controls.py，再看run_route1_controls.py如何串起数据、拟合与评测。这里的代码没有因为润色报告而改写，原执行版本、注释和来源信息都保留，运行哈希还能对上。\n\n核对交付结果用verify_readable_delivery.py。这个入口只补文稿版本检查，计算仍使用原来的verify_route1_delivery.py及其评测模块。早期PaDiM和OAS源码也保留，用来说明前面的试验为什么没有继续。")
    print(json.dumps({"status": "drafts_prepared", "stage_root": str(stage), "root_and_route1_documents": 7, "new_explanations": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
