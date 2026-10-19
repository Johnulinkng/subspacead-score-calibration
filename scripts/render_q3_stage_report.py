"""Render measured calibration evidence, with a separate live-compute status."""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha


def main():
    run=ROOT/"runs/research-q3-20261016";source=run/"contribution-refits";out=run/"stage-report";out.mkdir(exist_ok=True)
    if read(source/"run_state.json")["status"]!="completed":raise ValueError("Only a completed full calibration matrix can be reported")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats=read(source/"paired_statistics.json");macro=read(source/"dataset_macro_metrics.json")
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),sharey=True)
    for ax,metric,label in zip(axes,("aupro_0_30","pixel_ap"),("AUPRO@0.30","Pixel AP")):
        for i,dataset in enumerate(("mvtec_ad","visa")):
            r=next(r for r in stats if r["dataset"]==dataset and r["reference"]=="global_tail" and r["metric"]==metric)
            lo,hi=r["bootstrap_95_ci_pp"];point=r["mean_delta_pp"]
            ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt="o",color="#23638c",capsize=5,markersize=6)
        ax.axvline(0,color="#777",lw=1,ls="--");ax.set_title(label);ax.set_xlabel("Candidate minus global-tail control (pp)");ax.set_yticks([0,1],["MVTec AD","VisA"]);ax.grid(axis="x",alpha=.2)
    fig.suptitle("Independent content-conditioning effect: 3 calibration refits")
    fig.tight_layout();fig.savefig(out/"content_contribution_ci.png",dpi=180);fig.savefig(out/"content_contribution_ci.svg");plt.close(fig)
    names={"baseline":"原始评分","global_tail":"统一尾部校准","conditional_tail":"内容条件校准","random_group_tail":"随机分组对照","spatial_tail":"位置尾部校准","conditional_no_floor":"去掉全局下界","conditional_no_guard":"去掉高分保护","prototype_quantile_control":"原型分位数标准化"}
    table=[]
    for r in macro:
        values="".join(f"<td>{r['mean'][m]*100:.4f} ± {r['std'][m]*100:.4f}</td>" for m in ("image_auroc","pixel_auroc","pixel_ap","aupro_0_30"))
        table.append(f"<tr><td>{html.escape(r['dataset'])}</td><td>{names.get(r['variant'],r['variant'])}</td>{values}</tr>")
    ci_rows=[]
    for r in stats:
        if r["reference"]!="global_tail":continue
        lo,hi=r["bootstrap_95_ci_pp"]
        ci_rows.append(f"<tr><td>{r['dataset']}</td><td>{r['metric']}</td><td>{r['mean_delta_pp']:+.4f}</td><td>[{lo:+.4f}, {hi:+.4f}]</td><td>{r['positive_categories']}/{r['categories']}</td><td>{r['exact_two_sided_sign_flip_p']:.5f}</td><td>{r['holm_familywise_p']:.5f}</td></tr>")
    current={}
    for name in ("features","independent-partitions","trainonly-development","selected-candidate","final-analysis"):
        p=run/name/"run_state.json"
        current[name]=read(p) if p.exists() else {"status":"not_yet_reported"}
    active=read(run/"active_baseline_queue.json") if (run/"active_baseline_queue.json").exists() else None
    active_queue=Path(active["queue"]) if active else run/"baseline-queue-v2.json"
    if active and sha(active_queue)!=active["queue_sha256"]:raise ValueError("Active queue changed")
    first=read(active_queue)["training_jobs"][0]
    baseline=Path(first["receipt_dir"]).parent/"model/run_state.json"
    current["first_full_rd4ad"]=read(baseline) if baseline.exists() else {}
    if (run/"active_recovery.json").exists():
        recovery=read(run/"active_recovery.json")
        current["recovery_supervisor"]=read(recovery["state"])
    write(out/"status_snapshot.json",current)
    queue_link="../"+active_queue.name
    development_link="../trainonly-development/frozen_protocol.json" if (run/"trainonly-development/frozen_protocol.json").exists() else "../../../configs/q3_trainonly_development_20261016.json"
    status="".join(f"<tr><td>{html.escape(k)}</td><td><pre>{html.escape(json.dumps(v,ensure_ascii=False,indent=2))}</pre></td></tr>" for k,v in current.items())
    timestamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S 北京时间")
    body=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SCI 三区目标：实验推进记录</title><style>body{{max-width:1100px;margin:44px auto;padding:0 22px;font-family:'Microsoft YaHei',sans-serif;color:#18232c;line-height:1.75}}h1{{font-size:27px}}h2{{font-size:20px;margin-top:34px}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}}th,td{{border:1px solid #dbe2e7;padding:8px;text-align:left}}th{{background:#eef3f6}}pre{{white-space:pre-wrap;line-height:1.4;margin:0}}img{{width:100%;max-width:1000px}}a{{color:#165f8a}}.note{{background:#f4f7f9;padding:16px 20px;border-left:4px solid #4d7288}}small{{color:#59636b}}</style>
<h1>SCI 三区目标：实验推进记录</h1><small>{timestamp}；这是阶段记录，运行状态为生成报告时的快照。</small>
<p class="note">当前原始内容条件方案仍未通过研究验收。三种子校准重拟合显示，MVTec 的 AUPRO 有小幅正收益，VisA 则下降；VisA 上相对简单尾部校准的独立贡献区间仍跨过零。后续实验继续检验正常划分、既有校准机制和调整方案。</p>
<p>本批已完成 27 类、81 次主矩阵校准重拟合和 108 次敏感性拟合，共 756 条指标记录，其中 648 条来自三种子主矩阵，108 条来自首个种子的四项参数敏感性检查。PCA 和骨干固定；原始评分在三次校准对照中复用，不能算作三次模型训练。测试集此前已访问，这批结果属于回顾性验证。</p>
<p>已核对全部 97,175 张新生成异常图的哈希，并用另一套指标实现抽查 22 条记录，最大误差约 1.59×10⁻¹²。这是全量文件完整性检查与抽样独立复算，尚未完成 756 条记录的全量独立回放。</p>
<h2>内容条件的独立收益</h2><p>下图比较内容条件校准与统一尾部校准。区间按类别的三次拟合平均差值进行 10,000 次配对重采样，表示未调整的 95% 区间。表中另列精确符号翻转检验与全比较族 Holm 校正；它们不能替代独立测试。</p><img src="content_contribution_ci.png" alt="两数据集内容条件模块的 AUPRO 和 AP 独立收益及置信区间"><table><tr><th>数据集</th><th>指标</th><th>平均变化/百分点</th><th>95% 区间</th><th>正收益类别</th><th>精确 p</th><th>Holm p</th></tr>{''.join(ci_rows)}</table>
<h2>完整主矩阵</h2><p>数值为百分制的类别宏平均，均值 ± 三次校准拟合之间的样本标准差。逐类、逐种子和四项敏感性结果保存在原始 CSV 中，未用最好种子替代均值。</p><table><tr><th>数据集</th><th>方法</th><th>I-AUROC</th><th>P-AUROC</th><th>P-AP</th><th>AUPRO@0.30</th></tr>{''.join(table)}</table>
<h2>正在运行的后续实验</h2><p>新的正常数据划分将 PCA 拟合、校准统计、操作阈值和开发图像分开，并按内容哈希避免完全相同图像跨分区。调整方案先用正常训练图像与 DTD 构造的开发集筛选，再确认剩余种子；合格候选才进入全类别真实缺陷复核。合成结果仅用于开发选择。</p>
<p>强对照登记了 DRAEM、PatchCore、RD4AD 各 27 类 × 3 种子，共 243 个完整拟合或训练单元。RD4AD 保留 200 轮，DRAEM 保留 700 轮。已完成的资源检查不计入完整结果。原始 SCI 三区方案的主指标和数值要求保留，当前没有宣布达标。</p><table><tr><th>阶段</th><th>最近记录</th></tr>{status}</table>
<p><a href="../contribution-refits/category_metrics.csv">756 条原始指标</a>　<a href="../contribution-refits/paired_statistics.json">配对统计</a>　<a href="../contribution-refits/contribution_decision.json">阶段判定</a>　<a href="{queue_link}">完整强对照登记</a>　<a href="{development_link}">开发选择登记</a></p>
<p><small>基线方法、骨干预训练和校准思路的已有来源继续保留。新的数值证据用于判断具体设计的价值，不以文档措辞证明原创性或期刊录用。</small></p></html>"""
    (out/"report.html").write_text(body,encoding="utf-8")
    write(out/"generation_receipt.json",{"generated_at":timestamp,"source_sha256":{p.name:sha(p) for p in (source/"category_metrics.csv",source/"paired_statistics.json",source/"dataset_macro_metrics.json",out/"status_snapshot.json",Path(__file__))},"report_sha256":sha(out/"report.html"),"figures_sha256":{p.name:sha(p) for p in out.glob("content_contribution_ci.*")},"q3_goal_achieved":False})
    print(str(out/"report.html"))


if __name__=="__main__":main()
