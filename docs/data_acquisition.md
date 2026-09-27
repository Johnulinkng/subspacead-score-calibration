# 数据获取与许可记录

## DTD R1.0.1

DTD 已从 Oxford VGG 官方入口下载到 `data/archives/dtd-r1.0.1.tar.gz`，随后解压到 `data/dtd`。归档大小、SHA-256、归档测试与提取计数见 `manifests/dtd_r1.0.1.json`。

官方页面说明当前版本含 47 类、5,640 张图，本地计数与之相符。官方只声明面向研究用途提供，没有找到 CC、MIT 等标准许可证；图像来源还涉及 Google/Flickr。因此本项目按研究用途保存，交付包默认不再分发 DTD 原始归档。

来源：https://www.robots.ox.ac.uk/~vgg/data/dtd/

## MVTec AD

MVTec AD 官方页面确认数据含 15 类、超过 5,000 张高分辨率图像，许可为 CC BY-NC-SA 4.0，并明确禁止商业使用。

MVTec 官网当前交互入口显示下载表单；DRAEM 仓库中的旧 MyDrive 链接已失效。为避免未知镜像，项目采用 `open-edge-platform/anomalib` 官方主分支在 commit `d716bb94...` 登记的 MVTec AD 下载信息。该登记同时给出归档 SHA-256 `cf4313b1...98cd3d`，因此可对取得的数据做强校验。Anomalib 源码以子模块固定，登记文件自身的哈希也写入 `manifests/mvtec_source_manifest.json`。

HTTP 入口核对为 5,264,982,680 字节并支持 Range 续传。归档现已匹配登记 SHA-256 `cf4313b1...98cd3d`，通过 `tar.xz` 路径安全与 15 类结构审计；完整证据见 `manifests/mvtec_source_manifest.json` 和 `manifests/mvtec_prepared.json`。项目不会使用生成数据替代真实 MVTec AD，也不会在交付包中重新分发受非商业、相同方式共享条款约束的数据。

安全解压与机械审计由 `scripts/prepare_mvtec_ad.py` 完成，具体 fail-closed 规则见 `docs/mvtec_preparation.md`。脚本只检查归档 metadata、文件路径/大小和掩码文件名配对，不解码或展示正式 test 图像；成功后才生成 `manifests/mvtec_prepared.json`。

来源：https://www.mvtec.com/research-teaching/datasets/mvtec-ad

## VisA

VisA 是本项目第二个正式验证集。Amazon Science 官方仓库确认其包含 12 类、10,821 张高分辨率图像，其中 9,621 张正常图和 1,200 张异常图，并提供图像级及像素级标注。数据许可为 CC BY 4.0。

官方 AWS Open Data 归档：`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`。下载后使用作者仓库 `split_csv/1cls.csv` 的 1-class 划分转换为与 MVTec 相同的目录结构；不得自行重分测试异常样本后仍称官方 1-class 协议。

来源：https://github.com/amazon-science/spot-diff 及 https://registry.opendata.aws/visa/
