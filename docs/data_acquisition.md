# 数据获取与许可记录

## DTD R1.0.1

DTD 已从 Oxford VGG 官方入口下载到 `data/archives/dtd-r1.0.1.tar.gz`，随后解压到 `data/dtd`。归档大小、SHA-256、归档测试与提取计数见 `manifests/dtd_r1.0.1.json`。

官方页面说明当前版本含 47 类、5,640 张图，本地计数与之相符。官方只声明面向研究用途提供，没有找到 CC、MIT 等标准许可证；图像来源还涉及 Google/Flickr。因此本项目按研究用途保存，交付包默认不再分发 DTD 原始归档。

来源：https://www.robots.ox.ac.uk/~vgg/data/dtd/

## MVTec AD

MVTec AD 官方页面确认数据含 15 类、超过 5,000 张高分辨率图像，许可为 CC BY-NC-SA 4.0，并明确禁止商业使用。

截至 2026-09-27，官方下载入口要求提交 First Name、Last Name、Email 和非商业使用确认。DRAEM 作者仓库中的旧 MyDrive 链接当前返回 404，旧静态下载地址也不可用。项目不会使用未知镜像或生成数据替代真实 MVTec AD。

待办：由用户提供官方表单所需身份信息并授权提交，或者由用户自行提交表单后提供生成的官方下载链接。取得归档后，必须记录来源、大小、SHA-256、归档完整性和逐类文件清单，再解锁正式评测。

来源：https://www.mvtec.com/research-teaching/datasets/mvtec-ad

## VisA

VisA 是本项目第二个正式验证集。Amazon Science 官方仓库确认其包含 12 类、10,821 张高分辨率图像，其中 9,621 张正常图和 1,200 张异常图，并提供图像级及像素级标注。数据许可为 CC BY 4.0。

官方 AWS Open Data 归档：`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`。下载后使用作者仓库 `split_csv/1cls.csv` 的 1-class 划分转换为与 MVTec 相同的目录结构；不得自行重分测试异常样本后仍称官方 1-class 协议。

来源：https://github.com/amazon-science/spot-diff 及 https://registry.opendata.aws/visa/
