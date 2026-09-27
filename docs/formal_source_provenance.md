# 正式源码身份

正式实验必须能够回答“训练和推理究竟使用了哪一份代码”。项目在每次 `formal_full` 启动前执行 `clean_git_source_provenance_v1` 门禁：

- 项目仓库必须是干净的 Git HEAD；tracked 修改、staged 修改和未跟踪文件都会拒绝启动；
- DRAEM、PatchCore、RD4AD vendor 仓库必须分别位于登记目录、处于登记 commit，并且各自工作树干净；
- 记录项目及每个 vendor 仓库的 HEAD commit 和 HEAD tree；
- 单独记录项目配置、开发协议、正式登记、正式测试清单协议和正式训练输入根清单的大小及 SHA-256。

项目仓库的 clean HEAD 与 tree 绑定全部已跟踪代码，避免只列出少数入口文件而漏掉间接依赖。未提交的新源码会使工作树变脏，因此同样不能进入正式实验。正式训练把这份证据写入 checkpoint/fitted-state 清单，推理和聚合会重新采集当前状态并逐字段比较；代码 commit、vendor commit、配置或协议发生变化时，旧 checkpoint 不能混入同一正式矩阵。

登记范围在 [formal_source_provenance.json](../configs/formal_source_provenance.json) 中。当前工作区仍在开发且有未提交改动，所以这个门禁现在应当失败；只有完成实现、测试、提交并冻结正式方法和配方后才具备运行条件。
