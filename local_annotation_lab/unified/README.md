# 三类数据统一标注试点

本目录是首批试点的适配、初标构建和质检工具。结果是待独立复核的初标，不是训练金标。当前发布了三类各 3 个独立 episode，共 153 段；原折衣样本占 90 段。

## 页面和数据

- 网页：<http://127.0.0.1:53419/>
- 实际服务目录：`/Users/panmiao/Documents/ChatGPT/paper/outputs/hand_mesh_repair_20261020`
- 本轮报告、源数据摘录和检查明细：`/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_three_domain_20261020`
- 本目录上一级的 HTML/JS/CSS 是页面源码；视频和统一 JSON 保存在服务目录的 `unified/`。只服务源码目录会缺少视频。
- `assets/multi_manifest.json` 定义样本、源帧率、相机资源和标注路径。`?clip=<episode_id>` 可直接定位样本。

服务停止后，在项目目录启动（若端口占用，不要终止已有服务）：

```sh
python tools/serve_hand_review.py --directory /Users/panmiao/Documents/ChatGPT/paper/outputs/hand_mesh_repair_20261020 --port 53419
```

## 处理流程

1. `remote_extract.py` 在数据服务器读取 ABC 和 EgoDex。ABC 的候选列表为结果目录中的 `abc_selection.json`；每个相机使用自己的源时间偏移。UMI 使用已有本地导出，具体路径见各样本 `source.json`。源数据只读。
2. `contact_sheets.py` 生成观察图；`build_pilot.py` 的 `CONFIG` 保存本轮视觉初标。其 `main()` 重新构建 8 个新样本，并从 `manifest_before.json` 恢复原折衣资源后映射词表。路径和字体是当前主机配置，需要迁移时修改顶部常量。依赖 numpy、Pillow、FFmpeg；远端另需 pyarrow、tensorboard。
3. `validate_pilot.py` 完整解码新增视频并检查帧数、帧率、分段覆盖、词表及原生时序。通过只代表结构可用，不代表语义/几何精度通过。
4. 浏览器可以编辑边界和语义、拆分/合并。草稿仅保存到当前浏览器；“导出语义 JSON”生成下载链接和可复制内容。内置浏览器的下载事件未成功验证，已验证“复制完整 JSON”与导出内容一致，可粘贴到 `.json` 文件留档。不能将本地保存当作服务端发布。
5. 独立复核时，两人从原视频各自分段并标注；提交文件加上不同的 `reviewer_id` 和真实的 `independent_review: true`。然后运行：

```sh
python compare_reviews.py reviewer_a.json reviewer_b.json --output agreement.json
python -m unittest test_review_metrics.py
```

审核者标记只是来源声明，不是程序能独立证明的事实。比较工具永远不自动批准训练准入。原折衣允许 `source_episode`，新样本使用 `episode_id`；时间区间均为左闭右开帧区间。

## 发布与复核注意

`build_pilot.py` 会按初标配置重新生成派生文件，**不会自动导入浏览器导出的复核 JSON**。重新构建前保留审核版本；审核后需用该版本重新渲染字幕视频，运行结构检查，再更新发布清单。切勿仅替换 JSON 却把旧字幕视频当作新版本。本轮没有多人数据库、审核登录或自动发布服务。

`quality_plan.md` 是验收方案，`validation.json` 是实际检查结果；`browser_validation.json` 记录页面验证及下载限制；`release_manifest.json` 固定本轮资源 SHA256。新增物体 mask、bbox、6D 均为显式待补，源末端位姿不能代替物体位姿。

## 监控

TensorBoard：<http://127.0.0.1:16067/?tagFilter=completed&runFilter=unified_three_domain#timeseries>

远端日志：`/user/panmiao/human-robot-transfer/outputs/object_annotation_20261020/events/unified_three_domain`。

`selection=9`、`annotation=9`（初标）、`qc=9`（结构检查），`independent_review=0`（等待）。本轮未做模型训练，不记录伪造训练 loss。重跑任务时继续记录真实完成数，复核阶段必须等待实际独立结果，不能随构建自动置为完成。
