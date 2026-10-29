# 人类视频标注网页

按原项目 `human2robot/web/styles.css` 的配色、卡片与布局约定，以及共享聊天中的样本选择、独立标注、样本质检、数据集汇总结构重建。原 `local_annotation_lab` 源文件未找回，因此本目录是重建版本。

## 当前交付

本机运行入口：<http://127.0.0.1:53419/>。正在提供服务的目录为 `/Users/panmiao/Documents/ChatGPT/paper/outputs/hand_mesh_repair_20261020`；旧的独立手型比较页保留为 `hand_review_before_annotation_lab.html`。

如果本地服务停止，在项目根目录运行以下命令可恢复同一入口：

```sh
python3 tools/serve_hand_review.py --directory /Users/panmiao/Documents/ChatGPT/paper/outputs/hand_mesh_repair_20261020 --port 53419
```

本目录保留网页源码；视频与报告保存在上述输出目录。仅对本目录启动静态服务无法读取这些媒体。

六项固定展示：原始视频、离散手部关节点、HaMeR 手型、语义、SAM3 分割、物体 bounding box。前三项接入真实视频。后三项资源未找到，使用明确空状态，不能将原视频或启发式结果当作模型输出。

语义卡下方只引用共享聊天中的整体短语、动词、名词及操作含义；没有捏造七段子任务的时间边界。

## 资源接入

`assets/multi_manifest.json` 描述样本与六类标注。视频地址、缩略图均相对于网页目录，允许使用完整 HTTP(S) 地址。新增真实结果时更新对应层的 `video`、`poster`、`badge`、`badge_tone`、`assessment` 等字段；缺失时 `video` 必须为 `null`。质量状态不得因文件存在而自动提升为通过。

网页支持数据集和样本选择、同步播放/暂停、逐帧移动、时间拖动、播放速度、原生视频控制，以及跳至未检测或需复核帧。视频服务使用 `tools/serve_hand_review.py`，支持 HTTP Range。

## 验证

三个视频均在浏览器解码为 640×480、161.133333 秒；验证同步播放、暂停、逐帧移动、复核帧跳转、倍速、窄屏无横向溢出。所有已接入视频及缩略图通过实际 HTTP 206 Range 检查。没有重新运行模型或更改已有预测。

后续方案见 [物体位姿与机器人转换](../docs/OBJECT_POSE_AND_EGOMATRIX.md)。
