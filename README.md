# Human ↔ Robot Transfer Lab

人类视频标注、机器人重定向、动作/未来视频预测、机器人转人手视频编辑的研究代码。

[公开效果主页](https://robot-human-video-lab.panmiao307.chatgpt.site/)

| 入口 | 做什么 | 当前边界 |
|---|---|---|
| `human2robot/h2r/` | 手部标注、轨迹修复、URDF/FK/IK、参考导出 | 参考动作不等于机器人真实执行 |
| `human2robot/v2/annotate.py` | 双手/全身/物体模型标注 | 模型伪标注，需要人工审核 |
| `human2robot/v2/render_conversion.py` | 人手→Aloha、人体→G1 表面渲染 | 启发式掩码 + Telea 补背景，非生成模型 |
| `human2robot/v3/` | 视频、语言、动作数据准备与训练 | 见对应实验文档 |
| `human2robot/v4/` | Wan LoRA 未来视频和动作预测 | **不是人机视频编辑器，不能直接反向运行** |
| `robot2human/` | SAM2 + Wan VACE 视频编辑及配对导出 | 手指、接触、遮挡尚不可靠；默认禁止动作训练准入 |

## 安装和测试

```bash
cd human2robot
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,vision]'
python -m pytest -q
```

渲染、训练、导出需要分别安装 `.[render]`、`.[train]`、`.[export]`。Wan/SAM2 上游代码和模型需单独安装；见 `robot2human/README.md`。GPU/模型路径和集群脚本保留原实验设置，换环境时必须配置，不承诺零配置复现。未包含原始数据、模型权重、第三方机器人网格或认证信息。

## 实验状态

机器人→人手已有 6 个独立源片段、18 个推理版本。扩大掩码和切换随机种子只能提供候选，尚未解决五指解剖、物体形状和接触一致性。14B 在部分场景仍生成机械夹爪，不优于较小模型。

人→机器人结果属于几何合成预览；不是已有高质量视频生成系统。公开页区分标注、合成预览和扩散模型生成，不混算样例数量。

最新优化判断与实验路线见 [OPTIMIZATION.md](OPTIMIZATION.md)。文件哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。v4 来自服务器代码快照，其余 human2robot 模块来自现有工作区；robot2human 为本次实际使用脚本。代码仅归档当前研究状态，不将未运行的新方案标成有效结果。

## 第三方项目

依赖 [Wan2.1](https://github.com/Wan-Video/Wan2.1)、[VACE](https://github.com/ali-vilab/VACE)、[SAM2](https://github.com/facebookresearch/sam2)、MediaPipe、MuJoCo、RoboTwin 等；安装和使用遵循各自许可证。数据和模型权利不因本仓库上传而改变。仓库没有替第三方素材授予新许可证。

服务器实际运行的 v2/v3 副本另存于 `runtime_snapshots/`，与整理后的主源码区分。网页源码在 `website/`，旧标注查看器在 `human2robot/web/`；大体积媒体不在代码仓库中。
