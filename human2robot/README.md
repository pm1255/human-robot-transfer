## 第二轮：双手、全身质检与 RoboTwin 闭环

最新页面：[证据与闭环](http://127.0.0.1:8877/web/v2.html)。用 `python scripts/serve_review.py` 启动支持视频精确跳转的本地服务。

- 131 段真实视频：96 段 EPIC 瓶子相关视频参与人类辅助训练；35 段第三人称视频只用于人体标注审计。
- 两只手分别保存和跟踪；全身、物体框与原视频叠加，支持局部 3D 转角。第一人称错误人体预测保留在 raw 字段，经过来源门控后不绘制为合格全身。
- 四个条件：robot_only、explicit、implicit_joint、shuffled。两个初始化种子，机器人数据固定 40/5/5 episodes。最终各训练 2,000 + 8,000 步，以绝对关节目标输出。
- 最终指标为原生 adjust_bottle 闭环成功率；离线动作误差、物理关节跟踪误差和辅助损失分开。开发场景与正式测试场景分开。

详细协议与限制见 [V2实验协议](docs/V2实验协议.md)。实际训练/评估时使用的源代码快照在 `outputs/v2/versions/`，训练初始化、数据划分与曝光量审计在 `outputs/v2/matched_experiment_audit.json`。当前主源码经过格式整理，不能用文件文本哈希冒充运行时快照哈希。

```bash
python -m v2.annotate --root DATA_RGB --models MODELS --out ANNOTATIONS
python -m v2.train --robot ROBOT_EPISODES --human ANNOTATIONS --out RUN \
  --pre-steps 2000 --fine-steps 8000 --action-representation absolute --seed 17
python -m v2.evaluate --robotwin ROBOTWIN_ROOT --checkpoint RUN/robot_only/final.pt \
  --seeds TEST_SEEDS_JSON --out EVALUATION --single-active-arm
```

`--single-active-arm` 只适用于本次单臂任务：依据模型预测选择活动臂并保持闲置臂初始目标。真正双臂任务应关闭此约束。人类显式监督目前是二维双手运动辅助，不是已标定重定向的机器人伪动作；人体和物体标注目前用于审计，未加入策略损失。

---

# Human2Robot：RGB 人类操作的可审计参考轨迹

本项目把人类视频、模型伪标注、修复轨迹、机器人重定向参考和训练准入结果放在一起检查。第一版目标是两指夹爪机械臂。**参考轨迹不等于机器人执行过的动作。**

- [本轮验证结果](docs/验证结果.md)：27 项测试、543 帧 LeRobot 读回、单卡短训练及其限制。
- [详细技术方案](docs/方案.md)：每个步骤的做法、原因、公式、失败条件与字段。
- [训练与资源协议](docs/实验与资源.md)：显式/隐式对照、RoboTwin 微调、数据替代验证。
- [交互网页](http://127.0.0.1:8876/web/)：真实视频、关键点、轨迹修复、IK、失败例子。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test,vision]'
.venv/bin/python -m pytest -q
.venv/bin/python -m h2r.cli gallery --synthetic-only
.venv/bin/python -m http.server 8876 --bind 127.0.0.1
```

本次工作区虚拟环境实际上在项目父目录 `../.venv`。Mac 无图形上下文时 MediaPipe 可能即使指定 CPU 仍初始化 OpenGL 失败；这不是空视频，应在 Linux CPU 环境运行姿态模型，不伪造检测结果。通过 CLI `gallery` 生成/刷新网页数据。

模型使用 Google 官方 HandLandmarker bundle，保存于 `data/models/hand_landmarker.task`。真实视频清单放在 `data/raw/bootstrap/manifest.json`。`scripts/collect_epic_bootstrap.py` 为本次集群抽取脚本，路径需按自己环境修改；不自动上传到任何公共服务。

```bash
python -m h2r.cli annotate path/to/human.mp4 --out outputs/real/my_clip
python -m h2r.cli convert outputs/real/my_clip/annotation.json --urdf configs/demo_arm.urdf --out outputs/converted
python -m h2r.cli export-lerobot outputs/real/my_clip --out outputs/lerobot/my_clip
```

实际机器人可指定 `convert --base <base_link> --tip <tcp_link> --alignment T_robot_world.json`。
相机标定输入用 `annotate --intrinsics K.json --camera-trajectory camera.json --source-group <original_session>`。
`K.json` 是 3×3 数组，外参文件是 4×4 数组；相机文件包含 `timestamp`、`T_world_camera` 和明确的 `status`，采样时间必须一致。
这些参数传入已有标定结果，不会自动证明输入可信，也不会把估计尺度升级为已标定尺度。

`export-lerobot` 需要独立安装/配置官方 LeRobot。导出的是自定义人类辅助监督视图，没有捏造 action 或 measured state；不能直接当标准机器人 BC 数据使用。`export_references` 生成的是本项目 NPZ 参考协议，不是 LeRobot。

## 代码索引

| 文件 | 职责 |
|---|---|
| `h2r/schema.py` | 数据契约、来源、单位、时间戳与合法性 |
| `h2r/vision.py` | RGB 手部模型、2D/3D、PnP、身份与缺失记录 |
| `h2r/geometry.py` | 坐标、SO(3)、捏合候选、物体相对转移 |
| `h2r/repair.py` | 加权离线平滑、跳点和短缺失修复 |
| `h2r/kinematics.py` | URDF/FK/IK、限位、速度和碰撞代理 |
| `h2r/pipeline.py` | 端到端参考处理、准入掩码、未来窗口 |
| `h2r/export.py` | 审计 NPZ 和官方 LeRobot 写入/读回 |
| `h2r/fixtures.py` | 12 类可复现的合成压力测试 |
| `scripts/train_pilot.py` | 4 条人类预训练条件 + 5 条机器人微调条件 |
| `scripts/collect_robotwin_pilot.py` | 只读采集小型 RoboTwin 验证子集 |

## 训练命令

```bash
python scripts/train_pilot.py --human-root outputs/real \
  --out outputs/training --steps 120 --device cuda \
  --robot-data data/robotwin_pilot.npz --robot-steps 200
```

该模型是小型 CNN，不是完整 VLA。训练记录包括实际数据曝光、梯度、参数变化、标签置乱对照和机器人离线验证指标。它验证通路，不证明真机数据替代率。正式结论需要完整 VLA、多种子、仿真闭环及真机验证。

## 已知边界

真实片段的相机运动、米制尺度、接触和机器人外参尚未验证。网页中真实视频的机器人首帧对齐仅供演示，真实机器人动作准入应保持为零。示例臂是教学模型；碰撞只检查连杆代理对球/平面，不含完整自碰撞、网格、动力学、抓取力。左右手交叉时的身份关联仍可能失败。

本代码不读取人类采集硬件动作标签，不修改源数据，不默认将数据发布到 Hub。所有源路径、版本、缺失和失败原因都应随派生数据保留。

### 标注缺失、重定向与动作效果补充

第二版网页增加逐帧覆盖条、头部 11 点编号开关、20 个同步重定向回放（8 段真实 RGB 的示意对齐，12 个合成测试）、三段实际执行的开发失败录像及 18 类指标解释。真实重定向片段仍为 0 动作训练准入，开发录像不进入正式成功率。

- 指标说明：`docs/指标与可视化口径.md`
- 发布说明/开发录像：`python -m v2.build_review_details`
- 核对 FK、长缺失、执行指标：`python -m v2.audit_review_details`
- 审计结果：`outputs/v2/review_details_audit.json`

旧开发录像缺少瓶子三维轨迹，页面显示缺项。正式评估输出已增加 `bottle_point`，`build_results.py` 会把该字段与原始/执行动作一起导出。
