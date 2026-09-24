# Wan 语言、视频与夹爪动作实验（v4）

这是根据 2026-09-24 的要求重新建立的实验，独立于 v3 的 ImageNet/ResNet18 基线。不能把 v3 的训练步数或成功率算作 Wan 的结果。

## 代码和产物位置

- 本地源码：`human2robot/v4/`。
- embody-train2 源码：`/user/panmiao/workspace/human2robot-wan-v4-20260924/code/v4/`。
- 服务器实验根目录：`/user/panmiao/workspace/human2robot-wan-v4-20260924/`。
- 基座只读路径：`/user/liuhanyu/model_ckpt/Wan2.1-T2V-1.3B/`。
- 官方实现来源：<https://github.com/Wan-Video/Wan2.1>，固定提交 `9737cba9c1c3c4d04b33fcad41c111989865d315`。保留官方版权和许可证。
- `v4/native.py` 隔离加载官方核心模块，把其 FlashAttention 调用换为支持长度掩码的 PyTorch SDPA；模型权重严格加载。

| 文件 | 作用 |
|---|---|
| `prepare.py` | 审计语言来源、冻结数据划分与样本数、重新读取原始 RGB |
| `cache.py` | 缓存原配 UMT5 文字向量、原配 Wan VAE 视频 latent；验证 VAE 因果性 |
| `model.py` | Wan LoRA、动作头、动作条件适配器、未来视频和动作联合损失 |
| `train.py` | 配对种子训练、微调、验证集选模型、断点恢复、离线动作指标 |
| `smoke.py` | 真权重加载、语言敏感性、有限反向梯度与接口检查 |
| `policy.py`、`evaluate.py` | 相同 Wan 权重和动作头在原生 RoboTwin 中执行 |
| `visualize_predictions.py` | 留出集的真实/预测未来视频并排展示及数值动作预测 |
| `submit.py`、`continue_train.py` | 资源预检、GPU smoke、数据完成后提交正式训练 |
| `continue_eval.py`、`run_eval.py` | 成功训练后跨集群传输，再执行配对闭环评测 |

## 语言标注确实存在，且本轮进入网络

| 数据 | 文字字段 | 本次训练部分 | 标注性质 |
|---|---|---|---|
| 人类 EPIC/Ego4D | `action_config[].action_text` | 10,000 个窗口，3,068 个片段，2,622 条不同文字 | 上游自动标注，未人工逐条核验 |
| Bridge | `meta/episodes.tasks` | 10,000 个窗口，637 条演示，472 条不同文字 | 数据集提供的任务指令；不是本实验重新人工标注 |
| RoboTwin | `meta/episodes.tasks` | 40 条训练演示，40 条不同文字 | 数据集逐演示提供的指令 |

人类例子：`Place the blender container on the counter.`；Bridge 例子：`move the silver pot to the upper right burner`。RoboTwin 存在如 `Use the arm to position the rounded base blue cup onto the round coaster with light streaks.` 的描述。

文字经过 Wan 原配 UMT5-XXL 编码，然后通过 Wan 的文本交叉注意力影响视频预测和动作预测。文字编码器冻结，输出向量缓存；缺失指令直接报错，不自动用统一模板补齐。闭环使用固定指令 `Pick up the empty cup and place it on the coaster.`。

## 模型与监督

基座是原始 Wan2.1-T2V-1.3B 视频生成模型。我们添加当前帧条件、动作条件和动作预测头，把它改造成一个可训练的视频/动作模型；它不是现成的机器人世界模型，也不能仅凭基座名称宣称掌握了物理规律。

原始 Wan VAE、UMT5 和主干基础权重冻结。Wan 自注意力及文本交叉注意力的 q/k/v/o 使用 rank-16 LoRA；新动作头和动作条件适配器参与训练。所有组使用同一个真实预训练 checkpoint，每个 checkpoint 记录权重 SHA256、源码 SHA256 和数据审计 SHA256。

动作分支：当前图像 latent + 任务文字 → Wan 特征；保留 4×4 空间特征，再结合当前夹爪状态和左右槽位 → 4 步夹爪动作。该分支不接收未来图片、未来 latent 或真实未来动作。

视频分支：固定当前帧 latent，给未来 latent 加噪声，再输入文字及可用动作条件，学习 flow-matching 速度 `noise - clean_latent`。只在未来部分计算视频损失，不能通过重建当前图片取得虚假进展。

联合组损失：`L = L_future_video + L_gripper_action`。动作损失为经过训练集统计归一化后的 masked Smooth L1。人类伪动作根据观测有效性、重投影误差、姿态跳变和背景运动降权或屏蔽；不是校准后的真机动作真值。纯视频消融组预训练不输入人类状态/动作，也不训练动作头。

## 配对实验

| 组别 | 预训练 | 共同下游微调 |
|---|---|---|
| `human_joint` | 10k 人类视频窗口，文字＋未来视频＋弱夹爪轨迹 | RoboTwin 40 条演示，视频＋真实动作联合损失 |
| `bridge_joint` | 10k Bridge 窗口，文字＋未来视频＋真实夹爪轨迹 | 同上 |
| `human_video_only` | 同一批 10k 人类窗口，只训练文字条件未来视频 | 同上 |
| `robotwin_only` | 无额外预训练，直接使用同一 Wan 初始化 | 同上 |

`human_video_only` 是移除显式动作监督的消融，不等同于已经实现了某篇隐式动作论文。它与联合组使用相同视频和更新次数，但计算量较小；`robotwin_only` 也少了预训练算力。这些差异必须随结果报告。

每组种子 17/29/43/71，共 16 个模型。预训练 20,000 次参数更新；微调 10,000 次；micro-batch 2、梯度累积 8，有效 batch 16。10k 指独立窗口数，不是完整演示数，也不是训练步数；预训练约消费 32 遍样本。AdamW，初始峰值学习率 2e-4，500 步 warmup，余弦衰减，weight decay 0.01，梯度裁剪 1。每 500 步验证并保存 latest/best，不按测试集选模型。

输入由原始视频重新解码成 256×256，每个窗口包含 5 个真实帧、5 Hz、覆盖 0.8 秒；不把旧的 128×128 图片放大充当正式训练数据。Wan VAE 输出 2 个时刻的 latent。人类/Bridge 动作间隔 0.2 秒；RoboTwin 动作预测仍使用原生 15 Hz 的未来 4 个动作，视频目标每隔 3 帧采样。阶段切换保留网络权重、重置优化器和归一化统计。这个时间尺度变化属于迁移设置，不能把不同域的 loss 当成同物理单位误差比较。

人体硬件采集的动作标签未被使用。人类轨迹来自 RGB 几何伪标注。仅按容器搬运任务族匹配，杯子、锅、瓶子等比例仍有差别；不能称为严格同一任务分布。人类原视频文件隔离已经检查，但原始参与者/会话级划分尚未完全核实。

## 验证与启动顺序

1. 固定 10k 独立窗口与语言来源，严格分开训练/验证/测试；不足则失败，不重复填数。
2. CPU 解码原始视频，同时提交 Wan 真权重 GPU smoke。
3. 检查 tokenizer、基座严格加载、当前帧 VAE latent 因果性、文字变化导致预测变化、LoRA/动作头有限且非零反向梯度。
4. CPU 数据和 GPU smoke 均成功后，控制脚本只提交一次正式 8×H100、NORMAL、80 CPU、512 GiB 的 PYTORCHJOB。准备阶段失败则不进入正式训练。
5. 8 张 GPU 缓存冻结的 UMT5 和 VAE；之后每张 GPU 一个模型，两个波次完成 16 个模型。缓存只包含固定冻结编码器输出，Wan LoRA 每一步实际更新，不是缓存 ResNet 特征替代训练。
6. 训练成功后，传输真实 Wan 基座、VAE、任务文字向量与 16 个动作适配器到 embody-eval。先做实际策略加载/动作执行 smoke，再跑闭环。

断点保存优化器、随机数状态和适配器；评测按 checkpoint hash 和步数上限复用完整 episode。控制器不会遇到不确定的提交结果就反复申请资源；失败写入 `*_BLOCKED.json`。

## 评测与可视化

任务 `place_empty_cup`，所有模型使用此前由专家可行性检查冻结的同一组 100 个场景种子，每个模型最多 400 次控制调用。以 `env.check_success()`/原生评测成功信号为依据，报告每种子的成功次数、成功率和动作误差。不会把专家演示成功当成策略成功，也不会把评测超时当成已经完成。

保存真实闭环 MP4、原始预测动作、限幅后目标位姿、实际执行位姿。未来视频可视化采用当前图像和模型自己预测的动作，从噪声迭代生成，与真实未来视频并排；不是把真实未来帧喂进去后展示重建。动作数值使用明确字段和单位。图像生成效果与闭环成功率分别报告。

离线指标：TCP 位移误差（m）、95 分位位移误差、旋转测地角误差（deg）、开合 MAE。人类弱标注的伪米误差不与真机米制误差混比。是否能替代真机数据，必须看同等下游微调后的配对闭环结果；Wan 结果尚未完成前不作结论。

生成留出集视频/动作对比命令（训练后执行）：

```bash
cd /user/panmiao/workspace/human2robot-wan-v4-20260924/code
PYTHONPATH=../deps:. /user/panmiao/workspace/envs/windtunnel/bin/python -m v4.visualize_predictions --checkpoint ../training/seed17/human_joint/finetune_best.pt
```
