"""Build model identities, observed learning curves and matched protocol for review."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/v2/results_absolute/training"
models = [
    {
        "name": "人体 / 手部标注",
        "base": "MediaPipe Pose Landmarker Full / Hand Landmarker 官方预训练权重",
        "trained": "本项目训练 0 步；只推理",
        "output": "身体 33 点；每手 21 点及左右标签",
        "used": "手部二维运动用于 explicit；人体用于质检展示",
    },
    {
        "name": "物体标注",
        "base": "EfficientDet Lite0 / COCO 官方预训练权重",
        "trained": "本项目训练 0 步；只推理",
        "output": "物体类别与二维框",
        "used": "质检展示；未加入本轮策略损失",
    },
    {
        "name": "人 → 机器人画面转换",
        "base": "无生成基座：启发式掩码 + OpenCV Telea + 机械臂/G1 几何渲染",
        "trained": "训练 0 步；G1 每帧数值 IK 不是网络训练",
        "output": "原画 / 删除区域 / 替换画面三栏视频",
        "used": "新增可视化预览，机器人训练准入 0；未进入当前 8 个模型训练",
    },
    {
        "name": "未来图像预测分支",
        "base": "自定义 CNN 编码器 + MLP，随机初始化，无预训练视觉/语言基座",
        "trained": "图像 loss 仅前 2,000 步；共享编码器继续接受 8,000 步动作微调",
        "output": "当前帧 → +0.5s 的 32×32 RGB 残差",
        "used": "implicit_joint 的辅助监督；不是人变机器人模型，也不是视频扩散模型",
    },
    {
        "name": "机器人策略",
        "base": "同一 CNN 编码器 + 三视角特征拼接 + MLP 动作头；没有语言输入",
        "trained": "每条件 2,000 + 8,000 = 10,000 步，随机种子 17 / 29",
        "output": "16×14 绝对关节/夹爪目标；每次执行前 4 条",
        "used": "小型视觉行为克隆（BC）；未训练完整 VLA",
    },
]
conditions = [
    ("robot_only", "仅机器人动作"),
    ("explicit", "机器人动作 + 0.1×二维手部运动"),
    ("implicit_joint", "机器人动作 + 未来图像残差"),
    ("shuffled", "机器人动作 + 0.1×打乱的运动标签"),
]
runs = []
summary = []
for cond, label in conditions:
    vals = []
    for seed in [17, 29]:
        p = R / f"seed{seed}" / cond
        r = json.loads((p / "report.json").read_text())
        curves = json.loads((p / "curves.json").read_text())
        runs.append(
            {
                "name": f"{cond} / seed{seed}",
                "condition": cond,
                "seed": seed,
                "curves": curves,
                "report": r,
            }
        )
        vals.append(r["test"]["joint_rmse_rad"])
    summary.append(
        {
            "condition": cond,
            "label": label,
            "seed17": vals[0],
            "seed29": vals[1],
            "mean": float(np.mean(vals)),
        }
    )
meta = json.loads((R / "seed17/experiment.json").read_text())
params = {
    "encoder": sum(
        [
            3 * 32 * 25 + 32,
            32 * 48 * 9 + 48,
            48 * 64 * 9 + 64,
            64 * 64 * 9 + 64,
            2 * (32 + 48 + 64 + 64),
            1024 * 192 + 192,
            2 * 192,
        ]
    ),
    "action": 590 * 512 + 512 + 512 * 512 + 512 + 512 * 224 + 224,
    "motion": 192 * 192 + 192 + 192 * 6 + 6,
    "image": 192 * 384 + 384 + 384 * 3072 + 3072,
}
result = {
    "models": models,
    "runs": runs,
    "summary": summary,
    "parameters": params,
    "total_parameters": sum(params.values()),
    "source": "v2/policy.py; outputs/v2/results_absolute/training/seed*/experiment.json + */report.json + */curves.json",
    "protocol": {
        "robot_episodes": 50,
        "split": [40, 5, 5],
        "split_seed": 429,
        "human_clips": 96,
        "human_windows": meta["human_frames"],
        "human_valid_hand_targets": meta["human_valid_hand_targets"],
        "review_only_clips": 35,
        "image_input": "128×128 RGB; robot has head/left/right views",
        "horizon": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "robot_batch": 64,
        "human_batch": 32,
        "stage1": 2000,
        "stage2": 8000,
        "robot_exposures": 640000,
        "human_exposures": 64000,
        "seeds": [17, 29],
        "checkpoint_selection": "固定第 2,000/10,000 步；没有按测试分数挑选检查点",
        "validation": "按 episode 保留 5 条 validation；当前脚本未用它早停或选模型",
        "fairness": "同随机种子相同初始化/机器人 batch 序列/步数，人体辅助计算量额外增加，未做严格等 FLOPs 对照",
        "image_schedule": "第 2,001 步起图像/运动辅助 loss 停用。此时曲线的 0 表示未优化，不是完美重建。",
    },
    "conclusion": "随机标签组也改善离线 RMSE，目前不能把收益归因于正确人类运动。正式 8×18 场景闭环作业 1054089 最新查询仍 Queued；没有完整 VLA 成功率，也没有画面转换数据训练 VLA 的结果。",
    "training_on_converted_video": False,
    "language_conditioned": False,
}
(ROOT / "web/v2data/experiment_details.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(params, sum(params.values()), summary)
lines = [
    "# 模型身份、实验设计与实际训练记录\n",
    result["conclusion"] + "\n",
    "## 模型身份\n",
]
for m in models:
    lines.append(
        f"- **{m['name']}**：{m['base']}。{m['trained']}。输出：{m['output']}。用途：{m['used']}。\n"
    )
lines += (
    ["## 训练配置\n"]
    + [f"- {k}: {v}\n" for k, v in result["protocol"].items()]
    + [
        "## 离线留出关节 RMSE（rad）\n",
        "|条件|seed17|seed29|均值|\n|---|---:|---:|---:|",
    ]
)
for x in summary:
    lines.append(
        f"|{x['condition']}|{x['seed17']:.5f}|{x['seed29']:.5f}|{x['mean']:.5f}|"
    )
lines += [
    "\n共约 225 万参数。三维标注、未来图像预测、视觉替换、机器人策略是四个不同模块。当前没有生成式视觉转换基座，也没有使用转换视频训练的 VLA。图像预览展示第 2,000 步的辅助头输出，来自训练素材，不是人类留出集准确率。"
]
(ROOT / "docs/模型身份与训练明细.md").write_text("\n".join(lines))
if __name__ == "__main__":
    pass
