# 机器人视频转人手视频实验

源项目 `human2robot/v4/` 是动作/未来视频预测训练代码；本实验另建视频编辑流程，不加载 v4 策略 LoRA。

## 流程

1. 从 Bridge 机器人视频截取短片，保留原始时间和帧率。
2. SAM2.1 Hiera tiny 根据人工首帧点和框分割机器人，传播到整段，再膨胀 7 个源像素。
3. 用机器人首帧生成一张人手参考图。确切图像编辑提示词见 `reference_image_prompts.json`。
4. 按 VACE 官方 inpainting 预处理把掩码内像素置为灰色（128），再根据掩码外画面、参考图和任务提示词生成视频。
5. 同时保留模型直接输出和掩码内编辑、掩码外保留源画面的合成版本。

## 执行

脚本的 `ROOT` 指向服务器 `/user/panmiao/workspace/robot2human-vace-20260924`。该目录包含 inputs、human_references、models、deps、Wan2.1-main 和 sam2-code。移植时修改 ROOT 和启动脚本路径。

```bash
bash run_generation.sh
# 补齐本次罐子场景的上臂掩码并重跑：
bash run_can_refined.sh
# 大模型复跑（权重准备完成后，在 GPU 任务内执行）：
python generate.py --model 14B --steps 40 --size 512 --seed 2026
# 推理完成后在带 ffmpeg 的服务器上导出原始帧率配对版本：
python package_pairs.py --model 14B
```

正式上游代码：
- https://github.com/Wan-Video/Wan2.1
- https://github.com/ali-vilab/VACE
- https://github.com/facebookresearch/sam2

视频输出：`results/<模型>/<样例>/seed<种子>/`，包含 human.mp4、raw_generated.mp4、source_aligned.mp4、mask_overlay.mp4、contact_sheet.jpg 和 result.json。

## 配对含义

输出以 16 fps 编码，原片 5 fps 使用最近邻时间采样，未创造新的机器人观测或标签。每帧映射在 result.json 的 frame_indices、source_timestamps 中。

Bridge 标签在当前 v4 管线里是实际发生的 TCP 状态/增量，不能当成原始控制命令。合成人手像素也不能证明它与机器人动作物理等价。所有样例默认 paired_alignment_verified=false、training_eligible=false；需要对物体轨迹、接触时刻、遮挡与下游性能继续验证。

AI 人手参考图仅提供外观条件，不是逐帧真值。背景合成后的稳定性不能用来证明模型本身保持了全部背景。

配对导出保留每个源观察帧；`frame_alignment.json` 明确记录生成帧选择和时间误差。`paired_manifest.json` 保留原始 v4 uid、state、target，并指出四步未来时域是否完整包含在短片内。所有数据仍待验证，不能把生成样例直接当成可靠动作监督。

本次罐子场景发现后段上臂漏分割，使用最后一帧人工点/框反向传播，并与初始掩码取并集（`refine_can_mask.py`）。初始结果单独保留，网页展示修正后的版本。

生成前检查每段 `source_frames.npz` 与掩码帧数一致。`input_provenance.json` 记录源视频、参考图、掩码和原机器人标签的 SHA256；它只验证输入文件对应关系，不等于验证人手动作。

配对清单是独立的 `robot2human.paired-example.v1` 格式：原 v4 行完整保存在 `robot_sample` 中，人手视频另有相对路径、局部时间戳和 pair_id。它不能直接当成 v4 训练清单加载，以免复用机器人缓存或原视频绝对时间戳。接入前需显式转换并完成配对质量筛选。

## Expanded hand-room experiment

Added scripts: `segment_expanded.py`, `generate_expanded.py`, `run_expanded.sh`, `package_expanded.py`.

- Prepare the six episode IDs: `bridge_bridge_042977`, `bridge_bridge_007089`, `bridge_bridge_045160`, `bridge_bridge_041207`, `bridge_bridge_015993`, `bridge_bridge_040489`.
- `segment_expanded.py` retains existing masks and adds SAM2 prompts for the three new scenes.
- `generate_expanded.py --variant handroom --seeds 2026 2027` adds 10 native pixels to the prior edit mask and unions masks over the neighboring native frames. It writes a separate `results/1.3B_handroom` tree.
- Existing three reference images remain unchanged; three new reference images and the exact built-in image generation prompt are retained in the deliverables.
- `package_expanded.py --model 1.3B_handroom` exports native-fps media and uses the actual effective mask digest plus variant in pair IDs.
- Human alignment and training eligibility remain false. Manual gallery selection is a visual preference, not a geometric or downstream-training validation.

For a fresh reproduction of the gray-can condition, run the included `refine_can_mask.py` after SAM2 segmentation and before expanded generation. This experiment reused that already-refined can mask; the original results and masks were not overwritten by the expanded variant.
