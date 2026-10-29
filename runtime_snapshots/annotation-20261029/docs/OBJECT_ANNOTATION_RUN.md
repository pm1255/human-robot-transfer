# 当前折衣视频的可重复运行记录

此流程只对当前 EgoVerse 折衣视频验证过。结果保留形变与缺失；FoundationPose 输出为 `cloth_global_proxy`，不得作为柔性衣物真值。

## 输入和依赖

- 服务器目录：`/user/panmiao/human-robot-transfer`
- 输入：`inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4`
- SHA256：`24e0f2c0a3b99d553cc425de958f79141e4b4712e13cdd0a925a8e6750ceefea`
- SAM3.1：`/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt`
- MoGe-3：`models/moge-3-vitl/model.pt`，SHA256 `9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`
- FlexGEMM：MoGe 官方依赖锁定的 `b2fadb29d41846c7981ade6801ffc689fae119cf`
- FoundationPose refiner：`weights/2023-10-28-18-33-37/model_best.pth`，SHA256 `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60`
- FoundationPose scorer：`weights/2024-01-11-20-02-45/model_best.pth`，SHA256 `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26`

源码位于 `vendor/MoGe-main`、`vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf`、`vendor/FoundationPose-main`。SAM/MoGe 使用 `.venv-sam3`，FoundationPose 使用 `.venv-foundation`。GPU 任务使用 H100；CPU devspace 用于提交、检查和渲染。

## 运行入口

1. `tools/run_object_annotation_job.sh`：最终逐帧 SAM3.1 提示与分割 → 分开渲染 mask / bbox → 文件完整性验收 → 更新原六类网页。
2. `tools/run_cloth_depth_job.sh`：两帧预检查 → 前 180 帧深度和位姿试验 → 全片深度。支持原参数和同输入的深度断点续算；改变参数会拒绝复用。
3. `tools/run_cloth_pose_job.sh`：完整精度前 180 帧复核 → 全片代理位姿 → 双栏视频。运行中可等待独立深度任务生成后续帧。
4. `python -m human2robot.object_annotation.validate_geometry_outputs --output outputs/object_annotation_20261020`：逐帧审核完整深度、时间索引和有效变换，输出 `geometry_validation.json`。

第 30 帧的 `reference_review.json` 绑定源视频与掩码 SHA256，只表示看过该二维掩码并允许代理试验，不代表三维几何或尺度经过确认。参考网格只取该帧可见深度表面，顶点颜色来自原图，不补造隐藏面。

## 已修复的运行问题

- 原系统 Triton 3.3 缺少 FlexGEMM 使用的 dtype 字节宽度和按位 OR 归约接口，并存在指针注解、联合类型、链式布尔和 reshape 语法差异。`tools/prepare_flexgemm_compat.py` 保存源码适配，`flex_compat.py` 补充等价接口，不替换模型计算算法。
- `flex_verify.py` 比较 GPU 哈希查询、去重、3D 邻居索引与独立 Python 结果，覆盖 int8/int16/int32，共 9 项通过。此检查不验证 MoGe 模型精度。
- NumPy 标量使 FoundationPose 裁剪偏移提升至 FP64，入口将网格直径转为 Python float。
- 仅关闭 AMP 仍不足以消除旋转数值误差。最终同时设置 `NVIDIA_TF32_OVERRIDE=0`，关闭矩阵与 cuDNN TF32，保持 FP32。没有放宽旋转正交阈值或将异常矩阵强行投影为有效旋转。

## 筛查与缺失

轮廓 IoU 至少 0.45、相对深度中位残差至多 0.08；连续通过帧旋转跳变至多 30°，平移变化至多前帧深度的 0.1。拒绝后清除跟踪状态，每 30 帧尝试重新注册。等待、输入不足、初始化之前的帧不输出正式变换；候选及拒绝原因单独保留。阈值未用独立真值校准。

深度为单目模型估计，K 为逐帧预测；尚未与手部尺度、实测尺寸或相机标定对齐。相机运动和物体运动未分离，只提供相机坐标中的代理。没有稠密布料对应、接触重建、机器人重定向或 PPO。

## 网页和监控

- 六类标注：http://127.0.0.1:53419/
- 独立几何复核：http://127.0.0.1:53419/geometry.html
- TensorBoard：http://127.0.0.1:16067/

TensorBoard 显示分割、深度、位姿、计算适配验证和输出审核的实际进度；预检查与试验单独记录。模型误差验证与文件完整性审核分别说明。
