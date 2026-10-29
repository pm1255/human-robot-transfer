"""Publish the continuous neutral-material robot video after validation."""
import argparse
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    args = parser.parse_args()
    root, site = args.root, args.site
    project = Path(__file__).resolve().parents[1]
    web = project / 'local_annotation_lab'
    out = site / 'robot'
    out.mkdir(exist_ok=True)
    validation = read(root / 'full/composite_validation.json')
    summary = validation['summary']
    independent = read(root / 'trajectory/independent_validation.json')
    trajectory = read(root / 'trajectory/robot_trajectory.json')
    comparison = read(root / 'trajectory/before_after.json')
    visual = read(root / 'visual_review.json')
    assert summary['total'] == 4834 and summary['complete_decode']
    assert summary['unchanged_outside_edit'] and summary['frames_without_robot'] == 0
    assert all(v == 0 for v in summary['frames_without_each_arm'].values())
    assert independent['integrity_pass'] and visual['release_review_completed']
    assert len(trajectory['frames']) == 4834
    copies = {
        'full/robot_source_composite.mp4': 'robot_source_repaired.mp4',
        'full/composite_validation.json': 'repair_composite_validation.json',
        'trajectory/robot_trajectory.json': 'robot_continuous_trajectory.json',
        'trajectory/independent_validation.json': 'repair_independent_validation.json',
        'trajectory/before_after.json': 'repair_before_after.json',
        'masks_final/human_mask_validation.json': 'repair_human_mask_validation.json',
        'visual_review.json': 'repair_visual_review.json',
    }
    for source, destination in copies.items():
        shutil.copy2(root / source, out / destination)
    fields = ('error_m', 'rotation_metric_available', 'link_point_table_clearance_m',
              'source_hand_observed', 'display_inferred', 'status')
    review = {'frames': [dict(frame_index=row['frame_index'],
        hands={side: {key: hand[key] for key in fields} for side, hand in row['hands'].items()})
        for row in trajectory['frames']]}
    write(out / 'robot_continuous_review.json', review)
    masks = read(root / 'masks_final/human_mask_validation.json')['frames']
    write(out / 'repair_composite_review.json', {'frames': [dict(
        frame_index=row['frame_index'], mask_review=masks[row['frame_index']]['review_required'],
        human_pixels=row['human_mask_pixels'], robot_pixels=row['robot_pixels'],
        estimated_cloth_pixels=row['cloth_completion_pixels']) for row in validation['frames']]})
    selected = [0, 208, 209, 600, 781, 790, 791, 883, 884, 1200, 1744, 1745,
                2400, 2753, 3518, 3600, 3960, 4800, 4833]
    figures = []
    for frame in selected:
        candidates = list((root / 'qa').glob(f'audit_{frame:06d}.jpg'))
        candidates += list((root / 'full').glob(f'part_*/audit_{frame:06d}.jpg'))
        if not candidates:
            continue
        for kind in ('audit', 'composite'):
            source = candidates[0].with_name(f'{kind}_{frame:06d}.jpg')
            if source.exists():
                shutil.copy2(source, out / f'repair_{kind}_{frame:06d}.jpg')
        figures.append(f'<figure><figcaption>第 {frame} 帧 · {frame / 30:.2f} 秒</figcaption>'
                       f'<img src="repair_audit_{frame:06d}.jpg" loading="lazy" '
                       f'alt="第 {frame} 帧的原片、人体掩码、背景补全、合成结果"></figure>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>机械臂修复 · 画面对照</title><style>
body{font:16px/1.65 system-ui;background:#f4f4f0;color:#27343d;margin:0;padding:32px}
main{max-width:1600px;margin:auto}a{color:#1a6a6b}h1{font-size:28px}
figure{margin:28px 0;padding:16px;background:white;border:1px solid #d8dfdd;border-radius:10px}
img{display:block;width:100%;height:auto}.steps{display:grid;grid-template-columns:repeat(4,1fr);text-align:center;background:#e6ede9;padding:12px}
</style><main><a href="../#layer-robot">← 返回统一标注网页</a>
<h1>统一灰色材质 · 连续运动 · 加密人体分割</h1>
<p>全片 4,834 帧均能看到双臂；原手部漏检处使用补间估计。画面抽查范围见修复报告。
背景与衣物遮挡区仍为估计补全，尚未验证物理抓持或布料受力。</p>
<p><a href="../robot_repair_report_summary.md">修复报告</a> · <a href="robot_source_repaired.mp4">新版视频</a></p>
<div class="steps"><b>原片</b><b>人体移除掩码</b><b>背景补全层</b><b>新版合成</b></div>'''
    (out / 'repair_audit.html').write_text(html + ''.join(figures) + '</main></html>')
    masks_summary = summary['human_masks']
    report = f'''# 机械臂视频修复汇总

已统一为灰色材质，并修复机械臂整段消失问题，明显的关节跳变已减小；加密人体分割后，已复查片段中的人手重新出现问题已改善。全片 {summary['total']} 帧都有双臂可见。仍有中等程度的背景补全接缝、衣物纹理拉伸和接触不准确，影响合成真实感；当前适合视觉审阅，不可直接用于机械臂执行或动作训练。人体移除尚未完成全片逐帧人工真值验收。

## 本轮修改与证据

- 两臂使用相同的银灰色外壳、深灰关节和浅灰夹爪，不再用颜色区分左右。
- 全片 640 × 480、30 FPS、161.13 秒、4,834 帧，完整解码通过。最终渲染中，左臂缺失 {summary['frames_without_each_arm']['Left']} 帧、右臂缺失 {summary['frames_without_each_arm']['Right']} 帧；此处指每臂至少有可见像素，不代表完整机械臂始终在视野内。
- 按最短角差计算，旧版最大相邻帧关节跳变约 {max(v['old_wrapped_max_step_deg'] for v in comparison.values()):.2f}°；新版约 {max(v['new_max_step_deg'] for v in comparison.values()):.2f}°。最大关节速度约 1.80 rad/s，最大加速度约 9.99 rad/s²。这是显示轨迹连续性指标，不是控制器验收。
- 固定估计焦距与尺度，稳定桌面坐标；连续求解关节位置，限制每次变化并平滑到 30 FPS。避免漏检直接隐藏机械臂、换解跳变和波动深度使整臂突然遮掉。
- 人体分割由每 30 帧加密为每 10 帧，实际模型锚帧 {masks_summary['model_inference_frames']} 帧，其余 {masks_summary['flow_estimated_frames']} 帧使用双向光流；合并逐帧手网格轮廓与短时传播。衣物补全与背景供体均排除人体像素。
- 编辑区外在编码前保持原视频解码像素不变；MP4 压缩后不保证逐像素相同。
- 独立正向运动学复算一致，连杆采样点最低高于估计桌面 {independent['minimum_link_point_clearance_m'] * 1000:.2f} mm；未检查完整网格碰撞。

## 缺失观测与尚未验证

- 手部观测缺失：左臂 {comparison['Left']['inferred_frames']} 帧，最长连续 {comparison['Left']['longest_inferred_seconds']:.2f} 秒；右臂 {comparison['Right']['inferred_frames']} 帧，最长连续 {comparison['Right']['longest_inferred_seconds']:.2f} 秒。网页逐帧显示补间估计，数据保存 source_hand_observed 与 display_inferred。
- 夹爪朝向使用向下的简化目标，不恢复真实手掌旋转；未计算旋转准确率，网页不再把占位 0° 显示为精度。
- 相机与尺寸未经标定；拟合误差仅相对于估计目标，不是真实三维准确率。遮挡使用稳定平面近似，不能保证立体衣物遮挡正确。
- 衣物仍来自人类视频，隐藏内容使用估计补全。未执行布料物理、抓持、受力或硬件验证；physics_executed=false、training_eligible=false。不是 EgoMatrix 完整复现。
- 人体分割一致性提示 {masks_summary['review_frames']} 帧需复核。已完成的画面抽查及具体限制见下方记录；模型掩码覆盖率不等同于真实人体去除准确率。

## 输出与完整明细

- [统一网页](http://127.0.0.1:53419/#layer-robot)
- [新版视频](http://127.0.0.1:53419/robot/robot_source_repaired.mp4)
- [画面对照](http://127.0.0.1:53419/robot/repair_audit.html)
- [合成逐帧检查](full/composite_validation.json)
- [独立轨迹检查](trajectory/independent_validation.json)
- [前后对比与补间区间](trajectory/before_after.json)
- [人体分割明细](masks_final/human_mask_validation.json)
- [画面抽查记录](visual_review.json)
- [TensorBoard](http://127.0.0.1:16067/)：robot_repair_composite_total 与各段渲染进度、human_removal_flow_stride10、robot_continuous_trajectory。
'''
    (root / 'report_summary.md').write_text(report)
    site_report = report
    for source, destination in copies.items():
        site_report = site_report.replace(f']({source})', f'](robot/{destination})')
    (site / 'robot_repair_report_summary.md').write_text(site_report)
    (project / 'docs/ROBOT_CONTINUOUS_REPAIR.md').write_text(site_report)
    manifest_path = web / 'assets/multi_manifest.json'
    manifest = read(manifest_path)
    clip = manifest['clips'][0]
    old_layers = [layer.copy() for layer in clip['layers'] if layer['id'] != 'robot']
    clip['robot'] = {'trajectory': 'robot/robot_continuous_review.json',
                     'composite_review': 'robot/repair_composite_review.json'}
    layer = next(layer for layer in clip['layers'] if layer['id'] == 'robot')
    layer.update(video='robot/robot_source_repaired.mp4', poster='robot/repair_composite_000600.jpg',
        badge='连续运动 · 视觉试验', description='保留原场景、衣物外观与形变；统一灰色机械臂，平滑关节运动并补间缺失轨迹，使用加密人体分割移除原人手与前臂。',
        tags=['统一灰色材质', '全片双臂连续显示', '缺失观测标记补间'],
        assessment='4,834 帧均有双臂可见；最大相邻帧关节变化 3.43°。背景与衣物补全仍有接缝、纹理拉伸，接触位置未验收。补间最长 1.93 秒；尚未验证物理抓持或布料受力，不可直接用于控制或动作训练。',
        resources=[{'label': label, 'url': url} for label, url in [
            ('修复报告', 'robot_repair_report_summary.md'), ('新版画面对照', 'robot/repair_audit.html'),
            ('完整质检', 'robot/repair_composite_validation.json'),
            ('连续关节轨迹', 'robot/robot_continuous_trajectory.json'),
            ('修复前视频', 'robot/robot_source_composite.mp4')]])
    assert old_layers == [layer for layer in clip['layers'] if layer['id'] != 'robot']
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    index = web / 'index.html'
    import re
    html = re.sub(r'app\.js\?v=[^"\s]+', 'app.js?v=robot-repair-20261020', index.read_text())
    if 'href="robot_repair_report_summary.md"' not in html:
        html = html.replace('<a href="robot_composite_report_summary.md">',
            '<a href="robot_repair_report_summary.md">机械臂连续性修复 ↗</a><a href="robot_composite_report_summary.md">')
    index.write_text(html)
    for name in ('app.js', 'index.html', 'assets/multi_manifest.json'):
        shutil.copy2(web / name, site / name)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
