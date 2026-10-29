"""Publish a reviewed temporal-removal video without changing other layers."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
    root,site=a.root,a.site;out=site/'robot';web=Path(__file__).resolve().parent
    validation=json.loads((root/'full/composite_validation.json').read_text());summary=validation['summary']
    audit=json.loads((root/'encoded_temporal_audit.json').read_text())['summary']
    visual=json.loads((root/'visual_review.json').read_text())
    assert summary['complete_decode'] and summary['total']==4834 and summary['unchanged_outside_edit']
    assert all(x==0 for x in summary['frames_without_each_arm'].values())
    assert summary['temporal_removal']['observed_cloth_unchanged']
    assert visual['release_visual_review_complete'] and audit['reduction_fraction']>.25
    copies={'full/robot_source_composite.mp4':'robot_source_temporal_v3.mp4','full/composite_validation.json':'temporal_composite_validation.json','encoded_temporal_audit.json':'temporal_encoded_audit.json','report_summary.md':'temporal_report_summary.md','visual_review.json':'temporal_visual_review.json','temporal_comparison.mp4':'temporal_comparison.mp4','temporal_comparison_cut.mp4':'temporal_comparison_cut.mp4','review_sheet.jpg':'temporal_review_sheet.jpg'}
    for src,dst in copies.items():shutil.copy2(root/src,out/dst)
    shutil.copy2(root/'full/part_003600/composite_003600.jpg',out/'temporal_poster.jpg')
    report=(root/'report_summary.md').read_text()
    for src,dst in copies.items():report=report.replace(f']({src})',f']({dst})')
    (out/'temporal_report_summary.md').write_text(report)
    masks=json.loads((out/'repair_human_mask_validation.json').read_text())['frames']
    rows=[dict(frame_index=r['frame_index'],mask_review=masks[r['frame_index']].get('review_required',False),human_pixels=r['human_mask_pixels'],robot_pixels=r['robot_pixels'],estimated_cloth_pixels=r['cloth_completion_pixels'],temporal_reset=r['temporal_removal']['history_reset']) for r in validation['frames']]
    (out/'temporal_frame_review.json').write_text(json.dumps(dict(frames=rows),ensure_ascii=False,separators=(',',':'))+'\n')
    reduction=100*audit['reduction_fraction']
    (out/'temporal_review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原手区域抖动 · 时序修复对照</title><style>body{{font:16px/1.7 system-ui;background:#f3f5f2;color:#25383a;padding:24px}}main{{max-width:1280px;margin:auto}}video,img{{width:100%;height:auto}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:28px}}a{{color:#176c70}}.note{{background:#fff1d5;padding:16px;border-left:4px solid #b17d32}}</style><main><a href="../#layer-robot">← 原统一标注工作台</a><h1>原手区域抖动 · 时序修复对照</h1><p>已把时间连续性加入背景与局部衣物补全。主网页已更新完整 4,834 帧视频。</p><p class="note">原手覆盖桌面的共同可比较区域，抵消镜头运动后的相邻帧颜色差降低约 {reduction:.0f}%。此数字只衡量这些区域的时间稳定性，不是整幅画面的抖动消除率，也不代表隐藏纹理正确。衣物遮挡边缘、补全条纹与抓取对齐仍需复核。</p><p><a href="temporal_report_summary.md">结论与方法</a> · <a href="temporal_encoded_audit.json">编码视频独立检查</a> · <a href="temporal_visual_review.json">画面抽查范围</a></p><h2>115–125 秒：持续折叠</h2><p>左：上一版；右：时序修复。两侧帧号相同，请观察原手附近桌面的亮暗和纹理跳变。</p><video controls muted playsinline preload="metadata" src="temporal_comparison.mp4"></video><h2>17–27 秒：含演示切换</h2><p>左：上一版；右：时序修复。镜头突变、背景跟踪不可靠时清空历史，避免把上一场景带入。</p><video controls muted playsinline preload="metadata" src="temporal_comparison_cut.mp4"></video><h2>全片画面抽查</h2><img src="temporal_review_sheet.jpg" alt="完整视频跨阶段抽帧"><p><a href="robot_source_temporal_v3.mp4">新版完整视频</a> · <a href="robot_source_background_v2.mp4">上一版完整视频</a> · <a href="contact_review.html">末端对齐的未完成问题</a></p></main></html>''')
    path=web/'assets/multi_manifest.json';m=json.loads(path.read_text());clip=m['clips'][0];layer=next(x for x in clip['layers'] if x['id']=='robot')
    clip['robot']['composite_review']='robot/temporal_frame_review.json'
    layer.update(video='robot/robot_source_temporal_v3.mp4',poster='robot/temporal_poster.jpg',badge='时序修复 · 仍需复核',description='双 ARX5 模型与两指夹爪，保留原场景与衣物形变。对原手区域加入随镜头运动的背景历史融合、局部衣物光流补全与柔和边界，减轻补丁闪烁。',tags=['统一灰色 ARX5','原手区域时序修复','抓取对齐待完善'],assessment=f'4,834 帧双臂均可见。可比较的原手覆盖桌面区域，运动补偿后的帧间颜色差降低约 {reduction:.0f}%；不是整幅画面抖动消除率。被遮挡衣物仍为估计，边缘与纹理仍需复核，未验证真实抓持或物理执行。',resources=[dict(label=label,url=url) for label,url in [('抖动修复视频对照','robot/temporal_review.html'),('本轮报告','robot/temporal_report_summary.md'),('独立时序检查','robot/temporal_encoded_audit.json'),('末端对齐问题','robot/contact_review.html'),('上一版视频','robot/robot_source_background_v2.mp4')]])
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');shutil.copy2(path,site/'assets/multi_manifest.json')
    print(json.dumps(dict(frames=summary['total'],reduction_fraction=audit['reduction_fraction']),ensure_ascii=False))


if __name__=='__main__':main()
