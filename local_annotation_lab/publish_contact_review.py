"""Publish background improvement; rejected pose trial remains a separate review."""
import argparse,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args();root=a.root;site=a.site;out=site/'robot';project=Path(__file__).resolve().parents[1];web=project/'local_annotation_lab'
    v=json.loads((root/'background_full/composite_validation.json').read_text());s=v['summary'];ind=json.loads((root/'trajectory_v4/independent_validation.json').read_text());review=json.loads((root/'visual_review.json').read_text())
    assert s['complete_decode'] and s['total']==4834 and s['frames_without_robot']==0 and all(x==0 for x in s['frames_without_each_arm'].values())
    assert review['background_review_complete'] and not ind['release_pass']
    copies={'background_full/robot_source_composite.mp4':'robot_source_background_v2.mp4','background_full/composite_validation.json':'contact_background_validation.json','trajectory_v4/independent_validation.json':'contact_trial_independent.json','trajectory_v4/alignment_validation.json':'contact_trial_validation.json','trajectory_v4/robot_trajectory.json':'contact_trial_trajectory.json','alignment_motion/robot_source_composite.mp4':'contact_alignment_trial.mp4','background_comparison.jpg':'background_comparison.jpg','alignment_trial_contact_sheet.jpg':'contact_trial_sheet.jpg','report_summary.md':'contact_report_summary.md','visual_review.json':'contact_visual_review.json'}
    for src,dst in copies.items():shutil.copy2(root/src,out/dst)
    shutil.copy2(project/'docs/ROBOT_CONTACT_ALIGNMENT.md',site/'ROBOT_CONTACT_ALIGNMENT.md')
    report=(root/'report_summary.md').read_text()
    for src,dst in copies.items():report=report.replace(f']({src})',f']({dst})')
    for src,dst in [('base_search/base_search.json','contact_base_search.json'),('research_evidence.json','contact_research_evidence.json')]:
        shutil.copy2(root/src,out/dst);report=report.replace(f']({src})',f']({dst})')
    (out/'contact_report_summary.md').write_text(report)
    poster=next((root/'background_full').glob('part_*/composite_000600.jpg'));shutil.copy2(poster,out/'contact_background_poster.jpg')
    records={'frames':[dict(frame_index=r['frame_index'],mask_review=False,human_pixels=r['human_mask_pixels'],robot_pixels=r['robot_pixels'],estimated_cloth_pixels=r['cloth_completion_pixels']) for r in v['frames']]}
    masks=json.loads((site/'robot/repair_human_mask_validation.json').read_text())['frames']
    for r in records['frames']:r['mask_review']=masks[r['frame_index']]['review_required']
    (out/'contact_background_review.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
    (out/'contact_review.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>手形残影与末端对齐复核</title><style>body{font:16px/1.7 system-ui;background:#f3f5f2;color:#25383a;padding:28px}main{max-width:1280px;margin:auto}h1{font-size:28px}h2{font-size:21px;margin-top:32px}img{width:100%;height:auto}video{width:640px;max-width:100%}a{color:#176c70}.note{background:#fff1d5;padding:16px;border-left:4px solid #b17d32}</style><main><a href="../#layer-robot">← 统一标注工作台</a><h1>手形残影与末端对齐复核</h1><p>主视频已采用背景亮度匹配、多供体融合和边界修复，减少人手形状的硬边。当前机器人为 ARX5 模型资产，使用两指平行夹爪。</p><p><a href="contact_report_summary.md">本轮结论与明细</a> · <a href="../ROBOT_CONTACT_ALIGNMENT.md">坐标关系与 EgoMatrix 对照</a></p><h2>背景修复前后</h2><p>左侧为上一版，右侧只修背景，机器人姿态相同。第 781 帧可见手掌形状的亮度边界减弱；第 3600 帧的拼接色块也有所减轻。背景不是实际观测真值。</p><img src="background_comparison.jpg" alt="两组背景修复前后对照"><h2>完整姿态对齐试片（未通过发布门槛）</h2><p class="note">下面 10 秒对应原视频 17–27 秒，仅用于诊断。末端朝向已有变化，但部分整臂姿态不合理；全片朝向误差的 95 分位仍约 68° / 71°，没有替换主视频。误差参照估计目标，不是真值精度。</p><video controls muted playsinline preload="metadata" src="contact_alignment_trial.mp4"></video><p><a href="contact_trial_independent.json">独立复算与拒绝原因</a></p><img src="contact_trial_sheet.jpg" alt="完整对齐试验的跨阶段抽查"><p>当前没有通过衣物材料点跟踪、双臂完整网格碰撞、抓持与布料物理验证。EgoMatrix 使用接触优化、物理执行和机器人/物体共同渲染；本轮不声称完成该流程。</p></main></html>''')
    path=web/'assets/multi_manifest.json';m=json.loads(path.read_text());clip=m['clips'][0];clip['robot']['composite_review']='robot/contact_background_review.json';layer=next(x for x in clip['layers'] if x['id']=='robot')
    layer.update(video='robot/robot_source_background_v2.mp4',poster='robot/contact_background_poster.jpg',badge='残影改善 · 对齐待完善',description='双 ARX5 模型与两指平行夹爪，保留原场景和衣物形变。已改进人体移除区的亮度与边界融合，减少手形残影；主视频保留已验证连续的轨迹。',tags=['统一灰色 ARX5', '背景轮廓修复', '完整末端对齐待通过'],assessment='4,834 帧双臂均可见。背景仍有模糊与补全误差；完整腕部/夹爪对齐试验因误差过大单独提供，未替换主视频。不能把外观合成视为真实接触或可执行动作。',resources=[{'label':label,'url':url} for label,url in [('残影与对齐对照','robot/contact_review.html'),('本轮修复报告','robot/contact_report_summary.md'),('EgoMatrix 与坐标关系','ROBOT_CONTACT_ALIGNMENT.md'),('主视频质检','robot/contact_background_validation.json'),('上一版视频','robot/robot_source_repaired.mp4')]])
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');shutil.copy2(path,site/'assets/multi_manifest.json')
    print(json.dumps(s,ensure_ascii=False))

if __name__=='__main__':main()
