"""Publish validated source-view preview to the existing unified annotation page."""
import argparse,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[1];web=project/'local_annotation_lab';out=a.site/'robot';out.mkdir(exist_ok=True)
    validation=json.loads((a.root/'composite_validation.json').read_text());s=validation['summary'];assert s['total']==4834 and s['complete_decode'] and s['unchanged_outside_edit']
    masks=json.loads((a.root.parent/'human_mask_validation.json').read_text());maskrows=masks['frames']
    for name in ['robot_source_composite.mp4','composite_validation.json']:
        shutil.copy2(a.root/name,out/name)
    selected=[0,120,600,720,1200,2400,3600,3960,4800,4833]
    for i in selected:
        matches=list(a.root.glob(f'part_*/audit_{i:06d}.jpg'))
        if matches:
            for kind in ('audit','composite','clean'):
                source=matches[0].with_name(f'{kind}_{i:06d}.jpg')
                if source.exists():shutil.copy2(source,out/source.name)
    compact={'frames':[dict(frame_index=r['frame_index'],mask_review=maskrows[r['frame_index']].get('review_required',False),human_pixels=r['human_mask_pixels'],robot_pixels=r['robot_pixels'],estimated_cloth_pixels=r['cloth_completion_pixels']) for r in validation['frames']]}
    (out/'composite_review.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
    cards=''.join(f'<figure><figcaption>第 {i} 帧 · {i/30:.2f} 秒</figcaption><img src="audit_{i:06d}.jpg" alt="第 {i} 帧：原片、人体掩码、背景补全与最终合成" loading="lazy"></figure>' for i in selected if (out/f'audit_{i:06d}.jpg').exists())
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原场景合成 · 四步对照</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f4f0;color:#27343d;margin:0;padding:32px}main{max-width:1600px;margin:auto}a{color:#1a6a6b}h1{font-size:28px}p{max-width:900px}figure{margin:28px 0;padding:16px;background:white;border:1px solid #d8dfdd;border-radius:10px}figcaption{margin-bottom:10px}img{display:block;width:100%;height:auto}.steps{display:grid;grid-template-columns:repeat(4,1fr);text-align:center;gap:12px;padding:14px;background:#e6ede9}.note{padding:16px;border-left:4px solid #b58443;background:#fff8e9}</style><main><a href="../#layer-robot">← 返回统一标注网页</a><h1>原场景合成 · 四步对照</h1><p>背景保持原视频的房间、桌面和视角。中间背景层清除人体与衣物，最终放回衣物可见层和原视角机器人。</p><p class="note">试验版：补全接缝、纹理拉伸、阴影与局部人体残留仍需复核。衣物来自人类演示，尚未由机器人物理驱动；此视频不证明抓持或折衣成功。</p><div class="steps"><strong>① 原视频</strong><strong>② 手与前臂掩码</strong><strong>③ 背景补全层</strong><strong>④ 衣物 + 机械臂</strong></div>'''+cards+'''<p><a href="robot_source_composite.mp4">完整视频</a> · <a href="composite_validation.json">完整质检明细</a> · <a href="../ROBOT_SOURCE_COMPOSITE.md">方法与限制</a></p></main></html>'''
    (out/'composite_audit.html').write_text(html)
    shutil.copy2(project/'docs/ROBOT_SOURCE_COMPOSITE.md',a.site/'ROBOT_SOURCE_COMPOSITE.md')
    (web/'index.html').write_text((web/'index.html').read_text().replace('app.js?v=robot-20261020','app.js?v=source-composite-20261020'))
    for name in ('app.js','index.html'):shutil.copy2(web/name,a.site/name)
    shutil.copy2(web/'assets/multi_manifest.json',a.site/'assets/multi_manifest.json')
    report=f'''# 原背景合成修复汇总

原网页第九路已替换为原场景合成试验视频：保留房间、桌面与相机视角，移除手和前臂，再放回衣物可见层与机械臂。旧三维预览保留对照链接。画面补全存在接缝、拉伸、阴影和局部人体残留，影响视觉质量；适合审阅合成流程，不能作为机器人成功折衣或训练数据。

## 已检查

- 视频共 {s['total']} 帧，640 × 480、30 FPS，161.13 秒；全片解码通过。
- {s['total']} 帧在视频编码前，编辑区外像素与原解码帧完全一致。MP4 压缩后不保证逐像素相同。
- {s['frames_without_robot']} 帧没有可渲染的机械臂；保持缺失状态，没有补造轨迹。
- 人体掩码：166 帧模型推理，4668 帧双向光流传播；{s['human_masks']['review_frames']} 帧因传播一致性不足需复核。
- {sum(r['area']==0 for r in maskrows)} 帧人体掩码为空，包含演示切换和无手画面；尚未逐一确认是否有漏检。
- 抽查不同动作阶段及模型锚帧之间的传播帧，发现补全接缝等问题已在网页提示。未完成人工逐帧真值验收。

## 为什么上一版背景变了

上一版仅绘制固定视角三维场景，用于检查关节重定向；缺少源背景修复与相机视角合成步骤。这不符合用户要求，本次已增加对应流程。

衣物和人体在中间背景层一起移除；最终可见衣物使用原视频像素与局部遮挡估计补全。机械臂按原相机投影，使用衣物可见深度处理遮挡。隐藏衣物和背景没有真值保证。

## 与 EgoMatrix 的边界

EgoMatrix 从同一物理状态联合渲染机器人和物体，叠回补全背景并转移阴影。本次未运行 SVOR、布料物理、接触优化或 PPO，衣物仍由人类视频提供运动，不能宣称已经机器人物理驱动。全部轨迹保留 physics_executed=false、training_eligible=false。

## 输出与明细

- [统一网页](http://127.0.0.1:53419/#layer-robot)
- [四步对照](http://127.0.0.1:53419/robot/composite_audit.html)
- [原背景合成视频](http://127.0.0.1:53419/robot/robot_source_composite.mp4)
- [质检原始报告](full/composite_validation.json)
- [人体分割完整明细](human_mask_validation.json)
- [方法与参数](../../code/human-robot-transfer/docs/ROBOT_SOURCE_COMPOSITE.md)
- [TensorBoard](http://127.0.0.1:16067/)，source_composite_total；人体锚帧、传播与各片段分别记录。GPU 排队任务已撤销，实际通过 CPU 完成。
'''
    (a.root.parent/'report_summary.md').write_text(report)
    (a.site/'robot_composite_report_summary.md').write_text(report.replace('(full/composite_validation.json)','(robot/composite_validation.json)').replace('(human_mask_validation.json)','(robot/human_mask_validation.json)').replace('(../../code/human-robot-transfer/docs/ROBOT_SOURCE_COMPOSITE.md)','(ROBOT_SOURCE_COMPOSITE.md)'))
    shutil.copy2(a.root.parent/'human_mask_validation.json',out/'human_mask_validation.json')
    print(json.dumps(s,ensure_ascii=False))

if __name__=='__main__':main()
