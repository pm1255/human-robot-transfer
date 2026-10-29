"""Publish measured temporal evidence separately from annotation acceptance."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import markdown

BASE = Path('/Users/panmiao/Documents/ChatGPT/paper')
ROOT = BASE / 'outputs/annotation_confidence_20261021'
SITE = BASE / 'outputs/hand_mesh_repair_20261020'
CODE = BASE / 'code/human-robot-transfer/local_annotation_lab'


def main():
    result=json.loads((ROOT/'summary.json').read_text())
    path=SITE/'assets/multi_manifest.json';manifest=json.loads(path.read_text())
    target=SITE/'confidence';target.mkdir(exist_ok=True)
    backup=ROOT/'manifest_before_confidence.json'
    if not backup.exists():shutil.copy2(path,backup)
    before={c['id']:c['layers'] for c in manifest['clips']}
    totals=Counter()
    for c in manifest['clips']:
        sid=c['id']
        if sid not in result['samples']:continue
        data=json.loads((ROOT/'samples'/f'{sid}.json').read_text())
        assert data['sample']==sid and len(data['frames'])==c['frames']
        assert data['calibrated_probability'] is False and data['training_eligible'] is False
        totals.update(data['summary']['statuses'])
        c['confidence']=f'confidence/{sid}.json'
        shutil.copy2(ROOT/'samples'/f'{sid}.json',target/f'{sid}.json')
    text='''# 无人工标注时的自动可信度

**已为 8 个多模态样本的 2,481 帧计算逐实例时序证据分，并接入原网页。用于定位弱证据与排序，不是正确概率；已知误分仍阻断相关类别。语义、接触、绝对深度与物体 6D 没有足够独立证据，正确性置信度保持未知。** 本轮不需要人工输入，没有把既有助手观察冒充人工金标；也未自动批准训练准入。

当前最重要的限制：稳定的错误掩码也能得高分；类别/实例选错、持续漏掉某只手、真实接触和目标完成情况都不能单靠时序一致性判断。建议用低分排除弱标签、高分保留候选；在没有金标时，只能称为自动伪标签实验，不能声称达到某个真实准确率。

- [原统一网页](http://127.0.0.1:53419/?clip=umi_hifi_pot#confidence-review)
- [本轮 TensorBoard](http://127.0.0.1:16068/#scalars)：实际已处理帧数与各样本进度；CPU 计算，没有训练 loss。
- [程序汇总](summary.json)、[完成状态](progress.json)、[发布验证](validation.json)

## 分数怎么算

1. 原视频缩放到宽 320 像素，逐对计算前向、后向 Farneback 光流。将上一帧掩码按当前到上一帧光流搬到当前帧，计算与当前掩码的 IoU。
2. 在当前掩码内有纹理且对应未越界的像素上，检查前后向光流是否能返回原位置。误差容许值为 `1 + 0.05 × 后向流长度` 像素。通过像素比例为第二个证据值。
3. 每对帧的分数为 `100 × min(补偿后掩码 IoU, 前后向对应通过比例)`；一个实例帧同时有前后两对有效证据时，取更低的一对。只有一对时明确记录，证据缺失不填成高分。
4. 320 像素尺度下任一掩码少于 16 像素，或有纹理的有效对应少于 16 像素 / 当前掩码的 15%，不给分。纹理由 Sobel 梯度幅值大于 8 定义。小物体在这个分辨率下更易证据不足。
5. 80–100 标为一致性强，50–80 为一般，低于 50 为弱。这些是公开的启发式排序阈值，不是经准确率校准的验收门槛。几何/光流证据并非完全独立，不做概率乘法或虚构综合正确率。

框由掩码生成，因此共享轮廓证据，不能再把框一致性当第二份独立支持。光流页展示的是本轮 320 像素尺度的前后向自一致性，不代表既有全分辨率光流的准确率。

深度另给“变化稳定性”：`100 × exp(-max(相邻中位深度变化, 相邻焦距变化) / 0.1)`。首帧无前序证据时不给分。真实相机/场景变化也会降分，稳定的错误深度也会得高分；它不是米制精度置信度，不参与分割评分。

SAM3 保存的 9,757 个原始模型分数全部为 1.0，没有区分能力，本轮完全不使用它们打分。没有把 1.0 解释为 100% 正确。

## 已有问题如何处理

先前助手抽帧观察记录中的误分 / 漏检只作为独立列出的冲突证据，来源不是人类金标。由于尚无精确错误时间范围，保守阻断该样本的受影响类别；不表示该类别每帧都已确认错误。原始一致性分仍可查看，但状态优先显示“已有问题 · 暂不采用”。当前未检出类别为“可见性未知”，不赋零分，更不能当作阴性。

## 本轮实际结果

下表中位数只针对有掩码且可评分的实例帧。多实例会使实例帧数超过视频帧数；必须同时看有输出帧覆盖和已知问题，不能用中位数掩盖漏检。

| 样本 / 类别 | 一致性中位分 | 可评分 / 有输出实例帧 | 有该类别的帧 / 视频帧 | 已有问题阻断 |
|---|---:|---:|---:|---|
'''
    for sid,s in result['samples'].items():
        for cls,v in s['summary']['classes'].items():
            text+=f"| {sid} / {cls} | {v['median']} | {v['scored']} / {v['total']} | {v['detected_frames']} / {s['summary']['frames']} | {'是' if v['known_issue_codes'] else '未记录；不代表正确'} |\n"
    text+='\n实例帧状态：'+ '；'.join(f'{k}={v}' for k,v in totals.items())+'。同一目标的相邻帧不是独立样本，这些数量不能用于推断总体正确率。\n'
    text+='''
## 完全自动流程下一步

无需人工也可以继续提高筛选能力，但以下证据本轮尚未运行：对稀疏关键帧重新检测、从末帧反向运行分割模型、变换提示 / 图像增强后复测；比较一个不同架构视觉模型的对象与动作判断；把原生夹爪状态、运动和多视角作为语义边界及接触事件的外部佐证。模型的自报置信度不作为验收依据，多模型一致也不保证正确。

每项分别保留分数、缺失原因和证据覆盖。所有关键证据一致且无冲突时，可进入自动伪标签候选；冲突时重算或排除；证据不足时弃权。先做小规模伪标签训练对比以检查实用价值，避免直接扩大全量训练。没有独立参考时，真实精度仍无法保证，不强制要求人工才能浏览或开展明确标识的伪标签实验。

## 验证与可复现

评分测试覆盖静态纹理对应、错误区域跳动、无纹理弃权、已知问题覆盖高分、缺失证据和深度变化。验证程序逐帧检查范围、空值语义、来源哈希和网页资源。本轮不改原视频、分割或语义内容，只新增证据记录。原折衣样本尚未计算这一版分数，网页明确显示未计算。

模型输出、掩码包、深度检查和原视频哈希均记录在各样本 JSON。版本 `temporal-evidence-v1`，参数随文件保存；本轮结论不覆盖全数据集。
'''
    (ROOT/'report_summary.md').write_text(text)
    for f in ['summary.json','progress.json','report_summary.md']:shutil.copy2(ROOT/f,target/f)
    html=markdown.markdown(text,extensions=['tables','fenced_code'])
    (target/'report_summary.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自动可信度说明</title><style>body{font:16px/1.8 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#21343c}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #ddd;overflow-wrap:anywhere}a{color:#176c70}code{overflow-wrap:anywhere}</style><a href="../">返回工作台</a>'+html+'</html>')
    for f in ['index.html','app.js','styles.css']:shutil.copy2(CODE/f,SITE/f)
    assert before=={c['id']:c['layers'] for c in manifest['clips']}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));tmp.replace(path)
    print('Published',len(result['samples']),'confidence reports; existing annotation layers unchanged')


if __name__=='__main__':main()
