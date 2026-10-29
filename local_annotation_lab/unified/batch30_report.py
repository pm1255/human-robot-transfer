import json,shutil
from pathlib import Path
import markdown
B=Path('/Users/panmiao/Documents/ChatGPT/paper');R=B/'outputs/unified_30_20261021';S=B/'outputs/hand_mesh_repair_20261020'
m=json.loads((S/'assets/multi_manifest.json').read_text());clips=m['clips'];count=sum(bool(c.get('multimodal'))for c in clips);segments=sum(len(json.loads((S/c['semantics']).read_text())['segments'])for c in clips)
lines=[f'# 三类数据统一标注与自动质检\n\n已接入 **EgoSuite 人类 10 条、ABC-130K 真机 10 条、UMI 10 条**，共 {sum(c["frames"]for c in clips):,} 帧、{segments} 个动作片段。保留原统一网页，Ego 样本已替换为 yangqinglin 的 egosuite_part 产物。',
f'当前可以用于回放、自动筛查和初标分析；**不能把规则分或一致性分解释为标注正确概率**。分割、框、深度、光流和二维接触证据已接入 {count}/30 条；物体 6D、功能部位和碰撞证据仍缺失。',
'需要重点处理的质量问题：3/10 条 Ego 的目录帧率与逐帧时间戳不符，已在预览中按原生时间网格修正并保留全部帧；纸杯样本的独立图像对齐支持较弱，不能仅凭约 95 分轨迹规则分放行。一条源动作描述与抽帧观察冲突，原文与修订初标均保留。',
'[打开统一网站](http://127.0.0.1:53419/) · [实际计算进度](http://127.0.0.1:16069/)\n',
'## 数据来源与选择\n\nEgo 来自 `/user/yangqinglin/egosuite_part_{04,05,06,07}/lerobot_v3`，分别取 3、3、2、2 条。根据每个 part 的导出 info、episode 元数据、状态和双目文件核对，不使用 staging 原料或 egosuite_body。这里只确认所选 episode 可用；没有声称整批导出已通过质量验收。',
'固定种子 20261021；Ego 从 240–900 帧的完整 episode 中按任务去重抽取，ABC 复用已有 3 条并从已有 500 条候选池补 7 种任务，UMI 从本地完整样本选取并去除跨目录镜像重复。全部保留完整 episode，不通过切片凑数。此为覆盖工作流的小样本试验，不估计全数据集错误率。',
'## uni_cleaner_plus 能检查什么\n\n代码来源 `/user/yangqinglin/workspace/uni_cleaner_plus`，提交 `d06d30a0729bef6df0c8404364d8ce0332b654e7`。代码快照存于本批 vendor；结果全部写入我们的输出目录，未修改他人的代码或源数据。',
'本批实际运行 `state_self_check` 全帧轨迹检查、B0 几何投影检查、B2 MediaPipe 与原生手部投影对照，每条左右相机各抽 24 帧。运行前确认 148 维布局与字段含义一致，所选 parquet 文件只有一个 episode，视频偏移为 0，避免上游默认单 episode / 零偏移读取的假设误用。B2 使用独立模型但不是人工真值。',
'上游 B1 运动/时延检查尚未实现，本批未计为通过。视频全帧解码、帧率/时间网格、语义完整区间、掩码框一致性由本项目补充。B0 的跳变统计来自抽帧间隔，不能作为相邻原始帧速度。未运行 uni_cleaner 的全套模糊/曝光 Stage1，也未运行 ego_validation 的全部 CLI 检查；不能称为全套质检通过。',
'## 分数定义与实际结果\n\n轨迹规则分 = 100 × max(0, min(1, 双手训练有效帧比例 − 0.05 × 触发原因数))，规则阈值来自上游默认配置，未经人工校准。PCK@0.2 是与检测手框长边 20% 以内的匹配关节点比例，表中取两相机较低值，单位为百分比；它不是准确率。可比对覆盖低时，PCK 只代表少数可见成功匹配帧。',
'| Ego episode | 轨迹规则分 /100 | 图像 PCK@0.2 | 可比对覆盖 | 结论 |\n|---|---:|---:|---:|---|']
for c in clips:
 if c.get('cleaner_summary'):
  q=c['cleaner_summary'];lines.append(f'| {c["id"]} | {q["state_score"]} | {q["pck20"]}% | {q["match_coverage"]}% | {"；".join(q["notes"])or"规则未触发；图像抽检有支持"} |')
lines += ['\nSAM3 分割与框共用轮廓证据，不重复累计。分割时序分取光流补偿轮廓 IoU 与前后向光流可靠对应比例的较小值；纹理、可见对应或目标面积不足时不给分。SAM 交互输出的恒定 1.0 不进入评分。深度只评价相邻变化稳定性，不代表米制精度。已有确定问题优先于高分，稳定误分也可能高分。',
'## 语义与视频\n\n三类共用 approach/contact/align/transport/manipulate/release/retract/idle/unknown 阶段与统一动词词表。按可见动作及状态变化划分，使用 [start_frame, end_frame)；保留源任务指令、源片段描述、当前初标与边界不确定度。新增语义由助手查看抽帧后初标，边界约 ±0.65–1.5 秒，没有独立逐帧语义验收；contact 阶段也不代表接触传感真值。',
'Ego 原生 3D 手部关节投影单独输出视频，无效手部保持缺失，不伪造网格。真机和 UMI 保留原生末端/夹爪时序，坐标系和单位不强行混合。SAM3、框、MoGe 深度、光流和二维接触证据分别输出视频；接触最近点不等于真实接触或功能点。',
'## 无人工情况下的使用规则\n\n先检查结构和时间对应；明确异常直接排除对应帧或标注层。再按每种标签的独立证据和覆盖分层选择候选，低分自动重算或换模型，有冲突则弃权。无需把整条 episode 所有模态一并删除；维护逐帧、逐对象、逐标注层有效掩码。高一致性候选仍是伪标签，不承诺正确率。没有独立真值时不能校准成 90%/95% 正确概率。',
'## 完整明细\n\n[选样清单](selection_manifest.json) · [74 路源视频完整解码](source_decode_audit.json) · [Ego 原生时间轴修正](ego_time_audit.json) · [原生时序检查](native_audit.json) · [共享语义词表](taxonomy.json)\n\n逐条 `cleaner_qc.json`、`segments.json`、`time_audit.json` 和模型原始结果均在网站相应卡片中可下载。历史 Ego 页面清单保存到 `archives/manifest_before_egosuite.json`，原视频与标注文件未删除。']
reviews=json.loads((R/'visual_review.json').read_text())['samples']
issues=sum(bool(v.get('confirmed_errors'))for v in reviews.values())
validation=json.loads((R/'release_validation.json').read_text())if(R/'release_validation.json').exists()else{}
if validation.get('videos'):
 lines.insert(2,f'交付验证：{len(validation["videos"])} 个源视频与标注视频已完整解码，帧数/帧率匹配；30 条主视频内容互不重复，资源链接检查无缺失，并验证了三类视频的分段加载。结构检查不等于标注精度验收。')
lines.insert(3,f'分割抽查中 {issues}/30 条存在至少一类已确认局部问题，按类别保守标记“暂不采用”；这不是全数据集错误率。另有边界不确定与可见性不足的标签，不判为通过，也不赋零分。9 条样本已针对明显提示错误重算；具体改善和残留问题见下表。')
lines += ['## 逐条标注证据与处置\n\n下表的分数是各对象类别的逐帧时序分中位数；不同对象分别列出，不合成正确概率。抽帧检查由助手完成，没有独立人工真值。已确认问题优先于分数。', '| 样本 | 类别时序分中位数 /100 | 抽查与处理 |\n|---|---|---|']
for c in clips:
 score=S/c['confidence']if c.get('confidence')else None
 q=json.loads(score.read_text())if score and score.exists()else None
 values='；'.join(f'{k}: {v["median"] if v["median"]is not None else "证据不足"}'for k,v in q['summary']['classes'].items())if q else '等待计算'
 review=reviews.get(c['id'],{})
 lines.append(f'| {c["id"]} | {values} | '+('；'.join(review.get('observations',[]))or'尚未查看模型结果')+' |')
lines += ['## 模型产物与未完成项\n\n每个样本保留源视频和原生状态，另有逐片段语义、SAM3 分割、掩码派生框、MoGe 相对深度、二维光流及接触候选视频。Ego 另有原生手部关节投影。手型网格、人手替换机器人、可信物体 6D、功能部位及碰撞真值未在这批 30 条上完成；原折衣文件完整保留，不能把历史结果当作这些新样本的标注。FastUMI 的近黑像素从深度预览与质检中排除，此启发式也可能排除很暗的场景区域，原始预测数组保留。',
'## 可追溯原始报告\n\n[发布解码与资源验证](release_validation.json) · [分割结构检查](multimodal_audit.json) · [模型传输哈希](transfer_manifest.json) · [逐条抽查与已知问题](visual_review.json) · [上游代码来源](vendor_provenance.json) · [模型提示修正](prompt_corrections_full_coordinates.json)\n\n原始逐条评分可从网站各样本的“逐帧评分 JSON”下载；规则阈值、模型来源和逐帧失败原因均保留。']
text='\n\n'.join(lines);(R/'report_summary.md').write_text(text);d=S/'batch30';d.mkdir(exist_ok=True);shutil.copy2(R/'report_summary.md',d/'report_summary.md')
html=markdown.markdown(text,extensions=['tables','fenced_code']);(d/'report_summary.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三类数据标注与质检</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#20343c}table{border-collapse:collapse}td,th{border:1px solid #ccd8db;padding:9px}a{color:#176c70}code{overflow-wrap:anywhere}</style><body><a href="../">← 返回标注网站</a>'+html+'</body></html>')

for name in ['selection_manifest.json','source_decode_audit.json','ego_time_audit.json','native_audit.json','taxonomy.json','release_validation.json','multimodal_audit.json','transfer_manifest.json','visual_review.json','vendor_provenance.json','prompt_corrections_full_coordinates.json']:
 if(R/name).exists():shutil.copy2(R/name,d/name)
