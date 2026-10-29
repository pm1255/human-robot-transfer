'use strict';
// Manifest-driven review and native video controls follow human2robot/web/app.js.
const $ = selector => document.querySelector(selector);
let manifest, clip, quality, videos = [], master, synchronized = false, syncing = false;
let generation = 0, playRequest = 0, pendingFrame = 0;
let frameInputDirty = false;
let buffering = false;
let poseRows = null;
let semanticSegments = null, robotRows = null, compositeRows = null;
let semanticDocument = null, nativeState = null, semanticTaxonomy = null, editorIndex = -1;
let objectRows = null, objectIssues = [];
let confidenceData = null;
const poseLabels = {
  proxy_review_required: ['通过筛查，待复核', '符合当前轮廓、深度与跳变阈值，精度未验收', '#89b4ac'],
  rejected_proxy: ['候选被拒绝', '轮廓、深度或运动不一致，正式位姿留空', '#d6ac61'],
  waiting_for_reinitialization: ['等待重新初始化', '失败后每 30 帧重试，中间不填充位姿', '#aab6be'],
  insufficient_mask_or_depth: ['输入不足', '掩码或有效深度不足', '#85949e'],
  before_reference: ['参考帧之前', '尚未初始化', '#e1e7eb']
};
const rejectionLabels = {
  temporal_jump: '相邻位姿跳变',
  silhouette_mismatch_deformation_or_tracking_error: '轮廓不匹配',
  depth_mismatch: '深度不匹配', insufficient_depth_overlap: '深度重叠不足',
  invalid_transform: '变换矩阵无效'
};
const fmt = value => {
  const seconds = Math.max(0, value || 0);
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
};
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function message(text) {
  $('#playback-status').textContent = text;
  $('#playback-status').hidden = !text;
}
function syncButton() {
  $('#play').textContent = synchronized ? buffering ? 'Ⅱ 等待缓冲' : 'Ⅱ 同步暂停' : '▶ 同步播放';
  $('#play').setAttribute('aria-label', synchronized ? '同步暂停' : '同步播放');
}
function pauseAll() {
  const wasSynchronized = synchronized;
  synchronized = false;
  buffering = false;
  playRequest += 1;
  videos.forEach(video => video.pause());
  syncButton();
  if (wasSynchronized && master?.readyState >= 1 && clip) seekFrame(Math.floor(master.currentTime * clip.fps + .001));
}
function waitForGroup() {
  if (!synchronized || buffering) return;
  buffering = true;
  videos.forEach(video => video.pause());
  seekFrame(Math.floor(master.currentTime * clip.fps));
  message('等待全部视频缓冲；已暂停在同一帧。');
  syncButton();
  resumeGroup();
}
async function resumeGroup() {
  if (!synchronized || !buffering || videos.some(v => !v.error && (v.readyState < 3 || v.seeking))) return;
  buffering = false;
  message(''); syncButton();
  const request = playRequest;
  const results = await Promise.allSettled(videos.filter(v => !v.error).map(v => v.play()));
  if (request !== playRequest) return;
  if (results.some(r => r.status === 'rejected')) { pauseAll(); message('部分视频尚未就绪，请再次同步播放。'); }
}
function updateTime(seconds) {
  const frame = Math.min(clip.frames - 1, Math.max(0, Math.floor(seconds * clip.fps + .001)));
  pendingFrame = frame;
  $('#seek').value = frame;
  $('#time').textContent = `${fmt(seconds)} / ${fmt(clip.frames / clip.fps)}`;
  $('#frame').textContent = `帧号 ${frame} / ${clip.frames - 1} · 共 ${clip.frames} 帧`;
  if (!frameInputDirty && document.activeElement !== $('#frame-input')) $('#frame-input').value = frame;
  updateFrameStatus(frame);
}
function seekFrame(frame) {
  if (!clip || !Number.isFinite(frame)) return;
  pendingFrame = Math.min(clip.frames - 1, Math.max(0, Math.round(frame)));
  const seconds = pendingFrame / clip.fps;
  syncing = true;
  for (const video of videos) if (video.readyState >= 1 && !video.error) video.currentTime = seconds;
  syncing = false;
  updateTime(seconds);
}
function refreshControls() {
  const playable = videos.filter(video => video.readyState >= 1 && !video.error);
  if (!master || master.error) master = playable[0];
  for (const selector of ['#play', '#previous', '#next', '#seek']) $(selector).disabled = !playable.length;
}
function card(layer, index) {
  const selection = generation;
  const article = element('article', 'annotation-card');
  article.dataset.layer = layer.id;
  article.id = `layer-${layer.id}`;
  const header = element('div', 'card-heading');
  const title = element('h3');
  title.append(element('span', 'card-number', String(index + 1).padStart(2, '0')), document.createTextNode(layer.title));
  const badge = element('span', `badge ${layer.badge_tone}`, layer.badge);
  header.append(title, badge);
  const stage = element('div', 'video-stage');
  const description = element('p', 'card-description', layer.description);
  const body = element('div', 'card-body');
  body.append(description);
  const chips = element('div', 'card-meta');
  for (const text of layer.tags || []) chips.append(element('span', 'meta-chip', text));
  body.append(chips);
  const assessment = element('p', 'card-assessment', layer.assessment);
  body.append(assessment);
  if (layer.resources?.length) {
    const links = element('div', 'resource-links');
    for (const resource of layer.resources) {
      const link = element('a', '', `${resource.label} ↗`);
      link.href = resource.url;
      links.append(link);
    }
    body.append(links);
  }
  if (layer.video) {
    const video = document.createElement('video');
    video.controls = true;
    video.playsInline = true;
    video.muted = true;
    video.preload = 'metadata';
    video.setAttribute('aria-label', layer.title);
    if (layer.poster) video.poster = layer.poster;
    stage.append(video);
    videos.push(video);
    if (layer.id === 'source') master = video;
    video.addEventListener('loadedmetadata', () => {
      if (selection !== generation) return;
      video.currentTime = pendingFrame / clip.fps;
      video.playbackRate = Number($('#speed').value);
      refreshControls();
    });
    video.addEventListener('error', () => {
      if (selection !== generation) return;
      pauseAll();
      badge.textContent = '加载失败';
      badge.className = 'badge review';
      assessment.textContent = '视频未能加载，请检查资源是否可访问。此项不计为播放验证通过。';
      refreshControls();
      message(`${layer.title}加载失败；其他可用视频仍可查看。`);
    });
    video.addEventListener('timeupdate', () => {
      if (selection !== generation || video !== master || syncing || buffering) return;
      updateTime(video.currentTime);
      if (synchronized) for (const other of videos) {
        if (other !== master && !other.error && (other.readyState < 2 || Math.abs(other.currentTime - master.currentTime) > .18)) { waitForGroup(); break; }
      }
    });
    video.addEventListener('pause', () => { if (selection === generation && synchronized && !buffering && video.paused) pauseAll(); });
    video.addEventListener('waiting', () => { if (selection === generation) waitForGroup(); });
    for (const event of ['canplay', 'seeked']) video.addEventListener(event, () => { if (selection === generation) resumeGroup(); });
    video.addEventListener('ended', () => { if (selection === generation && synchronized) pauseAll(); });
    video.addEventListener('seeking', () => {
      if (selection === generation && synchronized && video === master && !syncing) {
        for (const other of videos) if (other !== master && other.readyState >= 1 && Math.abs(other.currentTime - master.currentTime) > .18) other.currentTime = master.currentTime;
      }
    });
    video.src = layer.video;
    const link = element('a', 'card-link', '单独打开视频 ↗');
    link.href = layer.video;
    link.target = '_blank';
    link.rel = 'noopener';
    body.append(link);
  } else {
    stage.classList.add('unavailable');
    const empty = element('div', 'empty-content');
    empty.append(element('div', 'empty-symbol', layer.symbol || '◌'), element('strong', '', layer.empty_title || '原标注资源待接回'), element('p', '', layer.missing_reason));
    stage.append(empty);
  }
  if (layer.semantic_summary) {
    const block = element('details', 'semantic-summary');
    block.append(element('summary', '', '整体语义 · 历史记录与来源'));
    const dl = document.createElement('dl');
    for (const [label, value] of Object.entries(layer.semantic_summary.fields)) dl.append(element('dt', '', label), element('dd', '', value));
    block.append(dl, element('p', 'semantic-source', layer.semantic_summary.source));
    body.append(block);
  }
  article.append(header, stage, body);
  return article;
}
function renderCoverage() {
  const canvas = $('#coverage'), ctx = canvas.getContext('2d');
  ctx.fillStyle = '#89b4ac';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  function spans(items, color) {
    ctx.fillStyle = color;
    for (const [start, end] of items || []) ctx.fillRect(start / clip.frames * canvas.width, 0, Math.max(1, (end - start + 1) / clip.frames * canvas.width), canvas.height);
  }
  spans(quality.review_spans, '#d6ac61');
  spans(quality.missing_spans, '#d6dde2');
  $('#coverage-summary').textContent = `${quality.mesh_frames} 帧有网格 · ${quality.missing_frames} 帧未检测 · ${quality.review_frames} 帧需复核`;
}
function selectClip(id) {
  pauseAll();
  generation += 1;
  for (const video of videos) { video.removeAttribute('src'); video.load(); }
  clip = manifest.clips.find(item => item.id === id);
  const selectedUrl = new URL(location.href);
  selectedUrl.searchParams.set('clip', clip.id);
  history.replaceState(null, '', selectedUrl);
  quality = clip.hand_quality || null;
  nativeState = semanticDocument = null; editorIndex = -1;
  objectRows = null; objectIssues = [];
  confidenceData = null;
  $('#cleaner-review').hidden = !clip.cleaner_summary;
  if (clip.cleaner_summary) {
    const q = clip.cleaner_summary;
    $('#cleaner-conclusion').textContent = q.notes.length ? `需要留意：${q.notes.join('；')}。可用于查看与筛查，不能据此直接认定标签已验收。` : '当前轨迹规则筛查未发现明显异常，抽检有图像对齐支持；可作为初标候选，尚未验证全部帧与全部标注。';
    const rows = [
      ['轨迹规则分', `${q.state_score} / 100`, '有效性、骨长、四元数、冻结与时间连续性；未校准'],
      ['手部图像一致性', q.pck20 === null ? '证据不足' : `${q.pck20}%`, '左右相机中较低的 PCK@0.2：误差小于检测手框长边 20% 的关节点比例'],
      ['独立模型可比对覆盖', `${q.match_coverage}%`, '有效原生手部槽位中能与独立检测匹配的比例；不是检测召回真值'],
      ['原生时间轴', q.time_fixed ? '已修正目录帧率' : '按 episode 时间戳导出', '每个源解码帧保留一次；物理时延尚未测量']
    ];
    $('#cleaner-rows').replaceChildren(...rows.map(values => { const tr=element('tr'); for(const value of values) tr.append(element('td','',value)); return tr; }));
    $('#cleaner-download').href=clip.cleaner_qc; $('#time-audit-download').href=clip.time_audit;
  }
  $('#confidence-review').hidden = false;
  $('#confidence-summary').textContent = clip.confidence ? '正在计算结果中读取证据…' : '此样本尚未计算自动证据分；不等于零置信度。';
  $('#confidence-rows').replaceChildren();
  $('#confidence-frame').textContent = '';
  $('#confidence-other').textContent = '';
  $('#confidence-download').hidden = !clip.confidence;
  if (clip.confidence) $('#confidence-download').href = clip.confidence;
  else $('#confidence-download').removeAttribute('href');
  $('#confidence-next').disabled = true;
  if (clip.confidence) loadConfidence(generation, clip);
  $('#multimodal-review').hidden = !clip.multimodal;
  $('#multimodal-findings').textContent = '';
  $('#multimodal-frame').textContent = '';
  $('#next-object-issue').disabled = true;
  if (clip.multimodal) loadMultimodal(generation, clip);
  frameInputDirty = false;
  $('#hand-coverage').hidden = !quality;
  $('#geometry-review').hidden = !clip.geometry;
  $('#geometry-rejected').hidden = !clip.geometry;
  $('#native-review').hidden = !clip.state;
  $('#native-current').textContent = '';
  $('#native-links').replaceChildren();
  $('#native-summary').textContent = clip.native_summary || '';
  $('#native-qc').textContent = '';
  $('#domain-badge').textContent = clip.dataset;
  $('#task-instruction').textContent = clip.task_instruction ? `源任务指令（不直接作为片段标签）：${clip.task_instruction}` : '原折衣样本：保留九类标注与历史修复结果。';
  $('#semantic-protocol').textContent = clip.semantic_protocol || '';
  $('#semantic-download').href = clip.semantics;
  $('#edit-status').textContent = '';
  clearExport();
  $('#edit-form').reset();
  if (clip.state) loadNative(generation, clip);
  videos = [];
  master = null;
  pendingFrame = clip.initial_frame || 0;
  poseRows = null;
  semanticSegments = robotRows = compositeRows = null;
  $('#semantic-summary').textContent = '正在读取动作片段…';
  $('#semantic-detail').textContent = '';
  $('#semantic-frame-status').textContent = '片段语义：读取中';
  $('#robot-frame-status').textContent = clip.robot ? '双臂转换：读取中' : '';
  $('#semantic-segment').replaceChildren();
  $('#semantic-rows').replaceChildren();
  for (const id of ['#semantic-segment', '#semantic-prev', '#semantic-next']) $(id).disabled = true;
  const semanticCanvas = $('#semantic-timeline');
  semanticCanvas.getContext('2d').clearRect(0, 0, semanticCanvas.width, semanticCanvas.height);
  $('#geometry-rejected').disabled = true;
  $('#geometry-counts').replaceChildren();
  $('#geometry-legend').replaceChildren();
  $('#geometry-summary').textContent = '正在读取位姿结果…';
  $('#geometry-frame-info').textContent = '位姿逐帧记录读取中。';
  $('#depth-quality-note').textContent = '';
  const poseCanvas = $('#geometry-timeline');
  poseCanvas.getContext('2d').clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  $('#pose-frame-status').textContent = '位姿：读取中';
  $('#frame-input').max = clip.frames - 1;
  message('');
  $('#seek').max = clip.frames - 1;
  $('#cards').replaceChildren(...clip.layers.map(card));
  $('#layer-links').replaceChildren(...clip.layers.map(layer => {
    const link = element('a', '', layer.short_title || layer.title);
    link.href = `#layer-${layer.id}`;
    return link;
  }));
  $('#sample-meta').textContent = `${clip.frames} 帧 · ${(clip.frames / clip.fps).toFixed(2)} 秒 · ${clip.width} × ${clip.height} · ${clip.fps} FPS`;
  const available = clip.layers.filter(layer => layer.video).length;
  const missing = clip.layers.filter(layer => !layer.video).map(layer => layer.title).join('、');
  const incomplete = clip.layers.filter(layer => layer.annotation_complete === false).map(layer => layer.short_title || layer.title).join('、');
  $('#page-status').textContent = clip.multimodal_summary || `已接入 ${available} / ${clip.layers.length} 路独立视频。${missing ? `${missing}视频缺失。` : ''}${clip.semantics ? '语义已按动作片段切换，时间边界待复核。' : incomplete ? `${incomplete}尚未补全。` : ''}${clip.robot ? '双臂转换为运动学试验，未完成物理执行验证。衣物位姿为整体近似。' : ''}`;
  $('#strict-quality').textContent = `${incomplete || missing ? `${incomplete || missing}尚未补全。` : ''}语义边界、手型与单目深度尚未独立验收。衣物整体 6D 位姿不能描述布料每个点的运动。双臂预览不证明抓持接触或折衣成功。`;
  $('#loose-quality').textContent = `关节点视频与手型视频均已接入。关节点 ${clip.skeleton_frames}/${clip.frames} 帧有检测；手型 ${quality?.mesh_frames ?? 0}/${clip.frames} 帧有输出，其中 ${quality?.review_frames ?? 0} 帧需复核，不能仅凭覆盖率认定标注准确。`;
  if (!quality) {
    $('#page-status').textContent = clip.multimodal_summary || clip.sample_note;
    $('#loose-quality').textContent = clip.native_summary;
    $('#strict-quality').textContent = clip.multimodal_quality || '语义为逐片段初标，边界精度与接触状态待独立复核。源位姿未完成跨数据集坐标标定；物体标注尚待补全。当前样本未获训练准入。';
  }
  updateTime(pendingFrame / clip.fps);
  if (quality) renderCoverage();
  refreshControls();
  loadGeometry(generation, clip);
  loadTemporalAnnotations(generation, clip);
  if (clip.robot?.composite_review) {
    const current = generation;
    fetch(clip.robot.composite_review).then(r => { if (!r.ok) throw new Error('合成记录读取失败'); return r.json(); }).then(data => {
      if (current !== generation) return;
      compositeRows = data.frames;
      updateFrameStatus(pendingFrame);
    }).catch(() => { if (current === generation) message('合成逐帧记录暂不可用，遮挡补全需复核。'); });
  }
}
function updateFrameStatus(frame) {
  if (confidenceData) renderConfidence(frame);
  if (objectRows?.[frame]) {
    const row = objectRows[frame];
    const counts = {};
    for (const instance of row.instances) counts[instance.class_id] = (counts[instance.class_id] || 0) + 1;
    const issue = objectIssues.filter(item => item.frame_index === frame);
    $('#multimodal-frame').textContent = `第 ${frame} 帧 · 有输出：${Object.entries(counts).map(([label,n]) => `${label} × ${n}`).join('、') || '无'}。未检出类别：${row.missing_classes.join('、') || '无'}。${issue.length ? `有 ${issue.length} 个轮廓变化或缺失提示。` : '此帧未触发自动提示，精度仍需复核。'}`;
  }
  const contains = spans => (spans || []).some(([start, end]) => frame >= start && frame <= end);
  $('#hand-frame-status').textContent = quality ? `手型：${contains(quality.missing_spans) ? '未检测' : contains(quality.review_spans) ? '有输出，需复核' : '有输出，精度未验收'}` : clip.domain === 'human' ? clip.cleaner_qc ? '人手：原生 21 点投影已接入，网格未标' : '人手：源手腕记录已接入，网格待标' : '执行器：原生末端记录';
  if (nativeState) updateNativeFrame(frame);
  if (semanticSegments) {
    const index = semanticSegments.findIndex(s => frame >= s.start_frame && frame < s.end_frame_exclusive);
    const s = semanticSegments[index];
    if (s) {
      $('#semantic-frame-status').textContent = `语义：${s.label}`;
      $('#semantic-detail').textContent = `片段 ${index + 1} / ${semanticSegments.length}${s.cycle ? ` · 第 ${s.cycle} 次演示` : ''} · ${fmt(s.start_s)}–${fmt(s.end_s)} · ${s.label} · ${s.verb} · 边界约 ±${s.boundary_uncertainty_s ?? '未定'} 秒，待复核`;
      if (editorIndex !== index) populateEditor(index);
      $('#semantic-segment').value = String(index);
    }
  }
  if (robotRows) {
    const row = robotRows[frame], hands = Object.entries(row?.hands || {});
    const needsReview = hands.some(([, h]) => h.status === 'tracking_error_review');
    const belowTable = hands.some(([, h]) => h.link_point_table_clearance_m < 0);
    const continuous = hands.some(([, h]) => h.rotation_metric_available === false);
    const inferred = hands.filter(([, h]) => h.display_inferred).length;
    $('#robot-frame-status').textContent = continuous
      ? `机械臂：连续重定向 · ${inferred ? `${inferred} 臂使用补间估计` : '依据手部观测估计'} · 未验证物理执行${belowTable ? ' · 连杆点低于桌面' : ''}`
      : hands.length ? `机械臂试验：${hands.map(([side, h]) => `${side === 'Left' ? '左' : '右'} ${(h.error_m * 100).toFixed(1)} cm / ${(h.rotation_error_rad * 180 / Math.PI).toFixed(0)}°`).join('；')} 拟合残差 · ${needsReview ? '偏差超阈值' : '通过拟合筛查'}${belowTable ? ' · 连杆点低于桌面' : ''}${hands.some(([, h]) => h.reinitialized) ? ' · 重初始化' : ''}` : '机械臂试验：手部或桌面数据不足';
    const comp = compositeRows?.[frame];
    if (comp) $('#robot-frame-status').textContent += ` · 原场景合成 · ${comp.mask_review ? '手臂抠图需复核' : '遮挡补全待复核'}`;
  }
  if (!poseRows) return;
  const row = poseRows[frame];
  $('#pose-frame-status').textContent = `位姿：${poseLabels[row.status]?.[0] || row.status}`;
  let details = `帧号 ${frame} · ${fmt(row.timestamp_s)} · ${poseLabels[row.status]?.[0] || row.status}`;
  if (Number.isFinite(row.quality?.mask_iou)) details += ` · 轮廓 IoU ${row.quality.mask_iou.toFixed(3)}`;
  if (Number.isFinite(row.quality?.relative_depth_error)) details += ` · 相对深度残差 ${(row.quality.relative_depth_error * 100).toFixed(2)}%`;
  if (row.status === 'rejected_proxy') {
    const reasons = (row.quality?.flags || []).map(flag => rejectionLabels[flag]).filter(Boolean);
    if (reasons.length) details += ` · 原因：${reasons.join('、')}`;
  }
  $('#geometry-frame-info').textContent = details;
}
const confidenceLabels = {pot:'锅具',induction_cooker:'电磁炉',gripper:'夹爪',hand:'人手',bread:'面包',plate:'餐盘',tongs:'夹持工具',screwdriver:'螺丝刀',container:'容器',whiteboard:'白板',eraser:'板擦',cabinet_door:'柜门',cabinet_handle:'柜门把手',lettuce:'生菜',mustard_bottle:'调味瓶',cup:'杯子',cup_lid:'杯盖',bottle:'瓶子',bin:'桶',cloth:'布料',tray:'托盘',plastic_bag:'塑料袋',headphones:'耳机',rack:'沥水架',fruit:'水果',bowl:'碗 / 果盘',microwave:'微波炉',pillow:'枕头',folder:'文件夹',suitcase:'箱包',rice_cooker:'电饭煲',remote_control:'遥控器',package:'盒装商品',refrigerator_door:'冰箱门',picture_frame:'画框',cabinet:'衣柜',curtain:'窗帘',paper:'纸张',pen_holder:'笔筒'};
const confidenceStatuses = {known_issue:'已有问题 · 暂不采用',insufficient:'证据不足',strong_consistency:'时序一致性强',medium_consistency:'时序一致性一般',weak_consistency:'时序一致性弱'};
const scoreText = score => Number.isFinite(score) ? `${score.toFixed(1)} / 100` : '无法评分';
async function loadConfidence(selection, selectedClip) {
  try {
    const response = await fetch(selectedClip.confidence, {cache:'no-store'});
    if (!response.ok) throw new Error('自动证据分读取失败');
    const data = await response.json();
    if (selection !== generation) return;
    if (data.sample !== selectedClip.id || data.frames.length !== selectedClip.frames) throw new Error('评分与当前视频不匹配');
    confidenceData = data;
    $('#confidence-summary').textContent = '已计算本视频逐帧时序证据。分数用于排序，不是正确概率；稳定的误分也可能得高分，已有问题优先于分数。';
    $('#confidence-download').href = selectedClip.confidence;
    $('#confidence-next').disabled = !data.frames.some(r => r.missing_classes.length || r.instances.some(o => o.temporal_score === null || o.temporal_score < 50));
    renderConfidence(pendingFrame);
  } catch (error) {
    if (selection === generation) $('#confidence-summary').textContent = error.message;
  }
}
function renderConfidence(frame) {
  const row = confidenceData.frames[frame]; if (!row) return;
  $('#confidence-frame').textContent = `第 ${frame} 帧 · ${fmt(row.timestamp_s)} · 分割与框共用轮廓证据，不重复加分`;
  const records = row.instances.map(o => ({name:`${confidenceLabels[o.class_id] || o.class_id} · ${o.instance_id.split(':').at(-1)}`,score:o.temporal_score,status:o.status,reason:o.reasons.join('；') || '有相邻图像运动支持；类别与实例身份仍未验证'}));
  records.push(...row.missing_classes.map(o=>{
    const known = confidenceData.summary.classes[o.class_id]?.known_issue_codes?.length;
    return {name:confidenceLabels[o.class_id] || o.class_id,score:null,status:known ? 'known_issue' : 'insufficient',reason:known ? '当前未检出；已有该类漏检 / 误分记录，保守停用。当前帧原因未自动确诊，不赋零分' : '当前未检出，遮挡还是漏检未知；不赋零分'};
  }));
  $('#confidence-rows').replaceChildren(...records.map(o => {
    const tr = element('tr',o.status === 'known_issue' ? 'confidence-blocked' : '');
    for (const value of [o.name,scoreText(o.score),confidenceStatuses[o.status],o.reason]) tr.append(element('td','',value));
    return tr;
  }));
  $('#confidence-other').textContent = `深度变化稳定性：${scoreText(row.depth.stability_score)}；光流前后向自一致性（320 px）：${scoreText(row.flow_roundtrip_score)}。这两项不代表几何精度。语义、物理接触、绝对位姿：证据不足，暂不给正确性分数。`;
}
$('#confidence-next').addEventListener('click', () => {
  if (!confidenceData) return;
  const current = Number($('#seek').value);
  const rows = confidenceData.frames.filter(r => r.missing_classes.length || r.instances.some(o=>o.temporal_score === null || o.temporal_score < 50));
  const row = rows.find(r=>r.frame_index>current) || rows[0];
  if (row) {pauseAll();seekFrame(row.frame_index);}
});
async function loadMultimodal(selection, selectedClip) {
  try {
    const responses = await Promise.all([fetch(selectedClip.multimodal.objects), fetch(selectedClip.multimodal.qc)]);
    if (responses.some(r => !r.ok)) throw new Error('物体复核记录读取失败');
    const [objects, audit] = await Promise.all(responses.map(r => r.json()));
    if (selection !== generation) return;
    objectRows = objects.frames; objectIssues = audit.flags || [];
    $('#multimodal-findings').textContent = selectedClip.multimodal_quality;
    $('#next-object-issue').disabled = !objectIssues.length;
    updateFrameStatus(pendingFrame);
  } catch (error) {
    if (selection === generation) $('#multimodal-frame').textContent = error.message;
  }
}
$('#next-object-issue').addEventListener('click', () => {
  if (!objectIssues.length) return;
  const current = Number($('#seek').value);
  const issue = objectIssues.find(item => item.frame_index > current) || objectIssues[0];
  pauseAll(); seekFrame(issue.frame_index);
});
async function loadTemporalAnnotations(selection, selectedClip) {
  const read = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`读取失败 ${r.status}`); return r.json(); };
  const [sem, robot] = await Promise.allSettled([
    selectedClip.semantics ? read(selectedClip.semantics) : Promise.reject(new Error('未接入片段语义')),
    selectedClip.robot ? read(selectedClip.robot.trajectory) : Promise.resolve(null)
  ]);
  if (selection !== generation) return;
  if (sem.status === 'fulfilled') {
    const rows = sem.value.segments;
    if (Array.isArray(rows) && rows.length && rows[0].start_frame === 0 && rows.at(-1).end_frame_exclusive === selectedClip.frames && rows.every((s,i) => Number.isInteger(s.start_frame) && Number.isInteger(s.end_frame_exclusive) && s.end_frame_exclusive > s.start_frame && (i === 0 || s.start_frame === rows[i-1].end_frame_exclusive))) {
      semanticDocument = sem.value;
      semanticSegments = rows;
      try {
        const draft = JSON.parse(localStorage.getItem(`unified-draft:${selectedClip.id}`) || 'null');
        if (draft && draft.base_source === selectedClip.semantics && validSegments(draft.segments, selectedClip.frames)) {
          semanticSegments = draft.segments; semanticDocument = {...sem.value, ...draft};
          $('#edit-status').textContent = '已恢复本地草稿；语义视频仍是发布版本。';
        }
      } catch (_) { $('#edit-status').textContent = '本地草稿不可读，正在显示发布版本。'; }
      renderSemanticRows();
    } else $('#semantic-summary').textContent = '片段边界与源视频不一致，暂不显示。';
  } else $('#semantic-summary').textContent = `片段语义不可用：${sem.reason.message}`;
  if (!semanticSegments) $('#semantic-frame-status').textContent = '片段语义：读取失败';
  if (robot.status === 'fulfilled' && robot.value) {
    if (robot.value.frames?.length === selectedClip.frames && robot.value.frames.every((row,i)=>row.frame_index===i)) robotRows = robot.value.frames;
    else $('#robot-frame-status').textContent = '双臂转换：记录与视频不一致';
  } else if (selectedClip.robot) $('#robot-frame-status').textContent = '双臂转换：逐帧记录读取失败';
  updateFrameStatus(pendingFrame);
}
async function loadGeometry(selection, selectedClip) {
  const fetchJson = async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`资源无法读取 (${response.status})`);
    return response.json();
  };
  if (!selectedClip.geometry) {
    $('#geometry-summary').textContent = '此样本尚未接入位姿逐帧记录。';
    $('#geometry-frame-info').textContent = '位姿记录缺失；已接入视频仍可播放。';
    $('#pose-frame-status').textContent = selectedClip.state ? '位姿：源末端记录，物体位姿待标' : '位姿：记录缺失';
    return;
  }
  const [poseResult, auditResult] = await Promise.allSettled([
    fetchJson(selectedClip.geometry.poses), fetchJson(selectedClip.geometry.validation)
  ]);
  if (selection !== generation) return;
  try {
    if (poseResult.status !== 'fulfilled') throw poseResult.reason;
    const rows = poseResult.value.frames;
    if (!Array.isArray(rows) || rows.length !== selectedClip.frames || rows.some((row, i) =>
      row.frame_index !== i || !Number.isFinite(row.timestamp_s) || Math.abs(row.timestamp_s - i / selectedClip.fps) > 1 / selectedClip.fps || !poseLabels[row.status])) {
      throw new Error('逐帧记录与当前视频不一致');
    }
    poseRows = rows;
    const counts = {};
    for (const row of rows) counts[row.status] = (counts[row.status] || 0) + 1;
    $('#geometry-summary').textContent = `完整 ${rows.length} 帧已处理；${counts.proxy_review_required || 0} 帧通过一致性筛查，仍待复核。点击色带或输入帧号，同步定位全部视频。`;
    for (const [key, [label, meaning, color]] of Object.entries(poseLabels)) {
      const tr = element('tr');
      for (const value of [label, counts[key] || 0, meaning]) tr.append(element('td', '', String(value)));
      $('#geometry-counts').append(tr);
      const legend = element('span', '', label), dot = element('i');
      dot.style.backgroundColor = color; legend.prepend(dot); $('#geometry-legend').append(legend);
    }
    const canvas = $('#geometry-timeline'), ctx = canvas.getContext('2d');
    rows.forEach((row, i) => {
      ctx.fillStyle = poseLabels[row.status][2];
      ctx.fillRect(i / rows.length * canvas.width, 0, Math.max(1, canvas.width / rows.length), canvas.height);
    });
    $('#geometry-rejected').disabled = !counts.rejected_proxy;
    updateFrameStatus(pendingFrame);
    if (auditResult.status === 'fulfilled') {
      const audit = auditResult.value;
      $('#depth-quality-note').textContent = `尺度来自单目模型估计，尚未标定。相邻焦距变化超过 10% 的帧对有 ${audit.adjacent_focal_changes_over_10pct?.length ?? '未知'} 对；全画面深度中位数变化超过 10% 的有 ${audit.adjacent_median_depth_changes_over_10pct?.length ?? '未知'} 对。这些是复核线索，尚未区分相机、场景变化与模型误差。深度视频的颜色范围逐帧缩放，不能直接当作米制运动。`;
    } else $('#depth-quality-note').textContent = '尺度与时序质检明细暂时无法读取；单目尺度未经标定。';
  } catch (error) {
    poseRows = null;
    $('#pose-frame-status').textContent = '位姿：记录读取失败';
    $('#geometry-summary').textContent = `逐帧位姿暂时无法显示：${error.message}。视频仍可独立查看。`;
    $('#geometry-frame-info').textContent = '逐帧状态不可用，不推断当前帧质量。';
  }
}
async function playAll() {
  if (synchronized) { pauseAll(); return; }
  const playable = videos.filter(video => video.readyState >= 1 && !video.error);
  if (!playable.length) return;
  message('');
  seekFrame(Number($('#seek').value));
  synchronized = true;
  ++playRequest;
  buffering = true;
  message('等待全部视频就绪…'); syncButton();
  await resumeGroup();
}
$('#play').addEventListener('click', playAll);
$('#seek').addEventListener('input', event => { pauseAll(); seekFrame(Number(event.target.value)); });
function jumpSegment(index) {
  if (!semanticSegments) return;
  const s = semanticSegments[Math.max(0,Math.min(semanticSegments.length-1,index))];
  pauseAll();seekFrame(s.start_frame);
}
$('#semantic-segment').addEventListener('change',event=>jumpSegment(Number(event.target.value)));
$('#semantic-prev').addEventListener('click',()=>jumpSegment(Number($('#semantic-segment').value)-1));
$('#semantic-next').addEventListener('click',()=>jumpSegment(Number($('#semantic-segment').value)+1));
$('#semantic-timeline').addEventListener('click',event=>{
  if (!semanticSegments) return;
  const rect=event.currentTarget.getBoundingClientRect();
  pauseAll();seekFrame(Math.floor((event.clientX-rect.left)/rect.width*clip.frames));
});
$('#frame-input').addEventListener('input', () => { frameInputDirty = true; });
$('#frame-jump-form').addEventListener('submit', event => {
  event.preventDefault();
  const frame = Number($('#frame-input').value);
  frameInputDirty = false;
  pauseAll(); seekFrame(frame);
});
$('#geometry-rejected').addEventListener('click', () => {
  if (!poseRows) return;
  const current = Number($('#seek').value);
  const row = poseRows.find(item => item.frame_index > current && item.status === 'rejected_proxy') || poseRows.find(item => item.status === 'rejected_proxy');
  if (row) { pauseAll(); seekFrame(row.frame_index); }
});
for (const selector of ['#coverage', '#geometry-timeline']) {
  $(selector).addEventListener('click', event => {
    if (!clip || (selector === '#geometry-timeline' && !poseRows)) return;
    const rect = event.currentTarget.getBoundingClientRect();
    pauseAll(); seekFrame(Math.floor((event.clientX - rect.left) / rect.width * clip.frames));
  });
}
$('#previous').addEventListener('click', () => { pauseAll(); seekFrame(Number($('#seek').value) - 1); });
$('#next').addEventListener('click', () => { pauseAll(); seekFrame(Number($('#seek').value) + 1); });
$('#speed').addEventListener('change', event => videos.forEach(video => { video.playbackRate = Number(event.target.value); }));
for (const [selector, key] of [['#jump-missing', 'missing_spans'], ['#jump-review', 'review_spans']]) {
  $(selector).addEventListener('click', () => {
    if (!quality) return;
    const spans = quality[key] || [];
    if (!spans.length) return;
    const current = Number($('#seek').value);
    pauseAll();
    seekFrame((spans.find(([start]) => start > current) || spans[0])[0]);
    $('#annotations').scrollIntoView({behavior: 'smooth'});
  });
}
$('#sample').addEventListener('change', event => selectClip(event.target.value));
$('#dataset').addEventListener('change', event => {
  const items = manifest.clips.filter(item => item.dataset === event.target.value);
  $('#sample').replaceChildren(...items.map(item => new Option(item.label, item.id)));
  selectClip(items[0].id);
});
async function init() {
  try {
    const response = await fetch('assets/multi_manifest.json', {cache: 'no-store'});
    if (!response.ok) throw new Error('资源清单无法读取');
    manifest = await response.json();
    if (!manifest.clips?.length) throw new Error('资源清单中没有样本');
    $('#dataset').replaceChildren(...[...new Set(manifest.clips.map(item => item.dataset))].map(name => new Option(name, name)));
    $('#sample').replaceChildren(...manifest.clips.filter(item => item.dataset === $('#dataset').value).map(item => new Option(item.label, item.id)));
    $('#dataset').disabled = $('#sample').disabled = false;
    $('#sample-count').textContent = `${manifest.clips.length} 个样本 · 人类 ${manifest.clips.filter(c=>c.domain==='human').length} / 真机 ${manifest.clips.filter(c=>c.domain==='robot').length} / UMI ${manifest.clips.filter(c=>c.domain==='umi').length}`;
    semanticTaxonomy = await (await fetch(manifest.taxonomy)).json();
    for (const [value, p] of Object.entries(semanticTaxonomy.phases)) $('#edit-phase').append(new Option(p.label, value));
    for (const verb of semanticTaxonomy.verbs) $('#edit-verb').append(new Option(verb, verb));
    const requested = new URLSearchParams(location.search).get('clip');
    if (requested && manifest.clips.some(c=>c.id===requested)) { const target=manifest.clips.find(c=>c.id===requested); $('#dataset').value=target.dataset; $('#sample').replaceChildren(...manifest.clips.filter(c=>c.dataset===target.dataset).map(c=>new Option(c.label,c.id))); $('#sample').value=requested; }
    selectClip($('#sample').value);
    const layers = manifest.clips.flatMap(item => item.layers);
    const stats = [[manifest.clips.length, '已接入三类样本'], [`${layers.filter(layer => layer.video).length} / ${layers.length}`, '视频资源已接入'], [manifest.clips.filter(item => item.loose_verified === true).length, '宽松标准已确认通过'], [manifest.clips.filter(item => item.strict_verified === true).length, '严格标准已确认通过']];
    $('#metrics').replaceChildren(...stats.map(([value, label]) => {
      const node = element('div', 'metric');
      node.append(element('strong', '', String(value)), element('span', '', label));
      return node;
    }));
    $('#dataset-rows').replaceChildren(...manifest.clips.map(item => {
      const row = document.createElement('tr');
      const issue = item.cleaner_summary?.notes?.join('；');
      for (const text of [`${item.dataset} · ${item.label}`, `${item.layers.filter(layer => layer.video).length} / ${item.layers.length}`, item.cleaner_summary ? `轨迹规则 ${item.cleaner_summary.state_score} / 100` : '自动证据筛查', '精度未验收', issue || '按标签层筛查；缺失与低分保留弃权']) row.append(element('td', '', text));
      return row;
    }));
  } catch (error) {
    $('#page-status').textContent = `页面暂时无法读取数据：${error.message}。请通过本地服务打开此网页。`;
    $('#sample-count').textContent = '读取失败';
  }
}
init();

function validSegments(rows, frames) {
  return Array.isArray(rows) && rows.length > 0 && rows[0].start_frame === 0 && rows.at(-1).end_frame_exclusive === frames && rows.every((s,i)=>Number.isInteger(s.start_frame) && Number.isInteger(s.end_frame_exclusive) && s.end_frame_exclusive>s.start_frame && (!i || s.start_frame===rows[i-1].end_frame_exclusive));
}
function renderSemanticRows() {
  const rows=semanticSegments;
  $('#semantic-segment').replaceChildren();$('#semantic-rows').replaceChildren();
  $('#semantic-summary').textContent=`${rows.length} 个连续动作片段 · 同一语义词表 · 全部待独立复核。点击时间带或片段同步定位。`;
  const canvas=$('#semantic-timeline'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  rows.forEach((s,i)=>{
    ctx.fillStyle=s.color || '#899ba2';ctx.fillRect(s.start_frame/clip.frames*canvas.width,0,(s.end_frame_exclusive-s.start_frame)/clip.frames*canvas.width,canvas.height);
    ctx.fillStyle='#fff';ctx.fillRect(s.start_frame/clip.frames*canvas.width,0,1,canvas.height);
    $('#semantic-segment').append(new Option(`${String(i+1).padStart(2,'0')} · ${fmt(s.start_s)} · ${s.label}`,String(i)));
    const tr=element('tr');for(const v of [String(i+1)+(s.cycle?` / 演示 ${s.cycle}`:''),`${fmt(s.start_s)}–${fmt(s.end_s)}`,s.label])tr.append(element('td','',v));
    const td=element('td'),b=element('button','text-button','跳转');b.addEventListener('click',()=>jumpSegment(i));td.append(b);tr.append(td);$('#semantic-rows').append(tr);
  });
  for(const id of ['#semantic-segment','#semantic-prev','#semantic-next'])$(id).disabled=false;
  editorIndex=-1;updateFrameStatus(pendingFrame);
}
function populateEditor(index){
  if(!semanticSegments?.[index])return;editorIndex=index;const s=semanticSegments[index];
  for(const [field,key] of [['start','start_frame'],['end','end_frame_exclusive'],['phase','phase'],['verb','verb'],['object','object'],['target','target'],['label','label'],['before','state_before'],['after','state_after']])$('#edit-'+field).value=s[key]??'unknown';
  $('#edit-note').value=s.reviewer_note || '';
  $('#segment-evidence').textContent=JSON.stringify({actor:s.actor,roles:s.roles,before:s.state_before,after:s.state_after,contact:s.contact_transition,evidence:s.evidence,boundary_uncertainty_s:s.boundary_uncertainty_s,review_status:s.review_status},null,2);
}
function saveDraft(rows,change){
  if(!validSegments(rows,clip.frames))throw new Error('片段必须连续覆盖完整视频，且起止帧合法。');
  clearExport();
  rows.forEach(s=>{s.start_s=s.start_frame/clip.fps;s.end_s=s.end_frame_exclusive/clip.fps;});
  const draft={...semanticDocument,base_source:clip.semantics,episode_id:clip.id,segments:rows,training_eligible:false,independent_human_verification:false,revision:(semanticDocument?.revision||0)+1,edit_history:[...(semanticDocument?.edit_history||[]),{at:new Date().toISOString(),change}]};
  localStorage.setItem(`unified-draft:${clip.id}`,JSON.stringify(draft));semanticDocument=draft;semanticSegments=rows;renderSemanticRows();$('#edit-status').textContent='草稿已保存到当前浏览器。时间带已更新；预渲染语义视频仍对应发布版。请导出 JSON 留档。';
}
$('#edit-form').addEventListener('submit',e=>{e.preventDefault();if(!semanticSegments)return;try{
  const rows=structuredClone(semanticSegments),i=editorIndex,s=rows[i],start=Number($('#edit-start').value),end=Number($('#edit-end').value);
  if((i===0&&start!==0)||(i===rows.length-1&&end!==clip.frames)||start>=end||!Number.isInteger(start)||!Number.isInteger(end))throw new Error('首段必须从 0 开始，末段必须到视频总帧数，起止帧需为递增整数。');
  if(i>0){if(start<=rows[i-1].start_frame)throw new Error('起始帧越过前一片段。');rows[i-1].end_frame_exclusive=start;rows[i-1].review_status='edited_draft';}
  if(i<rows.length-1){if(end>=rows[i+1].end_frame_exclusive)throw new Error('结束帧越过下一片段。');rows[i+1].start_frame=end;rows[i+1].review_status='edited_draft';}
  Object.assign(s,{start_frame:start,end_frame_exclusive:end,phase:$('#edit-phase').value,verb:$('#edit-verb').value,object:$('#edit-object').value,target:$('#edit-target').value,label:$('#edit-label').value,state_before:$('#edit-before').value,state_after:$('#edit-after').value,reviewer_note:$('#edit-note').value,review_status:'edited_draft',review_required:true});s.color=semanticTaxonomy.phases[s.phase].color;
  saveDraft(rows,{type:'edit',segment_id:s.id});
}catch(err){$('#edit-status').textContent=err.message;}});
$('#segment-split').addEventListener('click',()=>{if(!semanticSegments)return;try{const rows=structuredClone(semanticSegments),i=editorIndex,s=rows[i],f=pendingFrame;if(f<=s.start_frame||f>=s.end_frame_exclusive)throw new Error('请先把播放位置移到当前片段内部。');const next={...structuredClone(s),id:s.id+'-split-'+f,start_frame:f,review_status:'edited_draft'};s.end_frame_exclusive=f;s.review_status='edited_draft';rows.splice(i+1,0,next);saveDraft(rows,{type:'split',frame:f});}catch(e){$('#edit-status').textContent=e.message;}});
$('#segment-merge').addEventListener('click',()=>{if(!semanticSegments)return;try{const rows=structuredClone(semanticSegments),i=editorIndex;if(i===rows.length-1)throw new Error('当前已经是最后一段。');rows[i].end_frame_exclusive=rows[i+1].end_frame_exclusive;rows[i].review_status='edited_draft';rows[i].reviewer_note='合并片段后需重新核对描述与状态。';rows.splice(i+1,1);saveDraft(rows,{type:'merge',segment_id:rows[i].id});}catch(e){$('#edit-status').textContent=e.message;}});
let exportUrl = null;
function clearExport(){
  if(exportUrl)URL.revokeObjectURL(exportUrl);exportUrl=null;
  $('#export-panel').hidden=true;$('#export-json').value='';$('#export-file').removeAttribute('href');
}
$('#export-semantics').addEventListener('click',()=>{
  if(!semanticDocument)return;clearExport();
  const payload=JSON.stringify({...semanticDocument,segments:semanticSegments},null,2);
  exportUrl=URL.createObjectURL(new Blob([payload],{type:'application/json'}));
  $('#export-file').href=exportUrl;$('#export-file').download=clip.id+'-semantics.json';
  $('#export-json').value=payload;$('#export-panel').hidden=false;
  $('#edit-status').textContent='语义 JSON 已生成。点击下载链接保存；也可复制下方内容。';
});
$('#copy-export').addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText($('#export-json').value);$('#edit-status').textContent='已复制完整语义 JSON，可粘贴到文件留档。';}
  catch(_){$('#edit-status').textContent='浏览器未允许直接复制，请选中下方 JSON 手动复制。';}
});
async function loadNative(selection,c){
  const results=await Promise.allSettled([fetch(c.state).then(r=>{if(!r.ok)throw new Error();return r.json();}),fetch(c.qc).then(r=>{if(!r.ok)throw new Error();return r.json();})]);if(selection!==generation)return;
  if(results[0].status==='fulfilled'&&results[0].value.frames===c.frames){nativeState=results[0].value;updateNativeFrame(pendingFrame);}else $('#native-summary').textContent='原生时序读取失败，不推断当前帧。';
  if(results[1].status==='fulfilled')$('#native-qc').textContent=JSON.stringify(results[1].value,null,2);
  $('#native-links').replaceChildren(...[['数据来源',c.provenance],['逐样本质检',c.qc],['统一时序',c.state]].map(([t,u])=>{const a=element('a','',t+' ↗');a.href=u;return a;}));
}
function updateNativeFrame(f){
  if(!nativeState?.position_m[f])return;
  $('#native-current').textContent=`帧 ${f} · `+nativeState.actors.map((arm,j)=>`${arm}：${nativeState.valid[f][j]?(nativeState.source_kind==='umi'?'源标记有效':'数值可读，精度未验收'):'源记录无效'}，速度 ${nativeState.speed_valid[f][j]?nativeState.speed_m_s[f][j].toFixed(3)+' m/s':'证据不足'}${nativeState.gripper?'，夹爪 '+nativeState.gripper[f][j].toFixed(3)+' '+nativeState.gripper_unit:''}`).join('；');
}

$('#draft-reset').addEventListener('click',()=>{ if(!clip)return; clearExport();localStorage.removeItem(`unified-draft:${clip.id}`); editorIndex=-1; semanticSegments=null; semanticDocument=null; $('#edit-status').textContent='已恢复发布版；本地草稿已清除。';loadTemporalAnnotations(generation,clip); });
