'use strict';
// Manifest-driven review and native video controls follow human2robot/web/app.js.
const $ = selector => document.querySelector(selector);
let manifest, clip, quality, videos = [], master, synchronized = false, syncing = false;
let generation = 0, playRequest = 0, pendingFrame = 0;
let buffering = false;
let poseRows = null;
let semanticSegments = null, robotRows = null, compositeRows = null;
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
  if (document.activeElement !== $('#frame-input')) $('#frame-input').value = frame;
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
  quality = clip.hand_quality;
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
  $('#page-status').textContent = `已接入 ${available} / ${clip.layers.length} 路独立视频。${missing ? `${missing}视频缺失。` : ''}${clip.semantics ? '语义已按动作片段切换，时间边界待复核。' : incomplete ? `${incomplete}尚未补全。` : ''}${clip.robot ? '双臂转换为运动学试验，未完成物理执行验证。' : ''}衣物位姿为整体近似。`;
  $('#strict-quality').textContent = `${incomplete || missing ? `${incomplete || missing}尚未补全。` : ''}语义边界、手型与单目深度尚未独立验收。衣物整体 6D 位姿不能描述布料每个点的运动。双臂预览不证明抓持接触或折衣成功。`;
  $('#loose-quality').textContent = `关节点视频与手型视频均已接入。关节点 ${clip.skeleton_frames}/${clip.frames} 帧有检测；手型 ${quality.mesh_frames}/${clip.frames} 帧有输出，其中 ${quality.review_frames} 帧需复核，不能仅凭覆盖率认定标注准确。`;
  updateTime(pendingFrame / clip.fps);
  renderCoverage();
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
  const contains = spans => (spans || []).some(([start, end]) => frame >= start && frame <= end);
  $('#hand-frame-status').textContent = `手型：${contains(quality.missing_spans) ? '未检测' : contains(quality.review_spans) ? '有输出，需复核' : '有输出，精度未验收'}`;
  if (semanticSegments) {
    const index = semanticSegments.findIndex(s => frame >= s.start_frame && frame < s.end_frame_exclusive);
    const s = semanticSegments[index];
    if (s) {
      $('#semantic-frame-status').textContent = `语义：${s.label}`;
      $('#semantic-detail').textContent = `片段 ${index + 1} / ${semanticSegments.length} · 第 ${s.cycle} 次演示 · ${fmt(s.start_s)}–${fmt(s.end_s)} · ${s.label} · ${s.verb} · 边界约 ±0.5 秒，待复核`;
      $('#semantic-segment').value = String(index);
    }
  }
  if (robotRows) {
    const row = robotRows[frame], hands = Object.entries(row?.hands || {});
    const needsReview = hands.some(([, h]) => h.status === 'tracking_error_review');
    const belowTable = hands.some(([, h]) => h.link_point_table_clearance_m < 0);
    $('#robot-frame-status').textContent = hands.length ? `机械臂试验：${hands.map(([side, h]) => `${side === 'Left' ? '左' : '右'} ${(h.error_m * 100).toFixed(1)} cm / ${(h.rotation_error_rad * 180 / Math.PI).toFixed(0)}°`).join('；')} 拟合残差 · ${needsReview ? '偏差超阈值' : '通过拟合筛查'}${belowTable ? ' · 连杆点低于桌面' : ''}${hands.some(([, h]) => h.reinitialized) ? ' · 重初始化' : ''}` : '机械臂试验：手部或桌面数据不足';
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
      semanticSegments = rows;
      $('#semantic-summary').textContent = `${rows.length} 个动作片段，${new Set(rows.map(s => s.cycle)).size} 次重复演示。点击色带、选择片段或切换上一/下一段，全部视频同步定位。`;
      const canvas = $('#semantic-timeline'), ctx = canvas.getContext('2d');
      rows.forEach((s,i) => {
        ctx.fillStyle = s.color;ctx.fillRect(s.start_frame / clip.frames * canvas.width,0,(s.end_frame_exclusive-s.start_frame)/clip.frames*canvas.width,canvas.height);
        ctx.fillStyle = '#ffffff';ctx.fillRect(s.start_frame / clip.frames * canvas.width,0,1,canvas.height);
        $('#semantic-segment').append(new Option(`${String(i+1).padStart(2,'0')} · ${fmt(s.start_s)} · ${s.label}`,String(i)));
        const tr = element('tr');
        for (const value of [`${i+1} / 第 ${s.cycle} 次`,`${fmt(s.start_s)}–${fmt(s.end_s)}`,s.label]) tr.append(element('td','',value));
        const td = element('td'), button = element('button','text-button','跳转');
        button.addEventListener('click',()=>{pauseAll();seekFrame(s.start_frame);});td.append(button);tr.append(td);$('#semantic-rows').append(tr);
      });
      for (const id of ['#semantic-segment', '#semantic-prev', '#semantic-next']) $(id).disabled = false;
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
    $('#pose-frame-status').textContent = '位姿：记录缺失';
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
$('#frame-jump-form').addEventListener('submit', event => {
  event.preventDefault();
  const frame = Number($('#frame-input').value);
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
    $('#sample-count').textContent = `${manifest.clips.length} 个样本`;
    selectClip($('#sample').value);
    const layers = manifest.clips.flatMap(item => item.layers);
    const stats = [[manifest.clips.length, '已接入人类样本'], [`${layers.filter(layer => layer.video).length} / ${layers.length}`, '视频资源已接入'], [manifest.clips.filter(item => item.loose_verified === true).length, '宽松标准已确认通过'], [manifest.clips.filter(item => item.strict_verified === true).length, '严格标准已确认通过']];
    $('#metrics').replaceChildren(...stats.map(([value, label]) => {
      const node = element('div', 'metric');
      node.append(element('strong', '', String(value)), element('span', '', label));
      return node;
    }));
    $('#dataset-rows').replaceChildren(...manifest.clips.map(item => {
      const row = document.createElement('tr');
      for (const text of [item.id, `${item.layers.filter(layer => layer.video).length} / ${item.layers.length}`, '待复核', '待独立验收', '复核片段边界、遮挡、位姿与机器人接触']) row.append(element('td', '', text));
      return row;
    }));
  } catch (error) {
    $('#page-status').textContent = `页面暂时无法读取数据：${error.message}。请通过本地服务打开此网页。`;
    $('#sample-count').textContent = '读取失败';
  }
}
init();
