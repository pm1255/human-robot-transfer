'use strict';
(async () => {
  const $ = id => document.getElementById(id);
  const labels = {
    proxy_review_required: ['通过筛查，待复核', '仅符合当前轮廓、深度与跳变阈值', '#89b4ac'],
    rejected_proxy: ['候选被拒绝', '轮廓、深度或运动不一致，正式位姿留空', '#d6ac61'],
    waiting_for_reinitialization: ['等待重新初始化', '失败后每 30 帧重试一次，中间不填充位姿', '#aab6be'],
    insufficient_mask_or_depth: ['输入不足', '物体掩码或有效深度不足', '#85949e'],
    before_reference: ['参考帧之前', '尚未初始化', '#e1e7eb']
  };
  const video = $('geometry-video');
  video.addEventListener('error', () => { $('geometry-video-status').textContent = '视频暂时无法加载，请查看原六类标注页或完整质检记录。'; });
  try {
    const [audit, pose] = await Promise.all(['geometry/geometry_validation.json', 'geometry/pose6d.json'].map(async url => {
      const response = await fetch(url); if (!response.ok) throw new Error('结果文件尚未就绪'); return response.json();
    }));
    const rows = pose.frames;
    const fps = 30;
    $('geometry-summary').textContent = `完整 ${audit.frames} 帧已处理；${audit.accepted_proxy_count} 帧通过当前一致性筛查，其他帧保留拒绝或缺失状态。尚无独立真值精度验证。`;
    $('geometry-frame').max = rows.length - 1;
    for (const [key, count] of Object.entries(audit.status_counts)) {
      const tr = document.createElement('tr');
      for (const value of [labels[key]?.[0] || key, count, labels[key]?.[1] || '请查看原始记录']) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      $('geometry-counts').append(tr);
    }
    const canvas = $('geometry-timeline'), ctx = canvas.getContext('2d');
    rows.forEach((row, i) => { ctx.fillStyle = labels[row.status]?.[2] || '#ccc'; ctx.fillRect(i / rows.length * canvas.width, 0, Math.max(1, canvas.width / rows.length), canvas.height); });
    const describe = i => {
      const row = rows[i], q = row.quality;
      let details = `第 ${i} 帧 · ${row.timestamp_s.toFixed(2)} 秒 · ${labels[row.status]?.[0] || row.status}`;
      if (q?.mask_iou !== undefined) details += ` · 轮廓 IoU ${q.mask_iou.toFixed(3)}`;
      if (q?.relative_depth_error != null) details += ` · 相对深度残差 ${(q.relative_depth_error * 100).toFixed(2)}%`;
      if (row.status === 'rejected_proxy') {
        const reasons = {
          temporal_jump: '相邻位姿跳变', silhouette_mismatch_deformation_or_tracking_error: '轮廓不匹配',
          depth_mismatch: '深度不匹配', insufficient_depth_overlap: '深度重叠不足', invalid_transform: '变换矩阵无效'
        };
        details += ' · 原因：' + q.flags.filter(flag => reasons[flag]).map(flag => reasons[flag]).join('、');
      }
      $('geometry-frame-info').textContent = details;
    };
    const jump = i => { i = Math.max(0, Math.min(rows.length - 1, Math.round(i))); $('geometry-frame').value = i; video.currentTime = rows[i].timestamp_s; describe(i); };
    canvas.addEventListener('click', event => { const rect = canvas.getBoundingClientRect(); jump((event.clientX - rect.left) / rect.width * rows.length); });
    $('geometry-jump').addEventListener('click', () => { const i = Number($('geometry-frame').value); if (Number.isFinite(i)) jump(i); });
    $('geometry-rejected').addEventListener('click', () => { const current = Math.floor(video.currentTime * fps); const row = rows.find(r => r.frame_index > current && r.status === 'rejected_proxy') || rows.find(r => r.status === 'rejected_proxy'); if (row) jump(row.frame_index); });
    video.addEventListener('timeupdate', () => describe(Math.min(rows.length - 1, Math.floor(video.currentTime * fps))));
    describe(0);
  } catch (error) { $('geometry-summary').textContent = '完整结果尚未就绪。请通过 TensorBoard 查看实际进度。'; }
})();
