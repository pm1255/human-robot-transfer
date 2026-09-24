const $ = (id) => document.getElementById(id),
  esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const handEdges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
  ],
  bodyEdges = [
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [11, 23],
    [12, 24],
    [23, 24],
    [23, 25],
    [25, 27],
    [24, 26],
    [26, 28],
    [27, 29],
    [29, 31],
    [28, 30],
    [30, 32],
  ];
let entries = [],
  annotation = null,
  current = null,
  predictions = {},
  bench = {},
  lastDrawKey = "",
  loadSerial = 0,
  rolloutSerial = 0,
  executionTrace = null;
async function json(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw Error(`${path}: ${r.status}`);
  return r.json();
}
function visible(p) {
  return (
    p &&
    p[2] > 0.6 &&
    p[3] > 0.6 &&
    p[0] >= 0 &&
    p[0] <= 1 &&
    p[1] >= 0 &&
    p[1] <= 1
  );
}
function lines(ctx, points, edges, color, project, valid = () => true) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
  for (const [a, b] of edges) {
    if (!valid(points[a]) || !valid(points[b])) continue;
    const A = project(points[a]),
      B = project(points[b]);
    ctx.beginPath();
    ctx.moveTo(...A);
    ctx.lineTo(...B);
    ctx.stroke();
  }
  points.forEach((p) => {
    if (valid(p)) {
      ctx.beginPath();
      ctx.arc(...project(p), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}
function gallery() {
  const f = $("filter").value,
    list = entries.filter(
      (x) =>
        f === "all" ||
        (f === "two" && x.both_hand_frames > 0) ||
        (f === "body" && x.full_body_frames > 0) ||
        (f === "ego" && x.view === "egocentric") ||
        (f === "exo" && x.view === "third_person") ||
        (f === "bottle" && x.bottle_frames > 0),
    );
  $("gallery").innerHTML = list
    .map(
      (x) =>
        `<button class="tile ${x.id === current?.id ? "active" : ""}" data-id="${esc(x.id)}"><img loading="lazy" src="v2data/media/${esc(x.id)}.jpg"><strong>${esc(x.id)}</strong><small>双手 ${x.both_hand_frames}/${x.frames} · 全身 ${x.full_body_frames} · 瓶子 ${x.bottle_frames}</small></button>`,
    )
    .join("");
  document
    .querySelectorAll(".tile")
    .forEach((e) => (e.onclick = () => select(e.dataset.id)));
  if (!list.length)
    $("gallery").innerHTML = "<p>没有符合条件的已完成标注。</p>";
}
async function select(id) {
  const serial = ++loadSerial;
  current = entries.find((x) => x.id === id);
  gallery();
  const nextAnnotation = await json(`v2data/annotations/${id}.json`);
  if (serial !== loadSerial) return;
  annotation = nextAnnotation;
  lastDrawKey = "";
  $("video").src = `v2data/media/${id}.mp4`;
  $("title").textContent = id;
  $("caption").textContent = current.task || "未提供任务指令";
  $("audit").innerHTML =
    `<p><b>${current.view === "egocentric" ? "第一人称" : "第三人称"}</b> · ${current.frames} 帧 · ${current.training_eligible ? "用于人类辅助训练" : "仅用于标注展示"}</p><p>双手：${current.both_hand_frames} 帧 / 全身候选：${current.full_body_frames} 帧</p><p>物体：${current.object_frames} 帧 / 瓶子：${current.bottle_frames} 帧</p><p>左右手：模型标签，镜像未知，未核实</p><p>局部 3D：学习先验估计，无世界坐标标定</p><p>人体质检：${esc(current.body_use_gate)}</p><p>直接机器人动作准入：否</p><p>来源：${esc(current.source_uri)}</p><p>SHA256：${esc(current.sha256?.slice(0, 20))}…</p>`;
  draw();
}
function draw() {
  if (!annotation) return;
  const v = $("video"),
    c = $("overlay");
  const key = [
    current?.id,
    Math.floor(v.currentTime * 10 + 1e-4),
    v.videoWidth,
    v.clientWidth,
    v.clientHeight,
    $("hands").checked,
    $("body").checked,
    $("objects").checked,
    $("headpoints").checked,
    $("yaw").value,
  ].join(":");
  if (key === lastDrawKey) return;
  lastDrawKey = key;
  c.width = v.clientWidth;
  c.height = v.clientHeight;
  const ctx = c.getContext("2d"),
    ratio = Math.min(
      c.width / (v.videoWidth || 640),
      c.height / (v.videoHeight || 480),
    ),
    w = (v.videoWidth || 640) * ratio,
    h = (v.videoHeight || 480) * ratio,
    ox = (c.width - w) / 2,
    oy = (c.height - h) / 2,
    project = (p) => [ox + p[0] * w, oy + p[1] * h];
  const frame =
    annotation.frames[
      Math.min(
        annotation.frames.length - 1,
        Math.max(0, Math.floor(v.currentTime * 10 + 1e-4)),
      )
    ];
  if (!frame) return;
  if ($("hands").checked)
    for (const hand of frame.hands) {
      const color = hand.side === "Left" ? "#48e5d5" : "#ffbd70";
      lines(ctx, hand.xy, handEdges, color, project);
      const p = project(hand.xy[0]);
      ctx.font = "bold 13px sans-serif";
      ctx.fillStyle = "#13251e";
      ctx.fillRect(p[0] - 3, p[1] - 24, 150, 21);
      ctx.fillStyle = color;
      ctx.fillText(`ID ${hand.track_id} · model ${hand.side}`, p[0], p[1] - 8);
    }
  if ($("body").checked && frame.body.length) {
    lines(
      ctx,
      frame.body,
      bodyEdges,
      "#a9e36c",
      project,
      (p) =>
        visible(p) && ($("headpoints").checked || frame.body.indexOf(p) >= 11),
    );
    if ($("headpoints").checked)
      frame.body.slice(0, 11).forEach((p, i) => {
        if (visible(p)) {
          ctx.font = "12px sans-serif";
          ctx.fillText(String(i), project(p)[0] + 4, project(p)[1] - 4);
        }
      });
  }
  if ($("objects").checked)
    for (const o of frame.objects) {
      const [x, y, bw, bh] = o.box,
        p = project([x, y]);
      ctx.strokeStyle = o.target_candidate ? "#ffe772" : "#e4ecff";
      ctx.lineWidth = 2;
      ctx.strokeRect(...p, bw * w, bh * h);
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#13251e";
      ctx.fillRect(p[0], p[1] - 19, 160, 19);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(
        `${o.label} ${o.score.toFixed(2)}${o.target_candidate ? " · 候选目标" : ""}`,
        p[0] + 3,
        p[1] - 5,
      );
    }
  $("frameinfo").innerHTML =
    `<span class="pill">${frame.t.toFixed(1)} s</span><span class="pill">${frame.hands.length} 只手</span><span class="pill">全身 ${frame.full_body_visible ? "模型判为可见" : "不完整 / 未检测"}</span><span class="pill">${frame.objects.length} 个物体框</span>`;
  const lc = $("local3d"),
    l = lc.getContext("2d");
  l.clearRect(0, 0, lc.width, lc.height);
  if (frame.body_local_xyz.length)
    lines(
      l,
      frame.body.map((p, i) => [
        ...frame.body_local_xyz[i],
        p[2],
        p[3],
        p[0],
        p[1],
        i,
      ]),
      bodyEdges,
      "#237d61",
      (p) => [
        200 +
          (p[0] * Math.cos((+$("yaw").value * Math.PI) / 180) +
            p[2] * Math.sin((+$("yaw").value * Math.PI) / 180)) *
            150,
        125 + p[1] * 150,
      ],
      (p) =>
        ($("headpoints").checked || p[7] >= 11) &&
        p[3] > 0.6 &&
        p[4] > 0.6 &&
        p[5] >= 0 &&
        p[5] <= 1 &&
        p[6] >= 0 &&
        p[6] <= 1,
    );
  else {
    l.fillStyle = "#69796a";
    l.font = "14px sans-serif";
    l.fillText("该帧没有可靠人体姿态", 100, 130);
  }
  window.H2RReview?.diagnose(annotation, frame);
}
function drawNext() {
  draw();
  requestAnimationFrame(drawNext);
}
$("filter").onchange = gallery;
$("best").onclick = () => {
  if (!annotation) return;
  let best = 0,
    score = -1;
  annotation.frames.forEach((f, i) => {
    const s =
      f.hands.length * 3 +
      Number(f.full_body_visible) * 7 +
      f.objects.filter((o) => o.target_candidate).length;
    if (s > score) {
      score = s;
      best = i;
    }
  });
  $("video").currentTime = best / 10;
};
function plot() {
  const data = predictions[$("condition").value];
  if (!data) return;
  const sample = data.samples[+$("sample").value || 0],
    j = +$("joint").value || 0,
    c = $("actionplot"),
    ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const p = sample.predicted.filter((_, i) => sample.mask[i]).map((x) => x[j]),
    t = sample.target.filter((_, i) => sample.mask[i]).map((x) => x[j]),
    vals = [...p, ...t],
    min = Math.min(...vals) - 0.03,
    max = Math.max(...vals) + 0.03,
    X = (i) => 65 + (i * (c.width - 90)) / Math.max(1, p.length - 1),
    Y = (v) => c.height - 40 - ((v - min) / (max - min)) * (c.height - 70);
  ctx.font = "20px sans-serif";
  for (let k = 0; k <= 4; k++) {
    const val = min + ((max - min) * k) / 4,
      y = Y(val);
    ctx.strokeStyle = "#dce2d7";
    ctx.beginPath();
    ctx.moveTo(60, y);
    ctx.lineTo(c.width - 20, y);
    ctx.stroke();
    ctx.fillStyle = "#536555";
    ctx.fillText(val.toFixed(3), 5, y + 4);
  }
  for (const [arr, col, dash] of [
    [p, "#147b61", []],
    [t, "#cf773d", [7, 5]],
  ]) {
    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.setLineDash(dash);
    ctx.beginPath();
    arr.forEach((v, i) =>
      i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)),
    );
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "#536555";
  for (let k = 0; k < p.length; k += 3)
    ctx.fillText(String(k + 1), X(k) - 5, c.height - 22);
  ctx.fillText(
    `${$("joint").selectedOptions[0].textContent} · ${[6, 13].includes(j) ? "夹爪原生标量" : "角度（rad）"}`,
    80,
    20,
  );
  ctx.fillText("预测的未来动作序号", c.width / 2 - 90, c.height - 3);
  $("actioninfo").textContent =
    `episode ${sample.episode} / frame ${sample.frame} · 有效 ${p.length}/${sample.mask.length} 步 · 此维 MAE ${(p.reduce((s, v, i) => s + Math.abs(v - t[i]), 0) / p.length).toFixed(4)} ${[6, 13].includes(j) ? "夹爪原生单位" : "rad"}`;
}
function chooseCondition() {
  const d = predictions[$("condition").value];
  $("sample").innerHTML = (d?.samples || [])
    .map(
      (s, i) =>
        `<option value="${i}">片段 ${i + 1} · ep ${s.episode}/${s.frame}</option>`,
    )
    .join("");
  plot();
}
async function loadRollout() {
  const serial = ++rolloutSerial;
  executionTrace = null;
  const r = (bench.rollouts || [])[+$("rollout").value];
  if (!r) return;
  $("rolloutvideo").src = r.video;
  $("rolloutdownload").href = r.video;
  $("rolloutinfo").textContent =
    `${r.diagnostic ? "开发场景诊断，排除正式成功率；与左侧模型选择不自动对应。" : "正式留出评估。"}${r.condition} · seed ${r.seed} · ${r.valid ? (r.success ? "任务成功" : "任务未成功") : "环境/执行异常"} · ${r.steps ?? 0} 条动作命令。视频 15 帧/秒为回放速度，每条命令实际物理时长不固定。${r.checkpoint_sha256 ? " 权重 SHA256: " + r.checkpoint_sha256.slice(0, 16) + "…" : " 历史录像未记录权重哈希。"}`;
  $("executionmetrics").textContent = "此录像没有可用动作轨迹。";
  for (const id of ["executionplot", "bottleplot"]) {
    const c = $(id);
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
  }
  if (r.trace) {
    const d = await json(r.trace);
    if (serial !== rolloutSerial) return;
    executionTrace = d;
    execution(d);
    window.H2RReview?.execution(d, r);
  }
}
async function benchmark() {
  try {
    bench = await json("v2data/benchmark.json");
    try {
      const dev = await json("v2data/development_rollouts.json");
      bench.rollouts = [...(bench.rollouts || []), ...dev];
    } catch (e) {
      console.warn("开发录像未载入", e);
    }
    $("experiment").textContent = bench.protocol;
    $("interpretation").textContent = bench.interpretation;
    $("results").innerHTML =
      "<table><thead><tr><th>条件 / seed</th><th>闭环成功率</th><th>95% 区间</th><th>微调前 → 后 RMSE</th><th>P95 偏差</th><th>夹爪 MAE</th><th>保持当前姿态 RMSE</th></tr></thead><tbody>" +
      bench.rows
        .map(
          (r) =>
            `<tr><td>${esc(r.name)}</td><td>${r.closed_loop ? `${r.closed_loop.successes}/${r.closed_loop.valid} (${(r.closed_loop.success_rate * 100).toFixed(1)}%)` : "未完成，不代填"}</td><td>${r.ci ? r.ci.map((v) => (v * 100).toFixed(1)).join("–") + "%" : "—"}</td><td>${r.pretrain_offline?.joint_rmse_rad.toFixed(4) ?? "—"} → ${r.offline?.joint_rmse_rad.toFixed(4) ?? "—"} rad</td><td>${r.offline?.joint_p95_rad.toFixed(4) ?? "—"} rad</td><td>${r.offline?.gripper_mae_native.toFixed(4) ?? "—"}</td><td>${r.offline?.persistence.joint_rmse_rad.toFixed(4) ?? "—"} rad</td></tr>`,
        )
        .join("") +
      "</tbody></table>";
    for (const row of bench.rows)
      if (row.predictions) predictions[row.name] = await json(row.predictions);
    $("condition").innerHTML = Object.keys(predictions)
      .map((k) => `<option>${esc(k)}</option>`)
      .join("");
    $("joint").innerHTML = Array.from(
      { length: 14 },
      (_, i) =>
        `<option value="${i}">${i < 7 ? "左" : "右"}${[6, 13].includes(i) ? "夹爪" : "关节 " + ((i % 7) + 1)}</option>`,
    ).join("");
    chooseCondition();
    $("rollout").innerHTML = (bench.rollouts || [])
      .map(
        (r, i) =>
          `<option value="${i}">${r.diagnostic ? "[开发诊断] " : "[正式评估] "}${esc(r.condition)} · seed ${r.seed}</option>`,
      )
      .join("");
    loadRollout();
    $("futureimages").innerHTML = (bench.future_images || [])
      .map(
        (f) =>
          `<div><b>${esc(f.seed)}</b><img src="${esc(f.image)}" alt="当前、预测和真实未来图像比较"></div>`,
      )
      .join("");
  } catch (e) {
    $("experiment").textContent =
      "训练/闭环结果尚未同步完成；不使用旧版离线 loss 冒充本轮任务成功率。";
    console.warn(e);
  }
}
$("condition").onchange = chooseCondition;
$("sample").onchange = plot;
$("joint").onchange = () => {
  plot();
  if (executionTrace) execution(executionTrace);
};
$("rollout").onchange = loadRollout;
(async () => {
  try {
    entries = await json("v2data/manifest.json");
    $("stats").innerHTML = [
      [entries.length, "真实视频例子"],
      [entries.filter((x) => x.both_hand_frames > 0).length, "含双手检测片段"],
      [
        entries.filter((x) => x.full_body_frames > 0).length,
        "含全身候选帧片段",
      ],
      [
        entries.reduce((s, x) => s + x.frames, 0).toLocaleString(),
        "已标注视频帧",
      ],
    ]
      .map(
        ([n, t]) =>
          `<div class="stat"><strong>${n}</strong><span>${t}</span></div>`,
      )
      .join("");
    gallery();
    if (entries.length)
      await select(
        entries.find((x) => x.both_hand_frames > 0)?.id || entries[0].id,
      );
    drawNext();
    await benchmark();
  } catch (e) {
    $("stats").innerHTML =
      `<p class="error">数据读取失败：${esc(e.message)}</p>`;
  }
})();

function execution(d) {
  const c = $("executionplot"),
    ctx = c.getContext("2d"),
    j = +$("joint").value || 0;
  ctx.clearRect(0, 0, c.width, c.height);
  const a = d.action.map((x) => x[j]),
    s = (d.physical_after || d.state).map((x) => x[j]),
    raw = (d.raw_action || d.action).map((x) => x[j]);
  const lo = Math.min(...a, ...s, ...raw) - 0.03,
    hi = Math.max(...a, ...s, ...raw) + 0.03,
    X = (i) => 80 + (i * (c.width - 110)) / Math.max(1, a.length - 1),
    Y = (v) => c.height - 50 - ((v - lo) / (hi - lo)) * (c.height - 90);
  ctx.font = "16px sans-serif";
  ctx.fillStyle = "#536555";
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4,
      y = Y(v),
      n = Math.round(((a.length - 1) * k) / 4);
    ctx.strokeStyle = "#dce2d7";
    ctx.beginPath();
    ctx.moveTo(80, y);
    ctx.lineTo(c.width - 30, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(3), 5, y + 5);
    ctx.fillText(String(n), X(n) - 10, c.height - 25);
  }
  for (const [arr, color] of [
    [raw, "#9464b1"],
    [a, "#147b61"],
    [s, "#cf773d"],
  ]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    arr.forEach((v, i) =>
      i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)),
    );
    ctx.stroke();
  }
  const step = Math.min(
    a.length - 1,
    Math.floor($("rolloutvideo").currentTime * 15),
  );
  ctx.strokeStyle = "#263b30";
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(X(step), 30);
  ctx.lineTo(X(step), c.height - 50);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#536555";
  ctx.fillText(
    `${$("joint").selectedOptions[0].textContent} · ${[6, 13].includes(j) ? "夹爪原生标量" : "角度（rad）"}`,
    80,
    19,
  );
  ctx.fillText(`执行命令序号 · 当前 ${step}`, c.width / 2 - 110, c.height - 4);
}

$("rolloutvideo").addEventListener("timeupdate", () => {
  if (executionTrace) execution(executionTrace);
});
