/* Evidence views: no inferred missingness causes and no fabricated robot actions. */
(() => {
  const el = (id) => document.getElementById(id);
  const finite = (p) => Array.isArray(p) && p.every(Number.isFinite);
  const position = (T) =>
    T && T.length === 4 && finite([T[0][3], T[1][3], T[2][3]])
      ? [T[0][3], T[1][3], T[2][3]]
      : null;
  const fmt = (n, digits = 3) =>
    Number.isFinite(n) ? n.toFixed(digits) : "未记录";
  function path(ctx, points, color, width = 2, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let start = true;
    for (const p of points) {
      if (!p || !finite(p)) {
        start = true;
        continue;
      }
      if (start) ctx.moveTo(...p);
      else ctx.lineTo(...p);
      start = false;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function dot(ctx, p, color, r = 4) {
    if (!p) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(...p, r, 0, 2 * Math.PI);
    ctx.fill();
  }
  function projection(points, w, h) {
    const iso = (p) => [
      p[0] - 0.6 * p[1],
      -0.85 * p[2] + 0.25 * p[0] + 0.3 * p[1],
    ];
    const ps = points.filter(finite).map(iso);
    if (!ps.length) ps.push([0, 0], [0.6, 0.4]);
    const xs = ps.map((p) => p[0]),
      ys = ps.map((p) => p[1]),
      lo = [Math.min(...xs), Math.min(...ys)],
      hi = [Math.max(...xs), Math.max(...ys)];
    const s = Math.min(
      (w - 90) / Math.max(0.15, hi[0] - lo[0]),
      (h - 75) / Math.max(0.15, hi[1] - lo[1]),
    );
    return (p) =>
      finite(p)
        ? [
            w / 2 + (iso(p)[0] - (lo[0] + hi[0]) / 2) * s,
            h / 2 + (iso(p)[1] - (lo[1] + hi[1]) / 2) * s,
          ]
        : null;
  }
  let coverageAnnotation = null;
  function diagnose(a, f) {
    coverageAnnotation = a;
    const raw = f.body_raw_prediction || f.body || [],
      accepted = f.body || [],
      shown = accepted.filter(visible).length;
    const rejected = raw.length > 0 && accepted.length === 0;
    el("diagnosis").innerHTML =
      `<p><b>手：</b>模型输出 ${f.hands.length}/2 只；${f.hands.length < 2 ? "未输出的手是否确实在画面中，需要人工核对；不能自动判为漏检。" : "两个模型实例不保证来自同一个人。"} ${f.hands.map((h) => `ID ${h.track_id} / ${esc(h.side)}`).join("，")}</p><p><b>人：</b>${raw.length ? "模型有 1 个人体输出" : "模型没有人体输出"}；${rejected ? "来源质检剔除全部人体点，非模型未预测" : `33 点中 ${shown} 点达到绘制阈值，${accepted.length ? 33 - shown : 0} 点被隐藏`}。${shown ? `头部 ${accepted.slice(0, 11).filter(visible).length}/11 点达到阈值。` : ""}</p><p><b>后处理：</b>本栏没有插值补点，也没有多人身份关联。灰色缺失不会被当成真实零动作；“全身候选”只检查肩、髋、膝、踝是否可见，不证明标注准确。</p>`;
    const c = el("coverage"),
      ctx = c.getContext("2d"),
      frames = a.frames,
      left = 105,
      right = 980,
      width = (right - left) / frames.length;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.font = "17px sans-serif";
    ctx.fillStyle = "#496051";
    ["手数量", "人体状态", "全身候选"].forEach((s, i) =>
      ctx.fillText(s, 4, 30 + 38 * i),
    );
    frames.forEach((x, i) => {
      const raw = x.body_raw_prediction || x.body || [],
        count = (x.body || []).filter(visible).length;
      const colors = [
        x.hands.length === 2
          ? "#157861"
          : x.hands.length === 1
            ? "#6bb8ad"
            : "#d8ddd5",
        raw.length && !x.body.length
          ? "#d58a46"
          : count
            ? "#a5c794"
            : "#d8ddd5",
        x.full_body_visible ? "#157861" : "#d8ddd5",
      ];
      colors.forEach((col, k) => {
        ctx.fillStyle = col;
        ctx.fillRect(left + i * width, 12 + k * 38, Math.max(1, width), 25);
      });
    });
    const idx = frames.indexOf(f),
      x = left + (idx + 0.5) * width;
    path(
      ctx,
      [
        [x, 4],
        [x, 126],
      ],
      "#25382e",
      2,
    );
    ctx.fillStyle = "#496051";
    ctx.fillText("0 s", left, 148);
    ctx.fillText(`${frames.at(-1).t.toFixed(1)} s`, right - 50, 148);
  }
  el("coverage").onclick = (e) => {
    if (!coverageAnnotation) return;
    const c = el("coverage"),
      x =
        ((e.clientX - c.getBoundingClientRect().left) / c.clientWidth) *
        c.width;
    const i = Math.max(
      0,
      Math.min(
        coverageAnnotation.frames.length - 1,
        Math.floor(((x - 105) / 875) * coverageAnnotation.frames.length),
      ),
    );
    el("video").currentTime = coverageAnnotation.frames[i].t;
  };
  function execution(d, r) {
    const c = el("bottleplot"),
      ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const points = d.bottle_point || [];
    if (points.length) {
      const initial = points[0],
        last = (r.final_bottle_point || points.at(-1)).slice(0, 3);
      const displacement = points.map((p) =>
          Math.hypot(...p.map((v, k) => v - initial[k])),
        ),
        max = Math.max(0.001, ...displacement);
      path(
        ctx,
        displacement.map((v, i) => [
          55 + (i * (c.width - 75)) / Math.max(1, points.length - 1),
          c.height - 35 - (v / max) * (c.height - 75),
        ]),
        "#9464b1",
        3,
      );
      ctx.fillStyle = "#496051";
      ctx.font = "19px sans-serif";
      ctx.fillText(`瓶子相对初始位置位移（m）；最大 ${max.toFixed(4)}`, 18, 24);
      ctx.fillText("执行前采样 / 命令序号", 250, c.height - 7);
      el("executionmetrics").innerHTML =
        `<p>瓶子最终位移：${fmt(Math.hypot(...last.map((v, k) => v - initial[k])), 4)} m；距离任务目标区域：${fmt(r.goal_region_distance_m, 4)} m。</p>`;
    } else {
      el("executionmetrics").textContent =
        "旧版记录没有瓶子轨迹，无法补算物体位移。";
    }
    if (d.physical_after) {
      el("executionmetrics").innerHTML +=
        `<p>命令 → 物理关节 MAE：${fmt(r.command_physical_after_joint_mae_rad, 5)} rad；命令 → 当前驱动目标 MAE：${fmt(r.command_state_joint_mae_rad, 5)} rad。前者低只说明跟随命令，不能证明命令正确。</p>`;
    } else
      el("executionmetrics").innerHTML +=
        "<p>旧版未记录物理关节位置，橙线仅为驱动目标，不能解读为实测执行误差。</p>";
    if (d.raw_action?.length === d.action.length) {
      let sum = 0,
        count = 0;
      for (let i = 0; i < d.action.length; i++)
        for (const j of [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]) {
          sum += Math.abs(d.raw_action[i][j] - d.action[i][j]);
          count++;
        }
      el("executionmetrics").innerHTML +=
        `<p>模型原始关节目标 → 控制约束后目标平均改变量：${fmt(sum / count, 5)} rad。夹爪阈值化单独处理，未混入这个角度指标。</p>`;
    }
    el("executionmetrics").innerHTML +=
      `<p>命令三阶差分：${fmt(r.joint_third_difference_per_command, 6)} rad/命令步³；不是物理 jerk。${r.active_arm_inferred !== undefined ? `本例固定闲置臂，模型推断活动臂为${r.active_arm_inferred ? "右" : "左"}臂。` : ""}</p>`;
  }
  window.H2RReview = { diagnose, execution };
  fetch("v2data/metrics.json")
    .then((r) => r.json())
    .then((rows) => {
      el("metricglossary").innerHTML = rows
        .map(
          (x) =>
            `<details class="metric"><summary>${esc(x.name)} <span>${esc(x.unit)}</span></summary><p><b>怎么算：</b>${esc(x.formula)}</p><p><b>来源：</b>${esc(x.source)}</p><p><b>说明：</b>${esc(x.meaning)}</p><p><b>不能说明：</b>${esc(x.limit)}</p></details>`,
        )
        .join("");
    })
    .catch((e) => (el("metricglossary").textContent = e.message));

  let cases = [],
    item = null,
    source = null,
    robot = null,
    frame = 0,
    playing = false,
    start = 0,
    offset = 0,
    last = "",
    serial = 0;
  const video = el("retargetvideo");
  const names = [
    "J1 基座旋转",
    "J2 肩部",
    "J3 肘部",
    "J4 腕部 1",
    "J5 腕部 2",
    "J6 腕部 3",
  ];
  el("retargetjoint").innerHTML = names
    .map((n, i) => `<option value="${i}">${n}</option>`)
    .join("");
  async function choose() {
    const token = ++serial;
    video.pause();
    playing = false;
    item = cases.find((x) => x.id === el("retargetcase").value);
    const [s, r] = await Promise.all([
      fetch(item.data).then((r) => r.json()),
      fetch(item.aloha).then((r) => r.json()),
    ]);
    if (token !== serial) return;
    source = s;
    robot = r;
    frame = 0;
    last = "";
    el("retargetframe").max = r.frames.length - 1;
    video.parentElement.hidden = !item.video;
    el("syntheticnotice").hidden = !!item.video;
    video.removeAttribute("src");
    if (item.video) video.src = item.video;
    video.load();
    render(true);
  }
  function drawChart() {
    if (!robot) return;
    const c = el("retargetplot"),
      ctx = c.getContext("2d"),
      j = +el("retargetjoint").value,
      vals = robot.frames.map((r) => (r ? (r.q[j] * 180) / Math.PI : null)),
      v = vals.filter(Number.isFinite),
      lo = Math.floor((Math.min(...v) - 5) / 10) * 10,
      hi = Math.ceil((Math.max(...v) + 5) / 10) * 10,
      T = robot.timestamp.at(-1),
      X = (t) => 80 + (t / T) * (c.width - 110),
      Y = (x) => c.height - 50 - ((x - lo) / (hi - lo)) * (c.height - 95);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "#52616c";
    for (let i = 0; i <= 4; i++) {
      const y = lo + ((hi - lo) * i) / 4;
      path(
        ctx,
        [
          [80, Y(y)],
          [c.width - 30, Y(y)],
        ],
        "#dbe2e5",
        1,
      );
      ctx.fillText(y.toFixed(0) + "°", 12, Y(y) + 5);
      const t = (T * i) / 4;
      ctx.fillText(t.toFixed(1), X(t) - 10, c.height - 25);
    }
    path(
      ctx,
      vals.map((v, i) =>
        Number.isFinite(v) ? [X(robot.timestamp[i]), Y(v)] : null,
      ),
      "#16829d",
      3,
    );
    path(
      ctx,
      [
        [X(robot.timestamp[frame]), 35],
        [X(robot.timestamp[frame]), c.height - 50],
      ],
      "#1e2936",
      2,
      [5, 4],
    );
    dot(
      ctx,
      Number.isFinite(vals[frame])
        ? [X(robot.timestamp[frame]), Y(vals[frame])]
        : null,
      "#16829d",
      6,
    );
    ctx.fillText("关节角度（度）", 12, 20);
    ctx.fillText("时间（秒）", c.width / 2 - 45, c.height - 3);
    el("retargetchartread").textContent =
      `当前看 ${names[j]}：${Number.isFinite(vals[frame]) ? vals[frame].toFixed(1) + "°" : "本帧无逆解"}。蓝线是重新求解的 Aloha 关节角度；黑色虚线对应视频当前时刻。断线表示缺失，不是零角度。`;
  }
  function render(reset = false) {
    if (!robot) return;
    const r = robot.frames[frame];
    el("retargetframe").value = frame;
    el("retargettime").textContent =
      `${robot.timestamp[frame].toFixed(2)} 秒 / ${robot.timestamp.at(-1).toFixed(2)} 秒 · 第 ${frame + 1} 帧`;
    window.H2RRobotPending = [r, robot.frames];
    window.H2RRobotView?.setPose(r, reset ? robot.frames : undefined);
    el("retargetinfo").innerHTML = r
      ? `<div class="readouts"><span><b>${(r.error_m * 1000).toFixed(1)} mm</b>夹爪中心距目标</span><span><b>${((r.rotation_error_rad * 180) / Math.PI).toFixed(1)}°</b>朝向误差</span><span><b>${(r.gripper_joint_m * 1000).toFixed(1)} mm</b>每个夹爪滑块位移</span></div><p>${r.ik_valid ? "位置与朝向达到当前 IK 阈值。" : "本帧未达到 IK 阈值；仍展示求解结果，便于检查错误。"} 这不是抓取成功判定。</p>`
      : "<p>本帧缺少可用人体目标，机械臂隐藏；拖动滑块查看其他帧。</p>";
    el("retargetsourceinfo").textContent =
      `${source.observed[frame] ? "当前帧检测到人手" : "当前帧人手观测缺失"}；${source.interpolated_mask[frame] ? "使用了短缺口插值" : "未使用插值"}。原视频里青色为手部骨架、黄色为检测点；它们不是机器人关节。`;
    el("retargetsummary").textContent =
      `${item.title} · 按 Aloha/AgileX 的 URDF 重新求解 ${robot.frames.filter(Boolean).length} 帧，未复用参考臂关节角。这里只演示左侧从臂；目标通过首帧示意对齐放入工作区。训练准入 0：相机、尺度、夹爪接触与碰撞尚未标定。`;
    const o = el("retargetoverlay");
    o.width = video.clientWidth || 640;
    o.height = video.clientHeight || 360;
    const oc = o.getContext("2d"),
      points = source.keypoints?.[frame]?.landmarks_2d;
    if (points && el("showhand").checked) {
      const w = video.videoWidth || 640,
        h = video.videoHeight || 360,
        k = Math.min(o.width / w, o.height / h),
        ox = (o.width - w * k) / 2,
        oy = (o.height - h * k) / 2,
        ps = points.map((p) =>
          finite(p) ? [ox + p[0] * w * k, oy + p[1] * h * k] : null,
        );
      for (const [a, b] of handEdges) path(oc, [ps[a], ps[b]], "#80f3dc", 2);
      ps.forEach((p) => dot(oc, p, "#ffe88e", 2));
    }
    drawChart();
  }
  el("retargetcase").onchange = () =>
    choose().catch((e) => (el("retargetinfo").textContent = e.message));
  el("retargetjoint").onchange = drawChart;
  el("showhand").onchange = () => render();
  el("retargetframe").oninput = (e) => {
    playing = false;
    video.pause();
    frame = +e.target.value;
    if (item.video) video.currentTime = robot.timestamp[frame];
    render();
  };
  el("retargetplay").onclick = () => {
    if (!robot) return;
    if (item.video) {
      if (video.paused) {
        if (video.ended) video.currentTime = 0;
        video.play();
      } else video.pause();
    } else {
      playing = !playing;
      if (frame === robot.frames.length - 1) frame = 0;
      start = performance.now();
      offset = robot.timestamp[frame];
    }
  };
  function tick(now) {
    if (robot) {
      const t = item.video
        ? video.currentTime
        : playing
          ? offset + (now - start) / 1000
          : robot.timestamp[frame];
      frame = Math.max(
        0,
        robot.timestamp.findLastIndex((x) => x <= t + 1e-4),
      );
      if (frame === robot.frames.length - 1) playing = false;
      el("retargetplay").textContent = (item.video ? !video.paused : playing)
        ? "暂停"
        : "播放同步回放";
      const key = [item.id, frame, video.clientWidth, video.videoWidth].join(
        ":",
      );
      if (key !== last) {
        render();
        last = key;
      }
    }
    requestAnimationFrame(tick);
  }
  fetch("v2data/aloha/manifest.json")
    .then((r) => r.json())
    .then(async (rows) => {
      cases = rows.sort(
        (a, b) =>
          (a.kind === "human_video" ? 0 : 1) -
          (b.kind === "human_video" ? 0 : 1),
      );
      el("retargetcase").innerHTML = cases
        .map(
          (i) =>
            `<option value="${esc(i.id)}">${i.kind === "human_video" ? "真人视频" : "合成压力测试"} · ${esc(i.title)}</option>`,
        )
        .join("");
      await choose();
      requestAnimationFrame(tick);
    })
    .catch((e) => (el("retargetinfo").textContent = e.message));
})();
