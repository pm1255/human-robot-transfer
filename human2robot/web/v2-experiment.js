/* Read-only presentation of saved experiment reports and rejected composites. */
(async () => {
  const [cases, experiment] = await Promise.all([
    json("v2data/conversion/manifest.json"),
    json("v2data/experiment_details.json"),
  ]);
  $("conversioncase").innerHTML = cases
    .map(
      (c, i) =>
        `<option value="${i}">${c.kind === "body" ? "全身 → G1" : "单手 → Aloha 实体机械臂"} · ${esc(c.id)}</option>`,
    )
    .join("");
  function showConversion() {
    const c = cases[Number($("conversioncase").value)];
    const v = $("conversionvideo");
    v.pause();
    v.poster = c.poster + "?revision=surface2";
    v.src = c.video + "?revision=surface2";
    v.load();
    $("conversiondownload").href = c.video;
    $("conversionaudit").href = c.audit;
    $("conversioninfo").textContent =
      `${c.id} · ${c.frames} 帧，${c.robot_frames} 帧有机器人渲染，10 fps。左：原视频；中：红色删除区域；右：填补背景后叠加机器人。${c.kind === "hand" ? "这里只替换 v1 跟踪到的一只手，另一只手仍保留；机械臂已换成 RoboTwin Aloha / AgileX 的 URDF 与实体网格，并重新求解关节角。" : "使用 G1 29 关节模型，头部固定；根姿态和关节由人体预测点拟合，位置由弱透视投影放回原画。"} 训练准入：0。残留人体、物体误删、接触错位都需要检查；可见机器人不代表物理执行成功。`;
  }
  $("conversioncase").onchange = showConversion;
  showConversion();
  $("modelidentity").innerHTML =
    `<div class="table-scroll"><table><thead><tr><th>模块</th><th>基座 / 方法</th><th>实际训练</th><th>输出与用途</th></tr></thead><tbody>${experiment.models.map((m) => `<tr><td>${esc(m.name)}</td><td>${esc(m.base)}</td><td>${esc(m.trained)}</td><td>${esc(m.output)}<br><small>${esc(m.used)}</small></td></tr>`).join("")}</tbody></table></div>`;
  const p = experiment.protocol;
  $("trainingrecipe").innerHTML = `
    <div class="recipe-grid">
    <article class="card"><h3>数据与拆分</h3><p>RoboTwin adjust_bottle：50 条轨迹、7,188 帧；按 episode 拆为 40 训练 / 5 验证 / 5 测试，拆分种子 429。验证集当前未用于早停或选模型。</p><p>人类：96 条 EPIC 视频、9,433 个可采样起点、4,509 个有效手部运动目标。35 条 HMDB 只展示，不训练。新增替换视频也未训练。</p></article>
    <article class="card"><h3>同一初始化，两个阶段</h3><p>种子 17 / 29 × 四组，共 8 个模型。前 2,000 步共同训练机器人动作与对应的人类辅助任务；后 8,000 步全部只用相同机器人数据微调。机器人基线全程仅动作。</p><p>同种子使用相同初始化、机器人批次顺序和步数。固定使用第 10,000 步检查点。人类分支额外计算，尚非等 FLOPs 比较。</p></article>
    <article class="card"><h3>网络和优化器</h3><p>随机初始化 CNN，${experiment.total_parameters.toLocaleString()} 参数（含三个头），没有预训练视觉基座、没有语言输入。每张 128×128 图像编码为 192 维；机器人三视角 + 14 维状态预测 16×14 动作。</p><p>AdamW，学习率 2×10⁻⁴，weight decay 10⁻⁴，梯度范数裁剪 1；batch：机器人 64、人类 32。每模型 64 万机器人窗口重复采样，人类组 6.4 万人类窗口重复采样。</p></article>
    <article class="card"><h3>动作评估与适用边界</h3><p>离线：未参加训练的 5 条轨迹，预测绝对关节/夹爪目标，与示教比较。闭环：预测 16 条、执行前 4 条、重新观察，最多 400 条命令；目标为瓶子达到任务区域。</p><p>正式 8×18 = 144 次闭环尚无已回传结果。开发回放的三个单场景版本均失败，不是四组正式成功率。这个实验是小 CNN 行为克隆试验，不能代表完整 VLA 的能力。</p></article>
    </div>
    <div class="table-scroll"><table><thead><tr><th>组别：前 2,000 步目标</th><th>seed17 关节 RMSE</th><th>seed29</th><th>均值（rad，↓）</th></tr></thead><tbody>${experiment.summary.map((s) => `<tr><td>${esc(s.label)}</td><td>${s.seed17.toFixed(5)}</td><td>${s.seed29.toFixed(5)}</td><td>${s.mean.toFixed(5)}</td></tr>`).join("")}</tbody></table></div>
    <p>动作损失：标准化动作 MSE。显式分支：两手各 3 维的二维捏合中心位移 / 开度变化 MSE（权重 0.1），并非重定向关节监督；隐式分支：+0.5 秒的 32×32 RGB 残差 MSE（权重 1）。随机标签组打乱运动目标。只在同一种损失内比较曲线，不跨动作 / 图像损失比较数值大小。</p>`;
  $("trainingconclusion").textContent = experiment.conclusion;
  $("trainingrun").innerHTML = experiment.runs
    .map((r, i) => `<option value="${i}">${esc(r.name)}</option>`)
    .join("");
  function drawCurve() {
    const run = experiment.runs[Number($("trainingrun").value)],
      key = $("trainingloss").value;
    const enabled =
      key === "action_loss" ||
      (key === "image_loss"
        ? run.condition === "implicit_joint"
        : ["explicit", "shuffled"].includes(run.condition));
    const data = enabled
      ? run.curves.filter((x) => key === "action_loss" || x.step <= p.stage1)
      : [];
    const canvas = $("trainingcurve"),
      ctx = canvas.getContext("2d"),
      w = canvas.width,
      h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f5f7f1";
    ctx.fillRect(0, 0, w, h);
    const l = 75,
      r = w - 25,
      t = 30,
      b = h - 45,
      X = (s) => l + (s / 10000) * (r - l);
    const logs = data.map((x) => Math.log10(Math.max(1e-8, x[key]))),
      lo = Math.floor(Math.min(-3, ...logs)),
      hi = Math.ceil(Math.max(0, ...logs));
    const Y = (v) =>
      b - ((Math.log10(Math.max(1e-8, v)) - lo) / (hi - lo)) * (b - t);
    ctx.font = "13px sans-serif";
    ctx.fillStyle = "#53645a";
    for (let e = lo; e <= hi; e++) {
      const y = Y(10 ** e);
      ctx.strokeStyle = "#dce2d9";
      ctx.beginPath();
      ctx.moveTo(l, y);
      ctx.lineTo(r, y);
      ctx.stroke();
      ctx.fillText(`10^${e}`, 12, y + 4);
    }
    for (let s = 0; s <= 10000; s += 2000)
      ctx.fillText(String(s), X(s) - 15, h - 20);
    ctx.strokeStyle = "#a67938";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(X(2000), t);
    ctx.lineTo(X(2000), b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText("2,000：切换机器人微调", X(2000) + 8, 20);
    ctx.strokeStyle = "#167b64";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) =>
      i ? ctx.lineTo(X(d.step), Y(d[key])) : ctx.moveTo(X(d.step), Y(d[key])),
    );
    ctx.stroke();
    if (!enabled) ctx.fillText("该组没有启用这个辅助目标", l + 250, 140);
    else if (key !== "action_loss")
      ctx.fillText(
        "辅助监督已停用：这里没有绘制虚假的零损失",
        X(2000) + 70,
        140,
      );
    $("trainingcurveinfo").textContent =
      `${run.name} · 共 ${run.report.steps.toLocaleString()} 步；报告训练计时 ${run.report.seconds.toFixed(1)} 秒（小模型，H100；不含数据准备与评测）。纵轴为对数刻度，横轴为优化步。${key === "action_loss" ? "显示机器人动作训练损失；泛化表现看上方测试集 RMSE 和最终闭环。" : "辅助分支只在前 2,000 步启用；之后的日志零值表示停用，不代表预测完美。"}`;
  }
  $("trainingrun").onchange = drawCurve;
  $("trainingloss").onchange = drawCurve;
  drawCurve();
})().catch((e) => {
  $("trainingconclusion").textContent = `明细加载失败：${e.message}`;
  console.error(e);
});
