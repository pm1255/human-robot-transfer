import * as THREE from "./vendor/three/build/three.module.js";
import { OrbitControls } from "./vendor/three/examples/jsm/controls/OrbitControls.js";
const host = document.getElementById("retargetrobot");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setClearColor("#edf1f2");
host.appendChild(renderer.domElement);
const scene = new THREE.Scene(),
  camera = new THREE.PerspectiveCamera(38, 1, 0.01, 12);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.38);
controls.enableDamping = true;
controls.minDistance = 0.3;
controls.maxDistance = 3;
function view(name) {
  camera.position.set(
    ...{
      front: [1.65, -0.03, 0.8],
      side: [0.05, -1.65, 0.8],
      three: [1.2, -1.45, 1.0],
    }[name],
  );
  controls.target.set(0, 0, 0.38);
  controls.update();
}
view("three");
scene.add(new THREE.HemisphereLight(0xe6f0ff, 0x6e756f, 2.4));
const light = new THREE.DirectionalLight(0xffffff, 3.5);
light.position.set(0.8, -0.5, 1.7);
light.castShadow = true;
light.shadow.mapSize.set(2048, 2048);
Object.assign(light.shadow.camera, {
  left: -1,
  right: 1,
  top: 1,
  bottom: -1,
  near: 0.01,
  far: 4,
});
light.shadow.bias = -0.0004;
scene.add(light);
const floor = new THREE.Mesh(
  new THREE.BoxGeometry(1.2, 1, 0.04),
  new THREE.MeshStandardMaterial({ color: "#c7cccf", roughness: 0.9 }),
);
floor.position.z = -0.025;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(1.2, 12, 0x89999c, 0xb7c3c5);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.003;
scene.add(grid);
const base = new THREE.Group();
scene.add(base);
const links = {};
const meshData = await (await fetch("v2data/aloha/meshes.json")).json();
for (const g of meshData.groups) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(g.vertices.flat(), 3),
  );
  geom.setIndex(g.faces.flat());
  geom.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: g.link.endsWith("7") || g.link.endsWith("8") ? 0x8b939a : 0x253647,
    metalness: 0.6,
    roughness: 0.34,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.matrixAutoUpdate = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  links[g.link] = mesh;
  base.add(mesh);
}
const marker = (color, r) => {
  const x = new THREE.Mesh(
    new THREE.SphereGeometry(r, 20, 16),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.4,
      emissive: color,
      emissiveIntensity: 0.2,
    }),
  );
  scene.add(x);
  return x;
};
const target = marker(0xe37b20, 0.012),
  actual = marker(0x13a1c2, 0.008);
target.material.depthTest = false;
actual.material.depthTest = false;
target.material.wireframe = true;
target.renderOrder = 99;
actual.renderOrder = 100;
const selectedJoint = marker(0xe05293, 0.016);
selectedJoint.material.depthTest = false;
selectedJoint.renderOrder = 101;
const jointLabel = document.createElement("span");
jointLabel.className = "robot-joint-label";
host.appendChild(jointLabel);
let current = null,
  trajectory = null;
function setPose(frame, rows) {
  current = frame;
  base.visible = !!frame;
  actual.visible = target.visible = !!frame;
  if (frame) {
    for (const [k, m] of Object.entries(frame.links)) {
      links[k]?.matrix.set(...m.flat());
    }
    actual.position.set(...frame.actual.slice(0, 3).map((r) => r[3]));
    target.position.set(...frame.target.slice(0, 3).map((r) => r[3]));
  }
  if (rows) {
    if (trajectory) {
      scene.remove(trajectory);
      trajectory.geometry.dispose();
      trajectory.material.dispose();
    }
    const pts = [];
    for (let i = 1; i < rows.length; i++) {
      if (!rows[i - 1] || !rows[i]) continue;
      for (const r of [rows[i - 1], rows[i]])
        pts.push(new THREE.Vector3(...r.target.slice(0, 3).map((x) => x[3])));
    }
    trajectory = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xe37b20 }),
    );
    trajectory.visible = document.getElementById("showpath").checked;
    scene.add(trajectory);
  }
}
document
  .querySelectorAll("[data-robot-view]")
  .forEach((b) => (b.onclick = () => view(b.dataset.robotView)));
document.getElementById("showpath").onchange = (e) => {
  if (trajectory) trajectory.visible = e.target.checked;
};
window.H2RRobotView = { setPose };
if (window.H2RRobotPending) setPose(...window.H2RRobotPending);
new ResizeObserver(() => {
  const w = host.clientWidth,
    h = host.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}).observe(host);
function animate() {
  controls.update();
  const j = Number(document.getElementById("retargetjoint").value) + 1;
  selectedJoint.visible = !!current;
  jointLabel.hidden = !current;
  if (current) {
    const m = current.links["fl_link" + j];
    selectedJoint.position.set(m[0][3], m[1][3], m[2][3]);
    const p = selectedJoint.position.clone().project(camera);
    jointLabel.textContent =
      document.getElementById("retargetjoint").selectedOptions[0].textContent;
    jointLabel.style.left =
      Math.max(
        4,
        Math.min(
          host.clientWidth - 120,
          ((p.x + 1) * host.clientWidth) / 2 + 12,
        ),
      ) + "px";
    jointLabel.style.top =
      Math.max(
        4,
        Math.min(
          host.clientHeight - 25,
          ((1 - p.y) * host.clientHeight) / 2 - 15,
        ),
      ) + "px";
  }
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
document.getElementById("robotload").textContent =
  "已加载 Aloha / AgileX RoboTwin 原始网格 · 拖动旋转，滚轮缩放";
