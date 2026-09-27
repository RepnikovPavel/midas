/* PandaSet WebGL viewer frontend. Data: /api/* (aiohttp backend). */
"use strict";

const CAM_ORDER = ["front_left_camera", "front_camera", "front_right_camera",
                   "left_camera", "back_camera", "right_camera"];
const SEMSEG_PALETTE = {};
{ const base = {0:[128,128,128],1:[150,60,60],5:[90,90,220],6:[60,60,200],7:[80,80,180],
  11:[220,220,60],13:[160,100,40],15:[200,160,60],19:[60,160,220],20:[0,220,220],
  23:[30,30,220],24:[40,80,200],25:[60,60,160],29:[120,120,120],30:[200,120,120],
  34:[140,140,60],35:[90,180,90],41:[100,100,180]};
  for (let i = 0; i < 256; i++)
    SEMSEG_PALETTE[i] = base[i] || [(i*47)%255, (i*91)%255, (i*137)%255]; }

const state = {
  sweeps: [], counts: [], sweep: 0, frame: 0, frames: 1,
  playing: true, dataFps: 10, colorMode: "height",
  showLidar: true, showBoxes: true,
  cams: {}, semsegClasses: {},
  cache: new Map(),          // frame -> Promise<data>
  gpsTrack: [],
};
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- data
async function api(path) { const r = await fetch(path); return r.json(); }

async function loadFrame(sweep, frame) {
  const key = sweep * 100000 + frame;
  if (state.cache.has(key)) return state.cache.get(key);
  const p = (async () => {
    const r = await fetch(`/api/frame?sweep=${sweep}&frame=${frame}`);
    const buf = await r.arrayBuffer();
    const dv = new DataView(buf);
    const jl = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jl)));
    let off = 4 + jl;
    const n = header.n, nb = header.nb;
    const points = new Float32Array(buf.slice(off, off + n * 12)); off += n * 12;
    const intensity = new Uint8Array(buf.slice(off, off + n)); off += n;
    const semseg = new Uint8Array(buf.slice(off, off + n)); off += n;
    const boxes = new Float32Array(buf.slice(off, off + nb * 28));
    return { header, points, intensity, semseg, boxes };
  })();
  state.cache.set(key, p);
  if (state.cache.size > 24) state.cache.delete(state.cache.keys().next().value);
  return p;
}

function prefetch() {
  for (let f = state.frame; f < Math.min(state.frame + 6, state.frames); f++)
    loadFrame(state.sweep, f).catch(() => {});
}

// ---------------------------------------------------------------- colors
function zColorRGB(z) {           // height colormap, matches backend fastops
  let t = (z + 3.0) / 6.0; t = t < 0 ? 0 : (t > 1 ? 1 : t);
  return [255 * t, 102, 255 * (1 - t)];
}
function computeColors(data, mode) {
  const n = data.header.n, pts = data.points, out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let r, g, b;
    if (mode === "intensity") { r = g = b = data.intensity[i]; }
    else if (mode === "semseg") { [r, g, b] = SEMSEG_PALETTE[data.semseg[i]]; }
    else { [r, g, b] = zColorRGB(pts[i * 3 + 2]); }
    out[i * 3] = r / 255; out[i * 3 + 1] = g / 255; out[i * 3 + 2] = b / 255;
  }
  return out;
}

// ---------------------------------------------------------------- 3D scene
const canvas3d = $("view3d");
const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x0b0e11);
const cam3 = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
cam3.position.set(-25, -25, 18);
cam3.up.set(0, 0, 1);
const controls = new THREE.OrbitControls(cam3, canvas3d);
controls.target.set(10, 0, 0);
scene3.add(new THREE.AxesHelper(3));

const ptsGeo = new THREE.BufferGeometry();
ptsGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
ptsGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
const points3 = new THREE.Points(ptsGeo, new THREE.PointsMaterial({
  size: 0.08, vertexColors: true, sizeAttenuation: true }));
scene3.add(points3);

const boxGeo = new THREE.BufferGeometry();
boxGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
const boxes3 = new THREE.LineSegments(boxGeo, new THREE.LineBasicMaterial({ color: 0x00ff44 }));
scene3.add(boxes3);

const arrowGeo = new THREE.BufferGeometry();
arrowGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
const arrows3 = new THREE.LineSegments(arrowGeo, new THREE.LineBasicMaterial({ color: 0xff2222 }));
scene3.add(arrows3);

// 3D labels: sprites (scale with zoom natively)
const labelTexCache = new Map();
function labelTexture(text) {
  if (labelTexCache.has(text)) return labelTexCache.get(text);
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = "bold 42px system-ui";
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  c.width = w; c.height = 56;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(0,0,0,0.55)"; g.fillRect(0, 0, w, 56);
  g.font = "bold 42px system-ui"; g.fillStyle = "#ff4";
  g.textBaseline = "middle"; g.fillText(text, 8, 30);
  const tex = new THREE.CanvasTexture(c);
  tex.userData = { aspect: w / 56 };
  if (labelTexCache.size > 200) labelTexCache.delete(labelTexCache.keys().next().value);
  labelTexCache.set(text, tex);
  return tex;
}
const labelSprites = [];
for (let i = 0; i < 64; i++) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false }));
  sp.visible = false;
  scene3.add(sp);
  labelSprites.push(sp);
}

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function boxCorners(b) {
  const [x, y, z, dx, dy, dz, yaw] = b;
  const c = Math.cos(yaw), s = Math.sin(yaw), out = [];
  for (const [sx, sy, sz] of [[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],
                              [-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]]) {
    const lx = sx * dx, ly = sy * dy, lz = sz * dz;
    out.push([c * lx - s * ly + x, s * lx + c * ly + y, lz + z]);
  }
  return out;
}

function render3d(data) {
  const n = data.header.n;
  ptsGeo.setAttribute("position", new THREE.BufferAttribute(data.points, 3));
  ptsGeo.setAttribute("color", new THREE.BufferAttribute(computeColors(data, state.colorMode), 3));
  ptsGeo.computeBoundingSphere();
  const boxes = data.boxes, nb = data.header.nb;
  const pos = new Float32Array(nb * 24 * 3), apos = new Float32Array(nb * 2 * 3);
  for (let k = 0; k < nb; k++) {
    const b = boxes.subarray(k * 7, k * 7 + 7);
    const cor = boxCorners(b);
    let o = k * 24 * 3;
    for (const [a, bb] of EDGES) {
      pos.set(cor[a], o); pos.set(cor[bb], o + 3); o += 6;
    }
    const hl = b[3] / 2, c = Math.cos(b[6]), s = Math.sin(b[6]);
    apos.set([b[0], b[1], b[2]], k * 6);
    apos.set([b[0] + hl * c, b[1] + hl * s, b[2]], k * 6 + 3);
  }
  boxGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  arrowGeo.setAttribute("position", new THREE.BufferAttribute(apos, 3));
  for (let i = 0; i < labelSprites.length; i++) {
    const sp = labelSprites[i];
    if (i < nb) {
      const b = boxes.subarray(i * 7, i * 7 + 7);
      const text = data.header.labels[i] || "?";
      const tex = labelTexture(text);
      sp.material.map = tex; sp.material.needsUpdate = true;
      sp.position.set(b[0], b[1], b[2] + b[5] / 2 + 1.0);
      const h = 1.1;
      sp.scale.set(h * tex.userData.aspect, h, 1);
      sp.visible = true;
    } else sp.visible = false;
  }
}

// ---------------------------------------------------------------- cameras
const camPanels = {};   // cam -> {wrap, zoomwrap, img, renderer, scene, cam2d, pts, boxes, labels[], z, name}

function quatToMat(q) {
  const [w, x, y, z] = q, n = w*w + x*x + y*y + z*z, s = n > 0 ? 2 / n : 0;
  const wx = s*w*x, wy = s*w*y, wz = s*w*z, xx = s*x*x, xy = s*x*y, xz = s*x*z,
        yy = s*y*y, yz = s*y*z, zz = s*z*z;
  return [[1-(yy+zz), xy-wz, xz+wy], [xy+wz, 1-(xx+zz), yz-wx], [xz-wy, yz+wx, 1-(xx+yy)]];
}

function buildCamPanels(camNames) {
  const grid = $("camgrid");
  grid.innerHTML = "";
  for (const cam of CAM_ORDER) {
    if (!camNames.includes(cam)) continue;
    const div = document.createElement("div");
    div.className = "cam";
    const zoomwrap = document.createElement("div");
    zoomwrap.className = "zoomwrap";
    const img = document.createElement("img");
    img.draggable = false;
    const cvs = document.createElement("canvas");
    const name = document.createElement("div");
    name.className = "camname"; name.textContent = cam.replace("_camera", "");
    zoomwrap.append(img, cvs);
    div.append(zoomwrap, name);
    grid.append(div);
    const renderer2 = new THREE.WebGLRenderer({ canvas: cvs, alpha: true, antialias: false });
    renderer2.setClearColor(0x000000, 0);
    const scene2 = new THREE.Scene();
    const cam2d = new THREE.OrthographicCamera(0, 1, 0, 1, -10, 10);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    const pts2 = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 1.8, vertexColors: true, sizeAttenuation: false }));
    scene2.add(pts2);
    const bgeo = new THREE.BufferGeometry();
    bgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    const boxes2 = new THREE.LineSegments(bgeo, new THREE.LineBasicMaterial({ color: 0x00ff44 }));
    scene2.add(boxes2);
    const labels = [];
    for (let i = 0; i < 48; i++) {
      const l = document.createElement("div");
      l.className = "lbl"; l.style.display = "none";
      zoomwrap.append(l);
      labels.push(l);
    }
    const p = { div, zoomwrap, img, renderer: renderer2, scene: scene2, cam2d,
                pts: pts2, boxes: boxes2, labels, z: { s: 1, x: 0, y: 0 },
                w: 960, h: 540, cam };
    camPanels[cam] = p;
    // zoom to cursor + drag pan
    div.onwheel = e => {
      e.preventDefault();
      const r = div.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const ds = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      const s2 = Math.min(30, Math.max(1, p.z.s * ds));
      p.z.x = mx - (mx - p.z.x) * (s2 / p.z.s);
      p.z.y = my - (my - p.z.y) * (s2 / p.z.s);
      p.z.s = s2;
      applyZoom(p);
    };
    let drag = null;
    div.onpointerdown = e => { drag = { x: e.clientX, y: e.clientY, zx: p.z.x, zy: p.z.y }; div.setPointerCapture(e.pointerId); };
    div.onpointermove = e => {
      if (!drag) return;
      p.z.x = drag.zx + (e.clientX - drag.x);
      p.z.y = drag.zy + (e.clientY - drag.y);
      applyZoom(p);
    };
    div.onpointerup = () => { drag = null; };
  }
}

function applyZoom(p) {
  p.zoomwrap.style.transform = `translate(${p.z.x}px, ${p.z.y}px) scale(${p.z.s})`;
  // label font scales with zoom (dynamic text scaling)
  const fs = Math.min(48, Math.max(6, 11 * p.z.s));
  for (const l of p.labels) l.style.fontSize = fs + "px";
  // point size shrinks a bit with zoom for readability
  p.pts.material.size = Math.max(1.0, 1.8 / Math.sqrt(p.z.s));
}

function setupCamPanelSize(p, w, h) {
  p.w = w; p.h = h;
  p.img.width = w; p.img.height = h;
  p.zoomwrap.style.width = w + "px"; p.zoomwrap.style.height = h + "px";
  p.renderer.setSize(w, h, false);
  p.cam2d.right = w; p.cam2d.top = h;
  p.cam2d.updateProjectionMatrix();
}

function renderCamPanel(p, data) {
  const cd = data.header.cams[p.cam];
  if (!cd) { p.div.style.display = "none"; return; }
  p.div.style.display = "";
  const w0 = cd.w, h0 = cd.h;
  if (p.w !== w0 || p.h !== h0) setupCamPanelSize(p, w0, h0);
  const imgUrl = `/api/camimg?sweep=${state.sweep}&cam=${p.cam}&ts=${cd.ts}`;
  if (p.img.dataset.cur !== imgUrl) { p.img.src = imgUrl; p.img.dataset.cur = imgUrl; }

  const n = data.header.n, pts = data.points;
  const R = quatToMat(cd.R), t = cd.t;   // sensor2ego; cam = R^T (ego - t)
  const uv = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let m = 0;
  if (state.showLidar) {
    for (let i = 0; i < n; i++) {
      const ex = pts[i*3] - t[0], ey = pts[i*3+1] - t[1], ez = pts[i*3+2] - t[2];
      const cx = R[0][0]*ex + R[1][0]*ey + R[2][0]*ez;
      const cy = R[0][1]*ex + R[1][1]*ey + R[2][1]*ez;
      const cz = R[0][2]*ex + R[1][2]*ey + R[2][2]*ez;
      if (cz < 0.5) continue;
      const K = cd.K;
      const u = (K[0][0]*cx + K[0][2]*cz) / cz, v = (K[1][1]*cy + K[1][2]*cz) / cz;
      if (u < 0 || u >= w0 || v < 0 || v >= h0) continue;
      uv[m*3] = u; uv[m*3+1] = h0 - v; uv[m*3+2] = 0;
      const [r, g, b] = zColorRGB(pts[i*3+2]);   // Z-colored projection
      col[m*3] = r/255; col[m*3+1] = g/255; col[m*3+2] = b/255;
      m++;
    }
  }
  p.pts.geometry.setAttribute("position", new THREE.BufferAttribute(uv.subarray(0, m*3), 3));
  p.pts.geometry.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, m*3), 3));
  p.pts.visible = state.showLidar && m > 0;

  const nb = data.header.nb, boxes = data.boxes;
  const bpos = new Float32Array(nb * 24 * 3);
  let segs = 0;
  const labelInfo = [];
  if (state.showBoxes) {
    for (let k = 0; k < nb; k++) {
      const b = boxes.subarray(k*7, k*7+7);
      const cor = boxCorners(b);
      const cc = cor.map(([x, y, z]) => {
        const ex = x - t[0], ey = y - t[1], ez = z - t[2];
        return [R[0][0]*ex + R[1][0]*ey + R[2][0]*ez,
                R[0][1]*ex + R[1][1]*ey + R[2][1]*ez,
                R[0][2]*ex + R[1][2]*ey + R[2][2]*ez];
      });
      let zmin = Infinity;
      for (const c of cc) zmin = Math.min(zmin, c[2]);
      if (zmin < 0.2) continue;
      const K = cd.K;
      const uv8 = cc.map(([x, y, z]) => [(K[0][0]*x + K[0][2]*z)/z, h0 - (K[1][1]*y + K[1][2]*z)/z, 0]);
      for (const [a, bb] of EDGES) {
        bpos.set(uv8[a], segs*3); bpos.set(uv8[bb], segs*3+3); segs += 2;
      }
      let top = uv8[0];
      for (const q of uv8) if (q[1] > top[1]) top = q;
      labelInfo.push({ k, x: top[0], y: h0 - top[1] });
    }
  }
  p.boxes.geometry.setAttribute("position", new THREE.BufferAttribute(bpos.subarray(0, segs*3), 3));
  p.boxes.visible = state.showBoxes && segs > 0;
  p.labels.forEach((l, i) => {
    if (state.showBoxes && i < labelInfo.length) {
      const { k, x, y } = labelInfo[i];
      l.textContent = data.header.labels[k] || "?";
      l.style.left = x + "px"; l.style.top = (h0 - y) + "px";
      l.style.display = "";
    } else l.style.display = "none";
  });
  p.renderer.render(p.scene, p.cam2d);
}

// ---------------------------------------------------------------- map
let leaflet = null, trackLine = null, posMarker = null;
function initMap() {
  leaflet = L.map("map", { zoomControl: true, attributionControl: false })
            .setView([37.422, -122.16], 16);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 })
   .addTo(leaflet);
  trackLine = L.polyline([], { color: "#3af", weight: 3 }).addTo(leaflet);
  posMarker = L.circleMarker([37.422, -122.16], { radius: 7, color: "#f30" }).addTo(leaflet);
}
function updateMap(gps, sweepChanged) {
  if (!leaflet || !gps) return;
  const ll = [gps.lat, gps.long];
  if (!gps.lat && !gps.long) return;
  state.gpsTrack.push(ll);
  if (state.gpsTrack.length > 2000) state.gpsTrack.shift();
  trackLine.setLatLngs(state.gpsTrack);
  posMarker.setLatLng(ll);
  if (sweepChanged) leaflet.setView(ll, 16);
}

// ---------------------------------------------------------------- UI
async function selectSweep(i, keepFrame = false) {
  state.sweep = i;
  state.gpsTrack = [];
  const meta = await api(`/api/meta?sweep=${i}`);
  state.frames = meta.frames;
  state.cams = meta.cameras;
  state.semsegClasses = meta.semseg_classes;
  if (!keepFrame) state.frame = 0;
  $("slider").max = Math.max(0, state.frames - 1);
  buildCamPanels(meta.cameras);
  state.cache.clear();
  prefetch();
}

async function render() {
  const data = await loadFrame(state.sweep, state.frame).catch(() => null);
  if (!data || (data.header.frame !== state.frame && state.playing)) {}
  if (!data) return;
  render3d(data);
  for (const cam of CAM_ORDER) if (camPanels[cam]) renderCamPanel(camPanels[cam], data);
  updateMap(data.header.gps, false);
  $("speedo").textContent = `${((data.header.gps.speed || 0) * 3.6).toFixed(1)} km/h`;
  $("frameLbl").textContent = `f ${state.frame + 1}/${state.frames}`;
  $("slider").value = state.frame;
}

let lastAdvance = 0;
function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene3, cam3);
  const now = performance.now();
  if (state.playing && now - lastAdvance >= 1000 / state.dataFps && state.frames > 1) {
    lastAdvance = now;
    state.frame = (state.frame + 1) % state.frames;
    prefetch();
    render();
  }
}

function resize() {
  const w = canvas3d.clientWidth, h = canvas3d.clientHeight;
  renderer.setSize(w, h, false);
  cam3.aspect = w / h;
  cam3.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

$("btnPlay").onclick = () => {
  state.playing = !state.playing;
  $("btnPlay").textContent = state.playing ? "Stop" : "Start";
};
$("chkLidar").onchange = e => { state.showLidar = e.target.checked; render(); };
$("chkBoxes").onchange = e => { state.showBoxes = e.target.checked; render(); };
$("colorMode").onchange = e => { state.colorMode = e.target.value; render(); };
$("sweepSel").onchange = e => selectSweep(+e.target.value);
$("slider").oninput = e => {
  state.playing = false; $("btnPlay").textContent = "Start";
  state.frame = +e.target.value;
  prefetch(); render();
};
window.addEventListener("keydown", e => {
  if (e.key === " ") { e.preventDefault(); $("btnPlay").click(); }
  else if (e.key === "n") { state.playing = false; $("btnPlay").textContent = "Start";
    state.frame = (state.frame + 1) % state.frames; prefetch(); render(); }
  else if (e.key === "b") { state.playing = false; $("btnPlay").textContent = "Start";
    state.frame = (state.frame - 1 + state.frames) % state.frames; prefetch(); render(); }
  else if (e.key === "N") selectSweep((state.sweep + 1) % state.sweeps.length);
  else if (e.key === "B") selectSweep((state.sweep - 1 + state.sweeps.length) % state.sweeps.length);
  else if (e.key === "c") {
    const modes = ["height", "intensity", "semseg"];
    const m = modes[(modes.indexOf(state.colorMode) + 1) % 3];
    $("colorMode").value = m; state.colorMode = m; render();
  }
});

(async function init() {
  const d = await api("/api/sweeps");
  state.sweeps = d.sweeps; state.counts = d.counts;
  const sel = $("sweepSel");
  d.sweeps.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = s;
    sel.append(o);
  });
  const params = new URLSearchParams(location.search);
  const wantSeq = params.get("seq");
  let start = 0;
  if (wantSeq && d.sweeps.includes(wantSeq)) start = d.sweeps.indexOf(wantSeq);
  sel.value = start;
  initMap();
  resize();
  await selectSweep(start);
  setInterval(prefetch, 500);
  tick();
})();
