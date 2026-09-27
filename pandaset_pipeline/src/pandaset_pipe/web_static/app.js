/* PandaSet WebGL viewer frontend. Data: /api/* (aiohttp backend).
   Ego frame: X forward, Y left, Z up. */
"use strict";

const CAM_ORDER = ["front_left_camera", "front_camera", "front_right_camera",
                   "left_camera", "back_camera", "right_camera"];

// ------------------------------------------------------------ colormaps
const TURBO = new Uint8Array(256 * 3);
{ // polynomial approximation of Google's turbo colormap
  const cl = v => v < 0 ? 0 : (v > 255 ? 255 : v);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    TURBO[i * 3]     = cl(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))));
    TURBO[i * 3 + 1] = cl(23.06 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3220.32 - t * 1320.25)))));
    TURBO[i * 3 + 2] = cl(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814 + t * (-22569.18 + t * 6838.66)))));
  }
}
const SEMSEG_PALETTE = [];
{ const base = { 0: [128, 128, 128], 1: [150, 60, 60], 5: [90, 90, 220], 6: [60, 60, 200],
  7: [80, 80, 180], 11: [220, 220, 60], 13: [160, 100, 40], 15: [200, 160, 60],
  19: [60, 160, 220], 20: [0, 220, 220], 23: [30, 30, 220], 24: [40, 80, 200],
  25: [60, 60, 160], 29: [120, 120, 120], 30: [200, 120, 120], 34: [140, 140, 60],
  35: [90, 180, 90], 41: [100, 100, 180] };
  for (let i = 0; i < 256; i++)
    SEMSEG_PALETTE[i] = base[i] || [(i * 47) % 255, (i * 91) % 255, (i * 137) % 255]; }

function classColor(name) {  // deterministic color per class name
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  const hue = (h * 137.508) % 360;
  const a = 0.85, b = 0.55;  // HSV(hue, .85, .95) approx via HSL
  const c = (1 - Math.abs(2 * b - 1)) * a;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = b - c / 2;
  const seg = Math.floor(hue / 60) % 6;
  const rgb = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][seg]
    .map(v => Math.round((v + m) * 255));
  return { rgb, css: `rgb(${rgb[0]},${rgb[1]},${rgb[2]})` };
}

// ------------------------------------------------------------ state
const state = {
  sweeps: [], counts: [], sweep: 0, frame: 0, frames: 1,
  playing: true, dataFps: 5, colorMode: "height",
  showLidar: true, showBoxes: true, showCamLbl: true, showNms: true,
  showFrustum: true, showGrid: true, showLabels3d: true, showBev: true,
  followMap: true, ptSize: 0.06, rangeClip: Infinity,
  hasSemseg: false, semsegClasses: {}, boxClasses: {},
  hiddenClasses: new Set(), classColors: {},
  track: null,               // gps track of current sweep
  gpsTrackDrawn: [],
  cache: new Map(),          // key -> Promise<data>
  lastLoadMs: 0, netKB: 0,
};
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString("en-US");

function classColorsFor(labels) {
  const out = [];
  for (const l of labels)
    out.push(state.classColors[l] || (state.classColors[l] = classColor(l)));
  return out;
}

// ------------------------------------------------------------ data
async function api(path) { const r = await fetch(path); return r.json(); }

async function loadFrame(sweep, frame) {
  const key = `${sweep}:${frame}:${state.showNms ? 1 : 0}`;
  if (state.cache.has(key)) return state.cache.get(key);
  const t0 = performance.now();
  const p = (async () => {
    const r = await fetch(`/api/frame?sweep=${sweep}&frame=${frame}&nms=${state.showNms ? 1 : 0}`);
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
    state.netKB += buf.byteLength / 1024;
    state.lastLoadMs = performance.now() - t0;
    return { header, points, intensity, semseg, boxes, colors: {} };
  })();
  state.cache.set(key, p);
  if (state.cache.size > 24) state.cache.delete(state.cache.keys().next().value);
  return p;
}

function prefetch() {
  const a = Math.max(0, state.frame - 2), b = Math.min(state.frame + 6, state.frames - 1);
  for (let f = a; f <= b; f++) loadFrame(state.sweep, f).catch(() => {});
}

// point colors for a frame, cached per mode+clip in the data object
function pointColors(data, mode, clip2) {
  const ck = mode + ":" + clip2;
  if (data.colors[ck]) return data.colors[ck];
  const n = data.header.n, pts = data.points;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let r, g, b;
    if (mode === "intensity") { r = g = b = data.intensity[i]; }
    else if (mode === "semseg") {
      const p = SEMSEG_PALETTE[data.semseg[i]]; r = p[0]; g = p[1]; b = p[2];
    } else if (mode === "range") {
      const d = Math.min(1.999, Math.hypot(pts[i * 3], pts[i * 3 + 1]) / 40);
      const k = (d | 0) * 3; r = TURBO[k]; g = TURBO[k + 1]; b = TURBO[k + 2];
    } else {
      const z = pts[i * 3 + 2];
      let t = (z + 3.0) / 6.0; t = t < 0 ? 0 : (t > 0.999 ? 0.999 : t);
      const k = (t * 255 | 0) * 3; r = TURBO[k]; g = TURBO[k + 1]; b = TURBO[k + 2];
    }
    out[i * 3] = r / 255; out[i * 3 + 1] = g / 255; out[i * 3 + 2] = b / 255;
  }
  data.colors[ck] = out;
  return out;
}

// in-range mask (range clip from ego origin)
function rangeMask(data) {
  const ck = "mask:" + state.rangeClip;
  if (data.colors[ck]) return data.colors[ck];
  const n = data.header.n, pts = data.points, m = new Uint8Array(n);
  const r2 = state.rangeClip * state.rangeClip;
  for (let i = 0; i < n; i++)
    m[i] = (pts[i * 3] * pts[i * 3] + pts[i * 3 + 1] * pts[i * 3 + 1]) <= r2 ? 1 : 0;
  data.colors[ck] = m;
  return m;
}

function clippedPoints(data, colors) {
  const m = rangeMask(data), pts = data.points, n = data.header.n;
  let c = 0; for (let i = 0; i < n; i++) c += m[i];
  const pos = new Float32Array(c * 3), col = new Float32Array(c * 3);
  let j = 0;
  for (let i = 0; i < n; i++) if (m[i]) {
    pos[j * 3] = pts[i * 3]; pos[j * 3 + 1] = pts[i * 3 + 1]; pos[j * 3 + 2] = pts[i * 3 + 2];
    col[j * 3] = colors[i * 3]; col[j * 3 + 1] = colors[i * 3 + 1]; col[j * 3 + 2] = colors[i * 3 + 2];
    j++;
  }
  return { pos, col, n: c };
}

function boxVisible(label) { return !state.hiddenClasses.has(label); }

// ------------------------------------------------------------ 3D scene
const canvas3d = $("view3d");
const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x0b0e11);
const cam3 = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
cam3.position.set(-26, -26, 19);
cam3.up.set(0, 0, 1);
const controls = new THREE.OrbitControls(cam3, canvas3d);
controls.target.set(12, 0, 0);
scene3.add(new THREE.AxesHelper(3));

const grid3 = new THREE.GridHelper(160, 16, 0x2a3b2f, 0x1a2420);
grid3.rotation.x = Math.PI / 2;
scene3.add(grid3);

// ego vehicle wireframe (X fwd)
{
  const L = 4.9, W = 2.0, H = 1.6;
  const v = [];
  const c = [[-L/2,-W/2,0],[L/2,-W/2,0],[L/2,W/2,0],[-L/2,W/2,0],
             [-L/2,-W/2,H],[L/2,-W/2,H],[L/2,W/2,H],[-L/2,W/2,H]];
  const e = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  for (const [a, b] of e) v.push(...c[a], ...c[b]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  scene3.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x66aaff })));
}

const ptsMat = new THREE.PointsMaterial({ size: 0.06, vertexColors: true, sizeAttenuation: true });
const ptsGeo = new THREE.BufferGeometry();
ptsGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
ptsGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
const points3 = new THREE.Points(ptsGeo, ptsMat);
scene3.add(points3);

const boxGeo = new THREE.BufferGeometry();
boxGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
boxGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
const boxes3 = new THREE.LineSegments(boxGeo, new THREE.LineBasicMaterial({ vertexColors: true }));
scene3.add(boxes3);

const frustumGeo = new THREE.BufferGeometry();
frustumGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
const frustums3 = new THREE.LineSegments(frustumGeo,
  new THREE.LineBasicMaterial({ color: 0x33ccff, transparent: true, opacity: 0.55 }));
scene3.add(frustums3);

// box labels as sprites
const labelTexCache = new Map();
function labelTexture(text, css) {
  const ck = text + "|" + css;
  if (labelTexCache.has(ck)) return labelTexCache.get(ck);
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  g.font = "bold 44px system-ui";
  const w = Math.ceil(g.measureText(text).width) + 22;
  c.width = w; c.height = 58;
  const x = c.getContext("2d");
  x.fillStyle = "rgba(0,0,0,0.55)"; x.fillRect(0, 0, w, 58);
  x.font = "bold 44px system-ui"; x.fillStyle = css;
  x.textBaseline = "middle"; x.fillText(text, 11, 31);
  const tex = new THREE.CanvasTexture(c);
  tex.userData = { aspect: w / 58 };
  if (labelTexCache.size > 300) labelTexCache.delete(labelTexCache.keys().next().value);
  labelTexCache.set(ck, tex);
  return tex;
}
const labelSprites = [];
for (let i = 0; i < 96; i++) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false }));
  sp.visible = false;
  scene3.add(sp);
  labelSprites.push(sp);
}

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function boxCorners(b) {
  const x = b[0], y = b[1], z = b[2], dx = b[3], dy = b[4], dz = b[5], yaw = b[6];
  const c = Math.cos(yaw), s = Math.sin(yaw), out = new Float32Array(24);
  const signs = [[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],
                 [-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]];
  for (let i = 0; i < 8; i++) {
    const lx = signs[i][0] * dx, ly = signs[i][1] * dy, lz = signs[i][2] * dz;
    out[i * 3] = c * lx - s * ly + x;
    out[i * 3 + 1] = s * lx + c * ly + y;
    out[i * 3 + 2] = lz + z;
  }
  return out;
}

function visibleBoxes(data) {
  // indices of boxes with visible class
  const out = [];
  const labels = data.header.labels, nb = data.header.nb;
  for (let k = 0; k < nb; k++)
    if (state.showBoxes && boxVisible(labels[k])) out.push(k);
  return out;
}

function render3d(data) {
  const colors = pointColors(data, state.colorMode, state.rangeClip);
  const cp = clippedPoints(data, colors);
  ptsGeo.setAttribute("position", new THREE.BufferAttribute(cp.pos, 3));
  ptsGeo.setAttribute("color", new THREE.BufferAttribute(cp.col, 3));
  ptsGeo.computeBoundingSphere();
  ptsMat.size = state.ptSize;
  points3.visible = true;

  const idx = visibleBoxes(data);
  const labels = data.header.labels;
  const cols = classColorsFor(labels);
  const pos = new Float32Array(idx.length * 27 * 3), col = new Float32Array(idx.length * 27 * 3);
  let o = 0;
  for (let ii = 0; ii < idx.length; ii++) {
    const k = idx[ii], b = data.boxes.subarray(k * 7, k * 7 + 7);
    const cor = boxCorners(b);
    const c = cols[k].rgb.map(v => v / 255);
    for (const [a, bb] of EDGES) {
      pos.set(cor.subarray(a * 3, a * 3 + 3), o); col.set(c, o); o += 3;
      pos.set(cor.subarray(bb * 3, bb * 3 + 3), o); col.set(c, o); o += 3;
    }
    // heading arrow along box direction
    const hl = b[3] / 2, ca = Math.cos(b[6]), sa = Math.sin(b[6]);
    pos[o] = b[0]; pos[o + 1] = b[1]; pos[o + 2] = b[2]; col.set(c, o); o += 3;
    pos[o] = b[0] + hl * ca; pos[o + 1] = b[1] + hl * sa; pos[o + 2] = b[2]; col.set(c, o); o += 3;
  }
  boxGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  boxGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  boxes3.visible = true;

  // labels
  let li = 0;
  if (state.showLabels3d) {
    for (; li < Math.min(idx.length, labelSprites.length); li++) {
      const k = idx[li], b = data.boxes.subarray(k * 7, k * 7 + 7);
      const sp = labelSprites[li];
      const tex = labelTexture(labels[k] || "?", cols[k].css);
      sp.material.map = tex; sp.material.needsUpdate = true;
      sp.position.set(b[0], b[1], b[2] + b[5] / 2 + 1.2);
      const h = 1.3;
      sp.scale.set(h * tex.userData.aspect, h, 1);
      sp.visible = true;
    }
  }
  for (; li < labelSprites.length; li++) labelSprites[li].visible = false;

  // camera frustums
  const fp = [];
  if (state.showFrustum) {
    for (const cam of CAM_ORDER) {
      const cd = data.header.cams[cam];
      if (!cd) continue;
      const R = quatToMat(cd.R), t = cd.t, K = cd.K;
      const fwd = [R[0][2], R[1][2], R[2][2]];
      const rgt = [R[0][0], R[1][0], R[2][0]];
      const up  = [R[0][1], R[1][1], R[2][1]];
      const L = 3.4, ax = (cd.w / 2) / K[0][0] * L, ay = (cd.h / 2) / K[1][1] * L;
      const cor = [];
      for (const [sa, sb] of [[-1,-1],[1,-1],[1,1],[-1,1]])
        cor.push([t[0] + fwd[0]*L + rgt[0]*ax*sa + up[0]*ay*sb,
                  t[1] + fwd[1]*L + rgt[1]*ax*sa + up[1]*ay*sb,
                  t[2] + fwd[2]*L + rgt[2]*ax*sa + up[2]*ay*sb]);
      for (const c of cor) fp.push(t[0], t[1], t[2], c[0], c[1], c[2]);
      for (let i = 0; i < 4; i++) {
        const a = cor[i], b = cor[(i + 1) % 4];
        fp.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    }
  }
  frustumGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(fp), 3));
  grid3.visible = state.showGrid;
  return cp.n;
}

// ------------------------------------------------------------ BEV radar
const bev = $("bev"), bevCtx = bev.getContext("2d");
const BEV_RANGE = 60;   // meters radius
function renderBev(data) {
  const W = bev.width, cx = W / 2, cy = W / 2, s = (W / 2 - 12) / BEV_RANGE;
  bevCtx.fillStyle = "#080b0e";
  bevCtx.fillRect(0, 0, W, W);
  // range rings
  bevCtx.strokeStyle = "rgba(90,110,125,0.35)";
  bevCtx.lineWidth = 1;
  bevCtx.font = "16px ui-monospace, monospace";
  bevCtx.fillStyle = "rgba(140,160,175,0.55)";
  for (let r = 20; r <= BEV_RANGE; r += 20) {
    bevCtx.beginPath(); bevCtx.arc(cx, cy, r * s, 0, 2 * Math.PI); bevCtx.stroke();
    bevCtx.fillText(r + "m", cx + 3, cy - r * s - 3);
  }
  // cross axes
  bevCtx.strokeStyle = "rgba(90,110,125,0.25)";
  bevCtx.beginPath();
  bevCtx.moveTo(cx, 12); bevCtx.lineTo(cx, W - 12);
  bevCtx.moveTo(12, cy); bevCtx.lineTo(W - 12, cy);
  bevCtx.stroke();

  // points via ImageData
  const colors = pointColors(data, state.colorMode, state.rangeClip);
  const m = rangeMask(data), pts = data.points, n = data.header.n;
  const img = bevCtx.createImageData(W, W);
  const d = img.data;
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!m[i]) continue;
    const px = (cx - pts[i * 3 + 1] * s) | 0, py = (cy - pts[i * 3] * s) | 0;
    if (px < 0 || px >= W - 1 || py < 0 || py >= W - 1) continue;
    const r = Math.min(255, (colors[i * 3] * 255 * 1.45) | 0),
          g = Math.min(255, (colors[i * 3 + 1] * 255 * 1.45) | 0),
          b = Math.min(255, (colors[i * 3 + 2] * 255 * 1.45) | 0);
    for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const q = ((py + oy) * W + px + ox) * 4;
      d[q] = r; d[q + 1] = g; d[q + 2] = b; d[q + 3] = 255;
    }
    cnt++;
  }
  bevCtx.putImageData(img, 0, 0);

  // boxes
  const idx = visibleBoxes(data), labels = data.header.labels, cols = classColorsFor(labels);
  bevCtx.lineWidth = 2.5;
  for (const k of idx) {
    const b = data.boxes.subarray(k * 7, k * 7 + 7);
    const cor = boxCorners(b);
    bevCtx.strokeStyle = cols[k].css;
    bevCtx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = cor.subarray(i * 3, i * 3 + 3), bb = cor.subarray(((i + 1) % 4) * 3, ((i + 1) % 4) * 3 + 3);
      const x1 = cx - a[1] * s, y1 = cy - a[0] * s, x2 = cx - bb[1] * s, y2 = cy - bb[0] * s;
      if (i === 0) bevCtx.moveTo(x1, y1); else bevCtx.lineTo(x1, y1);
      bevCtx.lineTo(x2, y2);
    }
    bevCtx.stroke();
    // heading tick
    const hl = b[3] / 2, ca = Math.cos(b[6]), sa = Math.sin(b[6]);
    bevCtx.beginPath();
    bevCtx.moveTo(cx - b[1] * s, cy - b[0] * s);
    bevCtx.lineTo(cx - (b[1] + hl * sa) * s, cy - (b[0] + hl * ca) * s);
    bevCtx.stroke();
  }

  // ego triangle (points up = forward)
  bevCtx.fillStyle = "#66aaff";
  bevCtx.beginPath();
  bevCtx.moveTo(cx, cy - 10);
  bevCtx.lineTo(cx - 7, cy + 8);
  bevCtx.lineTo(cx + 7, cy + 8);
  bevCtx.closePath();
  bevCtx.fill();
  return cnt;
}

// ------------------------------------------------------------ cameras
const camPanels = {};   // cam -> panel object

function quatToMat(q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const n = w*w + x*x + y*y + z*z, s = n > 0 ? 2 / n : 0;
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
      size: 2.2, vertexColors: true, sizeAttenuation: false }));
    scene2.add(pts2);
    const bgeo = new THREE.BufferGeometry();
    bgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    bgeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    const boxes2 = new THREE.LineSegments(bgeo, new THREE.LineBasicMaterial({ vertexColors: true }));
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
  p.zoomwrap.style.transform = `translate(${p.z.x}px,${p.z.y}px) scale(${p.z.s})`;
  const fs = Math.min(48, Math.max(6, 11 * p.z.s));
  for (const l of p.labels) l.style.fontSize = fs + "px";
  p.pts.material.size = Math.max(1.2, 2.2 / Math.sqrt(p.z.s));
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

  const m = rangeMask(data), n = data.header.n, pts = data.points;
  const colors = pointColors(data, state.colorMode, state.rangeClip);
  const R = quatToMat(cd.R), t = cd.t, K = cd.K;
  const uv = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let mm = 0;
  if (state.showLidar) {
    for (let i = 0; i < n; i++) {
      if (!m[i]) continue;
      const ex = pts[i*3] - t[0], ey = pts[i*3+1] - t[1], ez = pts[i*3+2] - t[2];
      const cz = R[0][2]*ex + R[1][2]*ey + R[2][2]*ez;
      if (cz < 0.5) continue;
      const cx = R[0][0]*ex + R[1][0]*ey + R[2][0]*ez;
      const cy = R[0][1]*ex + R[1][1]*ey + R[2][1]*ez;
      const u = (K[0][0]*cx + K[0][2]*cz) / cz, v = (K[1][1]*cy + K[1][2]*cz) / cz;
      if (u < 0 || u >= w0 || v < 0 || v >= h0) continue;
      uv[mm*3] = u; uv[mm*3+1] = h0 - v; uv[mm*3+2] = 0;
      col[mm*3] = colors[i*3]; col[mm*3+1] = colors[i*3+1]; col[mm*3+2] = colors[i*3+2];
      mm++;
    }
  }
  p.pts.geometry.setAttribute("position", new THREE.BufferAttribute(uv.subarray(0, mm*3), 3));
  p.pts.geometry.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, mm*3), 3));
  p.pts.visible = state.showLidar && mm > 0;

  const idx = visibleBoxes(data);
  const labels = data.header.labels, cols = classColorsFor(labels);
  const bpos = new Float32Array(idx.length * 24 * 3), bcol = new Float32Array(idx.length * 24 * 3);
  let segs = 0;
  const labelInfo = [];
  if (state.showBoxes) {
    for (const k of idx) {
      const b = data.boxes.subarray(k*7, k*7+7);
      const cor = boxCorners(b);
      const cc = new Float32Array(24);
      for (let i = 0; i < 8; i++) {
        const ex = cor[i*3] - t[0], ey = cor[i*3+1] - t[1], ez = cor[i*3+2] - t[2];
        cc[i*3]   = R[0][0]*ex + R[1][0]*ey + R[2][0]*ez;
        cc[i*3+1] = R[0][1]*ex + R[1][1]*ey + R[2][1]*ez;
        cc[i*3+2] = R[0][2]*ex + R[1][2]*ey + R[2][2]*ez;
      }
      let zmin = Infinity;
      for (let i = 0; i < 8; i++) zmin = Math.min(zmin, cc[i*3+2]);
      if (zmin < 0.2) continue;
      const c = cols[k].rgb.map(v => v / 255);
      const uvs = [];
      for (let i = 0; i < 8; i++) {
        const z = cc[i*3+2];
        uvs.push([(K[0][0]*cc[i*3] + K[0][2]*z) / z, h0 - (K[1][1]*cc[i*3+1] + K[1][2]*z) / z]);
      }
      for (const [a, bb] of EDGES) {
        bpos.set([uvs[a][0], uvs[a][1], 0], segs*3); bcol.set(c, segs*3); segs++;
        bpos.set([uvs[bb][0], uvs[bb][1], 0], segs*3); bcol.set(c, segs*3); segs++;
      }
      let top = uvs[4];
      for (let i = 4; i < 8; i++) if (uvs[i][1] < top[1]) top = uvs[i];  // min screen-y = top
      labelInfo.push({ k, x: top[0], y: top[1] });
    }
  }
  p.boxes.geometry.setAttribute("position", new THREE.BufferAttribute(bpos.subarray(0, segs*3), 3));
  p.boxes.geometry.setAttribute("color", new THREE.BufferAttribute(bcol.subarray(0, segs*3), 3));
  p.boxes.visible = state.showBoxes && segs > 0;
  p.labels.forEach((l, i) => {
    if (state.showCamLbl && i < labelInfo.length) {
      const { k, x, y } = labelInfo[i];
      l.textContent = labels[k] || "?";
      l.style.left = x + "px"; l.style.top = y + "px";
      l.style.color = cols[k].css;
      l.style.display = "";
    } else l.style.display = "none";
  });
  p.renderer.render(p.scene, p.cam2d);
}

// ------------------------------------------------------------ map
let leaflet = null, trackLine = null, trackDone = null, posMarker = null;
function initMap() {
  leaflet = L.map("map", { attributionControl: false }).setView([37.422, -122.16], 16);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(leaflet);
  trackLine = L.polyline([], { color: "#3af", weight: 3, opacity: 0.45 }).addTo(leaflet);
  trackDone = L.polyline([], { color: "#3df", weight: 3 }).addTo(leaflet);
  const icon = L.divIcon({
    className: "",
    html: `<div id="ego-arrow"><svg width="26" height="26" viewBox="-13 -13 26 26">
      <polygon points="0,-10 7,9 0,5 -7,9" fill="#ff5533" stroke="#222" stroke-width="1"/></svg></div>`,
    iconSize: [26, 26], iconAnchor: [13, 13],
  });
  posMarker = L.marker([37.422, -122.16], { icon }).addTo(leaflet);
  leaflet.on("click", e => {          // jump to nearest track point
    if (!state.track) return;
    let best = -1, bd = 1e9;
    const pp = leaflet.latLngToLayerPoint(e.latlng);
    for (let i = 0; i < state.track.lat.length; i++) {
      const q = leaflet.latLngToLayerPoint([state.track.lat[i], state.track.lon[i]]);
      const d = (q.x - pp.x) ** 2 + (q.y - pp.y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0 && bd < 400) {      // within ~20 px
      setFrame(best);
    }
  });
}

function updateMap(gps) {
  if (!leaflet || !gps || (!gps.lat && !gps.long)) return;
  const ll = [gps.lat, gps.long];
  posMarker.setLatLng(ll);
  state.gpsTrackDrawn.push(ll);
  trackDone.setLatLngs(state.gpsTrackDrawn);
  const arrow = document.getElementById("ego-arrow");
  if (arrow && state.track) {
    const i = state.frame;
    const i0 = Math.max(0, i - 2), i1 = Math.min(state.track.lat.length - 1, i + 2);
    const dlat = state.track.lat[i1] - state.track.lat[i0];
    const dlon = state.track.lon[i1] - state.track.lon[i0];
    if (dlat || dlon) {
      const brg = Math.atan2(dlon * Math.cos(gps.lat * Math.PI / 180), dlat) * 180 / Math.PI;
      arrow.style.transform = `rotate(${brg}deg)`;
    }
  }
  if (state.followMap) {
    const p = leaflet.latLngToLayerPoint(ll);
    const size = leaflet.getSize();
    if (Math.abs(p.x - size.x / 2) > size.x * 0.3 || Math.abs(p.y - size.y / 2) > size.y * 0.3)
      leaflet.panTo(ll, { animate: true });
  }
}

// ------------------------------------------------------------ timeline
function drawTimeline() {
  const c = $("timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!state.track) return;
  const sp = state.track.speed, n = sp.length;
  const maxS = Math.max(1, ...sp);
  g.beginPath();
  g.moveTo(0, h);
  for (let i = 0; i < n; i++)
    g.lineTo(i / (n - 1) * w, h - sp[i] / maxS * (h - 2 * dpr) - dpr);
  g.lineTo(w, h);
  g.closePath();
  g.fillStyle = "rgba(80,160,120,0.30)";
  g.fill();
  g.strokeStyle = "rgba(110,220,160,0.8)";
  g.lineWidth = dpr;
  g.stroke();
}

function drawTimelineCursor() {
  drawTimeline();
  const c = $("timeline"), g = c.getContext("2d");
  const w = c.width, h = c.height;
  if (state.frames > 1) {
    const x = state.frame / (state.frames - 1) * w;
    g.strokeStyle = "#ffd34d";
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
}
$("timeline").addEventListener("click", e => {
  const r = e.target.getBoundingClientRect();
  setFrame(Math.round((e.clientX - r.left) / r.width * (state.frames - 1)));
});

// ------------------------------------------------------------ legend
function buildLegend() {
  const body = $("legendBody");
  body.innerHTML = "";
  const entries = Object.entries(state.boxClasses).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    body.innerHTML = "<div style='color:#7d8f9d;padding:2px 4px'>no boxes in sequence</div>";
    return;
  }
  for (const [name, cnt] of entries) {
    const col = state.classColors[name] || (state.classColors[name] = classColor(name));
    const chip = document.createElement("div");
    chip.className = "chip" + (state.hiddenClasses.has(name) ? " off" : "");
    chip.innerHTML = `<span class="sw" style="background:${col.css}"></span>` +
      `<span>${name}</span><span class="cnt">${fmt(cnt)}</span>`;
    chip.onclick = () => {
      if (state.hiddenClasses.has(name)) state.hiddenClasses.delete(name);
      else state.hiddenClasses.add(name);
      buildLegend();
      render();
    };
    body.append(chip);
  }
}

function buildSemsegLegend() {
  const body = $("legendBody");
  body.innerHTML = "";
  const entries = Object.entries(state.semsegClasses).sort((a, b) => a[0] - b[0]);
  if (!entries.length) {
    body.innerHTML = "<div style='color:#7d8f9d;padding:2px 4px'>no semseg classes</div>";
    return;
  }
  for (const [id, name] of entries) {
    const p = SEMSEG_PALETTE[+id] || [128, 128, 128];
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `<span class="sw" style="background:rgb(${p[0]},${p[1]},${p[2]})"></span>` +
      `<span>${name}</span>`;
    body.append(chip);
  }
}

function refreshLegend() {
  if (state.colorMode === "semseg") {
    $("legendTitle").textContent = "semseg classes";
    buildSemsegLegend();
  } else {
    $("legendTitle").textContent = "box classes";
    buildLegend();
  }
}

// ------------------------------------------------------------ sequencing
function syncUrl() {
  const name = state.sweeps[state.sweep];
  history.replaceState(null, "", `/?seq=${name}&frame=${state.frame}`);
}

function setFrame(f, keepUrl) {
  state.frame = Math.max(0, Math.min(f, state.frames - 1));
  $("slider").value = state.frame;
  prefetch();
  render();
  if (!keepUrl) syncUrl();
}

async function selectSweep(i, frame = 0) {
  state.sweep = i;
  state.gpsTrackDrawn = [];
  state.cache.clear();
  $("loading").classList.add("show");
  const meta = await api(`/api/meta?sweep=${i}`);
  state.frames = meta.frames;
  state.hasSemseg = meta.has_semseg;
  state.semsegClasses = meta.semseg_classes;
  state.boxClasses = meta.box_classes;
  if (!state.hasSemseg && state.colorMode === "semseg") {
    state.colorMode = "height";
    $("colorMode").value = "height";
  }
  state.frame = Math.min(frame, meta.frames - 1);
  $("slider").max = Math.max(0, meta.frames - 1);
  buildCamPanels(meta.cameras);
  refreshLegend();
  // gps track + timeline
  state.track = await api(`/api/gps_track?sweep=${i}`);
  const pts = [];
  for (let k = 0; k < state.track.lat.length; k++)
    pts.push([state.track.lat[k], state.track.lon[k]]);
  trackLine.setLatLngs(pts);
  trackDone.setLatLngs([]);
  const first = pts.find(p => p[0] || p[1]) || [37.422, -122.16];
  leaflet.setView(first, 16);
  drawTimeline();
  $("loading").classList.remove("show");
  prefetch();
  render();
  syncUrl();
}

// ------------------------------------------------------------ render
let renderSeq = 0;
async function render() {
  const s = state.sweep, f = state.frame, seq = ++renderSeq;
  const data = await loadFrame(s, f).catch(() => null);
  // discard only if the sweep changed or a newer render superseded this one;
  // while playing the frame may legitimately have advanced during the fetch
  if (!data || seq !== renderSeq || state.sweep !== s) return;
  const nShown = render3d(data);
  if (state.showBev) { $("bev-wrap").style.display = ""; renderBev(data); }
  else $("bev-wrap").style.display = "none";
  for (const cam of CAM_ORDER) if (camPanels[cam]) renderCamPanel(camPanels[cam], data);
  updateMap(data.header.gps);
  drawTimelineCursor();

  const spd = data.header.gps.speed || 0;
  $("speedo").textContent = (spd * 3.6).toFixed(1) + " km/h";
  $("frameLbl").textContent = `f ${f + 1}/${state.frames}`;
  $("tsLbl").textContent = `ts ${data.header.ts}`;
  $("ptsLbl").textContent = `pts ${fmt(nShown)}`;
  $("boxLbl").textContent = `boxes ${visibleBoxes(data).length}`;
  $("netLbl").textContent = `${state.lastLoadMs.toFixed(0)} ms · ${(state.netKB / 1024).toFixed(1)} MiB`;
}

// ------------------------------------------------------------ main loop
let lastAdvance = 0, fpsCnt = 0, fpsT0 = performance.now();
function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene3, cam3);
  fpsCnt++;
  const now = performance.now();
  if (now - fpsT0 > 1000) {
    $("fpsLbl").textContent = fpsCnt + " fps";
    fpsCnt = 0; fpsT0 = now;
  }
  if (state.playing && state.frames > 1 &&
      now - lastAdvance >= 1000 / state.dataFps) {
    lastAdvance = now;
    state.frame = (state.frame + 1) % state.frames;
    $("slider").value = state.frame;
    prefetch();
    render();
    syncUrl();
  }
}

function resize() {
  const w = canvas3d.clientWidth, h = canvas3d.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  cam3.aspect = w / h;
  cam3.updateProjectionMatrix();
  drawTimelineCursor();
}
window.addEventListener("resize", resize);

// ------------------------------------------------------------ UI wiring
$("btnPlay").onclick = () => {
  state.playing = !state.playing;
  $("btnPlay").textContent = state.playing ? "Stop" : "Start";
};
$("btnPrev").onclick = () => { pause(); setFrame(state.frame - 1); };
$("btnNext").onclick = () => { pause(); setFrame(state.frame + 1); };
function pause() {
  if (state.playing) { state.playing = false; $("btnPlay").textContent = "Start"; }
}
$("fpsSel").onchange = e => { state.dataFps = +e.target.value; };
$("chkLidar").onchange = e => { state.showLidar = e.target.checked; render(); };
$("chkBoxes").onchange = e => { state.showBoxes = e.target.checked; render(); };
$("chkCamLbl").onchange = e => { state.showCamLbl = e.target.checked; render(); };
$("chkFollow").onchange = e => { state.followMap = e.target.checked; };
$("chkNms").onchange = e => { state.showNms = e.target.checked; state.cache.clear(); render(); };
$("chkFrustum").onchange = e => { state.showFrustum = e.target.checked; render(); };
$("chkGrid").onchange = e => { state.showGrid = e.target.checked; render(); };
$("chkLabels3d").onchange = e => { state.showLabels3d = e.target.checked; render(); };
$("chkBev").onchange = e => { state.showBev = e.target.checked; render(); };
$("colorMode").onchange = e => {
  state.colorMode = e.target.value;
  refreshLegend();
  render();
};
$("ptSize").oninput = e => { state.ptSize = +e.target.value / 100; render(); };
$("rangeClip").oninput = e => {
  const v = +e.target.value;
  state.rangeClip = v >= 120 ? Infinity : v;
  $("rangeClipLbl").textContent = state.rangeClip === Infinity ? "∞" : v + " m";
  render();
};
$("sweepSel").onchange = e => selectSweep(+e.target.value);
$("slider").oninput = e => { pause(); setFrame(+e.target.value); };
$("btnHelp").onclick = () => $("help").hidden = !$("help").hidden;
$("btnHelpClose").onclick = () => $("help").hidden = true;

window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); $("btnPlay").click(); }
  else if (e.key === "n") { pause(); setFrame(state.frame + 1); }
  else if (e.key === "b") { pause(); setFrame(state.frame - 1); }
  else if (e.key === "N") selectSweep((state.sweep + 1) % state.sweeps.length);
  else if (e.key === "B") selectSweep((state.sweep - 1 + state.sweeps.length) % state.sweeps.length);
  else if (e.key === "c") {
    const modes = state.hasSemseg
      ? ["height", "intensity", "range", "semseg"] : ["height", "intensity", "range"];
    const m = modes[(modes.indexOf(state.colorMode) + 1) % modes.length];
    $("colorMode").value = m; state.colorMode = m;
    refreshLegend(); render();
  } else if (e.key === "h" || e.key === "?") $("help").hidden = !$("help").hidden;
  else if (e.key === "Escape") $("help").hidden = true;
});

// ------------------------------------------------------------ init
(async function init() {
  const d = await api("/api/sweeps");
  state.sweeps = d.sweeps; state.counts = d.counts;
  const sel = $("sweepSel");
  d.sweeps.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${s} (${d.counts[i]})`;
    sel.append(o);
  });
  const params = new URLSearchParams(location.search);
  const wantSeq = params.get("seq");
  let start = wantSeq && d.sweeps.includes(wantSeq) ? d.sweeps.indexOf(wantSeq) : 0;
  const wantFrame = parseInt(params.get("frame") || "0", 10) || 0;
  sel.value = start;
  initMap();
  resize();
  await selectSweep(start, wantFrame);
  setInterval(prefetch, 500);
  tick();
})();
