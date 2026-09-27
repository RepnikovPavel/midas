/* PandaSet WebGL viewer frontend. Data: /api/* (aiohttp backend).
   Ego frame: X forward, Y left, Z up. Map tiles (c) OpenStreetMap. */
"use strict";

const CAM_ORDER = ["front_left_camera", "front_camera", "front_right_camera",
                   "left_camera", "back_camera", "right_camera"];

// ------------------------------------------------------------ colormaps
const clamp255 = v => v < 0 ? 0 : (v > 255 ? 255 : v);
const TURBO = new Uint8Array(256 * 3);   // range mode (light background)
const JET = new Uint8Array(256 * 3);     // classic height (light background)
const VIRIDIS = new Uint8Array(256 * 3); // height over the light OSM raster
const INFERNO = new Uint8Array(256 * 3); // range over the light OSM raster
function rampLUT(stops) {  // linear interpolation over control points
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const a = Math.min(stops.length - 2, Math.floor(t)), f = t - a;
    for (let c = 0; c < 3; c++)
      lut[i * 3 + c] = clamp255(255 * (stops[a][c] * (1 - f) + stops[a + 1][c] * f));
  }
  return lut;
}
VIRIDIS.set(rampLUT([
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]]));
INFERNO.set(rampLUT([
  [0.001, 0.000, 0.014], [0.087, 0.044, 0.224], [0.258, 0.039, 0.406],
  [0.416, 0.090, 0.433], [0.578, 0.148, 0.404], [0.735, 0.215, 0.330],
  [0.866, 0.317, 0.226], [0.955, 0.464, 0.120], [0.988, 0.645, 0.040],
  [0.965, 0.843, 0.142], [0.988, 0.998, 0.645]]));
{ // polynomial approximation of Google's turbo
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    TURBO[i * 3]     = clamp255(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))));
    TURBO[i * 3 + 1] = clamp255(23.06 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3220.32 - t * 1320.25)))));
    TURBO[i * 3 + 2] = clamp255(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814 + t * (-22569.18 + t * 6838.66)))));
  }
}
{ // classic jet: dark blue -> cyan -> green -> yellow -> red
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    JET[i * 3]     = clamp255(255 * (1.5 - Math.abs(4 * t - 3)));
    JET[i * 3 + 1] = clamp255(255 * (1.5 - Math.abs(4 * t - 2)));
    JET[i * 3 + 2] = clamp255(255 * (1.5 - Math.abs(4 * t - 1)));
  }
}
const Z_MIN = -2.5, Z_MAX = 7.5;         // height colormap range (m)
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
  const a = 0.85, b = 0.55;
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
  showMap3d: true, showMapBev: true, showGrid: false, showLabels3d: true,
  showBev: true, showRoads: true, showBevScale: true, gridStep: 10, followMap: true,
  ptSize: 0.06, rangeClip: Infinity,
  hasSemseg: false, semsegClasses: {}, boxClasses: {},
  hiddenClasses: new Set(), classColors: {},
  track: null,               // gps track + world<->ENU alignment of current sweep
  roadsENU: null,            // OSM road graph of current sweep, ENU polylines
  gpsTrackDrawn: [],
  cache: new Map(),
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

// fp16 -> fp32 (points arrive half-size over the wire); proper bit conversion
const F16_SCALE = [];
for (let e = 0; e <= 30; e++) F16_SCALE[e] = Math.pow(2, e - 15);
const F16_SUB = Math.pow(2, -24);
function decodeF16(bits, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bits[i];
    const e = (h & 0x7c00) >> 10;
    const m = h & 0x03ff;
    if (e === 0) out[i] = (h & 0x8000 ? -1 : 1) * m * F16_SUB;
    else if (e === 31) out[i] = m ? NaN : (h & 0x8000 ? -Infinity : Infinity);
    else out[i] = (h & 0x8000 ? -1 : 1) * (m / 1024 + 1) * F16_SCALE[e];
  }
  return out;
}

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
    let points;
    if (header.pts_dtype === "float16") {
      const bytes = n * 6;
      const src = (off & 1) === 0
        ? new Uint16Array(buf, off, n * 3)
        : new Uint16Array(buf.slice(off, off + bytes));
      points = decodeF16(src, n * 3);
      off += bytes;
    } else {
      points = new Float32Array(buf.slice(off, off + n * 12));
      off += n * 12;
    }
    const intensity = new Uint8Array(buf, off, n); off += n;
    const semseg = new Uint8Array(buf, off, n); off += n;
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

// dark=true selects the high-contrast palette used over the light OSM raster
function pointColors(data, mode, clip2, dark) {
  const ck = mode + ":" + clip2 + ":" + (dark ? 1 : 0);
  if (data.colors[ck]) return data.colors[ck];
  const n = data.header.n, pts = data.points;
  const out = new Float32Array(n * 3);
  const hLut = dark ? VIRIDIS : JET;
  const rLut = dark ? INFERNO : TURBO;
  for (let i = 0; i < n; i++) {
    let r, g, b;
    if (mode === "intensity") { r = g = b = dark ? 40 + data.intensity[i] * 0.75 : data.intensity[i]; }
    else if (mode === "semseg") {
      const p = SEMSEG_PALETTE[data.semseg[i]]; r = p[0]; g = p[1]; b = p[2];
    } else if (mode === "range") {
      const d = Math.min(1.999, Math.hypot(pts[i * 3], pts[i * 3 + 1]) / 40);
      const k = (d | 0) * 3; r = rLut[k]; g = rLut[k + 1]; b = rLut[k + 2];
    } else {
      let t = (pts[i * 3 + 2] - Z_MIN) / (Z_MAX - Z_MIN);
      t = t < 0 ? 0 : (t > 0.999 ? 0.999 : t);
      const k = (t * 255 | 0) * 3; r = hLut[k]; g = hLut[k + 1]; b = hLut[k + 2];
    }
    out[i * 3] = r / 255; out[i * 3 + 1] = g / 255; out[i * 3 + 2] = b / 255;
  }
  data.colors[ck] = out;
  return out;
}

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

function clippedPoints(data, colors, dark) {
  const ck = "cp:" + state.colorMode + ":" + state.rangeClip + ":" + (dark ? 1 : 0);
  if (data.colors[ck]) return data.colors[ck];
  const m = rangeMask(data), pts = data.points, n = data.header.n;
  let c = 0; for (let i = 0; i < n; i++) c += m[i];
  const pos = new Float32Array(c * 3), col = new Float32Array(c * 3);
  let j = 0;
  for (let i = 0; i < n; i++) if (m[i]) {
    pos[j * 3] = pts[i * 3]; pos[j * 3 + 1] = pts[i * 3 + 1]; pos[j * 3 + 2] = pts[i * 3 + 2];
    col[j * 3] = colors[i * 3]; col[j * 3 + 1] = colors[i * 3 + 1]; col[j * 3 + 2] = colors[i * 3 + 2];
    j++;
  }
  const out = { pos, col, n: c };
  data.colors[ck] = out;
  return out;
}

function boxVisible(label) { return !state.hiddenClasses.has(label); }

function quatToMat(q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const n = w*w + x*x + y*y + z*z, s = n > 0 ? 2 / n : 0;
  const wx = s*w*x, wy = s*w*y, wz = s*w*z, xx = s*x*x, xy = s*x*y, xz = s*x*z,
        yy = s*y*y, yz = s*y*z, zz = s*z*z;
  return [[1-(yy+zz), xy-wz, xz+wy], [xy+wz, 1-(xx+zz), yz-wx], [xz-wy, yz+wx, 1-(xx+yy)]];
}

// ------------------------------------------------------------ OSM tiles
// Tiles (c) OpenStreetMap contributors; served pre-downloaded from the
// pipeline (pandaset_pipe.osmtiles) via /api/tile and positioned via the
// GPS<->world alignment. One trip-wide canvas per sequence: pre-rendered with
// margin over the whole route, then moved along the trajectory per frame.
const R_EARTH = 6378137, CIRC = 2 * Math.PI * R_EARTH;
const tiles = { img: new Map() };

function tileImg(z, x, y) {
  const k = z + "/" + x + "/" + y;
  let t = tiles.img.get(k);
  if (!t) {
    t = { img: new Image(), ok: false };
    t.img.onload = () => { t.ok = true; requestRender(); };
    t.img.src = `/api/tile/${z}/${x}/${y}.png`;
    tiles.img.set(k, t);
    if (tiles.img.size > 600) {  // simple LRU trim
      const first = tiles.img.keys().next().value;
      tiles.img.delete(first);
    }
  }
  return t;
}

function lon2px(lon, z) { return ((lon / 360 + 0.5) * 256 * 2 ** z); }
function lat2py(lat, z) {
  const s = Math.sin(lat * Math.PI / 180);
  return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * 256 * 2 ** z;
}
function gpx2mx(px, z) { return ((px / (256 * 2 ** z)) - 0.5) * CIRC; }
function gpx2my(py, z) { return (0.5 - (py / (256 * 2 ** z))) * CIRC; }

// per-frame ENU -> ego mapping (uses gps/world alignment + ego pose)
function frameGeo(header) {
  const aln = state.track && state.track.aln;
  if (!aln || !header.e2g) return null;
  const R = quatToMat(header.e2g.slice(3, 7));   // world_from_ego rotation
  const tw = header.e2g.slice(0, 3);             // ego position in world
  const cA = Math.cos(aln.rot), sA = Math.sin(aln.rot);
  const ca = Math.cos(aln.lat0 * Math.PI / 180);
  const mx0 = gpx2mx(lon2px(aln.lon0, 0), 0);
  const my0 = gpx2my(lat2py(aln.lat0, 0), 0);
  function map(e, n) {
    const de = e - aln.t[0], dn = n - aln.t[1];
    const x = ( cA * de + sA * dn) / aln.s;
    const y = (-sA * de + cA * dn) / aln.s;
    const dx = x - tw[0], dy = y - tw[1];
    return [R[0][0]*dx + R[1][0]*dy,
            R[0][1]*dx + R[1][1]*dy,
            R[0][2]*dx + R[1][2]*dy];
  }
  // ego ENU position (for the marker / debugging)
  const eE = aln.s * (cA * tw[0] - sA * tw[1]) + aln.t[0];
  const nE = aln.s * (sA * tw[0] + cA * tw[1]) + aln.t[1];
  return { aln, ca, mx0, my0, map, eE, nE };
}

// ---- trip-wide map: one canvas per sequence, pre-rendered over the route ----
const PLAN_MARGIN = 120, PLAN_MAX_PX = 3840, PLAN_ZMIN = 15, PLAN_ZMAX = 18;
let tripMap = null;   // {canvas, plan, eTL, nTL, mPerPx, painted, paintedCount, total, tex, lastUpload}

function computePlanFallback(aln) {
  // mirrors pandaset_pipe.osmtiles.sweep_plan (used only if the plan file is absent)
  const tr = state.track, ca = Math.cos(aln.lat0 * Math.PI / 180);
  let e0 = 1e18, e1 = -1e18, n0 = 1e18, n1 = -1e18;
  for (let i = 0; i < tr.lat.length; i++) {
    if (!tr.lat[i] && !tr.lon[i]) continue;
    const e = (tr.lon[i] - aln.lon0) * Math.PI / 180 * R_EARTH * ca;
    const n = (tr.lat[i] - aln.lat0) * Math.PI / 180 * R_EARTH;
    if (e < e0) e0 = e; if (e > e1) e1 = e;
    if (n < n0) n0 = n; if (n > n1) n1 = n;
  }
  e0 -= PLAN_MARGIN; e1 += PLAN_MARGIN; n0 -= PLAN_MARGIN; n1 += PLAN_MARGIN;
  const mpp = z => CIRC * ca / (256 * 2 ** z);
  let z = PLAN_ZMAX;
  while (z > PLAN_ZMIN && Math.max(e1 - e0, n1 - n0) / mpp(z) > PLAN_MAX_PX) z--;
  const lonAt = e => aln.lon0 + e / (R_EARTH * ca) * 180 / Math.PI;
  const latAt = n => aln.lat0 + n / R_EARTH * 180 / Math.PI;
  const px0 = lon2px(lonAt(e0), z), px1 = lon2px(lonAt(e1), z);
  const py0 = lat2py(latAt(n1), z), py1 = lat2py(latAt(n0), z);
  const x0 = Math.floor(px0 / 256), x1 = Math.floor((px1 - 1e-6) / 256);
  const y0 = Math.floor(py0 / 256), y1 = Math.floor((py1 - 1e-6) / 256);
  return { z, x0, y0, ntx: x1 - x0 + 1, nty: y1 - y0 + 1,
           lat0: aln.lat0, lon0: aln.lon0 };
}

async function buildTripMap() {
  const aln = state.track && state.track.aln;
  if (!aln) { tripMap = null; return; }
  let plan = null;
  try { plan = await api(`/api/osm_plan?sweep=${state.sweep}`); } catch (e) { /* offline */ }
  if (!plan || !plan.z) plan = computePlanFallback(aln);
  const canvas = document.createElement("canvas");
  canvas.width = plan.ntx * 256;
  canvas.height = plan.nty * 256;
  if (tripMap && tripMap.tex) tripMap.tex.dispose();
  const ca = Math.cos(plan.lat0 * Math.PI / 180);
  tripMap = {
    canvas, plan, painted: new Set(), paintedCount: 0,
    total: plan.ntx * plan.nty, tex: null, lastUpload: 0,
    mPerPx: CIRC * ca / (256 * 2 ** plan.z),
    eTL: (gpx2mx(plan.x0 * 256, plan.z) - gpx2mx(lon2px(plan.lon0, plan.z), plan.z)) * ca,
    nTL: (gpx2my(plan.y0 * 256, plan.z) - gpx2my(lat2py(plan.lat0, plan.z), plan.z)) * ca,
  };
}

function paintTrip() {  // progressively paint newly loaded tiles; true if changed
  const tm = tripMap;
  if (!tm || tm.paintedCount === tm.total) return false;
  const ctx = tm.canvas.getContext("2d"), p = tm.plan;
  let changed = false;
  for (let i = 0; i < p.ntx; i++) for (let j = 0; j < p.nty; j++) {
    const kk = i + "," + j;
    if (tm.painted.has(kk)) continue;
    const t = tileImg(p.z, p.x0 + i, p.y0 + j);
    if (t.ok) {
      ctx.drawImage(t.img, i * 256, j * 256);
      tm.painted.add(kk);
      tm.paintedCount++;
      changed = true;
    }
  }
  return changed;
}

// map patch for the current frame: trip canvas + geo mapping
function mapPatch(header) {
  const g = frameGeo(header);
  if (!g || !tripMap) return null;
  const changed = paintTrip();
  return { g, canvas: tripMap.canvas, eTL: tripMap.eTL, nTL: tripMap.nTL,
           mPerPx: tripMap.mPerPx, changed };
}

// ------------------------------------------------------------ 3D scene
const canvas3d = $("view3d");
const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true,
                                           preserveDrawingBuffer: true });
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x0b0e11);
const cam3 = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
cam3.position.set(-26, -26, 19);
cam3.up.set(0, 0, 1);
const controls = new THREE.OrbitControls(cam3, canvas3d);
controls.target.set(12, 0, 0);
scene3.add(new THREE.AxesHelper(3));

let grid3 = null;
function setGridStep(step) {
  if (grid3) scene3.remove(grid3);
  const size = Math.max(240, step * 24);
  grid3 = new THREE.GridHelper(size, Math.round(size / step), 0x2a3b2f, 0x1a2420);
  grid3.rotation.x = Math.PI / 2;
  grid3.material.transparent = true;
  grid3.material.opacity = 0.9;
  grid3.renderOrder = -1;
  grid3.visible = state.showGrid;
  scene3.add(grid3);
}
setGridStep(10);

// OSM road graph over the raster: flat quads (WebGL lines are 1px) + node dots
const roadsMat = new THREE.MeshBasicMaterial({ color: 0x101010, transparent: true,
  opacity: 0.85, depthWrite: false, depthTest: false, side: THREE.DoubleSide });
const roadsMesh = new THREE.Mesh(new THREE.BufferGeometry(), roadsMat);
roadsMesh.renderOrder = -1;
roadsMesh.visible = false;
scene3.add(roadsMesh);
const roadsNodes = new THREE.Points(
  new THREE.BufferGeometry(),
  new THREE.PointsMaterial({ color: 0x101010, size: 1.1, sizeAttenuation: true,
    transparent: true, opacity: 0.9, depthWrite: false, depthTest: false }));
roadsNodes.renderOrder = -1;
roadsNodes.visible = false;
scene3.add(roadsNodes);

// OSM map on the ground plane. Drawn first with depthTest off: it is a pure
// underlay — points/boxes always render over it, never dive under the raster
// where terrain slopes away from the ego-local ground height.
const mapPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9,
                                depthWrite: false, depthTest: false,
                                side: THREE.DoubleSide }));
mapPlane.renderOrder = -2;
mapPlane.visible = false;
scene3.add(mapPlane);

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
  const egoBox = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    color: 0x66aaff, transparent: true, opacity: 1 }));
  egoBox.renderOrder = -1;
  scene3.add(egoBox);
}

// transparent+renderOrder channels the draw order explicitly:
// map(-2) -> roads(-1) -> grid/ego/points/boxes(0) -> labels
const ptsMat = new THREE.PointsMaterial({ size: 0.06, vertexColors: true,
  sizeAttenuation: true, transparent: true, opacity: 1 });
const ptsGeo = new THREE.BufferGeometry();
ptsGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
ptsGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
const points3 = new THREE.Points(ptsGeo, ptsMat);
scene3.add(points3);

const boxGeo = new THREE.BufferGeometry();
boxGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
boxGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
const boxes3 = new THREE.LineSegments(boxGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 1 }));
scene3.add(boxes3);

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
for (let i = 0; i < 256; i++) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false }));
  sp.visible = false;
  scene3.add(sp);
  labelSprites.push(sp);
}

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const BOX_SIGNS = [[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],
                   [-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]];
function colorFor(label) {
  return state.classColors[label] || (state.classColors[label] = classColor(label));
}
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
  const out = [];
  const labels = data.header.labels, nb = data.header.nb;
  for (let k = 0; k < nb; k++)
    if (state.showBoxes && boxVisible(labels[k])) out.push(k);
  return out;
}

// local ground height in ego frame (median of near-field points), cached per frame
function groundLevel(data) {
  if (data.colors.gz !== undefined) return data.colors.gz;
  const pts = data.points, m = rangeMask(data), n = data.header.n;
  const zs = [];
  for (let i = 0; i < n; i++) {
    if (!m[i]) continue;
    const x = pts[i * 3], y = pts[i * 3 + 1];
    if (x * x + y * y < 100) zs.push(pts[i * 3 + 2]);   // within 10 m
  }
  let g;
  if (zs.length > 50) { zs.sort((a, b) => a - b); g = zs[zs.length >> 1] - 0.25; }
  else g = -1.9;
  data.colors.gz = g;
  return g;
}

function render3d(data) {
  const colors = pointColors(data, state.colorMode, state.rangeClip, state.showMap3d);
  const cp = clippedPoints(data, colors, state.showMap3d);
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
    const hl = b[3] / 2, ca = Math.cos(b[6]), sa = Math.sin(b[6]);
    pos[o] = b[0]; pos[o + 1] = b[1]; pos[o + 2] = b[2]; col.set(c, o); o += 3;
    pos[o] = b[0] + hl * ca; pos[o + 1] = b[1] + hl * sa; pos[o + 2] = b[2]; col.set(c, o); o += 3;
  }
  boxGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  boxGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  boxes3.visible = true;

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

  grid3.visible = state.showGrid;

  // ---- OSM map on the ground plane (trip-wide canvas) ----
  if (state.showMap3d) {
    const mp = mapPatch(data.header);
    if (mp) {
      const Wpx = mp.canvas.width, Hpx = mp.canvas.height;
      // flatten in the EGO frame (same projection the BEV uses): the pandaset
      // world frame is not gravity-aligned, so a world-horizontal plane would
      // appear tilted; take (x, y) only and pin z to the local ground
      const gz = groundLevel(data);
      const c00 = mp.g.map(mp.eTL, mp.nTL); c00[2] = gz;
      const c10 = mp.g.map(mp.eTL + Wpx * mp.mPerPx, mp.nTL); c10[2] = gz;
      const c01 = mp.g.map(mp.eTL, mp.nTL - Hpx * mp.mPerPx); c01[2] = gz;
      // texture: one per trip canvas; refresh (throttled) while tiles arrive
      const tm = tripMap;
      if (!tm.tex) {
        tm.tex = new THREE.CanvasTexture(tm.canvas);
        tm.tex.flipY = false;
        tm.lastUpload = performance.now();
      } else if (mp.changed && (tm.paintedCount === tm.total ||
                                performance.now() - tm.lastUpload > 400)) {
        tm.tex.needsUpdate = true;
        tm.lastUpload = performance.now();
      }
      mapPlane.material.map = tm.tex;
      mapPlane.material.needsUpdate = true;
      const u = [c10[0] - c00[0], c10[1] - c00[1], c10[2] - c00[2]];
      const v = [c01[0] - c00[0], c01[1] - c00[1], c01[2] - c00[2]];
      const lu = Math.hypot(...u) || 1, lv = Math.hypot(...v) || 1;
      const ux = new THREE.Vector3(u[0] / lu, u[1] / lu, u[2] / lu);
      const vy = new THREE.Vector3(v[0] / lv, v[1] / lv, v[2] / lv);
      const nz = new THREE.Vector3().crossVectors(ux, vy);
      const m = new THREE.Matrix4().makeBasis(ux, vy, nz);
      mapPlane.quaternion.setFromRotationMatrix(m);
      mapPlane.scale.set(lu, lv, 1);
      mapPlane.position.set(c00[0] + u[0] / 2 + v[0] / 2, c00[1] + u[1] / 2 + v[1] / 2,
                            c00[2] + u[2] / 2 + v[2] / 2);
      mapPlane.visible = true;
    } else mapPlane.visible = false;
  } else mapPlane.visible = false;

  // ---- OSM road graph (ego frame, just above the ground plane) ----
  const mp0 = mapPatch(data.header);
  if (state.showRoads && state.roadsENU && mp0) {
    const gz = groundLevel(data) + 0.35;
    const quad = [], nodes = [];
    const HALF_W = 0.45;  // road ribbon half-width, m
    for (const way of state.roadsENU) {
      let prev = null;
      for (const [e, n] of way) {
        const p = mp0.g.map(e, n);
        nodes.push(p[0], p[1], gz + 0.05);
        if (prev) {
          const dx = p[0] - prev[0], dy = p[1] - prev[1];
          const l = Math.hypot(dx, dy) || 1;
          const nx = -dy / l * HALF_W, ny = dx / l * HALF_W;
          // two triangles of a flat quad
          quad.push(prev[0] + nx, prev[1] + ny, gz,
                    prev[0] - nx, prev[1] - ny, gz,
                    p[0] - nx, p[1] - ny, gz,
                    prev[0] + nx, prev[1] + ny, gz,
                    p[0] - nx, p[1] - ny, gz,
                    p[0] + nx, p[1] + ny, gz);
        }
        prev = p;
      }
    }
    roadsMesh.geometry.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(quad), 3));
    roadsMesh.visible = quad.length > 0;
    roadsNodes.geometry.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(nodes), 3));
    roadsNodes.visible = nodes.length > 0;
  } else {
    roadsMesh.visible = false;
    roadsNodes.visible = false;
  }

  return cp.n;
}

// ------------------------------------------------------------ BEV radar
const bev = $("bev"), bevCtx = bev.getContext("2d");
// offscreen canvas for points: putImageData would wipe the map underlay,
// so points are composed with drawImage (alpha) instead
const bevPtsCv = document.createElement("canvas");
bevPtsCv.width = bev.width; bevPtsCv.height = bev.height;
const bevPtsCtx = bevPtsCv.getContext("2d");
let bevImgData = null;                     // reused point buffer (no per-frame alloc)
const bevView = { z: 1, x: 0, y: 0 };   // zoom factor + pan px
const BEV_BASE_R = 60;                   // meters radius at zoom 1

function renderBev(data) {
  const W = bev.width, cx = W / 2, cy = W / 2;
  const scale = (W / 2 - 12) * bevView.z / BEV_BASE_R;
  const px = (ex, ey) => [cx + bevView.x - ey * scale, cy + bevView.y - ex * scale];
  bevCtx.fillStyle = "#080b0e";
  bevCtx.fillRect(0, 0, W, W);

  // ---- OSM underlay (trip-wide canvas) ----
  if (state.showMapBev) {
    const mp = mapPatch(data.header);
    if (mp) {
      const [sx0, sy0] = px(...mp.g.map(mp.eTL, mp.nTL).slice(0, 2));
      const [sx1, sy1] = px(...mp.g.map(mp.eTL + mp.canvas.width * mp.mPerPx, mp.nTL).slice(0, 2));
      const [sx2, sy2] = px(...mp.g.map(mp.eTL, mp.nTL - mp.canvas.height * mp.mPerPx).slice(0, 2));
      const w = mp.canvas.width, h = mp.canvas.height;
      const a = (sx1 - sx0) / w, b = (sy1 - sy0) / w;
      const c = (sx2 - sx0) / h, d = (sy2 - sy0) / h;
      if ((a * a + b * b) > 1e-12) {
        bevCtx.save();
        bevCtx.setTransform(a, b, c, d, sx0, sy0);
        bevCtx.globalAlpha = 0.85;
        bevCtx.drawImage(mp.canvas, 0, 0);
        bevCtx.restore();
        bevCtx.setTransform(1, 0, 0, 1, 0, 0);
      }
    }
  }

  // ---- range rings ----
  const visR = (W / 2 + Math.max(Math.abs(bevView.x), Math.abs(bevView.y))) / scale;
  let step = 5;
  for (const s of [1, 2, 5, 10, 20, 50, 100, 200])
    { step = s; if (s * scale >= 60) break; }
  bevCtx.strokeStyle = "rgba(90,110,125,0.4)";
  bevCtx.lineWidth = 1;
  for (let r = step; r <= visR + step; r += step) {
    bevCtx.beginPath(); bevCtx.arc(cx + bevView.x, cy + bevView.y, r * scale, 0, 2 * Math.PI);
    bevCtx.stroke();
  }

  // ---- metric grid (same step as the 3D grid) ----
  {
    const step = state.gridStep * scale;
    if (step > 14) {
      bevCtx.strokeStyle = "rgba(70,90,110,0.35)";
      bevCtx.lineWidth = 1;
      const ox = (cx + bevView.x) % step, oy = (cy + bevView.y) % step;
      bevCtx.beginPath();
      for (let x = ox; x < W; x += step) { bevCtx.moveTo(x, 0); bevCtx.lineTo(x, W); }
      for (let y = oy; y < W; y += step) { bevCtx.moveTo(0, y); bevCtx.lineTo(W, y); }
      bevCtx.stroke();
    }
  }

  // ---- OSM road graph (black, thick) + graph nodes ----
  if (state.showRoads && state.roadsENU) {
    const g = frameGeo(data.header);
    if (g) {
      bevCtx.strokeStyle = "#000";
      bevCtx.lineWidth = 3;
      bevCtx.globalAlpha = 0.8;
      bevCtx.beginPath();
      for (const way of state.roadsENU) {
        let first = true;
        for (const [e, n] of way) {
          const p = g.map(e, n);
          const sx = cx + bevView.x - p[1] * scale;
          const sy = cy + bevView.y - p[0] * scale;
          if (first) bevCtx.moveTo(sx, sy); else bevCtx.lineTo(sx, sy);
          first = false;
        }
      }
      bevCtx.stroke();
      // nodes
      bevCtx.fillStyle = "#000";
      for (const way of state.roadsENU) {
        for (const [e, n] of way) {
          const p = g.map(e, n);
          const sx = cx + bevView.x - p[1] * scale;
          const sy = cy + bevView.y - p[0] * scale;
          if (sx < -4 || sx > W + 4 || sy < -4 || sy > W + 4) continue;
          bevCtx.beginPath();
          bevCtx.arc(sx, sy, 2.5, 0, 2 * Math.PI);
          bevCtx.fill();
        }
      }
      bevCtx.globalAlpha = 1;
    }
  }

  // ---- points (offscreen ImageData, composed over the map) ----
  const colors = pointColors(data, state.colorMode, state.rangeClip, state.showMapBev);
  const m = rangeMask(data), pts = data.points, n = data.header.n;
  if (!bevImgData || bevImgData.width !== W)
    bevImgData = bevPtsCtx.createImageData(W, W);
  const img = bevImgData, d = img.data;
  d.fill(0);
  for (let i = 0; i < n; i++) {
    if (!m[i]) continue;
    const pxx = (cx + bevView.x - pts[i * 3 + 1] * scale) | 0;
    const pyy = (cy + bevView.y - pts[i * 3 + 0] * scale) | 0;
    if (pxx < 0 || pxx >= W - 1 || pyy < 0 || pyy >= W - 1) continue;
    const r = Math.min(255, (colors[i * 3] * 255 * (state.showMapBev ? 1.1 : 1.45)) | 0),
          g = Math.min(255, (colors[i * 3 + 1] * 255 * (state.showMapBev ? 1.1 : 1.45)) | 0),
          b = Math.min(255, (colors[i * 3 + 2] * 255 * (state.showMapBev ? 1.1 : 1.45)) | 0);
    for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const q = ((pyy + oy) * W + pxx + ox) * 4;
      d[q] = r; d[q + 1] = g; d[q + 2] = b; d[q + 3] = 255;
    }
  }
  bevPtsCtx.putImageData(img, 0, 0);
  bevCtx.drawImage(bevPtsCv, 0, 0);

  // ---- boxes ----
  const idx = visibleBoxes(data), labels = data.header.labels, cols = classColorsFor(labels);
  bevCtx.lineWidth = 2.5;
  for (const k of idx) {
    const b = data.boxes.subarray(k * 7, k * 7 + 7);
    const cor = boxCorners(b);
    bevCtx.strokeStyle = cols[k].css;
    bevCtx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = cor.subarray(i * 3, i * 3 + 3);
      const bb = cor.subarray(((i + 1) % 4) * 3, ((i + 1) % 4) * 3 + 3);
      const [x1, y1] = px(a[0], a[1]), [x2, y2] = px(bb[0], bb[1]);
      if (i === 0) bevCtx.moveTo(x1, y1); else bevCtx.lineTo(x1, y1);
      bevCtx.lineTo(x2, y2);
    }
    bevCtx.stroke();
    const hl = b[3] / 2, ca = Math.cos(b[6]), sa = Math.sin(b[6]);
    bevCtx.beginPath();
    const [ax, ay] = px(b[0], b[1]);
    const [bx2, by2] = px(b[0] + hl * ca, b[1] + hl * sa);
    bevCtx.moveTo(ax, ay);
    bevCtx.lineTo(bx2, by2);
    bevCtx.stroke();
  }

  // ---- ego vehicle (self) box: 4.9 x 2.0 m, heading up (+X) ----
  {
    const L = 4.9, W = 2.0;
    const [fx, fy] = px(L / 2, 0), [bx, by] = px(-L / 2, 0);
    const [lx, ly] = px(-L / 2, W / 2), [rx, ry] = px(-L / 2, -W / 2);
    const [lfx, lfy] = px(L / 2 - 0.9, W / 2), [rfx, rfy] = px(L / 2 - 0.9, -W / 2);
    bevCtx.strokeStyle = "#66aaff";
    bevCtx.fillStyle = "rgba(102,170,255,0.25)";
    bevCtx.lineWidth = 2;
    bevCtx.beginPath();
    bevCtx.moveTo(fx, fy); bevCtx.lineTo(lfx, lfy); bevCtx.lineTo(lx, ly);
    bevCtx.lineTo(bx, by); bevCtx.lineTo(rx, ry); bevCtx.lineTo(rfx, rfy);
    bevCtx.closePath();
    bevCtx.fill();
    bevCtx.stroke();
    // windshield line
    bevCtx.beginPath();
    const [wx1, wy1] = px(L / 2 - 1.2, W / 2), [wx2, wy2] = px(L / 2 - 1.2, -W / 2);
    bevCtx.moveTo(wx1, wy1); bevCtx.lineTo(wx2, wy2);
    bevCtx.stroke();
  }

  // ---- compact radius scale: distance labels on all 4 semi-axes ----
  if (state.showBevScale) {
    const [ex0, ey0] = px(0, 0);
    const fs = 15;
    bevCtx.font = `bold ${fs}px ui-monospace, monospace`;
    bevCtx.textBaseline = "middle";
    for (let r = step; r <= visR; r += step) {
      for (const dir of [[1, 0], [-1, 0], [0, -1], [0, 1]]) {  // up/down/left/right
        const x = ex0 + dir[0] * r * scale;
        const y = ey0 + dir[1] * r * scale;
        if (x < 30 || x > W - 30 || y < 26 || y > W - 12) continue;
        bevCtx.strokeStyle = "rgba(160,180,195,0.9)";
        bevCtx.lineWidth = 2;
        bevCtx.beginPath();
        // tick across the axis direction
        bevCtx.moveTo(x - 5 * dir[1], y - 5 * dir[0]);
        bevCtx.lineTo(x + 5 * dir[1], y + 5 * dir[0]);
        bevCtx.stroke();
        const lab = String(r);
        bevCtx.textAlign = "center";
        const tw = bevCtx.measureText(lab).width;
        const lx = x + dir[0] * 10, ly = y + dir[1] * 10;
        bevCtx.fillStyle = "rgba(10,14,18,0.72)";
        bevCtx.fillRect(lx - tw / 2 - 3, ly - fs / 2 - 1, tw + 6, fs + 2);
        bevCtx.fillStyle = "#cfe0ee";
        bevCtx.fillText(lab, lx, ly);
      }
    }
    bevCtx.textAlign = "start";
    bevCtx.textBaseline = "alphabetic";
  }
}

// BEV navigation: wheel zoom to cursor, drag pan, double-click reset
{
  const wrap = $("bev-wrap");
  wrap.addEventListener("wheel", e => {
    e.preventDefault();
    const r = bev.getBoundingClientRect();
    const mx = (e.clientX - r.left) / r.width * bev.width - bev.width / 2;
    const my = (e.clientY - r.top) / r.height * bev.height - bev.height / 2;
    const z2 = Math.min(20, Math.max(0.2, bevView.z * (e.deltaY < 0 ? 1.25 : 0.8)));
    const k = z2 / bevView.z;
    bevView.x = mx + (bevView.x - mx) * k;
    bevView.y = my + (bevView.y - my) * k;
    bevView.z = z2;
    render();
  }, { passive: false });
  let drag = null;
  wrap.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY, bx: bevView.x, by: bevView.y };
    wrap.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener("pointermove", e => {
    if (!drag) return;
    const r = bev.getBoundingClientRect();
    const k = bev.width / r.width;
    bevView.x = drag.bx + (e.clientX - drag.x) * k;
    bevView.y = drag.by + (e.clientY - drag.y) * k;
    render();
  });
  wrap.addEventListener("pointerup", () => { drag = null; });
  wrap.addEventListener("dblclick", () => { bevView.z = 1; bevView.x = 0; bevView.y = 0; render(); });
}

// ------------------------------------------------------------ cameras
const camPanels = {};

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
    const cvs = document.createElement("canvas");       // WebGL: lidar points
    const ovs = document.createElement("canvas");       // 2D: box frames + labels
    ovs.className = "ovl";
    const name = document.createElement("div");
    name.className = "camname"; name.textContent = cam.replace("_camera", "");
    zoomwrap.append(img, cvs, ovs);
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
      size: 3.0, vertexColors: true, sizeAttenuation: false }));
    scene2.add(pts2);
    const p = { div, zoomwrap, img, renderer: renderer2, scene: scene2, cam2d,
                pts: pts2, ovs, octx: ovs.getContext("2d"),
                z: { s: 1, x: 0, y: 0 }, w: 960, h: 540, fit: 1, cam };
    camPanels[cam] = p;
    div.onwheel = e => {
      e.preventDefault();
      const r = div.getBoundingClientRect();
      const mx = e.clientX - (r.left + r.width / 2);   // vs cell center
      const my = e.clientY - (r.top + r.height / 2);
      const s2 = Math.min(30, Math.max(1, p.z.s * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
      const k = s2 / p.z.s;
      p.z.x = mx + (p.z.x - mx) * k;
      p.z.y = my + (p.z.y - my) * k;
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
    div.ondblclick = () => { p.z.s = 1; p.z.x = 0; p.z.y = 0; applyZoom(p); };
  }
}

// image letterbox-fits its cell at zoom 1; user zoom scales on top
function applyZoom(p) {
  const cell = p.div.getBoundingClientRect();
  p.fit = Math.min(cell.width / p.w, cell.height / p.h) || 1;
  const s = p.fit * p.z.s;
  p.zoomwrap.style.transform =
    `translate(-50%, -50%) translate(${p.z.x}px, ${p.z.y}px) scale(${s})`;
}

function setupCamPanelSize(p, w, h) {
  p.w = w; p.h = h;
  p.img.width = w; p.img.height = h;
  p.zoomwrap.style.width = w + "px"; p.zoomwrap.style.height = h + "px";
  p.renderer.setSize(w, h, false);
  p.ovs.width = w; p.ovs.height = h;
  p.cam2d.right = w; p.cam2d.top = h;
  p.cam2d.updateProjectionMatrix();
}

function renderCamPanel(p, data) {
  const cd = data.header.cams[p.cam];
  if (!cd) { p.div.style.display = "none"; return; }
  p.div.style.display = "";
  const w0 = cd.w, h0 = cd.h;
  if (p.w !== w0 || p.h !== h0) setupCamPanelSize(p, w0, h0);
  applyZoom(p);
  const imgUrl = `/api/camimg?sweep=${state.sweep}&cam=${p.cam}&ts=${cd.ts}`;
  if (p.img.dataset.cur !== imgUrl) { p.img.src = imgUrl; p.img.dataset.cur = imgUrl; }

  const m = rangeMask(data), n = data.header.n, pts = data.points;
  const colors = pointColors(data, state.colorMode, state.rangeClip, false);
  const R = quatToMat(cd.R), t = cd.t, K = cd.K;
  // ego-motion compensation: points live in the ego frame of the LIDAR frame
  // ts; the photo was taken at ts_cam (async sensors). Map every point
  // world -> ego(t_cam) so points, boxes and photo share the same instant.
  const e2g = data.header.e2g;
  let Mp = null;   // 3x4 ego(frame) -> ego(t_cam)
  if (e2g && cd.ib_e2g) {
    const Rfw = quatToMat(e2g.slice(3, 7)), tfw = e2g.slice(0, 3);
    const Rw2e = quatToMat(cd.ib_e2g.slice(3, 7)), tw2 = cd.ib_e2g.slice(0, 3);
    Mp = [
      [Rw2e[0][0]*Rfw[0][0] + Rw2e[1][0]*Rfw[1][0] + Rw2e[2][0]*Rfw[2][0],
       Rw2e[0][0]*Rfw[0][1] + Rw2e[1][0]*Rfw[1][1] + Rw2e[2][0]*Rfw[2][1],
       Rw2e[0][0]*Rfw[0][2] + Rw2e[1][0]*Rfw[1][2] + Rw2e[2][0]*Rfw[2][2]],
      [Rw2e[0][1]*Rfw[0][0] + Rw2e[1][1]*Rfw[1][0] + Rw2e[2][1]*Rfw[2][0],
       Rw2e[0][1]*Rfw[0][1] + Rw2e[1][1]*Rfw[1][1] + Rw2e[2][1]*Rfw[2][1],
       Rw2e[0][1]*Rfw[0][2] + Rw2e[1][1]*Rfw[1][2] + Rw2e[2][1]*Rfw[2][2]],
      [Rw2e[0][2]*Rfw[0][0] + Rw2e[1][2]*Rfw[1][0] + Rw2e[2][2]*Rfw[2][0],
       Rw2e[0][2]*Rfw[0][1] + Rw2e[1][2]*Rfw[1][1] + Rw2e[2][2]*Rfw[2][1],
       Rw2e[0][2]*Rfw[0][2] + Rw2e[1][2]*Rfw[1][2] + Rw2e[2][2]*Rfw[2][2]],
    ];
    const pw = [ // world position of ego(frame) origin = tfw; shift vector
      Mp[0][0]*tfw[0] + Mp[0][1]*tfw[1] + Mp[0][2]*tfw[2],
      Mp[1][0]*tfw[0] + Mp[1][1]*tfw[1] + Mp[1][2]*tfw[2],
      Mp[2][0]*tfw[0] + Mp[2][1]*tfw[1] + Mp[2][2]*tfw[2]];
    Mp[0][3] = pw[0] - tw2[0]; Mp[1][3] = pw[1] - tw2[1]; Mp[2][3] = pw[2] - tw2[2];
  }
  const uv = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let mm = 0;
  if (state.showLidar) {
    for (let i = 0; i < n; i++) {
      if (!m[i]) continue;
      let ex = pts[i*3], ey = pts[i*3+1], ez = pts[i*3+2];
      if (Mp) {
        const nx = Mp[0][0]*ex + Mp[0][1]*ey + Mp[0][2]*ez + Mp[0][3];
        const ny = Mp[1][0]*ex + Mp[1][1]*ey + Mp[1][2]*ez + Mp[1][3];
        const nz = Mp[2][0]*ex + Mp[2][1]*ey + Mp[2][2]*ez + Mp[2][3];
        ex = nx; ey = ny; ez = nz;
      }
      ex -= t[0]; ey -= t[1]; ez -= t[2];
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

  // ---- box frames + labels: 2D overlay (thick, crisp, aligned) ----
  // Cameras run at their own timestamps: cd.ib holds boxes interpolated onto
  // THIS camera's ts (world frame + ego pose at that ts) — project those;
  // fall back to the shared lidar-frame boxes if interpolation is absent.
  const ctx = p.octx;
  ctx.clearRect(0, 0, w0, h0);
  if (state.showBoxes) {
    const uiScale = 1 / Math.max(p.fit * p.z.s, 1e-3);   // constant screen px size
    const lw = Math.max(1.2, 2.6 * uiScale);
    const fs = Math.max(8, 12 * uiScale);
    const items = [];
    if (cd.ib) {
      const Rw2e = quatToMat(cd.ib_e2g.slice(3, 7));   // world_from_ego @ t_cam
      const tw = cd.ib_e2g.slice(0, 3);
      const withQuat = cd.ib.length && cd.ib[0].length === 10;
      for (let k = 0; k < cd.ib.length; k++) {
        const lab = cd.ib_labels[k] || "?";
        if (!boxVisible(lab)) continue;
        const b = cd.ib[k];
        // box axes in world: full quaternion (exact) or yaw fallback
        let Rb;
        if (withQuat) Rb = quatToMat([b[6], b[7], b[8], b[9]]);
        else {
          const cs = Math.cos(b[6]), sn = Math.sin(b[6]);
          Rb = [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]];
        }
        const cor = new Float32Array(24);
        for (let i = 0; i < 8; i++) {
          const lx = BOX_SIGNS[i][0] * b[3], ly = BOX_SIGNS[i][1] * b[4],
                lz = BOX_SIGNS[i][2] * b[5];
          // corner in world: full rotation, then world -> ego(t_cam)
          const vx = Rb[0][0] * lx + Rb[0][1] * ly + Rb[0][2] * lz + b[0];
          const vy = Rb[1][0] * lx + Rb[1][1] * ly + Rb[1][2] * lz + b[1];
          const vz = Rb[2][0] * lx + Rb[2][1] * ly + Rb[2][2] * lz + b[2];
          const dx = vx - tw[0], dy = vy - tw[1], dz = vz - tw[2];
          cor[i*3]   = Rw2e[0][0]*dx + Rw2e[1][0]*dy + Rw2e[2][0]*dz;
          cor[i*3+1] = Rw2e[0][1]*dx + Rw2e[1][1]*dy + Rw2e[2][1]*dz;
          cor[i*3+2] = Rw2e[0][2]*dx + Rw2e[1][2]*dy + Rw2e[2][2]*dz;
        }
        items.push({ cor, lab });
      }
    } else {
      const labels = data.header.labels;
      for (const k of visibleBoxes(data))
        items.push({ cor: boxCorners(data.boxes.subarray(k * 7, k * 7 + 7)),
                     lab: labels[k] || "?" });
    }
    ctx.lineWidth = lw;
    ctx.font = `bold ${fs}px system-ui`;
    ctx.textBaseline = "bottom";
    for (const it of items) {
      const cor = it.cor;
      const uvs = new Array(8);
      let zmin = Infinity;
      for (let i = 0; i < 8; i++) {
        const ex = cor[i*3] - t[0], ey = cor[i*3+1] - t[1], ez = cor[i*3+2] - t[2];
        const cz = R[0][2]*ex + R[1][2]*ey + R[2][2]*ez;
        const cx = R[0][0]*ex + R[1][0]*ey + R[2][0]*ez;
        const cy = R[0][1]*ex + R[1][1]*ey + R[2][1]*ez;
        uvs[i] = cz > 0.2 ? [(K[0][0]*cx + K[0][2]*cz) / cz,
                              (K[1][1]*cy + K[1][2]*cz) / cz] : null;
        if (cz < zmin) zmin = cz;
      }
      if (zmin < 0.2 || uvs.some(q => !q)) continue;
      let inside = false;
      for (const q of uvs) if (q[0] >= -50 && q[0] <= w0 + 50 && q[1] >= -50 && q[1] <= h0 + 50) inside = true;
      if (!inside) continue;
      const css = colorFor(it.lab).css;
      ctx.strokeStyle = css;
      ctx.beginPath();
      for (const [a, bb] of EDGES) {
        ctx.moveTo(uvs[a][0], uvs[a][1]);
        ctx.lineTo(uvs[bb][0], uvs[bb][1]);
      }
      ctx.stroke();
      if (state.showCamLbl) {
        let top = uvs[4];
        for (let i = 4; i < 8; i++) if (uvs[i][1] < top[1]) top = uvs[i];
        const text = it.lab;
        const tw2 = ctx.measureText(text).width;
        const pad = 3 * uiScale;
        const x = Math.min(Math.max(top[0], tw2 / 2 + 2), w0 - tw2 / 2 - 2);
        const y = Math.max(top[1] - 4 * uiScale, fs);
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(x - tw2 / 2 - pad, y - fs - pad * 0.5, tw2 + pad * 2, fs + pad);
        ctx.fillStyle = css;
        ctx.fillText(text, x - tw2 / 2, y);
      }
    }
  }
  p.renderer.render(p.scene, p.cam2d);
}

// ------------------------------------------------------------ map (Leaflet)
let leaflet = null, trackLine = null, trackDone = null, posMarker = null;
function initMap() {
  leaflet = L.map("map", { attributionControl: false }).setView([37.422, -122.16], 14);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(leaflet);
  trackLine = L.polyline([], { color: "#4aa8ff", weight: 4, opacity: 0.9 }).addTo(leaflet);
  trackDone = L.polyline([], { color: "#ffe14d", weight: 4 }).addTo(leaflet);
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
    if (best >= 0 && bd < 400) setFrame(best);
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
  if (!w || !h) return;
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

// ------------------------------------------------------------ sequencing
function syncUrl() {
  const name = state.sweeps[state.sweep];
  history.replaceState(null, "", `/?seq=${name}&frame=${state.frame}`);
}

function setFrame(f) {
  state.frame = Math.max(0, Math.min(f, state.frames - 1));
  $("slider").value = state.frame;
  prefetch();
  render();
  syncUrl();
}

async function selectSweep(i, frame = 0) {
  state.sweep = i;
  state.gpsTrackDrawn = [];
  state.cache.clear();
  bevView.z = 1; bevView.x = 0; bevView.y = 0;
  $("loading").classList.add("show");
  const [meta, track] = await Promise.all([
    api(`/api/meta?sweep=${i}`), api(`/api/gps_track?sweep=${i}`),
  ]);
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
  state.track = track;
  await buildTripMap();
  // OSM road graph -> ENU polylines (once per sweep)
  state.roadsENU = null;
  const chkR = $("chkRoads");
  chkR.disabled = true;
  chkR.title = "road graph not downloaded for this sequence yet";
  api(`/api/osm_roads?sweep=${i}`).then(rd => {
    if (!rd || !rd.ways || !rd.ways.length || !state.track.aln) return;
    const aln = state.track.aln;
    const ca = Math.cos(aln.lat0 * Math.PI / 180);
    const cA = Math.cos(aln.rot), sA = Math.sin(aln.rot);
    const out = [];
    for (const way of rd.ways) {
      const ptsENU = [];
      for (const [la, lo] of way) {
        const e = (lo - aln.lon0) * Math.PI / 180 * R_EARTH * ca;
        const n = (la - aln.lat0) * Math.PI / 180 * R_EARTH;
        if (Math.abs(e) < 4000 && Math.abs(n) < 4000) ptsENU.push([e, n]);
      }
      if (ptsENU.length > 1) out.push(ptsENU);
    }
    state.roadsENU = out;
    chkR.disabled = false;
    chkR.title = "draw the OSM road graph (black lines)";
    render();
  }).catch(() => {});
  const pts = [];
  for (let k = 0; k < track.lat.length; k++)
    if (track.lat[k] || track.lon[k]) pts.push([track.lat[k], track.lon[k]]);
  trackLine.setLatLngs(pts);
  trackDone.setLatLngs([]);
  if (pts.length > 1) leaflet.fitBounds(L.latLngBounds(pts).pad(0.08));
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
  // discard only if the sweep changed or a newer render superseded this one
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
function requestRender() { render(); }

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
  for (const cam of CAM_ORDER) if (camPanels[cam]) applyZoom(camPanels[cam]);
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
$("chkMap3d").onchange = e => { state.showMap3d = e.target.checked; render(); };
$("chkMapBev").onchange = e => { state.showMapBev = e.target.checked; render(); };
$("chkGrid").onchange = e => { state.showGrid = e.target.checked; grid3.visible = state.showGrid; };
$("gridStep").onchange = e => { state.gridStep = +e.target.value; setGridStep(state.gridStep); render(); };
$("chkRoads").onchange = e => { state.showRoads = e.target.checked; render(); };
$("chkBevScale").onchange = e => { state.showBevScale = e.target.checked; render(); };
$("chkLabels3d").onchange = e => { state.showLabels3d = e.target.checked; render(); };
$("chkBev").onchange = e => { state.showBev = e.target.checked; render(); };
$("colorMode").onchange = e => {
  state.colorMode = e.target.value;
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
    render();
  } else if (e.key === "h" || e.key === "?") $("help").hidden = !$("help").hidden;
  else if (e.key === "Escape") $("help").hidden = true;
});

// debugging aid (browser console)
window.__pandaset = { state, loadFrame, mapPatch, frameGeo, tripMap: () => tripMap };

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
