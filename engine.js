// Stand-in search engine for PAX's out-of-book play and for the trainer.
// 10x12 mailbox board, alpha-beta with quiescence, piece-square evaluation.
// Runs as a Web Worker; also loadable in Node for testing (module.exports).
'use strict';
const OFF = 7, MATE = 100000, INF = 1e9;
const VAL = [0, 100, 320, 330, 500, 900, 0];
// Piece-square tables from White's side, rank 8 first (Michniewski's simplified evaluation).
const PST = [null,
  [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5,
   0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
  [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30,
   -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
  [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10,
   -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
  [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
   -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
  [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5,
   0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
  [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20]];
// PAX 2.0's learnable evaluation: middlegame and endgame tables (piece value included), blended by game phase.
// They start as the hand-written tables above; setTables() loads the ones tuned on Stockfish-labelled games,
// and the point-of-no-return lessons nudge them further.
const MGT = new Float32Array(7 * 64), EGT = new Float32Array(7 * 64), PHW = [0, 0, 1, 1, 2, 4, 0];
for (let t = 1; t <= 6; t++) for (let i = 0; i < 64; i++) MGT[t * 64 + i] = EGT[t * 64 + i] = VAL[t] + PST[t][i];
let mg = 0, eg = 0, ph = 0;
const tix = (p, s) => { const r = ((s / 10) | 0) - 2, f = s % 10 - 1; return (p & 7) * 64 + ((p >> 3) ? (7 - r) * 8 + f : r * 8 + f); };
function put(p, s, sg) {                  // add (sg = 1) or remove (sg = -1) piece p on square s
  if (p === 0 || p === OFF) return;
  const i = tix(p, s), w = (p >> 3) ? -sg : sg;
  mg += w * MGT[i]; eg += w * EGT[i]; ph += sg * PHW[p & 7];
}
const pstW = () => { const q = ph > 24 ? 24 : ph; return (mg * q + eg * (24 - q)) / 24; };   // White's view
function setTables(w) {
  for (let t = 1; t <= 6; t++) for (let i = 0; i < 64; i++) { MGT[t * 64 + i] = w.mg[t - 1][i]; EGT[t * 64 + i] = w.eg[t - 1][i]; }
  ttClear();
}
const getTables = () => ({ mg: [1, 2, 3, 4, 5, 6].map(t => Array.from(MGT.slice(t * 64, t * 64 + 64), v => Math.round(v * 10) / 10)),
                           eg: [1, 2, 3, 4, 5, 6].map(t => Array.from(EGT.slice(t * 64, t * 64 + 64), v => Math.round(v * 10) / 10)) });
const KN = [-21, -19, -12, -8, 8, 12, 19, 21], DIAG = [-11, -9, 9, 11], ORTH = [-10, -1, 1, 10], ALL = [-11, -10, -9, -1, 1, 9, 10, 11];
const DIRS = [null, null, KN, DIAG, ORTH, ALL, ALL];
const SLIDE = [false, false, false, true, true, true, false];
const ABORT = { abort: true };

const bd = new Int8Array(120);
let side = 0, nodes = 0, deadline = 0, cap = 0;

// ---- hashing and the transposition table ----
let seed = 0x9e3779b9; const rnd32 = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed | 0; };
const Z1 = new Int32Array(16 * 120), Z2 = new Int32Array(16 * 120); for (let i = 0; i < Z1.length; i++) { Z1[i] = rnd32(); Z2[i] = rnd32(); }
const ZS1 = rnd32(), ZS2 = rnd32();
let h1 = 0, h2 = 0;
const TTB = 20, TTN = 1 << TTB, TTM = TTN - 1;
const tKey = new Int32Array(TTN), tVal = new Int32Array(TTN), tMove = new Int32Array(TTN), tDepth = new Int8Array(TTN), tFlag = new Int8Array(TTN);
const EXACT = 1, LOWER = 2, UPPER = 3;
function ttClear() { tFlag.fill(0); }
const killers = new Int32Array(128 * 2), hist = new Int32Array(120 * 120);
function zx(p, s) { if (p !== 0 && p !== OFF) { h1 ^= Z1[p * 120 + s]; h2 ^= Z2[p * 120 + s]; } }

// ---- PAX 2.0 neural evaluation (NNUE-style). Without a network the engine falls back to piece-square tables. ----
let NET = null;
const accW = new Float32Array(256), accB = new Float32Array(256), xin = new Float32Array(512), hid1 = new Float32Array(32), hid2 = new Float32Array(32);
function setNet(w) {
  const H = w.H, ft = new Float32Array(768 * H);
  for (let i = 0; i < 768; i++) ft.set(w.ft[i], i * H);
  const l1 = new Float32Array(32 * 2 * H);                 // stored input-major: l1[i * 32 + o], so the inner loop is contiguous
  for (let o = 0; o < 32; o++) for (let i = 0; i < 2 * H; i++) l1[i * 32 + o] = w.l1w[o][i];
  const l2 = new Float32Array(32 * 32); for (let o = 0; o < 32; o++) l2.set(w.l2w[o], o * 32);
  ttClear();
  NET = { residual: !!w.residual, H, ft, ftb: Float32Array.from(w.ftb), l1, l1b: Float32Array.from(w.l1b), l2, l2b: Float32Array.from(w.l2b), ow: Float32Array.from(w.ow), ob: w.ob };
}
function fidx(p, s, persp) {                     // feature index of piece p on square s, from one side's view
  const s64 = (7 - (((s / 10) | 0) - 2)) * 8 + (s % 10 - 1), c = p >> 3, t = (p & 7) - 1;
  return persp === 0 ? (c === 0 ? 0 : 384) + t * 64 + s64 : (c === 1 ? 0 : 384) + t * 64 + (s64 ^ 56);
}
function nAdd(p, s, sign) {
  if (!NET || p === 0 || p === OFF) return;
  const H = NET.H, ft = NET.ft, a = fidx(p, s, 0) * H, b = fidx(p, s, 1) * H;
  for (let i = 0; i < H; i++) { accW[i] += sign * ft[a + i]; accB[i] += sign * ft[b + i]; }
}
function nRefresh() {
  if (!NET) return;
  accW.set(NET.ftb); accB.set(NET.ftb);
  for (let s = 21; s < 99; s++) nAdd(bd[s], s, 1);
}
function nEval() {                               // centipawns for the side to move
  const H = NET.H, us = side ? accB : accW, them = side ? accW : accB;
  for (let i = 0; i < H; i++) { const u = us[i], v = them[i]; xin[i] = u < 0 ? 0 : u > 1 ? 1 : u; xin[H + i] = v < 0 ? 0 : v > 1 ? 1 : v; }
  const l1 = NET.l1, n2 = 2 * H;
  for (let o = 0; o < 32; o++) hid1[o] = NET.l1b[o];
  for (let i = 0; i < n2; i++) { const x = xin[i]; if (x === 0) continue; const b = i * 32; for (let o = 0; o < 32; o++) hid1[o] += l1[b + o] * x; }
  for (let o = 0; o < 32; o++) { let v = NET.l2b[o]; for (let i = 0; i < 32; i++) { const x = hid1[i]; v += NET.l2[o * 32 + i] * (x < 0 ? 0 : x > 1 ? 1 : x); } hid2[o] = v; }
  let out = NET.ob; for (let i = 0; i < 32; i++) { const x = hid2[i]; out += NET.ow[i] * (x < 0 ? 0 : x > 1 ? 1 : x); }
  const cp = out * 400 + (NET.residual ? (side ? -pstW() : pstW()) : 0); return cp > 3000 ? 3000 : cp < -3000 ? -3000 : cp;
}
const staticEval = () => NET ? nEval() : (side ? -pstW() : pstW());
// Lazy evaluation inside capture sequences: run the network once where the main search ends,
// then follow material changes with the piece-square score. hzW / pstH are set at that horizon.
let hzW = 0, pstH = 0;
const lazyEval = () => { const w = hzW + (pstW() - pstH); return side ? -w : w; };

function pv(p, s) {                       // value of piece p on square s, from White's side
  if (p === 0 || p === OFF) return 0;
  const t = p & 7, c = p >> 3, r = ((s / 10) | 0) - 2, f = s % 10 - 1;
  const v = VAL[t] + PST[t][c ? (7 - r) * 8 + f : r * 8 + f];
  return c ? -v : v;
}
function load(fen) {
  bd.fill(OFF);
  const [pl, st] = fen.split(' '); const rows = pl.split('/');
  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { for (let k = 0; k < +ch; k++) bd[21 + r * 10 + f++] = 0; }
      else { const t = 'pnbrqk'.indexOf(ch.toLowerCase()) + 1; bd[21 + r * 10 + f++] = t + (ch === ch.toLowerCase() ? 8 : 0); }
    }
  }
  side = st === 'w' ? 0 : 1; mg = eg = ph = 0; h1 = 0; h2 = 0;
  for (let s = 21; s < 99; s++) { put(bd[s], s, 1); zx(bd[s], s); }
  if (side) { h1 ^= ZS1; h2 ^= ZS2; }
  nRefresh();
}
function gen(caps) {
  const out = []; const fwd = side ? 10 : -10;
  for (let s = 21; s < 99; s++) {
    const p = bd[s]; if (p === 0 || p === OFF || (p >> 3) !== side) continue;
    const t = p & 7;
    if (t === 1) {
      const to = s + fwd, promo = (side ? to > 90 : to < 29) ? 1 : 0;
      if ((!caps || promo) && bd[to] === 0) {
        out.push(s | to << 7 | promo << 14);
        if (!caps && (side ? s < 39 : s > 80) && bd[to + fwd] === 0) out.push(s | (to + fwd) << 7);
      }
      for (const d of [fwd - 1, fwd + 1]) {
        const q = bd[s + d], tt = s + d;
        if (q !== 0 && q !== OFF && (q >> 3) !== side) out.push(s | tt << 7 | ((side ? tt > 90 : tt < 29) ? 1 : 0) << 14);
      }
    } else {
      for (const d of DIRS[t]) {
        let to = s + d;
        for (;;) {
          const q = bd[to]; if (q === OFF) break;
          if (q === 0) { if (!caps) out.push(s | to << 7); }
          else { if ((q >> 3) !== side) out.push(s | to << 7); break; }
          if (!SLIDE[t]) break; to += d;
        }
      }
    }
  }
  return out;
}
function order(ms, ply, ttm) {
  const k0 = ply !== undefined ? killers[ply * 2] : 0, k1 = ply !== undefined ? killers[ply * 2 + 1] : 0;
  const key = ms.map(m => {
    if (m === ttm) return 1e9;
    const v = bd[(m >> 7) & 127];
    if (v) return 1e7 + ((v & 7) === 6 ? 1e6 : VAL[v & 7] * 10 - VAL[bd[m & 127] & 7]);
    if ((m >> 14) & 1) return 9e6;
    if (m === k0) return 8e6; if (m === k1) return 7e6;
    return Math.min(hist[(m & 127) * 120 + ((m >> 7) & 127)], 6e6);
  });
  return ms.map((_, i) => i).sort((a, b) => key[b] - key[a]).map(i => ms[i]);
}
function make(m) {
  const from = m & 127, to = (m >> 7) & 127, pr = (m >> 14) & 1;
  const p = bd[from], c = bd[to];
  const u = { from, to, p, c, mg, eg, ph, rf: 0, rt: 0, ep: 0, epp: 0, h1, h2 };
  put(c, to, -1); put(p, from, -1);
  const np = pr ? (5 | (side << 3)) : p;
  bd[to] = np; bd[from] = 0; put(np, to, 1);
  if (NET) { nAdd(c, to, -1); nAdd(p, from, -1); nAdd(np, to, 1); u.np = np; }
  zx(c, to); zx(p, from); zx(np, to);
  const t = p & 7;
  if (t === 6 && Math.abs(to - from) === 2) {            // castling (only ever arrives as a root move)
    const rf = to > from ? from + 3 : from - 4, rt = to > from ? from + 1 : from - 1;
    u.rf = rf; u.rt = rt; const r = bd[rf]; put(r, rf, -1); bd[rt] = r; bd[rf] = 0; put(r, rt, 1);
    if (NET) { nAdd(r, rf, -1); nAdd(r, rt, 1); }
    zx(r, rf); zx(r, rt);
  } else if (t === 1 && c === 0 && (to - from) % 10 !== 0) { // en passant (root move only)
    const e = to + (side ? -10 : 10); u.ep = e; u.epp = bd[e]; put(bd[e], e, -1); if (NET) nAdd(bd[e], e, -1); zx(bd[e], e); bd[e] = 0;
  }
  side ^= 1; h1 ^= ZS1; h2 ^= ZS2; return u;
}
function unmake(u) {
  side ^= 1;
  if (NET) {
    nAdd(u.np, u.to, -1); nAdd(u.p, u.from, 1); nAdd(u.c, u.to, 1);
    if (u.rf) { nAdd(bd[u.rt], u.rt, -1); nAdd(bd[u.rt], u.rf, 1); }
    if (u.ep) nAdd(u.epp, u.ep, 1);
  }
  bd[u.from] = u.p; bd[u.to] = u.c;
  if (u.rf) { bd[u.rf] = bd[u.rt]; bd[u.rt] = 0; }
  if (u.ep) bd[u.ep] = u.epp;
  mg = u.mg; eg = u.eg; ph = u.ph; h1 = u.h1; h2 = u.h2;
}
function attacked(s, by) {
  const o = by << 3;
  if (by === 0) { if (bd[s + 9] === 1 || bd[s + 11] === 1) return true; }
  else if (bd[s - 9] === 9 || bd[s - 11] === 9) return true;
  for (const d of KN) if (bd[s + d] === (2 | o)) return true;
  for (const d of ALL) if (bd[s + d] === (6 | o)) return true;
  for (const d of DIAG) { let t = s + d; while (bd[t] === 0) t += d; if (bd[t] === (3 | o) || bd[t] === (5 | o)) return true; }
  for (const d of ORTH) { let t = s + d; while (bd[t] === 0) t += d; if (bd[t] === (4 | o) || bd[t] === (5 | o)) return true; }
  return false;
}
function inCheck() { const k = 6 | (side << 3); for (let s = 21; s < 99; s++) if (bd[s] === k) return attacked(s, side ^ 1); return false; }
function tick() { if ((++nodes & 1023) === 0 && ((deadline && Date.now() > deadline) || (cap && nodes > cap))) throw ABORT; }

function qsRoot(a, b, ply) {
  if (NET) { const e = nEval(); hzW = side ? -e : e; pstH = pstW(); }
  return qs(a, b, ply);
}
function qs(a, b, ply) {
  tick();
  const stand = NET ? lazyEval() : staticEval();
  if (stand >= b) return stand;
  if (stand > a) a = stand;
  for (const m of order(gen(true))) {
    if ((bd[(m >> 7) & 127] & 7) === 6) return MATE - ply;   // can take the king: the last move was illegal
    const u = make(m); const s = -qs(-b, -a, ply + 1); unmake(u);
    if (s >= b) return s;
    if (s > a) a = s;
  }
  return a;
}
function negamax(d, a, b, ply) {
  tick();
  if (d <= 0) return qsRoot(a, b, ply);
  const idx = h1 & TTM, a0 = a; let ttm = 0;
  if (tFlag[idx] && tKey[idx] === h2) {
    ttm = tMove[idx];
    if (tDepth[idx] >= d) {
      let v = tVal[idx]; if (v > MATE - 1000) v -= ply; else if (v < -MATE + 1000) v += ply;
      const f = tFlag[idx];
      if (f === EXACT) return v;
      if (f === LOWER && v > a) a = v; else if (f === UPPER && v < b) b = v;
      if (a >= b) return v;
    }
  }
  let best = -INF, legal = 0, bestM = 0;
  for (const m of order(gen(false), ply, ttm)) {
    if ((bd[(m >> 7) & 127] & 7) === 6) return MATE - ply;
    const quiet = bd[(m >> 7) & 127] === 0;
    const u = make(m); const s = -negamax(d - 1, -b, -a, ply + 1); unmake(u);
    if (s > -MATE + ply + 1) legal++;                       // a reply that took our king scores exactly -MATE+ply+1
    if (s > best) { best = s; bestM = m; }
    if (s > a) a = s;
    if (a >= b) {
      if (quiet && ply < 128) { if (killers[ply * 2] !== m) { killers[ply * 2 + 1] = killers[ply * 2]; killers[ply * 2] = m; } hist[(m & 127) * 120 + ((m >> 7) & 127)] += d * d; }
      break;
    }
  }
  if (!legal) return inCheck() ? -(MATE - ply) : 0;       // mated, or stalemate
  let sv = best; if (sv > MATE - 1000) sv += ply; else if (sv < -MATE + 1000) sv -= ply;
  tKey[idx] = h2; tVal[idx] = sv; tMove[idx] = bestM; tDepth[idx] = d; tFlag[idx] = best <= a0 ? UPPER : best >= b ? LOWER : EXACT;
  return best;
}
function parseUci(u) {
  const sq = (f, r) => 21 + (8 - r) * 10 + f;
  return sq(u.charCodeAt(0) - 97, +u[1]) | sq(u.charCodeAt(2) - 97, +u[3]) << 7 | (u.length > 4 ? 1 : 0) << 14;
}

// Search the given root moves (UCI strings) for up to timeMs. variety > 0 picks randomly among moves within that many centipawns.
function search(fen, moves, timeMs, variety, maxDepth, style) {
  load(fen); nodes = 0; cap = 0; deadline = Date.now() + timeMs;
  const rm = moves.map(u => ({ u, m: parseUci(u), s: 0 }));
  const byU = new Map(rm.map(r => [r.u, r.m]));
  let done = null, depth = 0; const margin = variety || 0;
  try {
    for (let d = 1; d <= (maxDepth || 64); d++) {
      let alpha = -INF;
      for (const r of rm) { const u = make(r.m); r.s = -negamax(d - 1, -INF, -(alpha - margin), 1); unmake(u); if (r.s > alpha) alpha = r.s; }
      rm.sort((x, y) => y.s - x.s);
      done = rm.map(r => ({ u: r.u, m: r.m, s: r.s })); depth = d;
      if (done[0].s > MATE - 1000) break;
    }
  } catch (e) { if (e !== ABORT) throw e; }
  if (!done) return { uci: moves[Math.floor(Math.random() * moves.length)], score: 0, depth: 0, nodes };
  let pool = margin ? done.filter(r => r.s >= done[0].s - margin) : [done[0]];
  // Temperament (Dorfman's Aggressive / Satvic / Passive): among moves within 30cp of the best, lean toward or away from captures.
  if (style && style !== 'satvic' && done[0].s < MATE - 1000) {
    load(fen);
    const near = done.filter(r => r.s >= done[0].s - 30);
    const isCap = r => bd[(r.m >> 7) & 127] !== 0 || ((bd[r.m & 127] & 7) === 1 && (((r.m >> 7) & 127) < 29 || ((r.m >> 7) & 127) > 90));
    const pref = near.filter(r => style === 'aggressive' ? isCap(r) : !isCap(r));
    if (pref.length) pool = pref;
  }
  const pick = pool[Math.floor(Math.random() * pool.length)];
  return { uci: pick.u, score: pick.s, depth, nodes };
}
// Can the side to move force mate within `plies` half-moves? Used to prove a position lost for the other side.
function proveMate(fen, plies, nodeCap) {
  load(fen); nodes = 0; cap = nodeCap || 400000; deadline = Date.now() + 5000;
  try { return negamax(plies, MATE - 201, INF, 0) >= MATE - 200; }
  catch (e) { if (e !== ABORT) throw e; return false; }
}

// ---- Learning from a lesson: one gradient step on a position whose true outcome we now know. ----
// target = how the game went for the side to move in `fen` (1 win, 0 loss). Updates the network in place.
const z1 = new Float32Array(32), z2 = new Float32Array(32), dz1 = new Float32Array(32), dz2 = new Float32Array(32);
function learnStep(fen, target, lr) {
  if (!NET) return learnTables(fen, target, lr);
  load(fen);
  const H = NET.H, n2 = 2 * H, us = side ? accB : accW, them = side ? accW : accB;
  for (let i = 0; i < H; i++) { const u = us[i], v = them[i]; xin[i] = u < 0 ? 0 : u > 1 ? 1 : u; xin[H + i] = v < 0 ? 0 : v > 1 ? 1 : v; }
  for (let o = 0; o < 32; o++) { let v = NET.l1b[o]; for (let i = 0; i < n2; i++) v += NET.l1[i * 32 + o] * xin[i]; z1[o] = v; hid1[o] = v < 0 ? 0 : v > 1 ? 1 : v; }
  for (let o = 0; o < 32; o++) { let v = NET.l2b[o]; for (let i = 0; i < 32; i++) v += NET.l2[o * 32 + i] * hid1[i]; z2[o] = v; hid2[o] = v < 0 ? 0 : v > 1 ? 1 : v; }
  let out = NET.ob; for (let i = 0; i < 32; i++) out += NET.ow[i] * hid2[i];
  if (NET.residual) out += (side ? -pstW() : pstW()) / 400;
  const p = 1 / (1 + Math.exp(-out)), dout = 2 * (p - target) * p * (1 - p);
  for (let i = 0; i < 32; i++) dz2[i] = (z2[i] > 0 && z2[i] < 1) ? dout * NET.ow[i] : 0;
  for (let i = 0; i < 32; i++) { let s = 0; for (let o = 0; o < 32; o++) s += dz2[o] * NET.l2[o * 32 + i]; dz1[i] = (z1[i] > 0 && z1[i] < 1) ? s : 0; }
  const dx = new Float32Array(n2);
  for (let o = 0; o < 32; o++) { const g = dz1[o]; if (!g) continue; for (let i = 0; i < n2; i++) dx[i] += g * NET.l1[i * 32 + o]; }
  // apply updates, output layer backwards to the inputs
  for (let i = 0; i < 32; i++) NET.ow[i] -= lr * dout * hid2[i];
  NET.ob -= lr * dout;
  for (let o = 0; o < 32; o++) { const g = dz2[o]; if (!g) continue; for (let i = 0; i < 32; i++) NET.l2[o * 32 + i] -= lr * g * hid1[i]; NET.l2b[o] -= lr * g; }
  for (let o = 0; o < 32; o++) { const g = dz1[o]; if (!g) continue; for (let i = 0; i < n2; i++) NET.l1[i * 32 + o] -= lr * g * xin[i]; NET.l1b[o] -= lr * g; }
  const dUs = new Float32Array(H), dThem = new Float32Array(H);
  for (let i = 0; i < H; i++) { dUs[i] = (us[i] > 0 && us[i] < 1) ? dx[i] : 0; dThem[i] = (them[i] > 0 && them[i] < 1) ? dx[H + i] : 0; }
  const dW = side ? dThem : dUs, dB = side ? dUs : dThem;
  for (let i = 0; i < H; i++) NET.ftb[i] -= lr * (dW[i] + dB[i]);
  for (let s = 21; s < 99; s++) {
    const pc = bd[s]; if (pc === 0 || pc === OFF) continue;
    const a = fidx(pc, s, 0) * H, b = fidx(pc, s, 1) * H;
    for (let i = 0; i < H; i++) { NET.ft[a + i] -= lr * dW[i]; NET.ft[b + i] -= lr * dB[i]; }
  }
  return (p - target) ** 2;
}
// One gradient step on the tables: every piece on the board shares the correction, so a lesson about one
// position also shifts PAX's judgement of positions that look like it.
function learnTables(fen, target, lr) {
  load(fen);
  const e = side ? -pstW() : pstW(), p = 1 / (1 + Math.exp(-e / 400));
  let n = 0; for (let s = 21; s < 99; s++) if (bd[s] && bd[s] !== OFF) n++;
  const q = (ph > 24 ? 24 : ph) / 24, g = lr * 1000 * (p - target) / n;
  for (let s = 21; s < 99; s++) {
    const pc = bd[s]; if (pc === 0 || pc === OFF || (pc & 7) === 6) continue;   // king squares stay as tuned
    const i = tix(pc, s), w = ((pc >> 3) ? -1 : 1) * (side ? -1 : 1);
    MGT[i] -= g * w * q; EGT[i] -= g * w * (1 - q);
  }
  return (p - target) ** 2;
}
function learn(lessons, epochs, lr) {             // replay lessons a few times; returns final mean error
  ttClear(); let err = 0;
  for (let e = 0; e < epochs; e++) { err = 0; for (const l of lessons) err += learnStep(l.fen, l.target, lr * (l.weight || 1)); }
  return lessons.length ? err / lessons.length : 0;
}
function evalFen(fen, depth) {                  // short search score for the side to move, in centipawns
  load(fen); nodes = 0; cap = 200000; deadline = Date.now() + 2000;
  try { return negamax(depth || 2, -INF, INF, 0); } catch (e) { if (e !== ABORT) throw e; load(fen); return staticEval(); }
}
function _check(fen, ucis) {                    // test hook: incremental eval after moves, and after undoing them
  load(fen); const e0 = nEval(); const us = [];
  for (const u of ucis) us.push(make(parseUci(u)));
  const e1 = nEval(); for (let i = us.length - 1; i >= 0; i--) unmake(us[i]);
  return { e0, e1, back: nEval() };
}
if (typeof module !== 'undefined') module.exports = { search, proveMate, evalFen, setNet, setTables, getTables, learn, MATE, _check, _staticAt: fen => { load(fen); return staticEval(); } };
else onmessage = e => {
  const q = e.data;
  if (q.type === 'search') postMessage({ id: q.id, ...search(q.fen, q.moves, q.timeMs, q.variety, q.maxDepth, q.style) });
  else if (q.type === 'prove') postMessage({ id: q.id, lost: proveMate(q.fen, q.plies, q.nodeCap) });
  else if (q.type === 'net') { setNet(q.weights); postMessage({ id: q.id, ok: true }); }
  else if (q.type === 'tables') { setTables(q.tables); postMessage({ id: q.id, ok: true }); }
  else if (q.type === 'eval') postMessage({ id: q.id, score: evalFen(q.fen, q.depth) });
  else if (q.type === 'evalGame') postMessage({ id: q.id, scores: q.fens.map(f => evalFen(f, q.depth)) });
  else if (q.type === 'learn') postMessage({ id: q.id, err: learn(q.lessons, q.epochs || 1, q.lr || 0.01) });
};
