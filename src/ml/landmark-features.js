// Camera landmarks -> the features the trained sign models were trained on.
//
// This is a line-by-line port of ml/scripts/clean_landmarks.py (clean_video), so
// a movement seen by the camera is described exactly as the WLASL training videos
// were. tests/ml-features.test.mjs checks it against the Python output for real
// videos. Change one, change the other.
//
// Input: the frames of one movement, each in the raw layout of
// ml/scripts/extract_landmarks.py (rawRow below builds it from a camera frame):
//   pose: 33 x (x, y, z, visibility), image-normalised
//   two hand slots: [label (0 = Left, 1 = Right), score, 21 x image xyz, 21 x world xyz]
// Missing values are NaN.

const N_POSE = 33;
const HAND_W = 2 + 21 * 3 * 2;
export const RAW_WIDTH = N_POSE * 4 + 2 * HAND_W;
const P = { NOSE: 0, LSH: 11, RSH: 12, LEL: 13, REL: 14, LWR: 15, RWR: 16 };
const TIPS = [0, 4, 8, 12, 16, 20];
const MAX_POSE_GAP = 5;
const MAX_HAND_GAP = 6;
const FPS = 25; // speeds are per second at WLASL's frame rate, as in training

/** One camera frame (CameraTracker.detect output) -> one raw row. */
export function rawRow(obs) {
	const row = new Float64Array(RAW_WIDTH).fill(NaN);
	if (obs.pose) {
		obs.pose.forEach((q, j) => {
			row[j * 4] = q.x;
			row[j * 4 + 1] = q.y;
			row[j * 4 + 2] = q.z ?? NaN;
			row[j * 4 + 3] = q.visibility ?? NaN;
		});
	}
	(obs.hands ?? []).slice(0, 2).forEach((h, s) => {
		const b = N_POSE * 4 + s * HAND_W;
		row[b] = h.side === 'Left' ? 0 : 1;
		row[b + 1] = h.score ?? 0;
		h.image.forEach((q, k) => {
			row[b + 2 + k * 3] = q.x;
			row[b + 2 + k * 3 + 1] = q.y;
			row[b + 2 + k * 3 + 2] = q.z ?? 0;
		});
		h.world.forEach((q, k) => {
			row[b + 65 + k * 3] = q[0];
			row[b + 65 + k * 3 + 1] = q[1];
			row[b + 65 + k * 3 + 2] = q[2];
		});
	});
	return row;
}

const nan2num = (rows) => rows.map((r) => Float64Array.from(r, (v) => (Number.isNaN(v) ? 0 : v)));

/** Linear interpolation across gaps of at most maxGap frames (rows of equal length). */
function interpGaps(rows, present, maxGap) {
	const x = rows.map((r) => Float64Array.from(r));
	const pres = [...present];
	const idx = pres.map((p, i) => (p ? i : -1)).filter((i) => i >= 0);
	for (let k = 0; k + 1 < idx.length; k++) {
		const a = idx[k];
		const b = idx[k + 1];
		const gap = b - a - 1;
		if (gap > 0 && gap <= maxGap) {
			for (let g = 1; g <= gap; g++) {
				const t = g / (gap + 1);
				for (let j = 0; j < x[a].length; j++) x[a + g][j] = (1 - t) * x[a][j] + t * x[b][j];
				pres[a + g] = true;
			}
		}
	}
	return { x, present: pres };
}

/** 3-frame centred moving average inside each visible stretch. */
function smooth(rows, present) {
	const n = rows.length;
	return rows.map((r, i) => {
		if (!present[i]) return Float64Array.from(r);
		const nb = [i - 1, i, i + 1].filter((j) => j >= 0 && j < n && present[j]);
		const out = new Float64Array(r.length);
		for (const j of nb) for (let k = 0; k < r.length; k++) out[k] += rows[j][k] / nb.length;
		return out;
	});
}

function median(values) {
	const s = [...values].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** 21 world points -> 78 numbers: wrist-relative points scaled by the palm, then fingertip distances. */
export function handShape(world) {
	const w = world.map((p) => [p[0] - world[0][0], p[1] - world[0][1], p[2] - world[0][2]]);
	const scale = Math.hypot(...w[9]);
	const s = scale > 1e-6 ? scale : 1;
	const n = w.map((p) => p.map((v) => v / s));
	const out = n.flat();
	for (let i = 0; i < TIPS.length; i++)
		for (let j = i + 1; j < TIPS.length; j++)
			out.push(Math.hypot(n[TIPS[i]][0] - n[TIPS[j]][0], n[TIPS[i]][1] - n[TIPS[j]][1], n[TIPS[i]][2] - n[TIPS[j]][2]));
	return out;
}

/** np.gradient of a 1-D array (central differences inside, one-sided at the ends). */
function gradient(v) {
	const n = v.length;
	if (n < 2) return v.map(() => 0);
	return v.map((_, i) => (i === 0 ? v[1] - v[0] : i === n - 1 ? v[n - 1] - v[n - 2] : (v[i + 1] - v[i - 1]) / 2));
}

/**
 * The cleaned features of one movement.
 * @param {Float64Array[]} raw  frames in the raw layout
 * @param {number} width  video width in pixels
 * @param {number} height video height in pixels
 * @param {string[]} names feature names, in the trained models' order
 * @returns {{ rows: Float64Array[], dominant: 'L'|'R' }|null}  null when the body is not seen in enough frames
 */
export function cleanSegment(raw, width, height, names) {
	const n = raw.length;
	const poseOk = raw.map((r) => !Number.isNaN(r[P.LSH * 4]) && !Number.isNaN(r[P.RSH * 4]));
	const poseRows = interpGaps(nan2num(raw.map((r) => r.slice(0, N_POSE * 4))), poseOk, MAX_POSE_GAP);
	const keep = poseRows.present;
	if (keep.filter(Boolean).length < 8) return null;
	const px = poseRows.x.map((r) => Array.from({ length: N_POSE }, (_, j) => [r[j * 4] * width, r[j * 4 + 1] * height]));
	const span = median(px.filter((_, i) => keep[i]).map((p) => Math.hypot(p[P.LSH][0] - p[P.RSH][0], p[P.LSH][1] - p[P.RSH][1])));
	if (span < 10) return null;
	const centre = smooth(px.map((p) => Float64Array.of((p[P.LSH][0] + p[P.RSH][0]) / 2, (p[P.LSH][1] + p[P.RSH][1]) / 2)), keep);
	const body = (i, x, y) => [(x - centre[i][0]) / span, -(y - centre[i][1]) / span];

	// Hands -> body sides by the nearer pose wrist.
	const side = { L: raw.map(() => null), R: raw.map(() => null) };
	const wr = { L: P.LWR, R: P.RWR };
	const cost = (d, s, i) => Math.hypot(d.img[0][0] * width - px[i][wr[s]][0], d.img[0][1] * height - px[i][wr[s]][1]);
	for (let i = 0; i < n; i++) {
		if (!keep[i]) continue;
		let dets = [];
		for (let s = 0; s < 2; s++) {
			const b = N_POSE * 4 + s * HAND_W;
			if (Number.isNaN(raw[i][b + 2])) continue;
			const pts = (o) => Array.from({ length: 21 }, (_, k) => [raw[i][o + k * 3], raw[i][o + k * 3 + 1], raw[i][o + k * 3 + 2]]);
			dets.push({ img: pts(b + 2), world: pts(b + 65), score: raw[i][b + 1] });
		}
		if (!dets.length) continue;
		if (dets.length === 2 && Math.hypot((dets[0].img[0][0] - dets[1].img[0][0]) * width, (dets[0].img[0][1] - dets[1].img[0][1]) * height) < 0.25 * span) {
			dets = [dets[0].score >= dets[1].score ? dets[0] : dets[1]];
		}
		let pairs;
		if (dets.length === 2) {
			const [a, b] = dets;
			pairs = cost(a, 'L', i) + cost(b, 'R', i) <= cost(a, 'R', i) + cost(b, 'L', i) ? [['L', a], ['R', b]] : [['R', a], ['L', b]];
		} else {
			pairs = [[cost(dets[0], 'L', i) <= cost(dets[0], 'R', i) ? 'L' : 'R', dets[0]]];
		}
		for (const [s, d] of pairs) side[s][i] = d;
	}

	const hand = {};
	for (const s of ['L', 'R']) {
		const present = side[s].map(Boolean);
		const rows = side[s].map((d) => (d ? Float64Array.from([...d.img.flat(), ...d.world.flat()]) : new Float64Array(126)));
		const filled = interpGaps(rows, present, MAX_HAND_GAP);
		const sm = smooth(filled.x, filled.present);
		hand[s] = {
			present: filled.present,
			img: sm.map((r, i) => (filled.present[i] ? Array.from({ length: 21 }, (_, k) => [r[k * 3], r[k * 3 + 1], r[k * 3 + 2]]) : null)),
			world: sm.map((r, i) => (filled.present[i] ? Array.from({ length: 21 }, (_, k) => [r[63 + k * 3], r[63 + k * 3 + 1], r[63 + k * 3 + 2]]) : null)),
		};
	}

	const poseBody = (i, j) => body(i, px[i][j][0], px[i][j][1]);
	const pos = {};
	for (const [s, pw] of [['L', P.LWR], ['R', P.RWR]]) {
		pos[s] = raw.map((_, i) => {
			if (!hand[s].present[i]) return { p: poseBody(i, pw), tip: [NaN, NaN] };
			const im = hand[s].img[i];
			return { p: body(i, im[0][0] * width, im[0][1] * height), tip: body(i, im[8][0] * width, im[8][1] * height) };
		});
	}
	const path = (s) => {
		const q = pos[s].filter((_, i) => keep[i]).map((x) => x.p);
		let tot = 0;
		for (let i = 1; i < q.length; i++) tot += Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]);
		return tot;
	};
	const dom = path('R') >= path('L') ? 'R' : 'L';
	const oth = dom === 'R' ? 'L' : 'R';
	const m = dom === 'L' ? -1 : 1;
	const elbow = { L: P.LEL, R: P.REL };

	const cols = {};
	for (const [role, s] of [['dom', dom], ['oth', oth]]) {
		cols[`${role}_x`] = pos[s].map((x) => x.p[0] * m);
		cols[`${role}_y`] = pos[s].map((x) => x.p[1]);
		cols[`${role}_tip_x`] = pos[s].map((x) => x.tip[0] * m);
		cols[`${role}_tip_y`] = pos[s].map((x) => x.tip[1]);
		cols[`${role}_elbow_x`] = raw.map((_, i) => poseBody(i, elbow[s])[0] * m);
		cols[`${role}_elbow_y`] = raw.map((_, i) => poseBody(i, elbow[s])[1]);
		cols[`${role}_present`] = hand[s].present.map((p) => (p ? 1 : 0));
		const shapes = hand[s].world.map((w) => (w ? handShape(dom === 'L' ? w.map((p) => [-p[0], p[1], p[2]]) : w) : null));
		for (let k = 0; k < 78; k++) cols[`${role}_shape${k}`] = shapes.map((sh) => (sh ? sh[k] : NaN));
	}
	cols.nose_x = raw.map((_, i) => poseBody(i, P.NOSE)[0] * m);
	cols.nose_y = raw.map((_, i) => poseBody(i, P.NOSE)[1]);

	const kept = keep.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
	for (const role of ['dom', 'oth']) {
		for (const a of ['x', 'y']) {
			const v = gradient(kept.map((i) => cols[`${role}_${a}`][i])).map((d) => d * FPS);
			cols[`${role}_v${a}`] = new Array(n).fill(0);
			kept.forEach((i, k) => (cols[`${role}_v${a}`][i] = v[k]));
		}
	}
	const rows = kept.map((i) => Float64Array.from(names, (name) => cols[name][i]));
	return { rows, dominant: dom };
}

/** Frames where the dominant hand is seen or raised (tabular_models.active_part). */
export function activePart(rows, names) {
	const iy = names.indexOf('dom_y');
	const ip = names.indexOf('dom_present');
	const act = rows.map((r) => r[ip] > 0.5 || (Number.isNaN(r[iy]) ? -9 : r[iy]) > -1.0);
	const first = act.indexOf(true);
	const last = act.lastIndexOf(true);
	if (first < 0 || act.filter(Boolean).length < 4) return rows;
	return rows.slice(first, last + 1);
}

/** Linear resampling in time per feature, ignoring missing values (tabular_models.resample). */
export function resample(rows, steps) {
	const n = rows.length;
	const d = rows[0].length;
	const out = Array.from({ length: steps }, () => new Float64Array(d).fill(NaN));
	const tOld = rows.map((_, i) => (n === 1 ? 0 : i / (n - 1)));
	for (let j = 0; j < d; j++) {
		const xs = [];
		const ys = [];
		for (let i = 0; i < n; i++) if (!Number.isNaN(rows[i][j])) (xs.push(tOld[i]), ys.push(rows[i][j]));
		if (!xs.length) continue;
		for (let s = 0; s < steps; s++) {
			const t = steps === 1 ? 0 : s / (steps - 1);
			out[s][j] = interp(t, xs, ys);
		}
	}
	return out;
}

/** numpy.interp: clamps outside the known range. */
function interp(t, xs, ys) {
	if (xs.length === 1 || t <= xs[0]) return ys[0];
	if (t >= xs[xs.length - 1]) return ys[ys.length - 1];
	let k = 1;
	while (xs[k] < t) k++;
	const u = (t - xs[k - 1]) / (xs[k] - xs[k - 1]);
	return ys[k - 1] + u * (ys[k] - ys[k - 1]);
}
