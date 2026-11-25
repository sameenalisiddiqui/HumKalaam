// Avatar motion from real signers (experimental).
//
// ml/scripts/pose_generation.py gives, for each sign the app shares with WLASL,
// a 32-step path of the dominant wrist and index fingertip, the other wrist and
// the dominant handshape (15 fingertip distances), in shoulder widths from the
// middle of the shoulders. "retrieved" is one real recording (the one closest to
// the sign's other recordings); "generated" is the decoder trained with a mean
// squared error, which averages the recordings.
//
// Here that path becomes keyframes for the engine's own arm solver: the wrist is
// put where the signers' wrists were, the fingers point from the wrist toward the
// recorded fingertip, and the handshape is the engine's nearest named handshape.
// What 2D landmarks cannot give (depth, palm direction) is filled with neutral
// defaults, so this is a data-driven draft of a sign, not a finished one.

import { SignTimeline, neutralPose, posePhase, restingPose } from '../vendor/three-ws/sign-clip.js';
import { HANDSHAPE_NAMES } from '../vendor/three-ws/sign-handshapes.js';
import { ANCHORS } from '../vendor/three-ws/sign-rig.js';
import { poseHandLandmarks } from '../camera/features.js';

const BASE = import.meta.env?.BASE_URL ?? '/';
const SPAN = ANCHORS.shoulderSpan;
const SHOULDER_Y = ANCHORS.shoulder.Right[1];
const TIPS = [0, 4, 8, 12, 16, 20];
const KEYFRAMES = 8;
const SECONDS_PER_KEY = 0.14;
const MISSING = -9; // how the exporter marks a value MediaPipe did not see

function tipDistances(lm) {
	const w = lm.map((p) => [p[0] - lm[0][0], p[1] - lm[0][1], p[2] - lm[0][2]]);
	const s = Math.hypot(...w[9]) || 1;
	const n = w.map((p) => p.map((v) => v / s));
	const out = [];
	for (let i = 0; i < TIPS.length; i++)
		for (let j = i + 1; j < TIPS.length; j++) out.push(Math.hypot(...[0, 1, 2].map((k) => n[TIPS[i]][k] - n[TIPS[j]][k])));
	return out;
}

let shapeTable = null;
/** Fingertip distances of every engine handshape, measured on the posed rig. */
function engineShapes() {
	if (!shapeTable) {
		shapeTable = HANDSHAPE_NAMES.map((name) => {
			const pose = posePhase({ right: { shape: name, at: 'sternum' } });
			return { name, d: tipDistances(poseHandLandmarks(pose, 'Right')) };
		});
	}
	return shapeTable;
}

/** The engine handshape whose fingertip distances are closest to the recorded ones. */
export function nearestHandshape(dists) {
	if (!dists || dists.some((v) => v === MISSING || Number.isNaN(v))) return null;
	let best = null;
	for (const s of engineShapes()) {
		const e = s.d.reduce((acc, v, i) => acc + (v - dists[i]) ** 2, 0);
		if (!best || e < best.e) best = { name: s.name, e };
	}
	return best.name;
}

/** Body units -> engine model space. Depth is a guess: in front of the chest, closer in near the face. */
function toModel(x, y) {
	const forward = y > 0.2 ? 0.16 : 0.28;
	return [x * SPAN, SHOULDER_Y + y * SPAN, ANCHORS.sternum[2] + forward];
}

/** Keyframe phases (engine notation) for one recorded or generated path. */
export function phasesFromPath(path, columns) {
	const col = (name) => columns.indexOf(name);
	const [ix, iy, itx, ity, iox, ioy] = ['dom_x', 'dom_y', 'dom_tip_x', 'dom_tip_y', 'oth_x', 'oth_y'].map(col);
	const shapeCols = columns.map((c, i) => (c.startsWith('dom_shape') ? i : -1)).filter((i) => i >= 0);
	const phases = [];
	let lastShape = 'RELAXED';
	for (let k = 0; k < KEYFRAMES; k++) {
		const step = path[Math.round((k * (path.length - 1)) / (KEYFRAMES - 1))];
		const x = step[ix];
		const y = step[iy];
		const shape = nearestHandshape(shapeCols.map((i) => step[i])) ?? lastShape;
		lastShape = shape;
		const tipSeen = step[itx] !== MISSING && step[ity] !== MISSING;
		const dx = tipSeen ? step[itx] - x : 0;
		const dy = tipSeen ? step[ity] - y : 0;
		const fingers = Math.hypot(dx, dy) > 0.12 ? [dx, dy, 0.35] : 'up';
		const right = { shape, at: toModel(x, y), fingers, palm: 'forward' };
		const ox = step[iox];
		const oy = step[ioy];
		const left = oy > -1.0 ? { shape: 'B', at: toModel(ox, oy), fingers: 'up', palm: 'in' } : 'rest';
		phases.push({ right, left, t: SECONDS_PER_KEY });
	}
	return phases;
}

/** A clip document for one word from a learned path, in the same format buildClip makes. */
export function learnedClip(entry, columns, { kind = 'retrieved', lead = true, settle = true, rate = 1 } = {}) {
	const base = restingPose();
	const tl = new SignTimeline({ base, open: lead ? base : neutralPose(base) });
	phasesFromPath(entry[kind], columns).forEach((phase, i) => {
		tl.to(posePhase(phase, base), (phase.t + (i === 0 && lead ? 0.25 : 0)) / rate, { ease: i === 0 ? 'out' : 'smooth' });
	});
	if (settle) tl.settle(0.45 / rate);
	return tl.build({ name: `learned-${entry.sign?.toLowerCase() ?? 'sign'}`, seed: `learned:${entry.sign}:${kind}:${rate}:${lead}:${settle}` });
}

let data = null;
/** The learned paths, keyed by the app's sign name. */
export async function loadLearnedMotion(url = `${BASE}ml/learned-motion.json`) {
	if (!data) {
		const json = await (await fetch(url)).json();
		const bySign = new Map();
		for (const entry of Object.values(json.signs)) if (entry.sign) bySign.set(entry.sign, entry);
		data = { columns: json.columns, bySign };
	}
	return data;
}
