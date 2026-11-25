// The avatar's own (hand-authored) signs, sampled in the same body coordinates as
// the real-signer landmarks (clean_landmarks.py), so the two can be compared:
//
//   positions  shoulder widths from the middle of the shoulders, x toward the
//              signer's left, y up (the engine's model space divided by its
//              shoulder span, measured from the shoulder line)
//   handshape  the 15 distances between the wrist and the five fingertips,
//              after scaling by the wrist-to-middle-knuckle length
//
// Only signs whose word is also a WLASL gloss in our data are written.
//
//   node ml/scripts/export_avatar_trajectories.mjs

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildClip, findSign } from '../../src/text-to-sign/lexicon.js';
import { evaluateClip } from '../../src/camera/templates.js';
import { poseHandLandmarks } from '../../src/camera/features.js';
import { ANCHORS } from '../../src/vendor/three-ws/sign-rig.js';

const here = dirname(fileURLToPath(import.meta.url));
const proc = resolve(here, '../data/processed');
const HZ = 25;
const SPAN = ANCHORS.shoulderSpan;
const SHOULDER_Y = ANCHORS.shoulder.Right[1];
const TIPS = [0, 4, 8, 12, 16, 20];

const body = (p) => [p[0] / SPAN, (p[1] - SHOULDER_Y) / SPAN];

function tipDistances(lm) {
	const w = lm.map((p) => [p[0] - lm[0][0], p[1] - lm[0][1], p[2] - lm[0][2]]);
	const s = Math.hypot(...w[9]) || 1;
	const n = w.map((p) => p.map((v) => v / s));
	const out = [];
	for (let i = 0; i < TIPS.length; i++)
		for (let j = i + 1; j < TIPS.length; j++) out.push(Math.hypot(...[0, 1, 2].map((k) => n[TIPS[i]][k] - n[TIPS[j]][k])));
	return out;
}

/** WLASL gloss -> the app's sign for it, if the app has one. */
function appSignFor(gloss) {
	for (const w of [gloss, gloss.replace(/ /g, ''), gloss.split(' ')[0]]) {
		const hit = findSign(w);
		if (hit) return hit.name;
	}
	return null;
}

const videos = readFileSync(resolve(proc, 'videos.csv'), 'utf8').trim().split('\n').slice(1).map((l) => l.split(','));
const glosses = [...new Set(videos.map((r) => r[1]))];
const out = {};
for (const gloss of glosses) {
	const name = appSignFor(gloss);
	if (!name) continue;
	const clip = buildClip(name, { dominant: 'Right', lead: true, settle: true });
	if (!clip) continue;
	const frames = [];
	for (let t = 0; t <= clip.duration + 1e-9; t += 1 / HZ) {
		const pose = evaluateClip(clip, t);
		const dom = poseHandLandmarks(pose, 'Right');
		const oth = poseHandLandmarks(pose, 'Left');
		const [dx, dy] = body(dom[0]);
		const [tx, ty] = body(dom[8]);
		const [ox, oy] = body(oth[0]);
		frames.push([dx, dy, tx, ty, ox, oy, ...tipDistances(dom)]);
	}
	out[gloss] = { sign: name, frames };
}
writeFileSync(resolve(proc, 'avatar_trajectories.json'), JSON.stringify(out));
console.log(`${Object.keys(out).length} of ${glosses.length} glosses have an avatar sign`);
