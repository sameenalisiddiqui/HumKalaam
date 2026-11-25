// Scores the knowledge-based baseline recogniser on real signers: every WLASL video
// of the 20 camera words is fed, frame by frame, through the same code the camera
// uses (features.js cameraFeatures, then recognizer.js classifyMovement against
// the avatar's own sign templates). It was never trained on WLASL, so every
// video is a fair test; the WLASL test split is reported separately too, to
// compare with the trained models.
//
//   node ml/scripts/eval_template_recognizer.mjs     (after export_test_raw.py)

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cameraFeatures, isActive } from '../../src/camera/features.js';
import { classifyMovement } from '../../src/camera/recognizer.js';
import { coreTemplates } from '../../src/camera/core-words.js';

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(resolve(here, '../data/processed/app20_raw.json'), 'utf8'));
const N_POSE = 33;
const HAND_W = 2 + 21 * 3 * 2;
const GLOSS_TO_SIGN = { 'thank you': 'THANK' };
const signFor = (gloss) => GLOSS_TO_SIGN[gloss] ?? gloss.toUpperCase();

/** A raw row (extract_landmarks.py layout) -> the camera tracker's observation object. */
function observation(row) {
	const v = (x) => (x == null ? NaN : x);
	const pose = Number.isNaN(v(row[0]))
		? null
		: Array.from({ length: N_POSE }, (_, j) => ({ x: v(row[j * 4]), y: v(row[j * 4 + 1]), z: v(row[j * 4 + 2]), visibility: v(row[j * 4 + 3]) }));
	const hands = [];
	for (let s = 0; s < 2; s++) {
		const b = N_POSE * 4 + s * HAND_W;
		if (row[b + 2] == null) continue;
		const image = Array.from({ length: 21 }, (_, k) => ({ x: row[b + 2 + k * 3], y: row[b + 2 + k * 3 + 1], z: row[b + 2 + k * 3 + 2] }));
		const world = Array.from({ length: 21 }, (_, k) => [row[b + 65 + k * 3], row[b + 65 + k * 3 + 1], row[b + 65 + k * 3 + 2]]);
		hands.push({ side: row[b] === 0 ? 'Left' : 'Right', image, world, score: row[b + 1] });
	}
	return { hands, pose };
}

const templates = { Right: coreTemplates('Right'), Left: coreTemplates('Left') };
const rows = [];
let poseCarry = null;
for (const vid of data.videos) {
	const dominant = vid.dominant === 'L' ? 'Left' : 'Right'; // the recogniser is told the signer's hand
	const state = {};
	const frames = [];
	poseCarry = null;
	for (const raw of vid.raw) {
		const obs = observation(raw);
		// The body model ran on every second frame; the camera runs it on every frame, so carry the last one over.
		if (obs.pose) poseCarry = obs.pose;
		else obs.pose = poseCarry;
		const f = cameraFeatures(obs, { width: vid.size[0], height: vid.size[1], dominant, state });
		if (f && (isActive(f.dom) || isActive(f.other))) frames.push(f);
	}
	const truth = signFor(vid.gloss);
	if (frames.length < 3) {
		rows.push({ video_id: vid.video_id, split: vid.split, truth, accepted: null, best: null, rank: null });
		continue;
	}
	const res = classifyMovement(frames, { dominant, templates: templates[dominant] });
	const rank = res.candidates.findIndex((c) => c.word === truth);
	rows.push({ video_id: vid.video_id, split: vid.split, truth, accepted: res.word, best: res.candidates[0]?.word ?? null, rank: rank >= 0 ? rank + 1 : null });
}

const summary = (sel) => {
	const n = sel.length;
	const pct = (k) => Math.round((1000 * k) / n) / 1000;
	return {
		videos: n,
		accepted_and_correct: pct(sel.filter((r) => r.accepted === r.truth).length),
		accepted_any: pct(sel.filter((r) => r.accepted).length),
		top1: pct(sel.filter((r) => r.best === r.truth).length),
		top5: pct(sel.filter((r) => r.rank && r.rank <= 5).length),
		no_movement_found: sel.filter((r) => r.best === null).length,
	};
};
const out = { all_app20_videos: summary(rows), test_split: summary(rows.filter((r) => r.split === 'test')), rows };
writeFileSync(resolve(here, '../results/template_recognizer_results.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ all: out.all_app20_videos, test: out.test_split }, null, 1));
