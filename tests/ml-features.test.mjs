// The trained sign models only work if the app describes a movement exactly as
// the training data was described. These tests run the JavaScript port
// (src/ml/landmark-features.js, src/ml/sign-model.js) on real WLASL frames and
// compare it with what the Python pipeline (ml/scripts/clean_landmarks.py,
// tabular_models.py) produced for the same frames.
//
// Fixture: tests/fixtures/ml-app20.json, written by ml/scripts/export_test_raw.py.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { cleanSegment } from '../src/ml/landmark-features.js';
import { LearnedSignModel } from '../src/ml/sign-model.js';

const FIXTURE = new URL('./fixtures/ml-app20.json', import.meta.url);
const fixture = existsSync(FIXTURE) ? JSON.parse(readFileSync(FIXTURE, 'utf8')) : { feature_names: [], videos: [] };
const names = fixture.feature_names;
const toRows = (rows) => rows.map((r) => Float64Array.from(r, (v) => (v == null ? NaN : v)));

test('fixture has real test videos', (t) => {
	if (!existsSync(FIXTURE)) return t.skip('no fixture yet: run ml/scripts/export_test_raw.py');
	assert.ok(fixture.videos.length >= 1);
	for (const v of fixture.videos) assert.ok(v.raw.length >= 8, `${v.video_id} has ${v.raw.length} frames`);
});

for (const v of fixture.videos) {
	test(`cleaned features match Python for WLASL video ${v.video_id} (${v.gloss})`, () => {
		const seg = cleanSegment(toRows(v.raw), v.size[0], v.size[1], names);
		assert.ok(seg, 'the body was found');
		const want = toRows(v.features);
		assert.equal(seg.rows.length, want.length, 'same frames kept');
		let worst = 0;
		let where = '';
		for (let i = 0; i < want.length; i++) {
			for (let j = 0; j < names.length; j++) {
				const a = seg.rows[i][j];
				const b = want[i][j];
				if (Number.isNaN(b)) {
					assert.ok(Number.isNaN(a), `${names[j]} frame ${i}: expected missing, got ${a}`);
					continue;
				}
				const d = Math.abs(a - b);
				if (d > worst) [worst, where] = [d, `${names[j]} frame ${i}`];
			}
		}
		assert.ok(worst < 2e-3, `largest difference ${worst} at ${where}`);
	});
}

test('the ONNX models in the app give the Python models\' probabilities', async (t) => {
	const v = fixture.videos.find((x) => x.gru && x.xgb_video);
	if (!v) return t.skip('no model probabilities in the fixture (train the models first)');
	const read = (p) => new Uint8Array(readFileSync(new URL(`../public/ml/${p}`, import.meta.url)));
	const model = await LearnedSignModel.load({
		meta: JSON.parse(readFileSync(new URL('../public/ml/signs_app20_meta.json', import.meta.url), 'utf8')),
		gru: read('signs_app20_gru.onnx'),
		xgb: read('signs_app20_xgb.onnx'),
	});
	const res = await model.classify(toRows(v.raw), v.size[0], v.size[1]);
	const maxDiff = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
	assert.ok(maxDiff(res.gru, v.gru) < 1e-3, `GRU differs by ${maxDiff(res.gru, v.gru)}`);
	assert.ok(maxDiff(res.xgb, v.xgb_video) < 1e-3, `XGBoost differs by ${maxDiff(res.xgb, v.xgb_video)}`);
});
