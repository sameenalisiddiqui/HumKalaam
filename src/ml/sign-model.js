// The sign models trained on real signers (ml/scripts/tabular_models.py), run in
// the browser with onnxruntime-web: a GRU over the movement and an XGBoost model
// over a summary of it, averaged. Both were trained on WLASL videos of the app's
// 20 camera words; their accuracy on WLASL test videos is in ml/results.

import * as ort from 'onnxruntime-web';
import { activePart, cleanSegment, resample } from './landmark-features.js';

const BASE = import.meta.env?.BASE_URL ?? '/';
// The runtime's WASM file is bundled by Vite (vite.config.js picks the CPU-only build).
// Threads need cross-origin isolation, which a plain static host does not give.
if (typeof window !== 'undefined') ort.env.wasm.numThreads = 1;

const STATS = ['dom_x', 'dom_y', 'oth_x', 'oth_y', 'dom_tip_x', 'dom_tip_y'];

function nanStats(v) {
	const ok = v.filter((x) => !Number.isNaN(x));
	if (!ok.length) return [NaN, NaN, NaN, NaN];
	const mean = ok.reduce((a, b) => a + b, 0) / ok.length;
	const sd = Math.sqrt(ok.reduce((a, b) => a + (b - mean) ** 2, 0) / ok.length);
	return [mean, sd, Math.min(...ok), Math.max(...ok)];
}

/** One row per movement for the XGBoost model (tabular_models.video_row). */
export function videoRow(rows, names, meta) {
	const a = activePart(rows, names);
	const cols = meta.video_row_compact.map((c) => names.indexOf(c));
	const r = resample(a.map((row) => Float64Array.from(cols, (j) => row[j])), meta.video_row_steps);
	const out = r.flatMap((row) => Array.from(row));
	const col = (c) => a.map((row) => row[names.indexOf(c)]);
	for (const c of STATS) out.push(...nanStats(col(c)));
	const vx = col('dom_vx');
	const vy = col('dom_vy');
	const speed = vx.map((x, i) => Math.hypot(x, vy[i]));
	const okSpeed = speed.filter((x) => !Number.isNaN(x));
	const oth = col('oth_present').filter((x) => !Number.isNaN(x));
	out.push(a.length / 25, okSpeed.reduce((s, x) => s + x, 0) / 25, okSpeed.length ? Math.max(...okSpeed) : NaN, oth.length ? oth.reduce((s, x) => s + x, 0) / oth.length : NaN);
	return Float32Array.from(out);
}

/** The GRU's input: the movement resampled to `seq_steps` frames, standardised, missing values 0. */
export function sequenceInput(rows, names, meta) {
	const r = resample(activePart(rows, names), meta.seq_steps);
	const out = new Float32Array(meta.seq_steps * names.length);
	r.forEach((row, t) => row.forEach((v, j) => (out[t * names.length + j] = Number.isNaN(v) ? 0 : (v - meta.seq_mean[j]) / meta.seq_std[j])));
	return out;
}

const softmax = (z) => {
	const m = Math.max(...z);
	const e = z.map((v) => Math.exp(v - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / s);
};

/** The probability output of a converted XGBoost model (a [1, classes] float tensor). */
function probabilities(result, n) {
	for (const t of Object.values(result)) if (t.type === 'float32' && t.data.length === n) return Array.from(t.data);
	throw new Error('the XGBoost model returned no probability tensor');
}

export class LearnedSignModel {
	/**
	 * @param {{ meta: object, gru: ArrayBuffer|Uint8Array|string, xgb: ArrayBuffer|Uint8Array|string }} files
	 *   model files as bytes, or URLs
	 */
	static async load(files = {}) {
		const meta = files.meta ?? (await (await fetch(`${BASE}ml/signs_app20_meta.json`)).json());
		const gru = await ort.InferenceSession.create(files.gru ?? `${BASE}ml/signs_app20_gru.onnx`);
		const xgb = await ort.InferenceSession.create(files.xgb ?? `${BASE}ml/signs_app20_xgb.onnx`);
		return new LearnedSignModel(meta, gru, xgb);
	}

	constructor(meta, gru, xgb) {
		this.meta = meta;
		this.names = meta.feature_names;
		this.gru = gru;
		this.xgb = xgb;
	}

	/**
	 * Classify one movement.
	 * @param {Float64Array[]} raw frames in the raw layout (landmark-features.js rawRow)
	 * @param {number} width
	 * @param {number} height
	 * @returns {Promise<{ ranked: {word: string, p: number}[] }|null>} null when the body was not seen
	 */
	async classify(raw, width, height) {
		const seg = cleanSegment(raw, width, height, this.names);
		if (!seg || seg.rows.length < 4) return null;
		const k = this.meta.classes.length;
		const seq = new ort.Tensor('float32', sequenceInput(seg.rows, this.names, this.meta), [1, this.meta.seq_steps, this.names.length]);
		const row = videoRow(seg.rows, this.names, this.meta);
		const [g, x] = await Promise.all([
			this.gru.run({ [this.gru.inputNames[0]]: seq }),
			this.xgb.run({ [this.xgb.inputNames[0]]: new ort.Tensor('float32', row, [1, row.length]) }),
		]);
		const pg = softmax(Array.from(g[this.gru.outputNames[0]].data));
		const px = probabilities(x, k);
		const p = pg.map((v, i) => (v + px[i]) / 2);
		const ranked = this.meta.classes.map((word, i) => ({ word, p: p[i] })).sort((a, b) => b.p - a.p);
		return { ranked, gru: pg, xgb: px };
	}
}
