// Adapter between the camera recogniser (src/camera/recognizer.js) and the sign
// models trained on real signers (sign-model.js). The recogniser hands over the
// frames of one movement; each frame carries its raw landmarks (`raw`, from
// landmark-features.js rawRow) and the video size, added in camera-ui.js.

import { LearnedSignModel } from './sign-model.js';

/**
 * A movement is reported as a word only when the models give that word at least
 * this probability; otherwise the closest words are offered to pick from.
 * Chosen on the WLASL validation videos (ml/results/tabular_results.json).
 */
export const ACCEPT_P = 0.4;

let loading = null;
export function loadLearnedModel() {
	if (!loading) {
		loading = LearnedSignModel.load();
		loading.catch(() => (loading = null)); // a failed load can be retried
	}
	return loading;
}

const asWord = (gloss) => gloss.toUpperCase();

/** recognizer `classify` callback: frames of one movement -> result in the recogniser's format. */
export async function classifyFrames(frames, model) {
	const withRaw = frames.filter((f) => f?.raw);
	const size = withRaw[0]?.size;
	if (!size || withRaw.length < 8) return null;
	const res = await model.classify(withRaw.map((f) => f.raw), size.width, size.height);
	if (!res) return null;
	const best = res.ranked[0];
	return {
		word: best.p >= ACCEPT_P ? asWord(best.word) : null,
		distance: 1 - best.p,
		candidates: res.ranked.slice(0, 5).map((r) => ({ word: asWord(r.word), distance: 1 - r.p, p: r.p })),
	};
}
