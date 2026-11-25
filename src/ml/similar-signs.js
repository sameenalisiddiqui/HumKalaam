// "Similar sign" suggestions for English words the vocabulary does not have.
//
// The table (public/ml/similar-signs.json) was computed offline by the BERT
// sign encoder (ml/scripts/nlp_text_to_gloss.py): BERT fine-tuned on the groups
// of words that share one sign in J. S. Long's "The Sign Language" (1910). Only
// suggestions at or above the similarity where the model was right at least 90%
// of the time on the app's own vocabulary are in it. A 440 MB model cannot run in
// the browser; its decisions for a fixed word list can.

const BASE = import.meta.env?.BASE_URL ?? '/';
let table = null;

/** word -> [SIGN, similarity], loaded once. */
export async function loadSimilarSigns(url = `${BASE}ml/similar-signs.json`) {
	if (!table) {
		const data = await (await fetch(url)).json();
		table = new Map(Object.entries(data.words));
		table.threshold = data.threshold;
	}
	return table;
}
