
import { compileUtterance, scaledTiming } from '../vendor/three-ws/sign-speech.js';
import { findSign, lexiconLookup, omittedReason, vocabulary as allWords } from './lexicon.js';
import { tokenize } from './normalize.js';
import { learnedClip } from '../ml/learned-motion.js';

export const SPEEDS = Object.freeze([0.5, 0.75, 1]);
export const MAX_SECONDS = 60;

/** Deduplicated, sorted list of every word form that has a lexical sign. */
export function vocabulary() {
	return allWords();
}

/** Plain-English description of a word's sign, or null when it has none. */
export function glossFor(word) {
	return findSign(word)?.sign.gloss ?? null;
}

/**
 * @param {string} text
 * @param {{ dominant?: 'Right'|'Left', rate?: number, similar?: Map<string, [string, number]>|null }} [opts]
 *   `similar`: optional word -> [sign name, similarity] table (src/ml/similar-signs.js). Only when the
 *   user turns it on, a word with no sign of its own is signed with the suggested sign (kind 'similar',
 *   with `substitute` and `similarity` set) instead of being fingerspelled.
 * @param {{ kind: 'retrieved'|'generated', data: { columns: string[], bySign: Map<string, object> } }|null} [opts.motion]
 *   experimental: perform the signs that have a path learned from real signers (src/ml/learned-motion.js)
 *   with that path instead of the hand-authored sign (right-handed signing only); such items get `motion` set.
 * @returns {{
 *   ok: boolean,
 *   message: string|null,
 *   items: Array<{
 *     raw: string, word: string|null,
 *     kind: 'signed'|'similar'|'spelled'|'omitted'|'unsupported'|'cut',
 *     gloss: string|null, reason: string|null,
 *     start: number|null, end: number|null,
 *     letters: Array<{letter:string,start:number,end:number}>|null,
 *   }>,
 *   clip: object|null,
 *   duration: number,
 * }}
 */
export function planUtterance(text, { dominant = 'Right', rate = 1, similar = null, motion = null } = {}) {
	if (!SPEEDS.includes(rate)) throw new Error(`unsupported rate ${rate}`);
	const side = dominant === 'Left' ? 'Left' : 'Right';
	const tokens = tokenize(text);

	const items = tokens.map((t) => {
		const omitted = t.supported ? omittedReason(t.word) : null;
		return {
			raw: t.raw,
			word: t.word,
			kind: !t.supported ? 'unsupported' : omitted ? 'omitted' : 'pending',
			gloss: null,
			reason: omitted ? `${omitted}, so this word is left out` : t.reason,
			start: null,
			end: null,
			letters: null,
		};
	});
	const performable = items.filter((i) => i.kind === 'pending');

	if (!tokens.length) {
		return { ok: false, message: 'Type a word or sentence first.', items, clip: null, duration: 0 };
	}
	if (!performable.length && items.some((i) => i.kind === 'omitted') && items.every((i) => i.kind !== 'unsupported')) {
		return { ok: false, message: 'These words are not signed in ASL (a, the, is, to…). Add a word with meaning.', items, clip: null, duration: 0 };
	}
	if (!performable.length) {
		return {
			ok: false,
			message: 'None of these words can be signed or fingerspelled. Only English letters A-Z and digits 0-9 are supported.',
			items,
			clip: null,
			duration: 0,
		};
	}

	// Opt-in: a word with no sign of its own is performed with a similar word's sign.
	for (const item of performable) {
		const hit = similar && !findSign(item.word) ? similar.get(item.word.toLowerCase()) : null;
		if (hit && findSign(hit[0])) {
			item.substitute = hit[0];
			item.similarity = hit[1];
		}
	}
	const performed = (i) => i.substitute ?? i.word;

	const authored = lexiconLookup({ dominant: side, rate });
	const learned = (word) => (motion && side === 'Right' ? motion.data.bySign.get(findSign(word)?.name) : null);
	const signs = (word, position = {}) => {
		const entry = learned(word);
		if (!entry) return authored(word, position);
		return learnedClip(entry, motion.data.columns, { kind: motion.kind, lead: position.first !== false, settle: position.last !== false, rate });
	};
	const result = compileUtterance(performable.map(performed).join(' '), {
		signs,
		timing: rate === 1 ? undefined : scaledTiming(rate),
		dominant: side,
		maxSeconds: MAX_SECONDS,
		name: 'hum-kalaam-utterance',
	});

	// compileUtterance performs words in order and stops early (truncated) when
	// the clip would pass MAX_SECONDS, so segments line up 1:1 with a prefix of
	// `performable`. Anything after that prefix was not performed.
	performable.forEach((item, i) => {
		const seg = result.segments[i];
		if (!seg) {
			item.kind = 'cut';
			item.reason = `not performed: the message is longer than ${MAX_SECONDS} seconds of signing`;
			return;
		}
		if (seg.word !== performed(item)) throw new Error(`segment mismatch: ${seg.word} vs ${performed(item)}`);
		item.kind = seg.signed ? (item.substitute ? 'similar' : 'signed') : 'spelled';
		if (seg.signed && learned(seg.word)) item.motion = motion.kind;
		item.gloss = seg.signed ? glossFor(seg.word) : null;
		item.start = seg.start;
		item.end = seg.end;
		item.letters = seg.letters;
	});

	return { ok: true, message: null, items, clip: result.clip, duration: result.clip.duration };
}
