
const SPLIT = /[\s\-‐-―/_]+/; // spaces, hyphens/dashes, slashes, underscores
const IGNORABLE = /[!"#$%&'()*+,.:;<=>?@[\\\]^`{|}~‘’“”…]/g;
const SPELLABLE = /^[A-Z0-9]+$/;

/**
 * @typedef {{ raw: string, word: string|null, supported: boolean, reason: string|null }} Token
 */

/**
 * @param {string} text
 * @returns {Token[]}
 */
export function tokenize(text) {
	const tokens = [];
	for (const raw of String(text ?? '').split(SPLIT)) {
		if (!raw) continue;
		const stripped = raw.replace(IGNORABLE, '');
		if (!stripped) continue; // pure punctuation ("!", "...") carries no word
		const upper = stripped.toUpperCase();
		if (SPELLABLE.test(upper)) {
			tokens.push({ raw, word: upper, supported: true, reason: null });
		} else {
			const bad = [...new Set(upper.replace(/[A-Z0-9]/g, ''))].join(' ');
			tokens.push({
				raw,
				word: null,
				supported: false,
				reason: `contains characters the signing engine cannot fingerspell (${bad})`,
			});
		}
	}
	return tokens;
}
