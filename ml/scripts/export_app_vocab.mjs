// Writes the app's sign vocabulary (every sign and every English spelling that
// reaches it) to ml/data/processed/app_vocab.json, so the Python models can be
// scored against exactly what the avatar can sign.
//
//   node ml/scripts/export_app_vocab.mjs

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { allSigns, findSign, vocabulary } from '../../src/text-to-sign/lexicon.js';

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, '../data/processed/app_vocab.json');

const signs = [...allSigns().keys()].sort();
const forms = {};
for (const word of vocabulary()) {
	const hit = findSign(word);
	if (hit) forms[word.toLowerCase()] = hit.name;
}
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ signs, forms }, null, 1));
console.log(`${signs.length} signs, ${Object.keys(forms).length} spellings -> ${out}`);
