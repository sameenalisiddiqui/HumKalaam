// Results dashboard: one short tab per part of the project, read from public/ml/dashboard.json
// (written by ml/scripts/export_for_app.py).

import { PAPERS, RELATED_WORK, TREE } from './dashboard-content.js';

const BASE = import.meta.env?.BASE_URL ?? '/';
const main = document.getElementById('main');
const tabsEl = document.getElementById('tabs');

// ── helpers ──────────────────────────────────────────────────────────────────
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(d)}%`);
const num = (x) => (x == null ? '–' : Number(x).toLocaleString('en-US'));
const fix = (x, d = 3) => (x == null ? '–' : Number(x).toFixed(d));
const fig = (data, name, caption) =>
	data.figures?.includes(name) ? `<figure><img src="${BASE}ml/figures/${name}" alt="${esc(caption)}" loading="lazy"><figcaption>${esc(caption)}</figcaption></figure>` : '';

const metric = (k, v, sub = '') => `<div class="metric"><div class="k">${esc(k)}</div><div class="v">${v}</div>${sub ? `<div class="d">${esc(sub)}</div>` : ''}</div>`;

function table(head, rows, { num: numeric = [], best = -1 } = {}) {
	const th = head.map((h, i) => `<th class="${numeric.includes(i) ? 'n' : ''}">${h}</th>`).join('');
	const tr = rows.map((r, ri) => `<tr class="${ri === best ? 'best' : ''}">${r.map((c, i) => `<td class="${numeric.includes(i) ? 'n' : ''}">${c}</td>`).join('')}</tr>`).join('');
	return `<div class="t-wrap"><table class="t"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

/** Grouped vertical bars, values 0..1 shown as %. */
function bars({ labels, series, height = 240 }) {
	const W = 720;
	const H = height;
	const pad = { l: 38, r: 10, t: 16, b: 46 };
	const n = labels.length;
	const groupW = (W - pad.l - pad.r) / n;
	const barW = Math.min(36, (groupW * 0.8) / series.length);
	const y = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
	let svg = '';
	for (const t of [0, 0.25, 0.5, 0.75, 1]) svg += `<line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${pad.l - 6}" y="${y(t) + 4}" font-size="11" text-anchor="end">${t * 100}</text>`;
	labels.forEach((lab, i) => {
		const gx = pad.l + i * groupW + (groupW - barW * series.length) / 2;
		series.forEach((s, j) => {
			const v = s.values[i];
			if (v == null) return;
			const x = gx + j * barW;
			svg += `<rect x="${x}" y="${y(v)}" width="${barW - 3}" height="${y(0) - y(v)}" rx="3" fill="${s.color}"><title>${esc(lab)} · ${esc(s.name)}: ${pct(v)}</title></rect>`;
			if (j === 0) svg += `<text x="${x + (barW - 3) / 2}" y="${y(v) - 4}" font-size="11" text-anchor="middle">${Math.round(v * 100)}</text>`;
		});
		String(lab).split('\n').forEach((w, k) => (svg += `<text x="${pad.l + i * groupW + groupW / 2}" y="${H - pad.b + 16 + k * 13}" font-size="11" text-anchor="middle">${esc(w)}</text>`));
	});
	const legend = `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>`;
	return `${legend}<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${svg}</svg>`;
}

const C = { a: '#0b8a6f', b: '#9fd9c6', blue: '#2f6fed', amber: '#e3a23b' };

// ── tabs ─────────────────────────────────────────────────────────────────────
function problem(d) {
	const cv = d.tabular_app20_cv;
	const img = d.images?.['6_fusion']?.test?.accuracy;
	return `
	<h1>Two-way sign language communication with AI</h1>
	<p class="lead">Few hearing people know sign language and interpreters are scarce, so Deaf and hearing people often cannot talk directly.
	Hum Kalaam is an assistive system that signs typed or spoken English with a 3D avatar, and turns signing in front of a webcam into text and speech.</p>
	<div class="metrics">
		${metric('Sign recognition', pct(cv?.gru?.top1_mean), 'GRU, 5-fold cross-validation')}
		${metric('Knowledge-based baseline', pct(d.template_recognizer?.all_app20_videos?.top1), 'same 171 videos')}
		${metric('Letter recognition', pct(img), 'CNN + landmark fusion')}
		${metric('Sign videos', num(d.wlasl?.videos_downloaded), `${num(d.wlasl?.signers)} real signers`)}
	</div>
	<div class="grid3">
		<div class="card"><h4>Objective</h4><p>Recognise everyday signs from a webcam, and show any English sentence as signing, in a browser on a phone or laptop.</p></div>
		<div class="card"><h4>Why it matters</h4><p>Hospitals, shops and classrooms rarely have an interpreter. A free tool that works on the device helps both sides start a conversation.</p></div>
		<div class="card"><h4>AI approach</h4><p>Hybrid: a knowledge base of sign descriptions drives the avatar, and machine-learning models trained on real signers read the camera.</p></div>
	</div>
	<div class="card ethics">
		<h4>Ethics, privacy and limits</h4>
		<ul>
			<li><b>Privacy:</b> the models use only hand and body points, never faces or pixels. Recognition runs on the device, so no video leaves it, and WLASL signers are IDs, not names.</li>
			<li><b>Fairness:</b> balanced class weights and macro-F1, so rare signs count equally; left-handed signing is mirrored, not treated as wrong.</li>
			<li><b>Transparency and accountability:</b> every number comes from a script with a fixed seed (361), and the app shows each sign's description. It is a communication aid, not a replacement for interpreters, and the signs have not yet been reviewed by Deaf signers.</li>
			<li><b>External factors:</b> poor light, camera angle, distance, and how fast or differently a person signs can lower accuracy; the models were tested on WLASL videos (4–16 per sign), not on every webcam.</li>
		</ul>
	</div>
	<h3>Inputs and outputs</h3>
	${table(['Input', 'AI step', 'Output'], [
		['Webcam video of a signer', 'MediaPipe hand and body points → GRU sign model', 'Word as text and speech'],
		['Photo of a hand (A–Z)', 'EfficientNet CNN + hand points (fusion)', 'Letter (tested on 260 photos)'],
		['Typed or spoken English', 'Text rules + knowledge base of 476 sign descriptions', 'Avatar signing'],
	])}`;
}

function data(d) {
	const w = d.wlasl;
	return `
	<h2>Datasets</h2>
	<p class="caption">Two public datasets of two types: videos turned into a large table, and images.</p>
	<div class="grid2">
		<div class="card"><h4>WLASL: word-level ASL videos</h4>
			<p><b>Source:</b> Li et al., WACV 2020 (Hugging Face mirror of Kaggle <code>wlasl-processed</code>). <b>Type:</b> video → tabular. <b>Relevance:</b> real people signing single words, the same task as the camera.</p>
			${w ? table(['', ''], [['Videos', num(w.videos_downloaded)], ['Signs (words)', num(w.glosses)], ['Signers', num(w.signers)], ['Landmark rows after cleaning', num(d.cleaning?.frames_kept)], ['Features per row', num(d.cleaning?.feature_columns)]], { num: [1] }) : ''}
		</div>
		<div class="card"><h4>ASL Alphabet: letter images</h4>
			<p><b>Source:</b> Kaggle <code>grassknoted/asl-alphabet</code>. <b>Type:</b> images. <b>Relevance:</b> handshapes of the 26 letters, for fingerspelling.</p>
			${table(['Use', 'Images'], [['Training', '387'], ['Validation', '113'], ['Unlabelled (semi-supervised)', '100'], ['Held-out test', '260'], ["Kaggle's own test photos", '26']], { num: [1] })}
		</div>
	</div>`;
}

function preprocessing(d) {
	const c = d.cleaning;
	if (!c) return '<h2>Preprocessing</h2>';
	const step = (n, title, problem, fix, count = '') => `
		<details class="step"><summary><span class="step-n">${n}</span><span class="step-t">${title}</span>${count ? `<span class="step-c">${count}</span>` : ''}</summary>
			<p><b>Problem:</b> ${problem}</p><p><b>Fix:</b> ${fix}</p>
		</details>`;
	const kept = c.frames_kept / c.frames_raw;
	return `
	<h2>Preprocessing</h2>
	<p class="caption">MediaPipe turns every video frame into 388 raw numbers (hand and body points). Raw landmarks are noisy and full of gaps, so they are cleaned in seven steps before any model sees them. Click a step for details.</p>
	<div class="metrics">
		${metric('Raw frames', num(c.frames_raw), `${num(c.videos_seen)} videos`)}
		${metric('Clean frames kept', num(c.frames_kept), `${pct(kept)} of all frames`)}
		${metric('Features per frame', num(c.feature_columns), 'from 388 raw numbers')}
		${metric('Left-handed videos', num(c.videos_left_dominant_mirrored), 'mirrored, not dropped')}
	</div>
	<h3>Landmark cleaning, step by step</h3>
	${step(1, 'Missing body frames', `the body model was run on every 2nd frame to save time (${num(c.frames_body_model_run)} frames), and sometimes found no body (${num(c.frames_without_body)} frames).`, `frames in between are filled from their neighbours (${num(c.body_frames_interpolated)} frames). If the body is missing for more than 5 frames in a row, those frames are dropped (${num(c.frames_dropped_no_body)}).`, `${num(c.frames_dropped_no_body)} dropped`)}
	${step(2, 'Which hand is which', "MediaPipe's own left/right label flips when the palm is hidden, and sometimes one hand is detected twice.", `each hand is given to the nearer body wrist (${num(c.mediapipe_hand_label_overruled)} labels corrected). Of two detections of the same hand, the more confident one is kept (${num(c.duplicate_hand_detections_removed)} removed).`, `${num(c.mediapipe_hand_label_overruled)} corrected`)}
	${step(3, 'Short tracking drop-outs', 'a hand can vanish for a few frames while it moves fast.', `gaps of at most 6 frames (0.24 s) are filled by straight-line interpolation (${num(c.hand_frames_interpolated)} frames). Longer gaps stay empty: the hand is down or out of view.`, `${num(c.hand_frames_interpolated)} filled`)}
	${step(4, 'Jitter', 'landmark points shake slightly from frame to frame.', 'a 3-frame centred moving average smooths every visible stretch.')}
	${step(5, 'Body size and camera distance', 'a tall signer far from the camera and a short one close to it give very different numbers for the same sign.', "positions are measured in shoulder widths from the middle of the shoulders (x toward the signer's left, y up). The video's median shoulder width is used, so one bad frame cannot rescale it.")}
	${step(6, 'Left-handed signers', 'a left-handed sign is a mirror image of the right-handed one.', `the hand that travels further is the dominant hand; ${num(c.videos_left_dominant_mirrored)} left-handed videos are mirrored so every video reads as right-handed.`, `${num(c.videos_left_dominant_mirrored)} mirrored`)}
	${step(7, 'Handshape', 'raw finger points depend on where the hand is and how big it is.', 'the 21 hand points are measured from the wrist and scaled by the palm length (63 numbers), plus the 15 distances between the wrist and fingertips: 78 handshape numbers per hand.')}
	<div class="grid2 quad">
		<div class="card"><h4>The 176 features</h4><p>Per hand (dominant and other): wrist, index fingertip and elbow position, speed, whether the hand is visible, and the 78 handshape numbers. That is 87 per hand, plus the nose position: 87 × 2 + 2 = 176.</p></div>
		<div class="card"><h4>Missing values that remain</h4><p>${c.missing_cells_pct}% of cells stay empty, mostly the second hand, which many signs do not use. A "hand visible" feature records this, XGBoost handles empty cells itself, and the GRU gets the average value.</p></div>
		<div class="card"><h4>Class imbalance</h4><p>Each sign has 4–16 videos. Every model uses balanced class weights, results report macro-F1, and cross-validation folds keep each sign's share.</p></div>
		<div class="card"><h4>Images (letters)</h4><p>Training, validation, unlabelled and test images come from separate parts of each letter's recording, so test photos are never near-copies. Training adds random crops, small rotations and colour changes, but never left-right flips: a mirrored hand is a different hand.</p></div>
	</div>`;
}

const SIGN_MODELS = [
	['rf', 'Random Forest', 'baseline'],
	['xgb_frame', 'XGBoost (frames)'],
	['xgb_video', 'XGBoost (per video)'],
	['gru', 'GRU (sequence)'],
	['ensemble', 'Ensemble (GRU + XGBoost)'],
];

function signs(d) {
	const cv = d.tabular_app20_cv;
	const kb = d.template_recognizer?.all_app20_videos;
	if (!cv) return '<h2>Sign recognition</h2>';
	const all = [
		['Knowledge-based matching <span class="tag base">baseline</span>', kb?.top1, null, kb?.top5, null],
		...SIGN_MODELS.map(([k, l, base]) => [`${l}${base ? ' <span class="tag base">baseline</span>' : ''}`, cv[k]?.top1_mean, cv[k]?.top1_sd, cv[k]?.top5_mean, cv[k]?.macro_f1_mean]),
	];
	const three = [
		['Knowledge-based\n(sign rules, no training)', kb?.top1],
		['Random Forest\n(machine learning)', cv.rf?.top1_mean],
		['GRU\n(deep learning)', cv.gru?.top1_mean],
	];
	return `
	<h2>Sign recognition</h2>
	<p class="caption">Which of the app's 20 words is being signed? Tested on ${cv.videos} real videos (5-fold cross-validation).</p>
	${bars({ labels: three.map(([l]) => l), series: [{ name: 'Accuracy', color: C.a, values: three.map(([, v]) => v) }], height: 220 })}
	<div class="grid3">
		<div class="card"><h4>1 · Knowledge-based: ${pct(kb?.top1)}</h4><p>Compares the movement with each sign's written description. No training, so it is the baseline.</p></div>
		<div class="card"><h4>2 · Random Forest: ${pct(cv.rf?.top1_mean)}</h4><p>Learns from real signers, one frame at a time, and votes over the video.</p></div>
		<div class="card"><h4>3 · GRU: ${pct(cv.gru?.top1_mean)}</h4><p>Reads the whole movement in order, like a sentence. The best model; the app uses it.</p></div>
	</div>
	<details><summary>See all models tested</summary>
		${table(['Model', 'Top-1 (mean ± sd)', 'Top-5', 'Macro-F1'], all.map(([l, t1, sd, t5, f1]) => [l, sd == null ? pct(t1) : `${pct(t1)} ± ${pct(sd)}`, pct(t5), fix(f1, 2)]), { num: [1, 2, 3], best: 4 })}
		<p class="caption">In the app the GRU is averaged with the per-video XGBoost, which puts the right word in its top 5 guesses ${pct(cv.ensemble?.top5_mean)} of the time.</p>
		${fig(d, 'tabular_app20_cv.png', 'Accuracy per fold, and the confusion matrix of the best model over all folds.')}
	</details>`;
}

function letters(d) {
	const r = d.images;
	if (!r) return '<h2>Letter recognition</h2>';
	const rows = [
		['Pixels + logistic regression', r['1_pixels_logreg'], 'baseline'],
		['EfficientNet-B0, linear probe', r['2_effnet_linear_probe']],
		['EfficientNet-B0, fine-tuned', r['3_effnet_finetuned']],
		['+ pseudo-labels', r['4_effnet_semi_supervised']],
		['Fusion: CNN + hand landmarks', r['6_fusion']],
	];
	const best = rows.reduce((b, row, i) => (row[1].test.accuracy > rows[b][1].test.accuracy ? i : b), 0);
	const pl = r['4_pseudo_labels'];
	const pix = r['1_pixels_logreg'].test.accuracy;
	const cnn = r['4_effnet_semi_supervised'].test.accuracy;
	const fus = r['6_fusion'].test.accuracy;
	return `
	<h2>Letter recognition</h2>
	<p class="caption">Which letter (A–Z) does a hand photo show? Tested on 260 photos the models never saw.</p>
	${bars({ labels: ['Pixels\n(baseline)', 'EfficientNet CNN\n(deep learning)', 'Fusion\n(CNN + hand points)'], series: [{ name: 'Accuracy', color: C.a, values: [pix, cnn, fus] }], height: 220 })}
	<div class="grid3">
		<div class="card"><h4>1 · Pixels: ${pct(pix)}</h4><p>A simple model on the raw pixels. It cannot see shapes, so it is the baseline.</p></div>
		<div class="card"><h4>2 · EfficientNet CNN: ${pct(cnn)}</h4><p>A network pre-trained on 1.2 million photos, taught the 26 letters, plus ${pl.kept} extra photos it labelled itself (pseudo-labels).</p></div>
		<div class="card"><h4>3 · Fusion: ${pct(fus)}</h4><p>Averages the CNN with a model that reads the hand's 21 points. The best model.</p></div>
	</div>
	<details><summary>See all models tested</summary>
		${table(['Model', 'Accuracy', 'Macro-F1'], rows.map(([l, x, base]) => [`${l}${base ? ' <span class="tag base">baseline</span>' : ''}`, pct(x.test.accuracy), fix(x.test.macro_f1, 2)]), { num: [1, 2], best })}
		<div class="grid2">${fig(d, 'image_confusion_cnn.png', 'EfficientNet-B0 with pseudo-labels.')}${fig(d, 'image_confusion_fusion.png', 'Fusion.')}</div>
	</details>`;
}

function papers() {
	const groups = [...new Set(PAPERS.map(([g]) => g))];
	const card = ([, title, authors, venue, url, summary, took]) => `
		<details class="paper"><summary><span class="p-title">${esc(title)}</span><span class="p-meta">${esc(authors)} · ${esc(venue)}</span></summary>
			<p>${esc(summary)}</p>
			<p class="took"><b>What we took:</b> ${esc(took)}</p>
			<a class="p-link" href="${esc(url)}" target="_blank" rel="noopener">Read the paper ↗</a>
		</details>`;
	return `
	<h2>Papers</h2>
	<div class="card related"><h4>Related work</h4><p>${esc(RELATED_WORK.replace(/\s+/g, ' '))}</p></div>
	${groups.map((g) => `<h3>${esc(g)}</h3>${PAPERS.filter(([pg]) => pg === g).map(card).join('')}`).join('')}
	<p class="caption">Click a paper for its summary and link.</p>`;
}

function files() {
	const node = ([name, desc, kids], depth = 0) =>
		kids
			? `<li><details${depth === 0 && (name === 'src/' || name === 'ml/') ? ' open' : ''}><summary><span class="f-name dir">${esc(name)}</span><span class="f-desc">${esc(desc)}</span></summary><ul>${kids.map((k) => node(k, depth + 1)).join('')}</ul></details></li>`
			: `<li><span class="f-name${name.endsWith('/') ? ' dir' : ''}">${esc(name)}</span><span class="f-desc">${esc(desc)}</span></li>`;
	return `
	<h2>Files</h2>
	<p class="caption">The project folder. Click a folder to open it.</p>
	<ul class="tree">${TREE.map((n) => node(n)).join('')}</ul>`;
}

const TABS = [
	['problem', 'Problem', problem],
	['papers', 'Papers', papers],
	['data', 'Data', data],
	['preprocessing', 'Preprocessing', preprocessing],
	['signs', 'Sign recognition', signs],
	['letters', 'Letter recognition', letters],
	['files', 'Files', files],
];

// ── page ─────────────────────────────────────────────────────────────────────
function show(id, focus = false) {
	const known = TABS.some(([t]) => t === id) ? id : TABS[0][0];
	for (const b of tabsEl.querySelectorAll('[role="tab"]')) {
		const on = b.dataset.id === known;
		b.setAttribute('aria-selected', String(on));
		b.tabIndex = on ? 0 : -1;
		if (on && focus) b.focus();
	}
	for (const p of main.querySelectorAll('[role="tabpanel"]')) p.hidden = p.id !== `panel-${known}`;
	if (location.hash.slice(1) !== known) history.replaceState(null, '', `#${known}`);
}

async function start() {
	let d;
	try {
		d = await (await fetch(`${BASE}ml/dashboard.json`, { cache: 'no-cache' })).json();
	} catch {
		main.innerHTML = '<p class="pending">The results file (public/ml/dashboard.json) is missing. Run <code>python ml/scripts/export_for_app.py</code>.</p>';
		return;
	}
	tabsEl.innerHTML = TABS.map(([id, label]) => `<button type="button" role="tab" id="tab-${id}" data-id="${id}" aria-controls="panel-${id}">${esc(label)}</button>`).join('');
	main.innerHTML = TABS.map(([id, , render]) => `<section role="tabpanel" id="panel-${id}" aria-labelledby="tab-${id}" hidden>${render(d)}</section>`).join('');
	tabsEl.addEventListener('click', (e) => {
		const b = e.target.closest('[role="tab"]');
		if (b) show(b.dataset.id);
	});
	tabsEl.addEventListener('keydown', (e) => {
		if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
		const ids = TABS.map(([id]) => id);
		const i = ids.indexOf(location.hash.slice(1));
		show(ids[(i + (e.key === 'ArrowRight' ? 1 : ids.length - 1)) % ids.length], true);
	});
	window.addEventListener('hashchange', () => show(location.hash.slice(1)));
	show(location.hash.slice(1));
}

start();
