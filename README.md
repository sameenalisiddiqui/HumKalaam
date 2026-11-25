# Hum Kalaam: ASL Avatar

**CT-361 Artificial Intelligence & Expert Systems · Complex Computing Activity (CCA)**

https://hum-kalaam-green.vercel.app/

## Problem

Few hearing people know sign language and interpreters are scarce, so Deaf and hearing
people often cannot talk directly. Hum Kalaam is an assistive system for two-way
communication in American Sign Language (ASL), running in the browser on a phone or laptop:

- **Sign to text:** sign in front of the webcam; the app writes the word and reads it aloud.
- **Text or voice to sign:** type or speak English; a 3D avatar signs it.

**AI approach (hybrid):** machine-learning models trained on real signers read the camera,
and a knowledge base of 476 ASL sign descriptions drives the avatar.

| Input | AI step | Output |
| --- | --- | --- |
| Webcam video of a signer | MediaPipe landmarks → GRU + XGBoost ensemble | Word as text and speech |
| Photo of a hand (A–Z) | EfficientNet-B0 + hand-landmark model, fused | Letter |
| Typed or spoken English | Text rules + sign knowledge base | Avatar signing |

## Datasets

| Dataset | Source | Type | Used for |
| --- | --- | --- | --- |
| WLASL | Li et al., WACV 2020 (Hugging Face mirror of Kaggle `wlasl-processed`) | Video → landmark table: 1,892 videos, 207 signs, 69 signers, 123,449 clean frames × 176 features | Sign recognition |
| ASL Alphabet | Kaggle `grassknoted/asl-alphabet` | Images: 387 train, 113 validation, 100 unlabelled, 260 test, plus Kaggle's 26 test photos | Letter recognition |

## Preprocessing

- **Missing values:** short gaps in hand and body tracking are interpolated; long gaps stay empty.
- **Noise:** 3-frame smoothing; duplicate hand detections merged; hand left/right labels corrected from the body.
- **Normalisation:** positions in shoulder widths from the chest; left-handed signers mirrored.
- **Class imbalance:** 4–16 videos per sign, so every model uses balanced class weights and macro-F1.
- **Leak-free image split:** train, validation, unlabelled and test images come from separate parts of each recording.

## Models and results

**Signs** (the app's 20 words, 171 videos, 5-fold cross-validation):

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| Knowledge-based matching (baseline, no training) | 35.1% | 81.9% |
| Random Forest (baseline) | 88.9% | 95.3% |
| XGBoost, per frame | 79.5% | 96.5% |
| XGBoost, per video | 76.0% | 94.8% |
| **GRU (sequence)** | **91.8%** | 95.3% |
| Ensemble (GRU + XGBoost), used in the app | 90.6% | **97.1%** |

**Letters** (26 classes, 260 held-out images):

| Model | Accuracy |
| --- | --- |
| Pixels + logistic regression (baseline) | 8.5% |
| EfficientNet-B0, linear probe | 59.2% |
| EfficientNet-B0, fine-tuned | 58.5% |
| + pseudo-labels (semi-supervised) | 72.3% |
| **Fusion: CNN + hand landmarks** | **75.0%** |

All results, charts and confusion matrices are in the **results dashboard**
(`dashboard.html`, the button in the app's header).

## Run it

```bash
npm install
npm run dev        # the app, and /dashboard.html
npm test           # automated tests
npm run build      # static site in dist/
```

The machine-learning pipeline is in `ml/` (Python, runs on a CPU); see `ml/README.md`.
`ml/scripts/export_for_app.py` copies the trained models and the dashboard's data into
`public/ml/`, where the app loads them.

## How the app works

| Part | Files | What happens |
| --- | --- | --- |
| Camera tracking | `src/camera/camera-tracker.js` | MediaPipe hand and body landmarks, in the browser. Video never leaves the device. |
| Features | `src/ml/landmark-features.js` | The same cleaning as `ml/scripts/clean_landmarks.py`, ported to JavaScript and tested against Python on real WLASL frames. |
| Sign models | `src/ml/sign-model.js`, `public/ml/` | GRU and XGBoost (ONNX, onnxruntime-web), averaged. Shows the top guesses; asks when unsure. |
| Letters | `src/camera/recognizer.js` | Held fingerspelled letters are read by comparing the handshape with each letter's description. |
| Teach mode | `src/camera/teach.js` | Users can record their own examples of a word. |
| Text to sign | `src/text-to-sign/` | Words are signed from the knowledge base, other words fingerspelled, and *a, an, the, am, is, are, be, to* left out as in ASL. |
| Avatar | `src/avatar/stage.js` | The signing is retargeted onto the 3D avatar and played. |
| Voice | `src/voice/` | Browser speech recognition (Chrome, Edge, Safari) and read-aloud. |

## Ethics and limits

- **Privacy:** the models use only skeleton points, never faces or pixels, and run on the device.
- **Fairness:** balanced class weights, macro-F1, left-handed signing supported. Tracking may work less well in poor light.
- **Transparency:** every number comes from a script in `ml/scripts` with a fixed seed (361).
- **Accountability:** a communication aid, not a replacement for interpreters. The signs have not yet been reviewed by Deaf signers.
- **Limits:** 4–16 videos per sign; ASL signs in English word order, not ASL grammar or Pakistani Sign Language.

## Credits

| Part | Source | Licence |
| --- | --- | --- |
| Sign notation, arm IK, clip builder, retargeting | [three.ws](https://github.com/nirholas/three.ws), in `src/vendor/three-ws/` | Apache-2.0 |
| Hand and body tracking | [MediaPipe Tasks Vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) | Apache-2.0 |
| 3D rendering | [three.js](https://threejs.org) | MIT |
| WLASL | Li et al., 2020 | Computational Use of Data Agreement (academic, non-commercial) |
| ASL Alphabet | Kaggle, grassknoted | GPL-2 |
