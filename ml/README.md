# Hum Kalaam ML: ASL Avatar (CT-361 CCA)

The Python pipeline behind Hum Kalaam's sign recognition: raw public data → preprocessing →
baseline → improved models → evaluation → models exported for the app. Every model is trained
on real, public data, and everything runs on a CPU. All random seeds are 361.

## Datasets

| Type | Source | What we use | Problems handled |
| --- | --- | --- | --- |
| Video → tabular | WLASL (Li et al., WACV 2020), Hugging Face mirror of Kaggle `wlasl-processed` (`Voxel51/WLASL`) | 1,892 videos of 207 signs by 69 signers. MediaPipe landmarks, one row per frame: 388 raw columns, cleaned to 123,449 rows × 176 features | Body not found, hands lost for a few frames, left/right hand labels flipping, a hand detected twice, jitter, left-handed signers, different body sizes and camera distances, class imbalance (4–16 videos per sign) |
| Images | Kaggle `grassknoted/asl-alphabet` | 500 labelled letter images (A–Z) and 100 unlabelled; 260 held-out test images plus Kaggle's 26 test photos | Neighbouring frames of one recording are near-duplicates, so each part comes from a separate stretch of each recording |

## Pipeline

```
python scripts/download_data.py all          # WLASL videos + ASL Alphabet images
python scripts/extract_landmarks.py          # MediaPipe hand + body landmarks per frame
python scripts/clean_landmarks.py            # cleaning + body-relative features -> wlasl_features.csv.gz
python scripts/tabular_models.py             # signs: Random Forest -> XGBoost -> GRU -> ensemble
python scripts/tabular_cv.py                 # signs: 5-fold cross-validation on the app's 20 words
python scripts/image_models.py               # letters: pixels + LR -> EfficientNet-B0 -> pseudo-labels -> fusion
python scripts/export_test_raw.py            # test fixtures for the JavaScript feature port
node   scripts/eval_template_recognizer.mjs  # knowledge-based baseline on the same real videos
python scripts/export_for_app.py             # models + dashboard data -> ../public/ml/
```

Results are written to `results/*.json` and figures to `figures/*.png`. Landmark extraction is
the slow step: about 5 frames per second on a 4-core laptop.

## Results

| Signs: 20 words, 5-fold CV | Top-1 | | Letters: 260 test images | Accuracy |
| --- | --- | --- | --- | --- |
| Knowledge-based matching (baseline) | 35.1% | | Pixels + logistic regression (baseline) | 8.5% |
| Random Forest (baseline) | 88.9% | | EfficientNet-B0, linear probe | 59.2% |
| XGBoost, per frame | 79.5% | | EfficientNet-B0, fine-tuned | 58.5% |
| XGBoost, per video | 76.0% | | + pseudo-labels | 72.3% |
| **GRU** | **91.8%** | | **Fusion: CNN + landmarks** | **75.0%** |
| Ensemble (GRU + XGBoost) | 90.6% | | | |

## What runs in the app

| File in `public/ml/` | Made by | Used by |
| --- | --- | --- |
| `signs_app20_gru.onnx`, `signs_app20_xgb.onnx`, `signs_app20_meta.json` | `tabular_models.py` | the camera's sign recognition (`src/ml/sign-model.js`) |
| `dashboard.json`, `figures/` | `export_for_app.py` | the results dashboard (`dashboard.html`) |

`src/ml/landmark-features.js` is a line-by-line JavaScript port of `clean_landmarks.py`.
`tests/ml-features.test.mjs` checks that it gives the same numbers as Python on real WLASL
frames, and that the ONNX models give the same probabilities as the Python models.
