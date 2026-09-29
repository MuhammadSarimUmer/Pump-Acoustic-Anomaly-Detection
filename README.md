# Pump Acoustic Anomaly Detection

Comparing deep learning and classical ML approaches for detecting abnormal pump operation from audio, on a small (622-sample) industrial acoustic dataset.

## Problem

Given a `.wav` recording of a pump in operation, classify it as **normal** or **abnormal**. This is a binary classification task relevant to predictive maintenance — catching mechanical faults from sound before they cause downtime.

**Dataset:** Kaggle "AI Paradox" competition — 622 labeled training clips (normal/abnormal folders), 156 unlabeled test clips, evaluated on F1 score for the abnormal class. Competition deadline (Oct 16, 2025) had already passed by the time of this project; submissions were scored as late/practice entries, not ranked on a live leaderboard.

**Class balance:** ~427 normal / ~135 abnormal in training data (roughly 3:1 imbalance).

## Approach 1: Baseline CNN (from scratch)

Audio converted to 128×128 mel spectrograms (Librosa, 128 mel bands, 1024 FFT window, 512 hop length), fed into a small 3-block CNN (16→32→64 filters, GlobalAveragePooling, Dropout) trained from scratch — 27,521 parameters total.

Minority class (abnormal) was augmented 3x (time shift, pitch shift, light noise) to roughly balance the training set before this split, giving 761 training / 125 validation spectrograms.

An early version of this model used BatchNormalization and collapsed to predicting a single class on validation (val_recall frozen at 1.0, val_precision frozen at the class prior) — traced to BatchNorm's running statistics being unstable on a dataset this small. Removing BatchNorm and lowering the learning rate to 0.0005 fixed this.

**Result (validation):**
| Metric | Normal | Abnormal |
|---|---|---|
| Precision | 0.89 | 0.74 |
| Recall | 0.91 | 0.70 |
| F1 | 0.90 | 0.72 |

Macro F1: 0.81

## Approach 2: Transfer Learning (MobileNetV2)

Same spectrograms, duplicated to 3 channels to match MobileNetV2's expected RGB input. ImageNet-pretrained MobileNetV2 used as a frozen feature extractor (2.34M total params, only 82K trainable in the new classification head) for initial training, then fine-tuned by unfreezing the last 30 layers of the base model (1.6M trainable params) at a much lower learning rate (1e-5, vs 5e-4 for the frozen phase) to avoid catastrophic forgetting of pretrained features.

Decision threshold tuned on validation via F1 sweep (best: 0.34, vs default 0.5).

**Result (validation, frozen-feature model — fine-tuning matched but did not improve this):**
| Metric | Normal | Abnormal |
|---|---|---|
| Precision | 0.96 | 0.94 |
| Recall | 0.98 | 0.88 |
| F1 | 0.97 | 0.91 |

Macro F1: 0.94

**Kaggle test score:** 0.72500 (frozen-feature model, threshold 0.5) → 0.74698 (fine-tuned model, tuned threshold 0.34)

## Approach 3: Classical ML (hand-crafted features + LightGBM)

Instead of spectrogram images, each clip was reduced to 34 hand-crafted numeric features via Librosa: 13 MFCCs (mean + std), spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, RMS energy, and chroma (mean + std each).

LightGBM classifier (300 estimators, learning_rate 0.05, max_depth 5, num_leaves 15) trained on these tabular features, with `scale_pos_weight` set from the train class ratio to handle imbalance directly rather than via augmentation. Decision threshold tuned on validation (best: 0.62).

**Result (validation):**
| Metric | Normal | Abnormal |
|---|---|---|
| Precision | 0.99 | 0.94 |
| Recall | 0.98 | 0.97 |
| F1 | 0.98 | 0.96 |

Macro F1: 0.97

**Kaggle test score: 0.88311** — best of the three approaches.

## Results Comparison

| Approach | Val Abnormal F1 | Val Macro F1 | Kaggle Score |
|---|---|---|---|
| CNN (scratch) | 0.72 | 0.81 | — |
| CNN + Transfer Learning (frozen) | 0.91 | 0.94 | 0.72500 |
| CNN + Transfer Learning (fine-tuned) | 0.91 | — | 0.74698 |
| **LightGBM (hand-crafted features)** | **0.96** | **0.97** | **0.88311** |

Note the gap between validation and Kaggle scores for the CNN approaches (0.91 val F1 → 0.725-0.747 test) versus LightGBM's smaller gap (0.96 val F1 → 0.883 test). This is discussed below.

## Analysis: why LightGBM outperformed the CNN approaches here

This is reasoned analysis based on the results observed above, not a proven root cause — a proper ablation (e.g. testing LightGBM on spectrogram-derived features, or a CNN on hand-crafted features) would be needed to isolate whether this is a representation effect, a model-family effect, or both.

1. **Dataset size.** 622 training samples is small for image-shaped input. CNNs, even with transfer learning, adapt many parameters and generally need more data to generalize reliably on raw pixel-like input than tabular models need on compact, pre-summarized features.
2. **Spectrograms-as-images carry more irrelevant surface area.** A 128×128 grid gives a CNN far more opportunity to latch onto spurious spatial patterns (exact timing, background noise position) than a 34-number feature vector allows.
3. **Hand-crafted features are domain-informed compressions.** MFCCs, spectral centroid, RMS, etc. discard exactly the kind of irrelevant variation a CNN has to learn to ignore the hard way, from data alone.
4. **Distribution shift between train and test.** The validation split came from the same recordings as training data; the real Kaggle test set may differ in recording conditions. Robust summary statistics (used by LightGBM) tend to be less sensitive to this than pixel-level CNN features.
5. This matches a known pattern in small-data audio ML: classical feature engineering + gradient boosting often outperforms deep learning until dataset size reaches into the thousands of samples per class.

## Limitations

- Validation set is small (125 samples, only 33 abnormal) — single-sample changes swing reported F1 by several points. Results should be read as directional, not precise.
- No k-fold cross-validation was used; a single 80/20 split was used throughout, which adds variance to the validation estimates above.
- The competition deadline had passed, so no live leaderboard ranking was obtained — scores are self-assessed against other public submissions on the same dataset.
- The CNN vs. LightGBM comparison conflates two variables at once (model family AND feature representation) — not a controlled experiment.

