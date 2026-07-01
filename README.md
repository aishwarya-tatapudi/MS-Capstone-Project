# Multilingual Prompt Injection Detection: A Fairness Audit in Hindi and Hinglish

**Author:** Aishwarya Anand Tatapudi
**Course:** M.S. Computer Science Capstone (Plan III)
**Affiliated with:** University of Colorado Denver
**Date:** May 2026
**Contact / full report:** aishanand1999@gmail.com · [LinkedIn](https://www.linkedin.com/in/aishwarya-anand-tatapudi-96b530267/)
**License:** CC BY-NC-ND 4.0 (see LICENSE)

This project does two things:

1. Re-implements a multilingual injection detector with a rule layer plus embedding model on a public Hindi/Hinglish dataset, and
2. Audits that detector for fairness: it measures if the classifier has a higher false positive rate for Hindi than for code-mixed Hinglish inputs.

The fairness audit is the contribution. The previous published work optimized for aggregate detection accuracy and reported a single number. It did not turn the lens back on its own classifier to ask whether the two user groups are treated equally. That is the question this project asks.

---

## The finding

The false-positive rate (FPR) on the same MiniLM+SVM model is 2.50% for Hindi, and 15.00% for Hinglish — a six-fold gap. The FPR gap is about 12 percentage points: a Hinglish user writing an innocuous prompt is wrongfully blocked far more often than a Hindi user, for no reason other than the script they wrote in.

Two additional results suggest this is a property of the task rather than a quirk of one model:

- All 22 Hinglish false positives were in `NoRule`, indicating the disparity comes from the differing embedding-space positions of code-mixed text, not the rule layer's capacity to learn rules.
- Adding the rule layer improves accuracy by 0.25% but also marginally widens the equity gap (12.00% → 12.50%). More machinery leads to marginally better accuracy, but at the expense of equity.

This is not a problem of accuracy — aggregate Hindi/Hinglish detection accuracy is 93.12%, and accuracy on the English (SPML) set is 98.92%. The point of this project is what that high aggregate number conceals.

An ablation study across three training data compositions confirms the gap is structural: increasing English training data from 0% to 65% widens the Hindi–Hinglish fairness gap from 11.50% to 14.50%, rather than closing it.

---

## Why a deliberately minimal model

This pipeline uses a frozen `paraphrase-multilingual-MiniLM-L12-v2` encoder (not fine-tuned on the dataset), a small rule layer, and a CPU-trained SVM. Compared to the fine-tuned transformer plus large rule dictionary in the previous work, this is a weaker system that gets several points lower headline accuracy.

That's the tradeoff of the approach: a well-fit model can hide which groups it's not serving well. In this bare-bones approach, per-language classifier decisions are visible, and the gap is there. Seeing the gap in a bare-bones model is a stronger indicator that the gap is real than seeing it survive in a well-sharpened one.

---

## Relationship to prior work

Srinivasan, J., Regi, S. A., Anbarasan, A. K., Suresh, A., Vetriselvi, T., & Venu, S. (2026). *Detection and analysis of prompt injection in Indian multilingual large language models.* Scientific Reports, 16, 16208. https://doi.org/10.1038/s41598-026-43883-0

The paper built a 4,000-prompt Hindi/Hinglish dataset and a hybrid rule-based + transformer (XLM-RoBERTa) classifier achieving ~99.7% aggregate accuracy. The detection architecture here is in the same family and should be read as a re-implementation. What is not in that paper — to the best of the author's reading of its results tables — is a per-language fairness breakdown of the detector's own false positives. This project fills that gap.

## Contribution vs. prior work

| Component | Origin | Notes |
|---|---|---|
| Hindi + Hinglish dataset | Prior work (Srinivasan et al. 2026, public on Kaggle) | Used as-is; not created here. |
| Rule layer + model hybrid | Prior work | Same architectural principle, but uses MiniLM + SVM rather than XLM-RoBERTa and a much smaller rule set. |
| English detection (Pipeline 2, SPML) | Standard baseline | MiniLM + SVM versus zero-shot NLI — not a new method. |
| Zero-shot NLI as a classifier | Existing technique | Used as a no-training comparison point. |
| Per-language fairness audit (Hindi vs. Hinglish FPR disparity) | This work | Original analysis: ~12-point FPR gap on the MiniLM+SVM pipeline. |
| Ablation study on training data composition | This work | Shows English augmentation widens the fairness gap. |

---

## Project structure

**Pipeline 1 — Hindi + Hinglish (fairness audit)**
- Loads the Hindi/Hinglish dataset
- Trains MiniLM + SVM, applies rule layer, outputs five-tier confidence score
- Computes per-language false-positive rates and reports the Hindi vs. Hinglish difference — the core analysis

**Pipeline 2 — English (baseline comparison)**
- Loads the SPML dataset
- Experiment A: MiniLM + SVM (trained) — 98.92% accuracy, 0.14% FPR
- Experiment B: zero-shot NLI (no training) — 56.84% accuracy, 86.31% FNR
- The 42-point accuracy gap shows reliable detection requires labeled training data

**Ablation Study — training data composition**
- Trains three model variants: Hindi+Hinglish only, +15% English, +65% English
- Measures how each composition affects the Hindi–Hinglish fairness gap
- Confirms indiscriminate English augmentation actively harms Hinglish users

---

## Files

| File | Description |
|---|---|
| `pipeline1.py` | Hindi + Hinglish detection and fairness audit |
| `pipeline2.py` | English detection (MiniLM+SVM vs. zero-shot NLI) |
| `ablation_study.py` | Three-model training composition comparison |
| `requirements.txt` | Dependencies |
| `LICENSE` | CC BY-NC-ND 4.0 |

The capstone report (`report.pdf`) is available upon request (see contact above) and is not included in this repository.

---

## Datasets required

1. Hindi + Hinglish dataset, Srinivasan et al. (2026) — also available at `sillaannregi/hindi-and-hinglish-prompts` (Kaggle)
2. SPML Chatbot Prompt Injection dataset — paired system and user prompts
3. `deepset/prompt-injections` — loaded automatically via Hugging Face (used as English training augmentation)

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Pipeline 1 (Hindi + Hinglish + fairness audit)
python pipeline1.py

# 3. Run Pipeline 2 (English baseline comparison)
python pipeline2.py

# 4. Run the ablation study (training data composition analysis)
python ablation_study.py
```

Each script prints step-by-step progress and saves its results as CSV files in the working directory (e.g. `pipeline1_hybrid_results.csv`, `pipeline2_summary.csv`). All experiments use a fixed random seed (42) and are fully reproducible.

---

## Scope and limitations

- The Hindi-vs-Hinglish gap was measured on the MiniLM + SVM pipeline; validating it on a stronger model (e.g. a fine-tuned multilingual transformer) is an obvious next step, left for future work. This could also be read as "this detector treats the two groups unequally" rather than "this disparity is intrinsic to the task."
- Training and evaluation are done on a single public dataset; no cross-dataset generalization is claimed.
- High aggregate accuracy on a clean, balanced dataset is expected and is not the point of this work — the fairness behavior is.

## Future directions

- Investigate whether per-language separability exists in the raw MiniLM embeddings before any classification decision is made — if so, the bias is inherited from the encoder itself by every downstream system built on it.
- Extend this framework to other Indian languages and their code-mixed varieties (Tamil, Telugu, Bengali, Malayalam).

## Reproducibility notes

- All experiments were run on CPU.
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (frozen, 384-dimensional)
- SVM: RBF kernel, C = 1.0, `class_weight=balanced`
- Train/test split: 80/20, random seed fixed at 42

---

Code and materials © 2026 Aishwarya Anand Tatapudi. CC BY-NC-ND 4.0: Attribution required for non-commercial use. No derivatives. Commercial use and adaptations require written permission.
