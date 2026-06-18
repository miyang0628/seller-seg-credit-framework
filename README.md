# Explainable Seller Segmentation Framework for Alternative Credit Scoring

> **Paper**: An Explainable Seller Segment Auto-Assignment Framework: Combining Product Name Classification and LLM-Based Notice Generation for Alternative Credit Scoring
>
> **Author**: [Anonymous for blind review]
>
> **Journal**: Under Review
>
> **Status**: Revision submitted
>
> **Citation**: *(to be added upon publication)*

---

## Overview

This repository contains the experimental code accompanying the above paper.

We propose **ESSF (Explainable Seller Segmentation Framework)**, a four-stage end-to-end pipeline that automatically assigns e-commerce sellers to one of 11 industry segments using **only product name text** as input, and generates LLM-based explanatory notices for credit evaluation purposes.

The framework is designed to address three practical challenges in alternative credit scoring for platform sellers:
- **Thin filer problem**: New sellers lack financial history, making traditional credit models inapplicable.
- **Manual segment assignment**: Existing systems rely on manual or rule-based segmentation, which does not scale.
- **Explainability gap**: Regulatory requirements (e.g., Financial Consumer Protection Act) mandate that credit evaluation rationale be communicated to borrowers.

---

## Framework

```
Input: Product name set of seller s → {x_1, x_2, ..., x_N}
              │
              ▼
┌─────────────────────────────────────────────┐
│  Stage 1. Product-Level Classification      │
│  Classifier f_θ: x_i → (predicted label,   │
│  confidence score)                          │
│  Models: TF-IDF + LR / XGBoost / RF /      │
│          klue/roberta-base (M4)             │
│          klue/bert-base (M5)                │
│          KoELECTRA-base-v3 (M6)            │
│          klue/roberta-large (M7)            │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 2. Store-Level Aggregation & SCS     │
│  Confidence-Weighted Voting → Assigned Seg  │
│  SCS (Seg Confidence Score) → AUTO /        │
│                                MANUAL_REVIEW│
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 3. Credit Model Assignment           │
│  Assigned Seg → Segment-specific credit     │
│  scoring model [Assumption A1]              │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 4. LLM-Based Notice Generation       │
│  (Seg, SCS, Flag, distribution) → GPT →    │
│  Explainable evaluation notice for borrower │
└─────────────────────────────────────────────┘
```

---

## Key Contributions

1. **End-to-end Seg assignment framework** using non-financial text (product names) as the sole input signal — the first framework to explicitly address automated segment assignment as a prerequisite step for alternative credit scoring, grounded in signalling theory.

2. **SCS (Seg Confidence Score)** — a novel metric that quantifies store-level segment assignment reliability by combining dominant category ratio and Shannon Entropy:

$$SCS = \frac{n_{dominant}}{N} \times \frac{1}{1 + H}$$

A threshold-based (τ) AUTO / MANUAL\_REVIEW branching mechanism enables flexible operational risk management.

3. **Systematic PLM comparison** — seven Stage 1 classifiers evaluated across TF-IDF-based models and four Korean PLMs (BERT, RoBERTa-base, RoBERTa-large, ELECTRA) on 110,000 Korean product names, establishing klue/roberta-large as the performance ceiling and klue/roberta-base as the practical deployment choice.

4. **LLM-based explainable notice generation** — Stage 4 automatically produces borrower-facing evaluation rationale notices satisfying six explainability criteria, supporting financial consumer protection compliance.

---

## Experimental Results (Summary)

### Stage 1 — Product-Level Classification (Test set: 22,011)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LR (Baseline) | 0.8564 | 0.8561 |
| TF-IDF + XGBoost | 0.8070 | 0.8080 |
| TF-IDF + Random Forest | 0.7882 | 0.7873 |
| klue/roberta-base (M4) | 0.8732 | 0.8731 |
| klue/bert-base (M5) | 0.8640 | 0.8639 |
| KoELECTRA-base-v3 (M6) | 0.8654 | 0.8649 |
| **klue/roberta-large (M7)** | **0.8797** | **0.8795** |

### Stage 2 — Store-Level Aggregation (500 stores per condition)

| Backend | Store Size | MV Acc | CWV Acc | Avg SCS |
|---|---|---|---|---|
| TF-IDF + LR | 5 | 0.956 | 0.972 | 0.414 |
| KoBERT | 5 | 0.964 | 0.972 | 0.405 |
| TF-IDF + LR | 20 | 1.000 | 1.000 | 0.296 |
| KoBERT | 20 | 1.000 | 1.000 | 0.322 |

### SCS Threshold (τ) Optimization

| Backend | Optimal τ | Auto-Rate | CWV Accuracy |
|---|---|---|---|
| TF-IDF + LR | 0.15 | 92.75% | 0.9973 |
| KoBERT | 0.20 | 80.60% | 0.9981 |

### Stage 4 — LLM Notice Quality

All 3 representative cases (SCS: 0.821 / 0.412 / 0.118) passed all 6 explainability checklist items: evaluation date, product count, category distribution, assigned segment, applied model, and objection procedure.

---

## Repository Structure

```
seller-seg-credit-framework/
│
├── README.md
├── data/
│   └── sample_data.csv                    # Small public sample (full data not shared)
│
├── 01_preprocessing.ipynb                 # Data cleaning & train/val/test split
├── 02_stage1_model_comparison.ipynb       # TF-IDF models + PLM fine-tuning (M4–M7)
├── 03_stage2_aggregation_scs.ipynb        # MV vs CWV, SCS computation, τ search
├── 04_stage4_llm_notice.ipynb             # LLM-based notice generation (GPT-4o-mini)
└── 05_full_pipeline_demo.ipynb            # End-to-end pipeline demo (STORE_EXP_001)
```

---

## Data

The full dataset (110,000 product names from Naver Smart Store, November 2023) is **not publicly released** due to the platform's terms of service. A small anonymized sample (`data/sample_data.csv`) is provided for code verification purposes.

| Split | Records | Ratio |
|---|---|---|
| Train | 76,996 | 70% |
| Validation | 10,988 | 10% |
| Test | 22,011 | 20% |
| **Total** | **109,995** | **100%** |

**11 segment classes**: 가구/인테리어 · 도서 · 디지털/가전 · 생활/건강 · 스포츠/레저 · 식품 · 여가/생활편의 · 출산/육아 · 패션의류 · 패션잡화 · 화장품/미용

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
pip install openai  # for Stage 4 LLM notice generation
```

Tested on:
- Python 3.12 (Miniconda)
- PyTorch 2.10.0 (CUDA 12.6)
- NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)
- Scikit-learn 1.9, Transformers (Hugging Face)

---

## Usage

Run notebooks in order:

```bash
# 1. Preprocessing
jupyter notebook 01_preprocessing.ipynb

# 2. Stage 1: Model comparison (M1–M7)
jupyter notebook 02_stage1_model_comparison.ipynb

# 3. Stage 2: Aggregation & SCS
jupyter notebook 03_stage2_aggregation_scs.ipynb

# 4. Stage 4: LLM notice generation
#    Set your OpenAI API key before running
jupyter notebook 04_stage4_llm_notice.ipynb

# 5. Full pipeline demo
jupyter notebook 05_full_pipeline_demo.ipynb
```

For Stage 4, set your API key as an environment variable:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Linux / macOS
export OPENAI_API_KEY="your-api-key-here"
```

---

## License

This repository is released for academic reproducibility purposes.
Commercial use is not permitted without the author's consent.

---

## Contact

[Anonymous for blind review]
*(Contact information available upon publication)*
