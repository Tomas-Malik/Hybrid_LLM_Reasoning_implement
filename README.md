# Learned Routing for Cost-Efficient LLM Reasoning

Learned query routing for cost-efficient LLM inference using a DeBERTa-v3-large classifier across eight reasoning benchmarks.

## Overview

This project implements and extends [Hybrid LLM](https://arxiv.org/abs/2404.14618) (Ding et al., 2024), a learned routing framework that dynamically decides whether an incoming query should be answered by a small, cheap model or escalated to a large, expensive fallback. Rather than always paying the cost of a frontier model, the router identifies queries the small model can handle correctly and reserves the large model for cases where it is strictly necessary.

**Small model:** Llama 3.1 8B Instruct  
**Large model:** Llama 3.3 70B Instruct  
**Router:** DeBERTa-v3-large fine-tuned as a binary classifier

## Key Contributions

- **Correctness-based routing label:** A query is labeled *route to small* if the small model answers correctly, or if both models fail (escalation saves nothing). It is labeled *escalate* only when the large model alone succeeds. This directly optimises the cost-saving objective.
- **DeBERTa-v3-large router:** Replaces the original DistilBERT backbone with a significantly stronger encoder, improving routing quality on harder benchmarks.
- **Broad evaluation:** Eight reasoning benchmarks spanning mathematical problem-solving and natural-language inference.

## Results

| Model / Specification | Avg. Accuracy | Avg. Cost |
|---|---|---|
| Llama 3.1 8B (L8) | 67.2% | 42 |
| Llama 3.3 70B (L70) | 82.4% | 384 |
| Random: L8 + L70 | 75.3% | 213 |
| **Hybrid LLM, t=0.50** | 68.2% | 52 |
| **Hybrid LLM, t=0.75** | 71.8% | 117 |
| **Hybrid LLM, t=0.95** | **80.2%** | **343** |

Cost is average token-based monetary cost per query scaled by 10⁶ (USD). At `t=0.95` the router recovers 86% of the accuracy gap between L8 and L70 while reducing inference cost by 11%.

## Benchmarks

| Dataset | Type | Test Size |
|---|---|---|
| GSM8K | Math (free-response) | 1,319 |
| GSM Symbolic P2 | Math (free-response) | 2,500 |
| MATH500 | Math (free-response) | 500 |
| AQuA | Math (multiple-choice) | 254 |
| CommonsenseQA | NLI (multiple-choice) | 1,221 |
| StrategyQA (SQA⁻ / SQA⁺) | NLI (boolean) | 687 |
| SciNLI500 | NLI (multiple-choice) | 500 |

## Repository Structure

```
.
├── hllm/
│   ├── Code/
│   │   ├── aqua/               # AQuA inference & evaluation
│   │   ├── CommonSenseQA/      # CSQA inference & evaluation
│   │   ├── GSM8K/              # GSM8K inference & evaluation
│   │   ├── GSM_Symbolic/       # GSM Symbolic P1/P2 inference & evaluation
│   │   ├── MATH/               # MATH/MATH500 inference & evaluation
│   │   ├── confidence/         # Router confidence scoring
│   │   ├── grading/            # Answer extraction & grading utilities
│   │   └── training_data/      # Labeled training CSVs + label construction scripts
│   └── Outputs/                # Evaluation results
```

> **Note:** Trained router checkpoints (`router_deberta_v3_large_*/`) are not included in this repository due to file size. To reproduce them, follow the Training section below — the labeled training data and all training scripts are provided.

## .gitignore

Add the following to your `.gitignore` to keep the router checkpoints out of the repository:

```
router_deberta_v3_large_*/
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Requirements:** `torch`, `transformers`, `pandas`, `scikit-learn`, `tqdm`, `datasets`

## Training a Router

```bash
python hllm/Code/train_router.py \
    --dataset gsm8k \
    --model microsoft/deberta-v3-large \
    --output_dir router_deberta_v3_large_gsm8k
```

Training uses an 80/20 stratified split, AdamW with lr=2e-5, batch size 8 (4 + 2-step gradient accumulation), and early stopping on validation F1.

## Running Evaluation

```bash
python hllm/Code/aqua/aqua_eval.py   # example: AQuA benchmark
```

Set `threshold` in the config section of each eval script to control the routing decision boundary.

## Routing Label Construction

For each training query `x` with ground-truth answer `y*`:

```
ℓ(x) = 1   if small model is correct
ℓ(x) = 1   if both models are wrong  (escalation saves nothing)
ℓ(x) = 0   if only the large model is correct  (must escalate)
```

## Citation

If you use this code, please cite the original Hybrid LLM paper:

```bibtex
@inproceedings{ding2024hybrid,
  title     = {Hybrid {LLM}: Cost-efficient and quality-aware query routing},
  author    = {Ding, Dujian and Mallick, Ankur and Wang, Chi and Sim, Robert and
               Mukherjee, Subhabrata and Ruhle, Victor and Lakshmanan, Laks V.S.
               and Awadallah, Ahmed Hassan},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024}
}
```

## Author

Tomas Malik — University of Illinois Chicago  
CS 533: Deep Learning for NLP, Spring 2026
