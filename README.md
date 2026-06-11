# HIPPO-2 Digital Twin — Neural-Network Virtual Sensors

Coursework for **Special Ship Automation Systems 2026** (System Modeling Using Experimental
Measurement Data), Laboratory of Marine Engineering, NTUA. Instructor: Assoc. Prof. G. Papalambrou.

Data-driven **virtual sensors / digital twin** for the HIPPO-2 hybrid diesel–electric engine:
neural networks that predict engine parameters from other measured signals, so expensive or
intrusive sensors can be replaced by software.

- **Author:** Dagklis Spyros (nm20275)
- **Experiment tracking (WandB):** https://wandb.ai/nm20275ntua/Advanced_Control_Systems
- **Single script:** `nm20275.py`

> The course dataset `Data2026.csv` is **not** redistributed in this repository.

## Pipeline
Clean (physical bounds + 3σ outlier removal) → feature engineering (`none` / `poly` /
`gplearn` symbolic regression) → deep MLP with **Optuna** (Bayesian TPE) hyperparameter search
→ **WandB** tracking → evaluation on a held-out **1400 RPM** band (Leave-One-RPM-Out, to test
generalization) → **SHAP** interpretability + architecture **ablation** table.

## Results (test set, physical units)
Two target parameters per the assignment: **NOx** and **Fuel Consumption** (Intake Pressure and
λ modeled additionally).

| Target | Best mode | Test MAE | R² | Shallow baseline (Part 1) |
|---|---|---|---|---|
| **NOx** | none (deep, raw) | **13.56 ppm** | 0.968 | 84.49 ppm (R² 0.59) |
| **Fuel Consumption** | poly | **0.285 kg/h** | 0.9995 | ≈0.40 kg/h |
| Intake Pressure | none | 1.21 mbar | 0.9991 | — |
| λ | none | 0.039 | 0.9978 | — |

**Key finding:** for NOx, the improvement over the shallow baseline (84.49 → 13.56 ppm) comes
from **network capacity/depth**, not from engineered features — the ablation and SHAP show the
raw physical drivers (rotational speed, torque) dominate, while engineered `gplearn` features
rank lower. For Fuel, engineered (polynomial) features **do** help (SHAP confirms they rank top).
Full per-mode numbers in `final_results.csv`; per-change sweep in `ablation_results.csv`.

## Repository structure
```
nm20275.py              # full pipeline (data → models → SHAP → ablation → plots)
requirements.txt        # pinned dependencies (Python 3.12)
final_results.csv       # all targets × 3 feature modes: MAE / R² / RMSE
ablation_results.csv    # width/depth sweep → test loss
plots_part_b/           # loss curves, scatter, SHAP (bar+beeswarm), ablation, heatmaps
PRESENTATION_PLOTS.pdf  # key figures bundled
nm20275.pptx            # presentation
```

## How to run
```bash
pip install -r requirements.txt
# obtain Data2026.csv (course data) and place it next to nm20275.py
export WANDB_MODE=offline          # or: wandb login
python nm20275.py                  # set FEATURE_ENG_MODE=none|poly|gplearn (env var) per run
```
