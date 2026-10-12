# AIoT-2025--TP2 — Optimization and deployment readiness

Simple, documented solution to [the course TP2](https://github.com/ahmed3991/AIoT-2025/tree/main/TP2).

[Open notebook in Colab](https://colab.research.google.com/github/ggdh17934/AIoT-2025--TP2/blob/main/TP2-Ghanabzia.ipynb)

This repository is **private**. Sign in with your GitHub account when Colab asks
for access. You can also download the notebook and use Colab's **Upload** option.

## Run locally

Use Python 3.11 or newer (TensorFlow builds depend on the supported Python version).
From this project folder:

```sh
python -m pip install -r requirements.txt
python tp2.py
```

## Run the notebook

Run all. The public smoke dataset is downloaded automatically if it is missing; no other project files are needed.

The first notebook cell installs missing dependencies. TP2/TP4 also ensure the
versions used for their pickle models. All notebooks contain the actual solution
code; they do not require downloading the Python script just to execute cells.
Saved outputs come from the local run; rerunning generates your own measurements.

## Files

- `TP2-Ghanabzia.ipynb`: notebook with code, outputs, and report display.
- `tp2.py`: Python solution.
- `EXPLICATION_AR.md`: Arabic explanation of the code, concepts and limitations.
- `results/REPORT.md`: measured results and conclusions.
- `results/VALIDATION.md`: what was actually checked.
- `requirements.txt`: runtime dependencies.


## Sources and attribution

Assignment and original wiring: Ahmed Ghenabzia,
[ahmed3991/AIoT-2025](https://github.com/ahmed3991/AIoT-2025).
Solution Python and firmware are implemented for this project; diagrams are
adapted from the course starter files. Smoke data source:
[Kaggle Smoke Detection](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset).
Fashion-MNIST is loaded through the official Keras dataset loader; source:
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

On the original local computer, XGBoost is also available in `.packages`.
PowerShell: `$env:PYTHONPATH = "$PWD\.packages"` then `python tp2.py`.
