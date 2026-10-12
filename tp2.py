"""TP2: compare two simple pipelines on the smoke-detection dataset."""

# %% 1. Imports and settings
import json
import pickle
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = PROJECT_DIR / "data" / "smoke_detection_iot.csv"
OUTPUT_DIR = PROJECT_DIR / "results"
RANDOM_STATE = 42
REPEATS = 5

# %% 2. Load and prepare the data
# On Kaggle, attach deepcontractor/smoke-detection-dataset using Add Input.
if not DATA_PATH.exists():
    uploaded = PROJECT_DIR / "smoke_detection_iot.csv"
    kaggle_files = list(Path("/kaggle/input").rglob("smoke_detection_iot.csv"))
    if uploaded.exists():
        DATA_PATH = uploaded
    elif kaggle_files:
        DATA_PATH = kaggle_files[0]
    else:
        # Allows a standalone Colab notebook to download the public dataset.
        from io import BytesIO
        from urllib.request import urlopen
        from zipfile import ZipFile
        url = "https://www.kaggle.com/api/v1/datasets/download/deepcontractor/smoke-detection-dataset"
        with urlopen(url, timeout=60) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                csv_name = next(name for name in archive.namelist() if name.endswith("smoke_detection_iot.csv"))
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                DATA_PATH.write_bytes(archive.read(csv_name))

data = pd.read_csv(DATA_PATH)
print("Dataset:", DATA_PATH.name)
print("Original shape:", data.shape)
print("Missing values:", int(data.isna().sum().sum()))

# Remove identifiers and time/counter columns; keep actual sensor readings.
data = data.drop(columns=["Unnamed: 0", "UTC", "CNT"], errors="ignore")
rows_before = len(data)
data = data.dropna()
print("Rows removed because of missing values:", rows_before - len(data))
if set(data["Fire Alarm"].unique()) != {0, 1}:
    raise ValueError("Fire Alarm must contain both binary labels 0 and 1.")

X = data.drop(columns="Fire Alarm")
y = data["Fire Alarm"].astype(int)
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
    raise ValueError("All sensor features must be numeric.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Sensors: {X.shape[1]}")

# %% 3. Create both required pipelines
# Scaling is fitted ONLY on the training data inside each pipeline.
pipelines = {
    "LR-Pipeline": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "XGB-Pipeline": Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )),
    ]),
}

# %% 4. Train, evaluate, serialize, and measure resources
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
measurements = []

for name, pipeline in pipelines.items():
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    training_s = perf_counter() - start

    # Warm up predict(), then use the median of five full-test predictions.
    pipeline.predict(X_test)
    timings = []
    for _ in range(REPEATS):
        start = perf_counter()
        predictions = pipeline.predict(X_test)
        timings.append(perf_counter() - start)

    inference_s = median(timings)
    single_ms = inference_s / len(X_test) * 1000

    model_path = OUTPUT_DIR / f"{name}.pkl"
    with model_path.open("wb") as file:
        pickle.dump(pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)
    size_bytes = model_path.stat().st_size

    measurements.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "Size_bytes": size_bytes,
        "Size_KB": size_bytes / 1024,
        "Training_s": training_s,
        "Test_inference_s": inference_s,
        "Single_inference_ms": single_ms,
    })

results = pd.DataFrame(measurements)
results.to_csv(OUTPUT_DIR / "results.csv", index=False)
print("\n", results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

metadata = {
    "dataset": DATA_PATH.name,
    "rows": len(data),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "features": list(X.columns),
    "random_state": RANDOM_STATE,
    "inference_repeats": REPEATS,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
}
(OUTPUT_DIR / "environment.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

# %% 5. Write the measured results and answer the ESP32 question
table = (
    "| Model | Accuracy | F1 | Precision | Recall | Size (KB) | Training (s) | Test inference (s) | Average per sample (ms) |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)
for row in measurements:
    table += (
        f"| {row['Model']} | {row['Accuracy']:.6f} | {row['F1']:.6f} | "
        f"{row['Precision']:.6f} | {row['Recall']:.6f} | {row['Size_KB']:.3f} | "
        f"{row['Training_s']:.6f} | {row['Test_inference_s']:.6f} | "
        f"{row['Single_inference_ms']:.6f} |\n"
    )
lr, xgb = measurements
size_ratio = xgb["Size_bytes"] / lr["Size_bytes"]
smaller = min(measurements, key=lambda row: row["Size_bytes"])["Model"]
faster = min(measurements, key=lambda row: row["Test_inference_s"])["Model"]
under_deadline = all(row["Single_inference_ms"] < 1000 for row in measurements)
deadline_text = "Both" if under_deadline else "Not both"

report = f"""# TP2 — Optimization and Deployment Readiness

## Setup

Data: [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset).
Target: `Fire Alarm` (0 = normal, 1 = alarm).
{len(data):,} complete rows, {X.shape[1]} sensor features; an 80/20 stratified split
with seed {RANDOM_STATE}: {len(X_train):,} training rows and {len(X_test):,} test rows.
`Unnamed: 0`, `UTC`, and `CNT` are excluded because they describe row identity,
time, or collection order rather than sensor measurements.
Both pipelines contain `StandardScaler`, fitted only on the training partition.
LR uses `max_iter=1000`; XGBoost uses 100 trees, depth 3, learning rate 0.1, and one worker.

## Measurements

{table}
Here KB means bytes / 1024 (technically KiB). Size is the actual serialized
**whole pipeline**, including the scaler. Training time includes scaling.
Inference time includes scaling and prediction but excludes metrics and disk I/O.
It is the median of {REPEATS} full-test calls after one warm-up call.
Average per sample = total test inference seconds / {len(X_test):,} × 1000.
This batch average is not the latency of an isolated one-row call.
F1, precision, and recall use the positive class `Fire Alarm = 1`.
These are real measurements on the computer recorded in `environment.json`;
timings can vary between runs.

## ESP32 deployment analysis (Task 2.3)

The LR pipeline occupies **{lr['Size_KB']:.3f} KB**, compared with
**{xgb['Size_KB']:.3f} KB** for XGBoost, a size ratio of **{size_ratio:.1f}×**.
The smaller serialized model is **{smaller}**. Both files fit within a typical
multi-megabyte Flash capacity, but pickle size alone does not establish runtime
RAM feasibility: the classic ESP32's approximately 520 KB SRAM is shared with
code, stacks, communication, and buffers, and not all of it is application heap.
Python pickle pipelines also require desktop Python libraries and cannot be
used directly as ESP32 firmware. A native implementation must be measured for
actual RAM usage; constant weights or tree nodes can be stored in Flash.
XGBoost is more constrained by memory because its many trees require more
parameters and node traversal than LR's small coefficient vector.
Measured average inference costs are **{lr['Single_inference_ms']:.6f} ms** for LR
and **{xgb['Single_inference_ms']:.6f} ms** for XGBoost; **{faster}** has the lower
full-test inference time in this run. {deadline_text} measured averages are below
the **1000 ms** budget for one prediction per second on this computer.
These batch desktop measurements do not guarantee the same deadline on an
ESP32: a native one-sample implementation must be timed on the board.
For an efficiency-focused ESP32 design, **Logistic Regression is the preferred
starting point**: it has a compact coefficient vector and requires only scaling,
a weighted sum, and a threshold, making a small C/C++ implementation straightforward.
Accuracy and recall must still be acceptable for the application.
To make XGBoost more practical, reduce tree count and depth, prune trees,
quantize thresholds and leaf values where accuracy allows, select fewer sensors,
and export to native C/C++ with constant data in Flash; then recheck F1/recall,
RAM, Flash, and actual single-sample latency.

## Limits

The random split follows the exercise. Sensor readings collected close in time
can be correlated, so it does not prove generalization to new collection sessions.
A future deployment should also evaluate a split by time or recording session.
This TP analyzes readiness; no ESP32 hardware deployment was performed.

## References

- [TP2 assignment](https://github.com/ahmed3991/AIoT-2025/blob/main/TP2/TP2.md)
- [TP1 dataset and context](https://github.com/ahmed3991/AIoT-2025/blob/main/TP1/TP1.md)
- [Espressif: ESP32 memory types](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/memory-types.html)
"""
(OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
print(f"\nSaved models, results, and report in: {OUTPUT_DIR}")
