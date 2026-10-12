# TP2 — Optimization and Deployment Readiness

## Setup

Data: [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset).
Target: `Fire Alarm` (0 = normal, 1 = alarm).
62,630 complete rows, 12 sensor features; an 80/20 stratified split
with seed 42: 50,104 training rows and 12,526 test rows.
`Unnamed: 0`, `UTC`, and `CNT` are excluded because they describe row identity,
time, or collection order rather than sensor measurements.
Both pipelines contain `StandardScaler`, fitted only on the training partition.
LR uses `max_iter=1000`; XGBoost uses 100 trees, depth 3, learning rate 0.1, and one worker.

## Measurements

| Model | Accuracy | F1 | Precision | Recall | Size (KB) | Training (s) | Test inference (s) | Average per sample (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR-Pipeline | 0.895098 | 0.928150 | 0.908964 | 0.948162 | 1.642 | 0.078648 | 0.001546 | 0.000123 |
| XGB-Pipeline | 0.999840 | 0.999888 | 1.000000 | 0.999777 | 120.317 | 0.354177 | 0.009114 | 0.000728 |

Here KB means bytes / 1024 (technically KiB). Size is the actual serialized
**whole pipeline**, including the scaler. Training time includes scaling.
Inference time includes scaling and prediction but excludes metrics and disk I/O.
It is the median of 5 full-test calls after one warm-up call.
Average per sample = total test inference seconds / 12,526 × 1000.
This batch average is not the latency of an isolated one-row call.
F1, precision, and recall use the positive class `Fire Alarm = 1`.
These are real measurements on the computer recorded in `environment.json`;
timings can vary between runs.

## ESP32 deployment analysis (Task 2.3)

The LR pipeline occupies **1.642 KB**, compared with
**120.317 KB** for XGBoost, a size ratio of **73.3×**.
The smaller serialized model is **LR-Pipeline**. Both files fit within a typical
multi-megabyte Flash capacity, but pickle size alone does not establish runtime
RAM feasibility: the classic ESP32's approximately 520 KB SRAM is shared with
code, stacks, communication, and buffers, and not all of it is application heap.
Python pickle pipelines also require desktop Python libraries and cannot be
used directly as ESP32 firmware. A native implementation must be measured for
actual RAM usage; constant weights or tree nodes can be stored in Flash.
XGBoost is more constrained by memory because its many trees require more
parameters and node traversal than LR's small coefficient vector.
Measured average inference costs are **0.000123 ms** for LR
and **0.000728 ms** for XGBoost; **LR-Pipeline** has the lower
full-test inference time in this run. Both measured averages are below
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
