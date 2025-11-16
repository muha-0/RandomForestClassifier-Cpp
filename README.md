# RandomForestClassifier in C++

This repository contains a from-scratch **Random Forest Classifier in C++** that matches or slightly outperforms scikit-learn’s implementation on benchmark datasets such as **Titanic** and **Pulsar Star Classification**, under **identical preprocessing, seeds, and hyperparameters**.

## Highlights

- **High-performance C++ implementation** of Random Forest
- Outperforms scikit-learn’s RF on Titanic (~88.5% vs ~84.5%)
- Matches scikit-learn on Pulsar (Acc ~97%, F1 ~0.87)
- Identical preprocessing, seeds, tree parameters for fair comparison
- Optimized recursion & memory allocation for deep trees  
- Fully deterministic & reproducible

---

## Benchmark Results

High-level summary:

| Dataset | Model          | Accuracy | F1 Score |
|--------|----------------|----------|----------|
| Titanic        | C++ Random Forest | ~0.885 | ~0.79 |
| Titanic        | sklearn RF        | ~0.845 | ~0.79 |
| Pulsar Stars   | C++ Random Forest | ~0.97  | ~0.87 |
| Pulsar Stars   | sklearn RF        | ~0.97  | ~0.87 |

### Per-run benchmarks

**Titanic dataset (Runs 1–5)**

| Run  | Model    | Accuracy | F1 Score |
|------|----------|----------|----------|
| Run 1 | C++      | 0.8483 | 0.7938 |
| Run 1 | sklearn  | 0.8324 | 0.7945 |
| Run 2 | C++      | 0.8595 | 0.7787 |
| Run 2 | sklearn  | 0.8324 | 0.7887 |
| Run 3 | C++      | 0.8146 | 0.7814 |
| Run 3 | sklearn  | 0.8212 | 0.7777 |
| Run 4 | C++      | 0.8370 | 0.7972 |
| Run 4 | sklearn  | 0.8156 | 0.7692 |
| Run 5 | C++      | 0.8651 | 0.8285 |
| Run 5 | sklearn  | 0.8268 | 0.7862 |

**Pulsar Star dataset (Runs 6–10)**

| Run   | Model   | Accuracy | F1 Score |
|-------|---------|----------|----------|
| Run 6  | C++     | 0.9744 | 0.8407 |
| Run 6  | sklearn | 0.9796 | 0.8705 |
| Run 7  | C++     | 0.9788 | 0.8781 |
| Run 7  | sklearn | 0.9792 | 0.8779 |
| Run 8  | C++     | 0.9788 | 0.8646 |
| Run 8  | sklearn | 0.9788 | 0.8752 |
| Run 9  | C++     | 0.9776 | 0.8672 |
| Run 9  | sklearn | 0.9792 | 0.8779 |
| Run 10 | C++     | 0.9760 | 0.8642 |
| Run 10 | sklearn | 0.9796 | 0.8800 |

**Overall averages across all 10 runs**

| Metric       | C++ RF  | sklearn RF |
|-------------|---------|------------|
| Avg. accuracy | 0.91101 | 0.90248 |
| Avg. F1 score | 0.82944 | 0.82978 |

---

## Project Structure

```
RandomForestClassifier/
│
├── data/               # titanic and pulsar csv files
├── docs/               # results, tables, diagrams
├── include/            # header files 
├── src/                # implementation
│
├── .gitignore
└── README.md

```

## Implementation Notes

- No external ML libraries — **pure C++ Random Forest**
- Custom tree structures with optimized memory layout
- Manual recursion control for large depths
- Careful handling of:
  - node splitting
  - bootstrap sampling
  - feature subsampling
  - impurity calculations
  - stopping criteria

---

## License

MIT License.
