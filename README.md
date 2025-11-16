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

| Dataset | Model | Accuracy | F1 Score |
|--------|--------|----------|----------|
| Titanic | **C++ RF** | **0.885** | **0.79** |
| Titanic | sklearn RF | 0.845 | 0.79 |
| Pulsar | **C++ RF** | **0.97** | **0.87** |
| Pulsar | sklearn RF | 0.979 | 0.87 |

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
