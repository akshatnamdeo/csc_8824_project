
# Structured Matrix Variants of Module-LWE in ML-KEM

**CSC 8824 - Graduate Project**  
**Georgia State University**  

**Authors:**  
Akshat Namdeo (anamdeo1@student.gsu.edu)  
Dheeraj Narsani (dnarsani1@student.gsu.edu)

---

## Project Overview

This project experimentally studies how imposing algebraic structure on the public matrix **A** affects the security of the Module Learning With Errors (MLWE) problem underlying **ML-KEM (FIPS 203)**.

We evaluate four structured matrix families, **low-rank**, **module-circulant**, **module-Toeplitz**, and **sparse**, under ML-KEM-768 parameters (`n=256`, `q=3329`, `k=3`). Using a coefficient variance distinguisher and logistic regression, we show that distinguishability in sparse matrices arises **solely from uncovered rows**. We propose **row-covered sparse matrices** as a practical structured variant that achieves a **2.70× speedup** in the dominant matrix-vector multiplication while remaining empirically indistinguishable from uniform MLWE.

## Repository Structure

| File/Folder                    | Description |
|-------------------------------|-----------|
| `dataset_generation.py`       | Core module for polynomial arithmetic (NTT), MLWE sampling, and structured matrix generation (low-rank, circulant, Toeplitz, sparse variants). |
| `evaluation.py`               | Implements the coefficient variance distinguisher and logistic regression classifier. |
| `main.py`                     | Main experiment script for baseline families (low-rank, circulant, Toeplitz). |
| `main2.py`                    | Experiments on sparse matrices with different placement strategies and row-coverage tests. |
| `main3.py`                    | Additional timing and performance experiments. |
| `main_logs.txt`, `main2_logs.txt`, `main3_logs.txt` | Raw experiment logs. |
| `graphs/`                     | Generated plots (including rho sweep vs. uncovered row probability). |

## Requirements

- Python 3.8+
- Required packages:
  ```bash
  numpy
  scipy
  scikit-learn
  matplotlib
  ```

## How to Run

### 1. Install dependencies
```bash
pip install numpy scipy scikit-learn matplotlib
```

### 2. Reproduce main experiments
```bash
# Baseline experiments (low-rank, circulant, Toeplitz)
python main.py

# Sparse matrix experiments + row coverage analysis
python main2.py

# Timing / performance experiments
python main3.py
```

### 3. Generate graphs
After running the scripts, plots will be saved in the `graphs/` folder.

## Key Results

- Low-rank, module-circulant, and module-Toeplitz matrices show **no detectable deviation** from uniform MLWE.
- Random sparse matrices are distinguishable **only** when they contain uncovered rows.
- **Row-covered sparse matrices** (`nz = k = 3`) are empirically indistinguishable while providing **2.70× speedup** in `A·s`.
