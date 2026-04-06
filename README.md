# QuantOm: Performance vs Productivity in HPC

This repository accompanies the paper:

**"On the Efficacy of PyTorch for High-Performance Computing:  
A Case Study in Computational Physics"**

It provides:
- Implementations of the LOITS sampler in:
  - PyTorch
  - C++
  - OpenMP
  - SYCL (AdaptiveCpp, DPC++)
- Scripts to reproduce all experiments and figures
- Precomputed results used in the paper (for fast reproduction)

---

# 🚀 Quick Start (Recommended)

If you just want to reproduce the figures from the paper:

```bash
make plot
```

This uses **archived results** (included in the repository) and regenerates all figures.

👉 This is the **default path for reviewers** — no compilation required.

---

# ⚙️ Environment Setup

We provide Conda/Mamba environments:

```bash
mamba create -f quantom.yaml
mamba activate quantom
```

If dependency resolution fails:

```bash
mamba create -f quantom-no-build-info.yaml
```

> **Note:** `mamba` and `conda` are interchangeable, but `mamba` is faster.

---

# 🧪 Running Experiments

## 1. Install Backends (SYCL / LLVM / etc.)

```bash
make install
```

This installs:
- LLVM toolchains
- DPC++
- AdaptiveCpp
- UniSYCL (optional)
- IRIS runtime

All installs are host-specific and placed in:

```
sycl-implementations/<hostname>/
```

---

## 2. Run Experiments (⚠️ Expensive)

```bash
make run
```

This will:
- compile all implementations
- run CPU + GPU experiments
- overwrite `results/*.csv`

---

## 3. Plot Results

```bash
make plot
```

This generates:
- CPU scaling plots
- GPU scaling plots
- Strong/weak scaling figures

---

# 📦 Results Management

We separate **paper results** from **new runs**:

```
results/
├── archived/     # results used in the paper (DO NOT MODIFY)
├── latest/       # results from your runs
```

### Default behavior
- `make plot` → uses `archived/`
- `make run` → writes to `latest/`

### To replot new results:

```bash
make plot RESULTS=latest
```

---

# 🔁 Reproducing Paper Results

To fully reproduce:

```bash
make install
make run        # may take hours depending on hardware
make plot RESULTS=latest
```

Then compare:

```
results/archived/ vs results/latest/
```

---

# 🧠 What This Artifact Demonstrates

This artifact supports the paper’s key findings:

- PyTorch is **4–5× more productive** (SLOC)
- CPU performance:
  - PyTorch achieves ~50–72% of optimized C++/OpenMP
- GPU performance:
  - PyTorch outperforms SYCL by:
    - ~5–6× (CUDA)
    - ~15× (HIP)
    - up to ~16× (Intel XPU)

These results arise from:
- kernel fusion
- optimized backend primitives (e.g., CUB)
- reduced synchronization overhead

---

# 📊 Experiments Included

- Strong scaling (CPU threads)
- Weak scaling (per-core workload)
- Fixed-resource scaling (CPU + GPU)
- Profiling breakdown of LOITS pipeline

---

# 📁 Repository Structure

```
.
├── cpp/                 # C++ implementation
├── omp/                 # OpenMP implementation
├── sycl/                # SYCL implementations + installer
├── examples/            # Jupyter notebooks
├── results/
│   ├── archived/        # paper results
│   └── latest/          # new runs
├── utils/               # plotting scripts
├── setup_backends.sh    # environment configuration
├── utils.sh             # logging helpers
└── Makefile
```

---

# 📓 Example Usage (Notebook)

To explore LOITS interactively:

```bash
cd examples/2d_loits
jupyter-notebook
```

---

# ⚠️ Notes on Reproducibility

- GPU results depend on:
  - CUDA / ROCm / Level Zero versions
  - hardware (A100, MI300A, Intel Max)
- CPU scaling depends on:
  - NUMA configuration
  - thread pinning

We attempt to normalize this via:
- fixed seeds
- explicit thread control
- consistent build flags

---

# 🛠 Troubleshooting

### DPC++ / TBB issues
Ensure TBB is correctly built and linked:
```
sycl-implementations/<host>/tbb/
```

### CUDA / HIP not detected
Check:
```bash
echo $BACKENDS
echo $CUDA_PATH
echo $ROCM_PATH
```

### Build failures
Try:
```bash
make reinstall
```

---

# 📜 License (MIT)

Copyright 2026 Beau Johnston <beau@inbeta.org>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



---

# 📎 Citation

If you use this artifact, please cite:

```
Beau Johnston, Niteya Shah, Wu-chun Feng.
``On the Efficacy of PyTorch for High-Performance Computing:
A Case Study in Computational Physics.''
Proceedings of the 23rd ACM International Conference on Computing Frontiers (CF 26')
10.1145/3801487.3801838
```

