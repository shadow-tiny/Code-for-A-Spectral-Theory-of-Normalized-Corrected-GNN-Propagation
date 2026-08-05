# Code for A Spectral Theory of Normalized Corrected GNN Propagation

This repository contains the experiment code used in the paper on normalized corrected graph propagation for oversmoothing analysis in GNNs.

The code covers:

- synthetic CSBM experiments,
- real-data node classification experiments,
- balanced two-class subset experiments on real datasets,
- baseline comparisons against representative anti-oversmoothing methods,
- cached `.npz` result files and exported `.pdf` figures used in the paper.

## Repository Structure

```text
paper_code/
├── synthetic.py
├── synthetic_compare.py
├── real.py
├── real_compare.py
├── real_sample.py
├── real_sample_compare.py
└── result/
    ├── synthetic/
    ├── synthetic_1/
    ├── real/
    ├── real_1/
    ├── real_sample/
    └── real_sample_1/
```

## Environment

The code was developed with Python and PyTorch/PyG style dependencies.

Recommended packages:

- `python >= 3.9`
- `numpy`
- `matplotlib`
- `tqdm`
- `torch`
- `torch-geometric`
- `ogb`

A typical installation is:

```bash
pip install numpy matplotlib tqdm torch torch-geometric ogb
```

Depending on your platform and CUDA version, `torch` and `torch-geometric` may need to be installed from their official wheels.

## Datasets

The real-data scripts use the following public datasets:

- `Cora`
- `CiteSeer`
- `PubMed`
- `Reddit`
- `ogbn-arxiv`
- `ogbn-products`

Datasets are downloaded automatically by PyG / OGB into a local `data/` directory when first used.

All real-data experiments convert graphs to undirected form before evaluation.

## Main Scripts

### 1. Synthetic experiments

`synthetic.py`

- compares standard GCN propagation, normalized corrected propagation (`\hat A`), and unnormalized corrected propagation (`\tilde A`) on synthetic CSBM data,
- runs the feature-SNR sweep,
- runs the graph-signal sweep,
- saves cached results in `result/synthetic/*.npz`,
- exports figures to `result/synthetic/*.pdf`.

Run:

```bash
python synthetic.py
```

### 2. Synthetic baseline comparisons

`synthetic_compare.py`

- runs synthetic CSBM comparisons with oversmoothing baselines,
- includes methods such as PairNorm, DropEdge, APPNP, GCNII, MbaGCN, and Reverse-GNN,
- saves cached results in `result/synthetic_1/*.npz`,
- exports figures to `result/synthetic_1/*.pdf`.

Run:

```bash
python synthetic_compare.py
```

### 3. Real-data full-dataset experiments

`real.py`

- compares standard GCN, normalized corrected propagation (`\hat A`), and unnormalized corrected propagation (`\tilde A`) on the full real datasets,
- uses full-batch training on smaller datasets,
- uses `NeighborLoader` mini-batch training for `Reddit` and `ogbn-products`,
- saves cached results in `result/real/*.npz`,
- exports figures to `result/real/*.pdf`.

Run:

```bash
python real.py
```

### 4. Real-data full-dataset baseline comparisons

`real_compare.py`

- compares the proposed corrected operator with anti-oversmoothing baselines on the full datasets,
- saves cached results in `result/real_1/*.npz`,
- exports figures to `result/real_1/*.pdf`.

Run:

```bash
python real_compare.py
```

### 5. Balanced two-class subset experiments

`real_sample.py`

- constructs balanced two-class subsets for each real dataset,
- selects the two largest classes,
- downsamples the majority class to obtain a `1:1` class ratio,
- keeps the full graph topology intact,
- creates a fresh `60/20/20` train/validation/test split on the selected nodes,
- compares standard GCN, normalized corrected propagation (`\hat A`), and unnormalized corrected propagation (`\tilde A`),
- saves cached results in `result/real_sample/*.npz`,
- exports figures to `result/real_sample/*.pdf`.

Run:

```bash
python real_sample.py
```

### 6. Balanced two-class subset baseline comparisons

`real_sample_compare.py`

- runs baseline comparisons on the balanced two-class subsets,
- saves cached results in `result/real_sample_1/*.npz`,
- exports figures to `result/real_sample_1/*.pdf`.

Run:

```bash
python real_sample_compare.py
```

## Reproducibility Notes

- The scripts cache intermediate numeric results as `.npz` files.
- If a result file already exists, the script loads it instead of recomputing from scratch.
- Figures are exported as PDF files.
- The synthetic scripts, including synthetic baseline comparisons, store both mean accuracy and standard deviation arrays in the `.npz` files.
- The real-data scripts currently store mean accuracy curves across repeated runs.

## Hardware Notes

- The code automatically uses GPU if available.
- For very large datasets, especially `ogbn-products`, the scripts switch to mini-batch training with neighbor sampling.
- Some evaluation paths use CPU inference to avoid GPU out-of-memory issues on deep models.

## Expected Runtime

Runtime depends strongly on:

- whether cached `.npz` files already exist,
- whether a GPU is available,
- whether you run all datasets and all depth sweeps,
- whether you run the baseline comparison scripts.

The real-data baseline comparison scripts are the most time-consuming part of the repository.

## Output Files

Each script saves:

- cached numeric results in `.npz`,
- publication-ready figures in `.pdf`.

The output directories are:

- `result/synthetic/`
- `result/synthetic_1/`
- `result/real/`
- `result/real_1/`
- `result/real_sample/`
- `result/real_sample_1/`

## Anonymous Release Notes

This repository is prepared for anonymous review.

- No author names are included in this README.
- No institutional information is required to run the code.
- Dataset downloads rely only on public sources from PyG / OGB.
