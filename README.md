# DSE-DTI

This is the implementation of DSE-DTI: Decoupling decision direction and evidence scale for drug-target interaction prediction.

The direction-scale (DS) head separates the decision direction, which gives the binding probability, from the evidence scale, which gives the uncertainty. A training-only support bank adjusts the evidence scale without changing the binding probability.

## Catalogue

- [Installation](#installation)
- [Data](#data)
- [Train](#train)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Installation

DSE-DTI is built on Python 3.10. We recommend a virtual conda environment:

```bash
conda create -n dsedti python==3.10
conda activate dsedti
```

Install a PyTorch and DGL build that matches your CUDA version. The reported results were obtained with PyTorch 2.1.0 and DGL 2.2.1 under CUDA 12.1. Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` records the versions used for the reported results.

A numerical demo of the head runs without molecular data:

```bash
python demo_head.py
python -m unittest discover -s tests -v
```

## Data

Training and inference read prepared molecular graphs, protein graphs and frozen language-model features from a data root. The four benchmark collections are Article, Patent, Human and *C. elegans*. The directory layout, feature dimensions and column names are described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Label tables are CSV files with the columns `SMILES`, `Protein` and `Y`.

## Train

The protocol covers four datasets, five splits and five training seeds. The parameter values used for the reported results are in `paper_config.json`.

One setting:

```bash
python train_dse.py --data-root prepared_data --datasets human --splits random --seeds 42 --out runs --device auto
```

All settings:

```bash
python train_dse.py --data-root prepared_data --out runs --device auto
```

The loop refits the support bank from the training embeddings and selects the checkpoint by validation AUROC. Each checkpoint stores the backbone, the head and the support bank with its distance threshold.

## Prediction

```bash
python predict.py --checkpoint runs/models/human/random/seed_42/best_model.pth --data-root prepared_data --dataset human --csv pairs.csv --output outputs/predictions.csv --device auto --trust-input-files
```

Keep `manifest.json` beside the checkpoint; it carries the raw-scale bounds and the neighbour count. The output preserves the input row order and gives the binding probability `p1`, the uncertainty `u`, the gate-adjusted scale `m`, the raw gate, the nearest-neighbour distance and the three direction fractions.

Batch size and input order affect the backbone, so keep them unchanged when reproducing a reported evaluation.

## Evaluation

```bash
python evaluate.py --predictions outputs/predictions.csv --output outputs/metrics.json
```

The evaluator reports prediction AUROC, average precision, accuracy, error-detection AUROC, probability-stratified error detection, the area under the risk-coverage curve, accuracy in the most-certain quarter and the missed-interaction flag rate. It evaluates one prediction set and does not pool independently trained runs.

## Citation

Niu, W., Qian, Z., Zhang, Z., Lei, Z. & Gao, S. Decoupling decision direction and evidence scale for drug-target interaction prediction.

Contact: Shangce Gao, gaosc@eng.u-toyama.ac.jp
