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

Training and inference read prepared molecular graphs, protein contact graphs and frozen language-model features from a data root. `backbone/` holds the interaction backbone of this work; it is bundled here rather than installed, so that a checkout runs without a separate package. The four benchmark collections are Article, Patent, Human and *C. elegans*. The directory layout, feature dimensions and column names are described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Label tables are CSV files with the columns `SMILES`, `Protein` and `Y`.

## Train

The protocol covers four datasets, five splits and five training seeds. `train_dse.py` carries these values as its defaults, and `paper_config.json` records the same set in one place for reference.

Three entries of that record do not read across to the manuscript on their own:

| Entry | What it is |
|---|---|
| `k` | The number of graph-attention layers, written L_g in Supplementary Table 28, where its value is 5. It is not the number of direction-scale components, which is 3, nor the convolution branch count, which is 4. |
| `cl_weight` | The outer factor of the backbone regularisation weight. The term it multiplies carries a further factor of 1e-4 inside the backbone, so the weight applied to the parameter norms is 1e-8, the value lambda_aux given in Supplementary Table 28. |
| `fusion_mode`, `vlm_ablation` | `VLM-XA` selects cross-attention fusion between the graph and language pathways, which is the fusion the Methods describe. `no_attn` turns off the separate attention inside the virtual-mapping module, which the reported model does not use. The two settings act on different modules. |

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

The evaluator reports prediction AUROC, average precision, accuracy, error-detection AUROC, the probability-stratified error detection A(u | p_1), the area under the risk-coverage curve, accuracy in the lowest-uncertainty quarter and the missed-interaction flag rate. It evaluates one prediction set and does not pool independently trained runs.

## Citation

Niu, W., Qian, Z., Zhang, Z., Lei, Z., Luo, X., Qiao, Y., Wang, X., Zhao, W., Guo, L., Wang, Y., Xu, Z. & Gao, S. Decoupling decision direction and evidence scale for drug-target interaction prediction.

Contact: Shangce Gao, gaosc@eng.u-toyama.ac.jp

## License

Released under the MIT License. See [LICENSE](LICENSE).
