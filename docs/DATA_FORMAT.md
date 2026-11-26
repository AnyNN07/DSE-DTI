# Prepared data format

The training and inference pipeline consumes prepared molecular graphs, protein graphs and frozen language-model features. It does not obtain these assets automatically from raw SMILES and protein sequences. This source release does not include the full benchmark datasets, feature dictionaries or model checkpoints.

## Data root and dataset names

Set `DTI_DATA` to a prepared data root. The dataset directory names are case sensitive on systems with case-sensitive file names:

| Directory | Manuscript name |
|---|---|
| `Article` | Article |
| `Patent` | Patent |
| `human` | Human |
| `celegans` | C. elegans |

Each dataset has the following layout:

```text
<prepared-data>/
  <dataset>/
    Modalities/
      smiles_all.npy
      d_LM.pkl
      p_LM.pkl
    sequence_to_pdb.json
    CompoundGraph/
      <compound-index>.bin
    ProteinGraph/
      package_bin/
        <structure-id>.bin
    random/
      train/samples.csv
      valid/samples.csv
      test/samples.csv
    cold_smiles/
      source_train.csv
      target_train.csv
      target_valid.csv
      target_test.csv
    cold_protein/
      source_train.csv
      target_train.csv
      target_valid.csv
      target_test.csv
    cold_both/
      source_train.csv
      target_train.csv
      target_valid.csv
      target_test.csv
    cluster/
      source_train.csv
      target_train.csv
      target_valid.csv
      target_test.csv
```

For non-random splits, `source_train.csv` is optional. When present, its rows are appended to `target_train.csv` for training. Validation and test use their respective target files. Random splits use the three `samples.csv` files.

## Split names

| On-disk identifier | Manuscript term |
|---|---|
| `random` | random |
| `cold_smiles` | drug cold-start |
| `cold_protein` | target cold-start |
| `cold_both` | pair cold-start |
| `cluster` | cross-domain |

These identifiers are the ones the paper-associated checkpoints were trained under. The five splits are five different experimental settings, not five folds of a cross-validation, and how far drug and target identities overlap between training and test depends on which one is selected.

The reported numbers come from the prepared splits and the retained evaluation rows of the paper. A source dataset downloaded afresh, or splits built anew, gives a different record set.

## Pair tables

The canonical columns are `SMILES`, `Protein` and `Y`. The training reader also normalises lower-case `smiles`, `protein`, and `label` or `y` column names. Labels are binary, with zero indicating a negative interaction label and one a positive interaction label. Exact label meaning follows the source dataset or assay curation rules.

`SMILES` is the compound key. `Protein` is the full protein-sequence key. The loader strips leading and trailing whitespace, but it does not canonicalise a chemical structure or resolve a new protein identifier. Keys must match the prepared feature dictionaries and mapping files.

## Frozen language-model features

| File | Contents |
|---|---|
| `smiles_all.npy` | Ordered compound strings. The zero-based array index identifies the compound graph file. |
| `d_LM.pkl` | Dictionary from a compound string to an array of shape `[number_of_compound_tokens, 768]`. |
| `p_LM.pkl` | Dictionary from a protein sequence to an array of shape `[number_of_protein_tokens, 320]`. |

The collate function pads feature sequences within each minibatch and supplies their true token lengths to the model, which are a different quantity from the graph-node counts. Non-finite feature entries are replaced the same way as in the implementation these features were prepared with.

The project preparation scripts identify the protein feature model as `facebook/esm2_t6_8M_UR50D` and the compound feature model as `seyonec/PubChem10M_SMILES_BPE_450k`. Features regenerated from the same two model names can still differ numerically, because the cached model revision, the tokenisation and the processing settings all enter the result. The prepared-feature runtime does not import Transformers or download these encoders.

## Graph files and structure mapping

Compound graph files are DGL `.bin` files. A compound with array index `i` in `smiles_all.npy` is loaded from `CompoundGraph/i.bin`. The first graph in that file is used. The backbone consumes node features under `atom` with width 120 and bond features under `bond` with width 12.

Despite its file name, `sequence_to_pdb.json` is read as a mapping from a structure identifier to a sequence or a list of sequences. The loader reverses that mapping to obtain a structure identifier for each protein sequence. Its graph is loaded from `ProteinGraph/package_bin/<structure-id>.bin`. The backbone consumes protein node features under `feats` with width 41 and edge features under `feats` with width 5. Preserve the other graph fields produced by the original preparation pipeline when redistributing graph assets.

When an asset is missing the loader falls back to zero features, or to a single-node placeholder graph, and does so without raising. A run over an incomplete dataset therefore finishes normally while resting on substituted inputs, so asset coverage is worth checking before training or inference.

## Checkpoints and support assets

A trained DSE-DTI checkpoint includes the model state and the support-gate state. Inference reads the model state together with the training support bank and its distance threshold, so a backbone weight file on its own is not enough.

The prepared inputs, checkpoint and training configuration must agree on feature dimensions, dataset, split and seed.
