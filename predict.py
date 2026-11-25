import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

def modality_paths(dataset_root):
    root = Path(dataset_root)
    return tuple(root / name for name in ("Modalities/smiles_all.npy", "Modalities/p_LM.pkl", "Modalities/d_LM.pkl", "sequence_to_pdb.json", "CompoundGraph", "ProteinGraph/package_bin"))

def normalise_columns(frame):
    aliases = {"smiles": "SMILES", "protein": "Protein", "seq": "Protein", "label": "Y", "y": "Y"}
    rename = {column: aliases[column.strip().lower()] for column in frame.columns if column.strip().lower() in aliases}
    frame = frame.rename(columns=rename)
    if frame.columns.duplicated().any():
        raise ValueError("Input has duplicate columns after normalising names")
    if not {"SMILES", "Protein"}.issubset(frame.columns):
        raise ValueError("Input CSV requires SMILES and Protein columns")
    if not len(frame):
        raise ValueError("Input CSV is empty")
    for name in ("SMILES", "Protein"):
        if frame[name].isna().any():
            raise ValueError(f"Input has missing {name} values")
        frame[name] = frame[name].astype(str).str.strip()
        if (frame[name].str.len() == 0).any():
            raise ValueError(f"Input has empty {name} values")
    if "Y" in frame.columns and not frame["Y"].isin((0, 1)).all():
        raise ValueError("Provided labels must all be 0 or 1")
    return frame

def verify_inputs(frame, dataset_root):
    import numpy as np
    import dgl
    paths = modality_paths(dataset_root)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required feature asset is missing: {path}")
    smiles = np.load(paths[0], allow_pickle=True)
    smiles_index = {str(value): index for index, value in enumerate(smiles)}
    with paths[1].open("rb") as stream:
        protein_features = pickle.load(stream)
    with paths[2].open("rb") as stream:
        compound_features = pickle.load(stream)
    with paths[3].open(encoding="utf-8") as stream:
        mapping = json.load(stream)
    sequence_to_pdb = {}
    for pdb, sequences in mapping.items():
        for sequence in sequences if isinstance(sequences, list) else [sequences]:
            sequence_to_pdb[str(sequence).strip()] = str(pdb).strip()
    errors = []
    compound_graphs = {}
    protein_graphs = {}
    for row_number, row in enumerate(frame.itertuples(index=False), start=1):
        compound, protein = str(row.SMILES).strip(), str(row.Protein).strip()
        index = smiles_index.get(compound)
        pdb = sequence_to_pdb.get(protein)
        absent = []
        if index is None or not (paths[4] / f"{index}.bin").is_file():
            absent.append("compound graph")
        if not pdb or not (paths[5] / f"{pdb}.bin").is_file():
            absent.append("protein graph")
        for key, collection, width, label in ((compound, compound_features, 768, "compound embedding"), (protein, protein_features, 320, "protein embedding")):
            if key not in collection:
                absent.append(label)
                continue
            array = np.asarray(collection[key])
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width or not np.isfinite(array).all():
                absent.append(f"invalid {label} shape or values")
        if absent:
            errors.append(f"row {row_number}: {', '.join(absent)}")
        else:
            compound_graphs[index] = paths[4] / f"{index}.bin"
            protein_graphs[pdb] = paths[5] / f"{pdb}.bin"
    if errors:
        raise ValueError(f"Incomplete inputs in {len(errors)} rows. Missing inputs are not replaced with zero features. First problems: {'; '.join(errors[:8])}")
    for collection, node_key, node_width, edge_key, edge_width in ((compound_graphs, "atom", 120, "bond", 12), (protein_graphs, "feats", 41, "feats", 5)):
        for path in collection.values():
            graphs, _ = dgl.load_graphs(str(path))
            if len(graphs) != 1 or graphs[0].num_nodes() < 1:
                raise ValueError(f"Expected one nonempty graph in {path}")
            graph = graphs[0]
            if node_key not in graph.ndata or graph.ndata[node_key].ndim != 2 or graph.ndata[node_key].shape[1] != node_width:
                raise ValueError(f"Invalid {node_key} node features in {path}")
            if graph.num_edges() and (edge_key not in graph.edata or graph.edata[edge_key].ndim != 2 or graph.edata[edge_key].shape[1] != edge_width):
                raise ValueError(f"Invalid {edge_key} edge features in {path}")
            for array in (graph.ndata[node_key], graph.edata.get(edge_key)):
                if array is not None and not np.isfinite(array.detach().cpu().numpy()).all():
                    raise ValueError(f"Non-finite graph features in {path}")
    return tuple(str(path) for path in paths)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DSE-DTI inference using a trusted checkpoint and precomputed feature assets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--trust-input-files", action="store_true", help="Acknowledge that checkpoint and pickle/NumPy/DGL assets come from a trusted source.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.trust_input_files:
        parser.error("Feature pickles and checkpoint assets must be trusted. Add --trust-input-files only after checking their origin.")
    if not args.overwrite and (output.exists() or metadata_path.exists()):
        parser.error("Output or metadata already exists. Use another filename or --overwrite.")
    checkpoint_path = Path(args.checkpoint).resolve()
    csv_path = Path(args.csv).resolve()
    dataset_root = Path(args.data_root).resolve() / args.dataset
    manifest_path = Path(args.manifest).resolve() if args.manifest else checkpoint_path.parent / "manifest.json"
    protected = {checkpoint_path, csv_path, manifest_path}
    asset_paths = modality_paths(dataset_root)
    protected.update(path.resolve() for path in asset_paths[:4])
    asset_directories = [dataset_root / "Modalities", *asset_paths[4:]]
    for target in (output.resolve(), metadata_path.resolve()):
        if target in protected or any(target.is_relative_to(directory.resolve()) for directory in asset_directories):
            parser.error("Output cannot overwrite an input, checkpoint, manifest or prepared feature asset")
    os.environ["DTI_DATA"] = str(Path(args.data_root).resolve())
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from dse_model import DSENet, readout, VLM_CODE
        from train_dse import PROTOCOL, HEAD
        if VLM_CODE not in sys.path:
            sys.path.insert(0, VLM_CODE)
        from unified_dataset_v2 import UnifiedDtiDatasetV2, unified_collate_fn_v2
        frame = normalise_columns(pd.read_csv(csv_path))
        modality_args = verify_inputs(frame, dataset_root)
        protocol, head = dict(PROTOCOL), dict(HEAD)
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            protocol.update(manifest.get("protocol", {}))
            head.update(manifest.get("head", {}))
        elif args.manifest:
            raise FileNotFoundError(manifest_path)
        else:
            print("Warning: checkpoint manifest not found. Using packaged paper defaults.", file=sys.stderr)
        if head.get("num_classes", 3) != 3:
            raise ValueError("This release supports only the three-component DSE-DTI model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not {"gate.bank", "gate.filled", "gate.tau"}.issubset(state):
            raise ValueError("Checkpoint must include model parameters and fitted support-bank buffers")
        if int(state["gate.filled"].item()) < 1 or not math.isfinite(float(state["gate.tau"].item())) or float(state["gate.tau"].item()) <= 0:
            raise ValueError("Checkpoint has an unfitted or invalid support gate")
        if state["gate.bank"].ndim != 2 or state["gate.bank"].shape[1] != 16:
            raise ValueError("Checkpoint support bank must contain 16-dimensional representations")
        if int(state["gate.filled"].item()) > state["gate.bank"].shape[0]:
            raise ValueError("Checkpoint filled count exceeds its bank capacity")
        model = DSENet(gnn_hidden=protocol["gnn_hidden"], k=protocol["k"], fusion_mode=protocol["fusion_mode"], vlm_ablation=protocol["vlm_ablation"], m_min=head["m_min"], m_max=head["m_max"], bank_size=state["gate.bank"].shape[0], knn=head["knn"], head_hidden=head["head_hidden"], dropout=head["dropout"])
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        order = list(range(len(frame)))
        if args.order_seed is not None:
            random.Random(args.order_seed).shuffle(order)
        reordered = frame.iloc[order].copy().reset_index(drop=True)
        has_labels = "Y" in reordered.columns
        if not has_labels:
            reordered["Y"] = 0
        dataset = UnifiedDtiDatasetV2(reordered, *modality_args, load_llm=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=unified_collate_fn_v2, num_workers=0)
        collected = {key: [] for key in ("p1", "u", "m", "gate", "dist", "pi_nobind", "pi_bind", "pi_unknown")}
        with torch.no_grad():
            for bg_d, bg_p, drug_features, protein_features, _, drug_lengths, protein_lengths in loader:
                values = model(bg_d.to(device), bg_p.to(device), drug_features.to(device), protein_features.to(device), drug_lengths.to(device), protein_lengths.to(device))
                result = readout(values["pi"], values["m"])
                arrays = {"p1": result["p1"], "u": result["u"], "m": result["m"], "gate": values["gate"], "dist": model.gate.distance(values["h"]), "pi_nobind": values["pi"][:, 0], "pi_bind": values["pi"][:, 1], "pi_unknown": values["pi"][:, 2]}
                for key, array in arrays.items():
                    collected[key].append(array.detach().cpu().numpy())
        predictions = frame.copy().rename(columns={"Y": "y"})
        predictions.insert(0, "row_index", np.arange(len(frame)))
        for key, arrays in collected.items():
            restored = np.empty(len(frame), dtype=np.float64)
            restored[np.asarray(order)] = np.concatenate(arrays)
            if not np.isfinite(restored).all():
                raise ValueError(f"Non-finite model output in {key}")
            if key in {"p1", "u", "gate", "pi_nobind", "pi_bind", "pi_unknown"} and not ((restored >= 0) & (restored <= 1)).all():
                raise ValueError(f"Model output {key} is outside [0, 1]")
            predictions[key] = restored
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"model": "DSE-DTI", "classes": {"0": "non-binding", "1": "binding", "2": "unknown"}, "checkpoint_sha256": sha256(checkpoint_path), "input_csv_sha256": sha256(csv_path), "rows": len(frame), "labels_supplied": has_labels, "batch_size": args.batch_size, "order_seed": args.order_seed, "order_algorithm": "original CSV order" if args.order_seed is None else "Python random.Random.shuffle", "dataset": args.dataset, "device": str(device), "torch_version": torch.__version__, "protocol": protocol, "head": head, "configuration_source": str(manifest_path) if manifest is not None else "packaged defaults; original checkpoint manifest unavailable", "support_bank_filled": int(state["gate.filled"].item()), "support_gate_tau": float(state["gate.tau"].item()), "scope": "One input order. Outputs are restored to original CSV row order. m is the gate-adjusted effective scale. Batch composition and padded feature lengths can affect outputs. Exact published-number reproduction also requires the original row filtering, batch/order protocol and assets."}
        metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        temporary_paths = []
        try:
            for target in (output, metadata_path):
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".dse-output-", suffix=".tmp", delete=False) as stream:
                    temporary_paths.append(Path(stream.name))
            predictions.to_csv(temporary_paths[0], index=False, float_format="%.17g")
            temporary_paths[1].write_text(metadata_text, encoding="utf-8")
            os.replace(temporary_paths[0], output)
            os.replace(temporary_paths[1], metadata_path)
        finally:
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(f"Wrote {len(predictions)} predictions to {output}")
    print(f"Wrote inference metadata to {metadata_path}")

if __name__ == "__main__":
    main()
