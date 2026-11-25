import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

class InferenceTests(unittest.TestCase):
    def test_evaluation_cannot_overwrite_predictions(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'predictions.csv'
            original = 'y,p1,u\n0,0.1,0.2\n1,0.9,0.3\n'
            source.write_text(original, encoding='utf-8')
            run = subprocess.run([sys.executable, '-B', str(root / 'evaluate.py'), '--predictions', str(source), '--output', str(source), '--overwrite'], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('cannot overwrite', run.stderr)
            self.assertEqual(source.read_text(encoding='utf-8'), original)

    @unittest.skipUnless(importlib.util.find_spec('dgl') and importlib.util.find_spec('pandas'), 'Graph runtime is not installed')
    def test_prepared_input_inference_and_unlabelled_export(self):
        import dgl
        import numpy as np
        import pandas as pd
        import torch
        from dse_model import DSENet
        root = Path(__file__).resolve().parents[1]
        torch.manual_seed(31)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dataset = folder / 'data' / 'human'
            for name in ('Modalities', 'CompoundGraph', 'ProteinGraph/package_bin'):
                (dataset / name).mkdir(parents=True, exist_ok=True)
            compounds = ['CCO', 'CCN']
            proteins = ['ACDEFG', 'ACDEFGH']
            np.save(dataset / 'Modalities/smiles_all.npy', np.asarray(compounds))
            for name, keys, width in [('d_LM.pkl', compounds, 768), ('p_LM.pkl', proteins, 320)]:
                with (dataset / 'Modalities' / name).open('wb') as stream:
                    pickle.dump({key: np.random.default_rng(index).normal(size=(7 + index, width)).astype(np.float32) for index, key in enumerate(keys)}, stream)
            (dataset / 'sequence_to_pdb.json').write_text(json.dumps({'fixture_a': [proteins[0]], 'fixture_b': [proteins[1]]}), encoding='utf-8')
            for index in range(2):
                for location, name, node_key, node_width, edge_key, edge_width in [('CompoundGraph', str(index), 'atom', 120, 'bond', 12), ('ProteinGraph/package_bin', ['fixture_a', 'fixture_b'][index], 'feats', 41, 'feats', 5)]:
                    graph = dgl.graph(([0, 1, 2], [1, 2, 0]))
                    graph.ndata[node_key] = torch.randn(3, node_width)
                    graph.edata[edge_key] = torch.randn(3, edge_width)
                    dgl.save_graphs(str(dataset / location / (name + '.bin')), [graph])
            checkpoint = folder / 'fixture.pth'
            model = DSENet(bank_size=32)
            model.gate.fit(torch.randn(24, 16))
            torch.save(model.state_dict(), checkpoint)
            inputs = folder / 'pairs.csv'
            pd.DataFrame({'SMILES': compounds, 'Protein': proteins}).to_csv(inputs, index=False)
            output = folder / 'predictions.csv'
            command = [sys.executable, '-B', str(root / 'predict.py'), '--checkpoint', str(checkpoint), '--data-root', str(folder / 'data'), '--dataset', 'human', '--csv', str(inputs), '--output', str(output), '--batch-size', '2', '--device', 'cpu', '--order-seed', '7', '--trust-input-files']
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            run = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = pd.read_csv(output)
            self.assertEqual(result['SMILES'].tolist(), compounds)
            self.assertEqual(result['row_index'].tolist(), [0, 1])
            self.assertNotIn('y', result.columns)
            self.assertNotIn('Y', result.columns)
            self.assertTrue(np.isfinite(result[['p1', 'u', 'm']].values).all())
            metadata = json.loads(output.with_suffix('.csv.metadata.json').read_text(encoding='utf-8'))
            self.assertFalse(metadata['labels_supplied'])
            self.assertEqual(metadata['support_bank_filled'], 24)
            original = inputs.read_bytes()
            unsafe_command = command.copy()
            unsafe_command[unsafe_command.index('--output') + 1] = str(inputs)
            unsafe = subprocess.run(unsafe_command + ['--overwrite'], capture_output=True, text=True, env=environment, timeout=30)
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertEqual(inputs.read_bytes(), original)

if __name__ == '__main__':
    unittest.main()
