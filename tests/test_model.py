import importlib.util
import unittest

import torch

class ModelTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('dgl'), 'DGL is not installed')
    def test_synthetic_graph_forward_and_saved_gate(self):
        import dgl
        from dse_model import DSENet, readout
        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = DSENet(bank_size=32).eval()
        model.gate.fit(torch.randn(24, 16))

        def graph(nodes, node_key, node_width, edge_key, edge_width):
            indices = torch.arange(nodes)
            result = dgl.graph((indices, (indices + 1) % nodes))
            result.ndata[node_key] = torch.randn(nodes, node_width)
            result.edata[edge_key] = torch.randn(nodes, edge_width)
            return result

        drug = dgl.batch([graph(3, 'atom', 120, 'bond', 12), graph(4, 'atom', 120, 'bond', 12)])
        protein = dgl.batch([graph(6, 'feats', 41, 'feats', 5), graph(5, 'feats', 41, 'feats', 5)])
        drug_features = torch.randn(2, 7, 768)
        protein_features = torch.randn(2, 8, 320)
        drug_lengths = torch.tensor([7, 6])
        protein_lengths = torch.tensor([8, 7])
        with torch.no_grad():
            raw = model(drug, protein, drug_features, protein_features, drug_lengths, protein_lengths, use_gate=False)
            gated = model(drug, protein, drug_features, protein_features, drug_lengths, protein_lengths, use_gate=True)
        self.assertEqual(tuple(gated['pi'].shape), (2, 3))
        self.assertEqual(tuple(gated['h'].shape), (2, 16))
        torch.testing.assert_close(raw['pi'], gated['pi'])
        torch.testing.assert_close(readout(raw['pi'], raw['m'])['p1'], readout(gated['pi'], gated['m'])['p1'])
        self.assertTrue(bool(torch.isfinite(gated['alpha']).all()))
        self.assertTrue(bool(torch.all(gated['m'] <= raw['m'])))
        restored = DSENet(bank_size=32).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(int(restored.gate.filled.item()), 24)
        torch.testing.assert_close(restored.gate.tau, model.gate.tau)

if __name__ == '__main__':
    unittest.main()
