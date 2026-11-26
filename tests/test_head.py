import io
import math
import unittest

import torch

from dse_head import BIND, K, NOBIND, UNKNOWN, DirectionScaleHead, SupportGate, dse_loss, readout

class DirectionScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_label_order_and_readout(self):
        self.assertEqual((NOBIND, BIND, UNKNOWN, K), (0, 1, 2, 3))
        pi = torch.tensor([[0.1, 0.8, 0.1]])
        mass = torch.tensor([7.0])
        result = readout(pi, mass)
        torch.testing.assert_close(result['p1'], torch.tensor([8.0 / 9.0]))
        torch.testing.assert_close(result['u'], torch.tensor([0.3]))
        torch.testing.assert_close(result['S'], torch.tensor([10.0]))

    def test_probability_is_invariant_to_scale(self):
        pi = torch.tensor([[0.1, 0.8, 0.1]]).expand(4, -1)
        mass = torch.tensor([0.01, 1.0, 10.0, 1000.0])
        result = readout(pi, mass)
        torch.testing.assert_close(result['p1'], result['p1'][0].expand(4))
        self.assertTrue(bool(torch.all(result['u'][1:] < result['u'][:-1])))

    def test_uncertainty_is_invariant_to_direction(self):
        pi = torch.softmax(torch.randn(32, K), dim=-1)
        result = readout(pi, torch.full((32,), 7.0))
        torch.testing.assert_close(result['u'], torch.full((32,), 0.3))
        self.assertGreater((result['p1'].max() - result['p1'].min()).item(), 0.1)

    def test_gating_changes_mass_not_direction(self):
        head = DirectionScaleHead(in_dim=16, hidden=64, dropout=0.1).eval()
        features = torch.randn(12, 16)
        with torch.no_grad():
            pi_raw, mass_raw, alpha_raw = head(features)
            pi_gated, mass_gated, alpha_gated = head(features, gate=torch.full((12,), 0.2))
        torch.testing.assert_close(pi_raw, pi_gated)
        torch.testing.assert_close(mass_gated, mass_raw * 0.2)
        torch.testing.assert_close(alpha_raw, 1.0 + mass_raw[:, None] * pi_raw)
        torch.testing.assert_close(alpha_gated, 1.0 + mass_gated[:, None] * pi_gated)
        raw = readout(pi_raw, mass_raw)
        gated = readout(pi_gated, mass_gated)
        torch.testing.assert_close(raw['p1'], gated['p1'])
        self.assertTrue(bool(torch.all(gated['u'] > raw['u'])))

    def test_loss_is_finite_and_backpropagates(self):
        head = DirectionScaleHead(in_dim=16, hidden=64, dropout=0.0)
        features = torch.randn(20, 16)
        labels = torch.arange(20) % 2
        _, _, alpha = head(features)
        loss = dse_loss(alpha, labels, lam=0.1)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        gradients = [p.grad for p in head.parameters()]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.0)

    def test_support_threshold_excludes_self_distances(self):
        gate = SupportGate(dim=1, bank_size=3, k=1)
        gate.fit(torch.tensor([[0.0], [2.0], [5.0]]))
        self.assertEqual(int(gate.filled.item()), 3)
        torch.testing.assert_close(gate.tau, torch.tensor([2.0]))
        queries = torch.tensor([[1.0], [8.0]])
        torch.testing.assert_close(gate.distance(queries), torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(gate(queries), torch.tensor([1.0, math.exp(-0.25)]))

    def test_state_dict_roundtrip(self):
        head = DirectionScaleHead(in_dim=16, hidden=64, dropout=0.0).eval()
        gate = SupportGate(dim=16, bank_size=32, k=3)
        gate.fit(torch.randn(24, 16))
        features = torch.randn(6, 16)
        stream = io.BytesIO()
        torch.save({'head': head.state_dict(), 'gate': gate.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        restored_head = DirectionScaleHead(in_dim=16, hidden=64, dropout=0.0).eval()
        restored_gate = SupportGate(dim=16, bank_size=32, k=3)
        restored_head.load_state_dict(state['head'], strict=True)
        restored_gate.load_state_dict(state['gate'], strict=True)
        with torch.no_grad():
            original = head(features, gate=gate(features))
            restored = restored_head(features, gate=restored_gate(features))
        for expected, actual in zip(original, restored):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(restored_gate.tau, gate.tau)
        torch.testing.assert_close(restored_gate.bank, gate.bank)

if __name__ == '__main__':
    unittest.main()
