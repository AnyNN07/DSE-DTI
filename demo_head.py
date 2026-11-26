import json

import torch

from dse_head import DirectionScaleHead, SupportGate, dse_loss, readout

def main():
    torch.manual_seed(42)
    head = DirectionScaleHead().eval()
    gate = SupportGate(dim=16, bank_size=64, k=8)
    gate.fit(torch.randn(64, 16))
    features = torch.randn(4, 16)
    with torch.no_grad():
        pi, raw_mass, alpha = head(features)
        factor = torch.tensor([1.0, 0.5, 0.1, 0.01])
        pi_adjusted, effective_mass, _ = head(features, gate=factor)
        before = readout(pi, raw_mass)
        after = readout(pi_adjusted, effective_mass)
    torch.testing.assert_close(before['p1'], after['p1'])
    result = {
        'demo': 'Synthetic head calculation, not a trained molecular prediction',
        'class_order': ['non_binding', 'binding', 'unknown'],
        'binding_probability': before['p1'].tolist(),
        'binding_probability_after_scale_adjustment': after['p1'].tolist(),
        'raw_uncertainty': before['u'].tolist(),
        'adjusted_uncertainty': after['u'].tolist(),
        'finite_training_loss': float(dse_loss(alpha, torch.tensor([0, 1, 0, 1]))),
        'fitted_support_threshold': float(gate.tau.item()),
    }
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
