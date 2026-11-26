import os
import sys

import torch
import torch.nn as nn

def find_vlm_code():
    here = os.path.dirname(os.path.abspath(__file__))
    backbone = os.path.join(here, 'backbone')
    if not os.path.isfile(os.path.join(backbone, 'VLMNET', 'model.py')):
        raise FileNotFoundError('The bundled DSE-DTI backbone is missing')
    return backbone

VLM_CODE = find_vlm_code()
for p in (VLM_CODE, os.path.join(VLM_CODE, 'VLMNET')):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import UniQCM2Net
from dse_head import DirectionScaleHead, SupportGate, readout

JOINT_DIM = 16

class DSENet(nn.Module):
    def __init__(self, gnn_hidden=128, k=5, fusion_mode='VLM-XA',
                 vlm_ablation='no_attn', m_min=1e-2, m_max=1e3,
                 bank_size=4096, knn=8, head_hidden=64, dropout=0.1,
                 gate_in_training=False):
        super().__init__()
        self.backbone = UniQCM2Net(gnn_hidden=gnn_hidden, k=k,
                                   fusion_mode=fusion_mode,
                                   vlm_ablation=vlm_ablation)

        self.backbone.fc = nn.Identity()
        self.head = DirectionScaleHead(JOINT_DIM, hidden=head_hidden,
                                       dropout=dropout, m_min=m_min, m_max=m_max)
        self.gate = SupportGate(dim=JOINT_DIM, bank_size=bank_size, k=knn)
        self.gate_in_training = gate_in_training

        self._joint = None
        self.backbone.ban.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._joint = output

    def forward(self, bg_drug, bg_prot, d_feat, p_feat, d_lens, p_lens,
                use_gate=None):

        self._joint = None
        _, cl_loss = self.backbone(bg_drug, bg_prot, d_feat, p_feat, d_lens, p_lens)
        h = self._joint
        if h is None:
            raise RuntimeError('BAN hook did not fire -- the released forward '
                               'no longer routes through self.ban')
        if use_gate is None:
            use_gate = self.gate_in_training or (not self.training)
        gate = self.gate(h) if use_gate else None
        pi, m, alpha = self.head(h, gate=gate)
        return dict(pi=pi, m=m, alpha=alpha, cl=cl_loss, h=h,
                    gate=(gate if gate is not None else torch.ones_like(m)))

    @torch.no_grad()
    def refit_gate(self, loader, device, max_batches=None):

        was_training = self.training
        self.eval()
        embs = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            bg_d, bg_p, df_, pf_, _, dl_, pl_ = batch
            self._joint = None
            self.backbone(bg_d.to(device), bg_p.to(device), df_.to(device),
                          pf_.to(device), dl_.to(device), pl_.to(device))
            embs.append(self._joint.detach().float().cpu())
        if embs:
            self.gate.fit(torch.cat(embs).to(next(self.parameters()).device))
        if was_training:
            self.train()

__all__ = ['DSENet', 'readout', 'JOINT_DIM']
