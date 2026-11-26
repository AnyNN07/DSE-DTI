import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NOBIND, BIND, UNKNOWN = 0, 1, 2
K = 3

class DirectionScaleHead(nn.Module):

    def __init__(self, in_dim=16, hidden=64, dropout=0.1,
                 m_min=1e-2, m_max=1e3):
        super().__init__()
        if not 0 < m_min < m_max:
            raise ValueError('need 0 < m_min < m_max')
        self.lo = math.log(m_min)
        self.hi = math.log(m_max)

        self.direction = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, K))
        self.K = K

        self.scale = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, h, gate=None):

        pi = F.softmax(self.direction(h), dim=-1)
        log_m = self.lo + (self.hi - self.lo) * torch.sigmoid(self.scale(h).squeeze(-1))
        m = torch.exp(log_m)
        if gate is not None:
            m = m * gate.clamp_min(1e-6)
        alpha = m.unsqueeze(-1) * pi + 1.0
        return pi, m, alpha

def readout(pi, m):

    S = m + K
    denom = (pi[..., BIND] + pi[..., NOBIND]).clamp_min(1e-12)
    return dict(S=S, u=K / S, m=m,
                p1=pi[..., BIND] / denom,
                p_unknown=(pi[..., UNKNOWN] if pi.shape[-1] > 2
                           else torch.zeros_like(pi[..., BIND])),

                p1_dirichlet_mean=(m * pi[..., BIND] + 1.0) / S)

class SupportGate(nn.Module):

    def __init__(self, dim=16, bank_size=4096, k=8):
        super().__init__()
        self.k = k
        self.register_buffer('bank', torch.zeros(bank_size, dim))
        self.register_buffer('filled', torch.zeros(1, dtype=torch.long))
        self.register_buffer('tau', torch.ones(1))

    @torch.no_grad()
    def fit(self, embeddings):

        n = min(len(embeddings), self.bank.shape[0])
        idx = torch.randperm(len(embeddings), device=embeddings.device)[:n]
        self.bank[:n] = embeddings[idx].to(self.bank.dtype)
        self.filled[0] = n
        bank = self.bank[:n].float()
        kk = min(self.k + 1, n)
        d = torch.cdist(bank, bank).topk(kk, dim=1, largest=False).values[:, 1:].mean(1)
        self.tau[0] = d.median().clamp_min(1e-6)

    @torch.no_grad()
    def _dist(self, h):
        bank = self.bank[:int(self.filled[0])]
        d = torch.cdist(h.float(), bank.float())
        kk = min(self.k, bank.shape[0])
        return d.topk(kk, dim=1, largest=False).values.mean(1)

    def forward(self, h):
        if int(self.filled[0]) == 0:
            return torch.ones(h.shape[0], device=h.device)
        d = self._dist(h)
        return torch.exp(-((d - self.tau).clamp_min(0.0) / self.tau).pow(2))

    @torch.no_grad()
    def distance(self, h):

        if int(self.filled[0]) == 0:
            return torch.zeros(h.shape[0], device=h.device)
        return self._dist(h)

def dse_loss(alpha, y, lam=0.1):

    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    onehot = F.one_hot(y.long(), K).float()
    fit = ((onehot - p) ** 2).sum(-1)
    var = (p * (1.0 - p) / (S + 1.0)).sum(-1)

    keep = 1.0 - onehot
    if alpha.shape[-1] > 2:
        keep[:, UNKNOWN] = 0.0
    a_t = 1.0 + keep * (alpha - 1.0)
    kl = _kl_dirichlet_uniform(a_t)
    return (fit + var + lam * kl).mean()

def _kl_dirichlet_uniform(a):
    S = a.sum(-1, keepdim=True)
    t1 = torch.lgamma(S).squeeze(-1) - torch.lgamma(a).sum(-1)
    t2 = torch.lgamma(torch.tensor(float(K), device=a.device))
    t3 = ((a - 1.0) * (torch.digamma(a) - torch.digamma(S))).sum(-1)
    return t1 - t2 + t3
