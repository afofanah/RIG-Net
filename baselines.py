"""
baselines.py - compact PyTorch re-implementations of 15 forecasting baselines (2020-2026).
All share one interface:   forward(x) -> y_hat
    x     : [B, L, N]  standardised log1p weekly incidence (lookback L, N regions)
    y_hat : [B, H, N]  same space, H horizons
Graph models additionally receive the fixed geographic adjacency `A` (N x N) and
static node features `S` (N x F) at construction.

These are faithful to each paper's *core mechanism* but sized for tiny epidemic
panels (N<=61, T<=99). They are not the authors' official code.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def row_norm(A, self_loop=True):
    A = A + torch.eye(A.shape[-1], device=A.device) if self_loop else A
    return A / A.sum(-1, keepdim=True).clamp_min(1e-8)


class MovingAvg(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k
    def forward(self, x):                      # x [B, L, C]
        p = (self.k - 1) // 2
        front = x[:, :1].repeat(1, self.k - 1 - p, 1); end = x[:, -1:].repeat(1, p, 1)
        xx = torch.cat([front, x, end], 1)
        return F.avg_pool1d(xx.transpose(1, 2), self.k, 1).transpose(1, 2)


# ============================================================ 1. N-BEATS (ICLR 2020)
class NBEATS(nn.Module):
    def __init__(self, L, H, N, d=64, n_blocks=3, **kw):
        super().__init__(); self.L, self.H = L, H
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(L, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(),
                                                   nn.Linear(d, L + H)) for _ in range(n_blocks)])
    def forward(self, x):
        r = x.transpose(1, 2); f = 0                 # channel-independent [B,N,L]
        for b in self.blocks:
            o = b(r); r = r - o[..., :self.L]; f = f + o[..., self.L:]
        return f.transpose(1, 2)


# ============================================================ 2. N-HiTS (AAAI 2023)
class NHITS(nn.Module):
    def __init__(self, L, H, N, d=64, pools=(3, 2, 1), **kw):
        super().__init__(); self.L, self.H, self.pools = L, H, pools
        self.blocks = nn.ModuleList()
        for p in pools:
            Lp = math.ceil(L / p); Hc = max(1, math.ceil(H / p))
            self.blocks.append(nn.Sequential(nn.Linear(Lp, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, L + Hc)))
    def forward(self, x):
        r = x.transpose(1, 2); f = 0; B, N, L = r.shape
        for p, b in zip(self.pools, self.blocks):
            rp = F.max_pool1d(r.reshape(B * N, 1, L), p, p, ceil_mode=True).reshape(B, N, -1)
            o = b(rp); back, fc = o[..., :L], o[..., L:]
            fc = F.interpolate(fc.reshape(B * N, 1, -1), size=self.H, mode="linear", align_corners=True).reshape(B, N, self.H)
            r = r - back; f = f + fc
        return f.transpose(1, 2)


# ============================================================ 3. Autoformer (NeurIPS 2021)
class AutoCorrelation(nn.Module):
    def __init__(self, d, top_k=2):
        super().__init__(); self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4)); self.top_k = top_k
    def forward(self, x):                      # [B, L, d]
        q, k, v = self.q(x), self.k(x), self.v(x); L = x.shape[1]
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=1) * torch.conj(torch.fft.rfft(k, dim=1)), n=L, dim=1).mean(-1)  # [B,L]
        k_ = min(self.top_k, L); w, lag = torch.topk(corr, k_, -1); w = torch.softmax(w, -1)
        out = 0
        for i in range(k_):
            idx = (torch.arange(L, device=x.device)[None] + lag[:, i:i + 1]) % L
            out = out + w[:, i, None, None] * torch.gather(v, 1, idx[..., None].expand_as(v))
        return self.o(out)

class Autoformer(nn.Module):
    def __init__(self, L, H, N, d=32, **kw):
        super().__init__(); self.decomp = MovingAvg(3); self.emb = nn.Linear(N, d)
        self.ac = AutoCorrelation(d); self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.seas_head = nn.Linear(L * d, H * N); self.trend_head = nn.Linear(L, H); self.N, self.H = N, H
    def forward(self, x):
        trend = self.decomp(x); seas = x - trend
        h = self.emb(seas); h = h + self.ac(h); h = h - self.decomp(h); h = h + self.ff(h); h = h - self.decomp(h)
        s = self.seas_head(h.flatten(1)).view(-1, self.H, self.N)
        t = self.trend_head(trend.transpose(1, 2)).transpose(1, 2)
        return s + t


# ============================================================ 4. DLinear (AAAI 2023)
class DLinear(nn.Module):
    def __init__(self, L, H, N, **kw):
        super().__init__(); self.decomp = MovingAvg(3); self.ls = nn.Linear(L, H); self.lt = nn.Linear(L, H)
    def forward(self, x):
        t = self.decomp(x); s = x - t
        return (self.ls(s.transpose(1, 2)) + self.lt(t.transpose(1, 2))).transpose(1, 2)


# ============================================================ 5. PatchTST (ICLR 2023)
class PatchTST(nn.Module):
    def __init__(self, L, H, N, d=32, patch=2, stride=1, **kw):
        super().__init__(); self.patch, self.stride = patch, stride
        P = (L - patch) // stride + 1
        self.emb = nn.Linear(patch, d); self.pos = nn.Parameter(torch.randn(P, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 2)
        self.head = nn.Linear(P * d, H)
    def forward(self, x):
        B, L, N = x.shape; z = x.transpose(1, 2).reshape(B * N, L)
        mu = z.mean(-1, keepdim=True); sd = z.std(-1, keepdim=True) + 1e-4; z = (z - mu) / sd   # RevIN
        p = z.unfold(-1, self.patch, self.stride)
        h = self.enc(self.emb(p) + self.pos)
        y = self.head(h.flatten(1)) * sd + mu
        return y.view(B, N, -1).transpose(1, 2)


# ============================================================ 6. TimesNet (ICLR 2023)
class TimesBlock(nn.Module):
    def __init__(self, d, k=2):
        super().__init__(); self.k = k
        self.conv = nn.ModuleList([nn.Conv2d(d, d, ks, padding=ks // 2) for ks in (1, 3)])
    def forward(self, x):                      # [B, T, d]
        B, T, d = x.shape
        amp = torch.fft.rfft(x, dim=1).abs().mean(0).mean(-1); amp[0] = 0
        top = torch.topk(amp, min(self.k, len(amp) - 1)).indices.clamp_min(1)
        periods = [max(1, T // int(f)) for f in top]; w = torch.softmax(amp[top], 0)
        out = 0
        for p, wi in zip(periods, w):
            Tp = math.ceil(T / p) * p
            z = F.pad(x, (0, 0, 0, Tp - T)).reshape(B, Tp // p, p, d).permute(0, 3, 1, 2)
            z = sum(c(z) for c in self.conv) / 2
            out = out + wi * F.gelu(z).permute(0, 2, 3, 1).reshape(B, Tp, d)[:, :T]
        return x + out

class TimesNet(nn.Module):
    def __init__(self, L, H, N, d=32, **kw):
        super().__init__(); self.L, self.H = L, H
        self.emb = nn.Linear(N, d); self.extend = nn.Linear(L, L + H)
        self.blocks = nn.ModuleList([TimesBlock(d), TimesBlock(d)]); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, N)
    def forward(self, x):
        h = self.emb(x); h = self.extend(h.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks: h = self.ln(b(h))
        return self.out(h)[:, -self.H:]


# ============================================================ 7. TSMixer (TMLR 2023)
class TSMixer(nn.Module):
    def __init__(self, L, H, N, d=64, n_layers=2, **kw):
        super().__init__()
        self.time = nn.ModuleList([nn.Sequential(nn.Linear(L, L), nn.ReLU(), nn.Dropout(0.1)) for _ in range(n_layers)])
        self.feat = nn.ModuleList([nn.Sequential(nn.Linear(N, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, N)) for _ in range(n_layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(N) for _ in range(n_layers)]); self.ln2 = nn.ModuleList([nn.LayerNorm(N) for _ in range(n_layers)])
        self.proj = nn.Linear(L, H)
    def forward(self, x):
        for t, f, a, b in zip(self.time, self.feat, self.ln1, self.ln2):
            x = x + t(a(x).transpose(1, 2)).transpose(1, 2)
            x = x + f(b(x))
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


# ============================================================ 8. iTransformer (ICLR 2024)
class iTransformer(nn.Module):
    def __init__(self, L, H, N, d=32, **kw):
        super().__init__(); self.emb = nn.Linear(L, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), 2)
        self.head = nn.Linear(d, H)
    def forward(self, x):
        mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-4; z = (x - mu) / sd
        h = self.enc(self.emb(z.transpose(1, 2)))        # tokens = regions
        return self.head(h).transpose(1, 2) * sd + mu


# ============================================================ 9. AGCRN (NeurIPS 2020)
class AVWGCN(nn.Module):
    def __init__(self, cin, cout, emb, cheb_k=2):
        super().__init__(); self.k = cheb_k
        self.W = nn.Parameter(torch.randn(emb, cheb_k, cin, cout) * 0.05); self.b = nn.Parameter(torch.zeros(emb, cout))
    def forward(self, x, E):                   # x [B,N,cin]
        A = torch.softmax(F.relu(E @ E.T), -1)
        sup = [torch.eye(A.shape[0], device=x.device), A]
        xg = torch.stack([torch.einsum("nm,bmc->bnc", s, x) for s in sup[:self.k]], 2)   # [B,N,k,cin]
        W = torch.einsum("nd,dkio->nkio", E, self.W); b = E @ self.b
        return torch.einsum("bnki,nkio->bno", xg, W) + b

class AGCRN(nn.Module):
    def __init__(self, L, H, N, d=32, emb=8, **kw):
        super().__init__(); self.E = nn.Parameter(torch.randn(N, emb) * 0.3); self.d = d
        self.gate = AVWGCN(1 + d, 2 * d, emb); self.upd = AVWGCN(1 + d, d, emb)
        self.end = nn.Conv2d(1, H, (1, d))
    def forward(self, x):
        B, L, N = x.shape; h = x.new_zeros(B, N, self.d)
        for t in range(L):
            xt = x[:, t, :, None]
            zr = torch.sigmoid(self.gate(torch.cat([xt, h], -1), self.E)); z, r = zr.chunk(2, -1)
            c = torch.tanh(self.upd(torch.cat([xt, r * h], -1), self.E)); h = z * h + (1 - z) * c
        return self.end(h[:, None]).squeeze(-1)


# ============================================================ 10. MTGNN (KDD 2020)
class GraphLearner(nn.Module):
    def __init__(self, N, d=16, alpha=3.0, k=8):
        super().__init__(); self.e1 = nn.Embedding(N, d); self.e2 = nn.Embedding(N, d)
        self.l1 = nn.Linear(d, d); self.l2 = nn.Linear(d, d); self.a, self.k = alpha, min(k, N)
    def forward(self):
        idx = torch.arange(self.e1.num_embeddings, device=self.e1.weight.device)
        m1 = torch.tanh(self.a * self.l1(self.e1(idx))); m2 = torch.tanh(self.a * self.l2(self.e2(idx)))
        A = F.relu(torch.tanh(self.a * (m1 @ m2.T - m2 @ m1.T)))
        mask = torch.zeros_like(A).scatter_(1, A.topk(self.k, 1).indices, 1.)
        return A * mask

class MixProp(nn.Module):
    def __init__(self, c, depth=2, beta=0.05):
        super().__init__(); self.depth, self.beta = depth, beta; self.mlp = nn.Conv2d((depth + 1) * c, c, 1)
    def forward(self, x, A):                   # x [B,c,N,T]
        A = row_norm(A); h = x; out = [h]
        for _ in range(self.depth):
            h = self.beta * x + (1 - self.beta) * torch.einsum("bcnt,mn->bcmt", h, A); out.append(h)
        return self.mlp(torch.cat(out, 1))

class MTGNN(nn.Module):
    def __init__(self, L, H, N, d=16, **kw):
        super().__init__(); self.gl = GraphLearner(N); self.start = nn.Conv2d(1, d, 1)
        self.f = nn.ModuleList([nn.Conv2d(d, d // 2, (1, k), padding=(0, k // 2)) for k in (1, 3)])
        self.g = nn.ModuleList([nn.Conv2d(d, d // 2, (1, k), padding=(0, k // 2)) for k in (1, 3)])
        self.gc1 = MixProp(d); self.gc2 = MixProp(d); self.ln = nn.LayerNorm(d)
        self.end = nn.Sequential(nn.Conv2d(d * L, 2 * d, 1), nn.ReLU(), nn.Conv2d(2 * d, H, 1))
    def forward(self, x):
        A = self.gl(); h = self.start(x.transpose(1, 2)[:, None])        # [B,d,N,L]
        res = h
        filt = torch.tanh(torch.cat([c(h) for c in self.f], 1)); gate = torch.sigmoid(torch.cat([c(h) for c in self.g], 1))
        h = filt * gate
        h = self.gc1(h, A) + self.gc2(h, A.T) + res
        h = self.ln(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, d, N, L = h.shape
        return self.end(h.permute(0, 1, 3, 2).reshape(B, d * L, N, 1)).squeeze(-1)


# ============================================================ 11. Cola-GNN (CIKM 2020)
class ColaGNN(nn.Module):
    def __init__(self, L, H, N, A=None, d=32, **kw):
        super().__init__(); self.register_buffer("Ageo", row_norm(A, False) if A is not None else torch.zeros(N, N))
        self.rnn = nn.GRU(1, d, batch_first=True); self.W1 = nn.Linear(d, d, bias=False); self.W2 = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1); self.gate = nn.Linear(d, 1)
        self.conv = nn.ModuleList([nn.Conv1d(1, 8, k, dilation=dl) for k, dl in ((2, 1), (2, 2))])
        self.gcn = nn.Linear(16, d); self.out = nn.Linear(2 * d, H)
    def forward(self, x):
        B, L, N = x.shape; z = x.transpose(1, 2).reshape(B * N, L, 1)
        _, h = self.rnn(z); h = h[-1].view(B, N, -1)
        a = self.v(torch.tanh(self.W1(h)[:, :, None] + self.W2(h)[:, None])).squeeze(-1)       # cross-location attention
        a = torch.softmax(a, -1); g = torch.sigmoid(self.gate(h))
        Am = g * a + (1 - g) * self.Ageo
        feats = torch.cat([F.relu(c(z.transpose(1, 2))).amax(-1) for c in self.conv], -1).view(B, N, -1)
        hg = F.elu(self.gcn(torch.einsum("bnm,bmc->bnc", Am, feats)))
        return self.out(torch.cat([h, hg], -1)).transpose(1, 2)


# ============================================================ 12. STID (CIKM 2022)
class STID(nn.Module):
    def __init__(self, L, H, N, d=32, **kw):
        super().__init__(); self.node = nn.Parameter(torch.randn(N, d) * 0.1); self.inp = nn.Linear(L, d)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2 * d, 2 * d)) for _ in range(3)])
        self.out = nn.Linear(2 * d, H)
    def forward(self, x):
        h = torch.cat([self.inp(x.transpose(1, 2)), self.node.expand(x.shape[0], -1, -1)], -1)
        for m in self.mlp: h = h + m(h)
        return self.out(h).transpose(1, 2)


# ============================================================ 13. EpiGNN (ECML-PKDD 2022)
class EpiGNN(nn.Module):
    def __init__(self, L, H, N, A=None, d=32, **kw):
        super().__init__()
        self.register_buffer("Ageo", (A > 0).float() if A is not None else torch.zeros(N, N))
        self.local = nn.ModuleList([nn.Conv1d(1, 8, k, padding=k // 2) for k in (1, 3, 5)])     # local transmission risk
        self.glob = nn.Conv1d(1, 8, L)                                                       # global transmission risk
        dd = 32; self.proj = nn.Linear(24 + 8, dd)
        self.q = nn.Linear(dd, dd); self.k = nn.Linear(dd, dd)
        self.deg = nn.Linear(1, 1)
        self.gcn1 = nn.Linear(dd, dd); self.gcn2 = nn.Linear(dd, dd)
        self.out = nn.Linear(2 * dd + L, H)
    def forward(self, x):
        B, L, N = x.shape; z = x.transpose(1, 2).reshape(B * N, 1, L)
        loc = torch.cat([F.relu(c(z))[..., :L].amax(-1) for c in self.local], -1); gl = F.relu(self.glob(z)).squeeze(-1)
        h = self.proj(torch.cat([loc, gl], -1)).view(B, N, -1)
        att = torch.sigmoid(self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(h.shape[-1]))
        degs = self.Ageo.sum(-1, keepdim=True); prior = self.Ageo + torch.sigmoid(self.deg(degs))  # region-aware prior
        A = row_norm(att * prior[None])
        g = F.elu(self.gcn1(A @ h)); g = F.elu(self.gcn2(A @ g))
        return self.out(torch.cat([h, g, x.transpose(1, 2)], -1)).transpose(1, 2)


# ============================================================ 14. MepoGNN (ECML-PKDD 2022 / KAIS 2023)
class MepoGNN(nn.Module):
    """Spatio-temporal NN predicts SIR parameters; learned mobility graph couples a
    metapopulation SIR that is integrated forward. Works in count space internally."""
    def __init__(self, L, H, N, A=None, d=32, pop=None, scaler=None, **kw):
        super().__init__(); self.H = H; self.scaler = scaler
        self.logpop = nn.Parameter(torch.log(pop) if pop is not None else torch.full((N,), math.log(1e5)))
        self.E1 = nn.Parameter(torch.randn(N, 8) * 0.3); self.E2 = nn.Parameter(torch.randn(N, 8) * 0.3)
        self.rnn = nn.GRU(1, d, batch_first=True); self.par = nn.Linear(d, 2 * H)
    def forward(self, x):
        B, L, N = x.shape; mu, sd = self.scaler
        y = torch.expm1((x * sd + mu).clamp_min(0))                           # counts
        _, h = self.rnn(x.transpose(1, 2).reshape(B * N, L, 1)); h = h[-1].view(B, N, -1)
        p = self.par(h).view(B, N, 2, self.H); beta = 2.0 * torch.sigmoid(p[:, :, 0]); gamma = torch.sigmoid(p[:, :, 1])
        M = torch.softmax(F.relu(self.E1 @ self.E2.T), -1)                    # mobility (learned)
        pop = self.logpop.exp(); I = y[:, -2:].sum(1); R = y.sum(1); S = (pop - I - R).clamp_min(1.0)
        outs = []
        for t in range(self.H):
            force = (I @ M) / pop                                             # mobility-coupled infectious prevalence
            new = (beta[..., t] * S * force).clamp_max(S); rec = gamma[..., t] * I
            S = S - new; I = I + new - rec; outs.append(new)
        out = torch.stack(outs, 1)
        return (torch.log1p(out) - mu) / sd


# ============================================================ 15. STAEformer (CIKM 2023)
class STAEformer(nn.Module):
    def __init__(self, L, H, N, d=24, da=16, **kw):
        super().__init__(); self.inp = nn.Linear(1, d); self.adp = nn.Parameter(torch.randn(L, N, da) * 0.1)
        D = d + da
        self.t_att = nn.TransformerEncoderLayer(D, 4, 2 * D, 0.1, batch_first=True)
        self.s_att = nn.TransformerEncoderLayer(D, 4, 2 * D, 0.1, batch_first=True)
        self.out = nn.Linear(L * D, H)
    def forward(self, x):
        B, L, N = x.shape; h = torch.cat([self.inp(x[..., None]), self.adp.expand(B, -1, -1, -1)], -1)   # [B,L,N,D]
        D = h.shape[-1]
        h = self.t_att(h.permute(0, 2, 1, 3).reshape(B * N, L, D)).view(B, N, L, D)
        h = self.s_att(h.permute(0, 2, 1, 3).reshape(B * L, N, D)).view(B, L, N, D)
        return self.out(h.permute(0, 2, 1, 3).reshape(B, N, L * D)).transpose(1, 2)


class MASTNet(nn.Module):
    """MAST-Net (Fofanah, Wen & Chen, JNCA 2026), compact re-implementation.

    Built from the published description of the three stated innovations, since no code or
    full text was available: (1) a manifold-preserving learnable augmentation whose strength
    adapts to the regime, with a penalty that keeps augmented node states close to their
    pre-augmentation neighbourhood geometry; (2) a *parallel* multi-scale graph convolution,
    i.e. several propagation orders computed side by side and fused, rather than stacked;
    (3) a Lipschitz-constrained temporal module, here a GRU whose recurrent weight is rescaled
    to a spectral-norm budget so the temporal map is non-expansive.
    """

    def __init__(self, L, H, N, d=48, scales=(1, 2, 3), lip=0.95, **kw):
        super().__init__()
        self.N, self.L, self.H, self.scales, self.lip = N, L, H, tuple(scales), lip
        self.inp = nn.Linear(1, d)
        self.emb = nn.Parameter(torch.randn(N, d) * 0.02)
        # (1) learnable augmentation: per-node shift/scale with an adaptive gate
        self.aug_mu = nn.Parameter(torch.zeros(N, d))
        self.aug_gate = nn.Linear(d, 1)
        # (2) parallel multi-scale graph convolution over a learned adjacency
        self.e1 = nn.Parameter(torch.randn(N, d) * 0.02)
        self.e2 = nn.Parameter(torch.randn(N, d) * 0.02)
        self.branch = nn.ModuleList([nn.Linear(d, d) for _ in self.scales])
        self.fuse = nn.Linear(d * len(self.scales), d)
        # (3) Lipschitz-constrained temporal model
        self.gru = nn.GRU(d, d, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, H)

    def _lipschitz_(self):
        """Rescale the recurrent weight so its spectral norm stays within the budget."""
        with torch.no_grad():
            W = self.gru.weight_hh_l0
            sn = torch.linalg.matrix_norm(W, 2)
            if sn > self.lip: W.mul_(self.lip / sn)

    def forward(self, x):
        B, L, N = x.shape
        self._lipschitz_()
        h = self.inp(x[..., None]) + self.emb                       # [B,L,N,d]
        # (1) manifold-preserving augmentation, strength adapted per node and step
        a = torch.sigmoid(self.aug_gate(h))                          # [B,L,N,1]
        h_aug = h + a * self.aug_mu
        self.aug_pen = (a * self.aug_mu.pow(2).sum(-1, keepdim=True)).mean()
        h = h_aug
        # (2) parallel multi-scale propagation on a learned adjacency
        A = torch.softmax(F.relu(self.e1 @ self.e2.T), -1)
        outs, Ak = [], torch.eye(N, device=x.device)
        for k, lin in zip(self.scales, self.branch):
            Ak = A if k == 1 else Ak @ A
            outs.append(lin(torch.einsum("nm,blmd->blnd", Ak, h)))
        h = self.norm(h + F.gelu(self.fuse(torch.cat(outs, -1))))
        # (3) non-expansive temporal pass
        z = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        o, _ = self.gru(z)
        return self.out(o[:, -1]).view(B, N, self.H).transpose(1, 2)


BASELINES = {
    # name: (class, year, venue, family)
    "N-BEATS": (NBEATS, 2020, "ICLR", "temporal"),
    "AGCRN": (AGCRN, 2020, "NeurIPS", "graph"),
    "MTGNN": (MTGNN, 2020, "KDD", "graph"),
    "Cola-GNN": (ColaGNN, 2020, "CIKM", "graph"),
    "Autoformer": (Autoformer, 2021, "NeurIPS", "temporal"),
    "EpiGNN": (EpiGNN, 2022, "ECML-PKDD", "graph"),
    "STID": (STID, 2022, "CIKM", "graph"),
    "MepoGNN": (MepoGNN, 2022, "ECML-PKDD", "graph"),
    "DLinear": (DLinear, 2023, "AAAI", "temporal"),
    "N-HiTS": (NHITS, 2023, "AAAI", "temporal"),
    "PatchTST": (PatchTST, 2023, "ICLR", "temporal"),
    "TimesNet": (TimesNet, 2023, "ICLR", "temporal"),
    "TSMixer": (TSMixer, 2023, "TMLR", "temporal"),
    "STAEformer": (STAEformer, 2023, "CIKM", "graph"),
    "iTransformer": (iTransformer, 2024, "ICLR", "temporal"),
    "MAST-Net": (MASTNet, 2026, "JNCA", "graph"),
}
