"""
RIG-Net : Renewal-Importation Graph Network for sub-national Ebola forecasting.

Idea. Deep spatio-temporal GNNs (Cola-GNN, EpiGNN, MTGNN...) map past counts straight
to future counts; mechanistic hybrids (MepoGNN) couple a GNN to an SIR model that needs
population denominators and a recovery rate that are poorly identified for Ebola.
RIG-Net instead couples a multi-graph GNN to the *renewal equation* with the
Ebola-specific generation-interval distribution, which is the quantity field
epidemiologists actually estimate (WHO Ebola Response Team, NEJM 2014).

For region i and forecast week t+h:
    P_i(t)   = sum_s w_s * y_i(t-s)                         (local infection pressure)
    Q_i(t)   = (1-rho_i) P_i(t) + rho_i * sum_j A_ij P_j(t)   (importation through fused graph A)
    mu_i(t+h)= [ R_i(t+h) * Q_i + iota_i ] * exp(delta_i(h))
    y_i      ~ NegBin(mu, phi_i)
Future y are rolled out autoregressively with mu, so the whole decoder is differentiable.

Components
 (C1) Renewal decoder with fixed Ebola generation interval  -> interpretable R_i(t), rho_i
 (C2) Tri-graph fusion: geographic kernel (cross-border), learned static graph,
      state-dependent attention graph, mixed by per-node softmax gates
 (C3) Negative-binomial likelihood for zero-inflated, over-dispersed counts
 (C4) Bounded residual correction delta for reporting effects not captured by C1
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ebola_generation_weights(n_weeks=5, mean_days=15.3, sd_days=9.3):
    k = (mean_days / sd_days) ** 2; theta = sd_days ** 2 / mean_days
    edges = torch.arange(0, n_weeks + 1, dtype=torch.float64) * 7.0
    cdf = torch.special.gammainc(torch.tensor(k, dtype=torch.float64), edges / theta)
    w = (cdf[1:] - cdf[:-1]); w = w / w.sum()
    return w.float()                                         # w[0] -> lag 1 week


class RIGNet(nn.Module):
    count_model = True

    def __init__(self, L, H, N, A=None, S=None, scaler=None, d=32, emb=8,
                 use_renewal=True, use_geo=True, use_adaptive=True, use_dynamic=True,
                 n_gi=5, R_max=4.0, clip_k=None, delta_scale=0.5, G=None, n_interv=0, gen_kl=0.0, **kw):
        super().__init__()
        self.L, self.H, self.N, self.scaler = L, H, N, scaler
        self.use_renewal, self.R_max = use_renewal, R_max
        self.clip_k, self.delta_scale = clip_k, delta_scale
        self.flags = dict(geo=use_geo, adp=use_adaptive, dyn=use_dynamic, gen=G is not None)
        # ---- genomic prior: transmission-linkage graph from genomes sampled before the fold cut-off
        Gm = torch.log1p(G) if G is not None else torch.zeros(N, N)
        self.register_buffer("Agen", Gm / Gm.sum(-1, keepdim=True).clamp_min(1e-8))
        self.register_buffer("gen_mask", (Gm.sum(-1) > 0))             # regions with genomic links
        self.gen_kl = gen_kl
        # ---- intervention-aware transmission modulation (beta_k >= 0 by projection)
        self.n_interv = n_interv
        if n_interv: self.beta_raw = nn.Parameter(torch.full((n_interv,), 0.05))  # small positive start; projection keeps it >= 0
        self.register_buffer("w", ebola_generation_weights(n_gi))
        Ageo = A if A is not None else torch.zeros(N, N)
        self.register_buffer("Ageo", Ageo / Ageo.sum(-1, keepdim=True).clamp_min(1e-8))
        S = S if S is not None else torch.zeros(N, 0)
        self.register_buffer("S", S)
        # ---- node identity: static covariates + free embedding
        self.E = nn.Parameter(torch.randn(N, emb) * 0.3)
        self.Es = nn.Linear(S.shape[1], emb) if S.shape[1] > 0 else None
        # ---- temporal encoder (per region): GRU + dilated causal conv
        self.gru = nn.GRU(1, d, batch_first=True)
        self.tconv = nn.Conv1d(1, d, 2, dilation=2)
        self.tproj = nn.Linear(2 * d + emb, d)
        # ---- graph modules
        self.E1 = nn.Linear(emb, emb, bias=False); self.E2 = nn.Linear(emb, emb, bias=False)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.gate = nn.Linear(d, 4)
        self.gnn = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        # ---- heads
        self.head_R = nn.Linear(2 * d, H)
        self.head_rho = nn.Linear(2 * d, 1)
        self.head_iota = nn.Linear(2 * d, 1)
        self.head_delta = nn.Linear(2 * d, H)
        self.head_direct = nn.Linear(2 * d, H)                 # used only when use_renewal=False
        self.log_phi = nn.Parameter(torch.zeros(N))
        nn.init.zeros_(self.head_delta.weight); nn.init.zeros_(self.head_delta.bias)
        nn.init.constant_(self.head_R.bias, -0.4)              # R_max*sigmoid(-0.4) ~ 1.6 at init

    def project(self):
        """Projected gradient step: keep intervention effects non-negative (monotone suppression)."""
        if self.n_interv:
            with torch.no_grad(): self.beta_raw.clamp_(min=0.0)

    def node_emb(self):
        e = self.E
        if self.Es is not None: e = e + self.Es(self.S)
        return e

    def graphs(self, h, e):
        B, N, _ = h.shape; mats = []
        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        if self.flags["geo"]:
            mats.append(self.Ageo.expand(B, N, N))
        else:
            mats.append(h.new_zeros(B, N, N))
        a = F.relu(self.E1(e) @ self.E2(e).T).masked_fill(eye, -1e9)
        mats.append(torch.softmax(a, -1).expand(B, N, N) if self.flags["adp"] else h.new_zeros(B, N, N))
        s = (self.q(h) @ self.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        mats.append(torch.softmax(s.masked_fill(eye, -1e9), -1) if self.flags["dyn"] else h.new_zeros(B, N, N))
        mats.append(self.Agen.expand(B, N, N) if self.flags["gen"] else h.new_zeros(B, N, N))
        avail = torch.tensor([self.flags["geo"], self.flags["adp"], self.flags["dyn"], self.flags["gen"]], device=h.device)
        avail = avail[None, None].expand(B, N, 4).clone()
        if self.flags["gen"]: avail[..., 3] = self.gen_mask[None].expand(B, N)    # only regions with genomes
        g = self.gate(h).masked_fill(~avail, -1e9); pi = torch.softmax(g, -1)          # [B,N,4]
        A = sum(pi[..., k:k + 1] * mats[k] for k in range(4))
        self._A_adp = mats[1][0] if self.flags["adp"] else None
        return A, pi

    def forward(self, x, u=None, return_diag=False):
        B, L, N = x.shape; mu_s, sd_s = self.scaler
        y = torch.expm1((x * sd_s + mu_s).clamp_min(0))                     # counts [B,L,N]
        z = x.transpose(1, 2).reshape(B * N, L, 1)
        _, hg = self.gru(z); hc = F.relu(self.tconv(z.transpose(1, 2))).mean(-1)
        e = self.node_emb()
        h = self.tproj(torch.cat([hg[-1].view(B, N, -1), hc.view(B, N, -1), e.expand(B, N, -1)], -1))
        A, pi = self.graphs(h, e)
        g = h
        for lin in self.gnn:
            g = self.norm(g + F.gelu(lin(A @ g)))
        r = torch.cat([h, g], -1)

        if not self.use_renewal:
            mu = F.softplus(self.head_direct(r)).transpose(1, 2) * (y.mean(1, keepdim=True) + 1)
            out = {"mu": mu, "phi": F.softplus(self.log_phi) + 1e-3}
            return out

        R = self.R_max * torch.sigmoid(self.head_R(r))                      # [B,N,H] intrinsic
        beta = None
        if self.n_interv and u is not None:                                 # u: [B,N,K] response intensity at t-1
            beta = self.beta_raw          # kept >= 0 by project(); clamping here would zero the gradient
            R = R * torch.exp(-(u * beta).sum(-1, keepdim=True).clamp(-3, 6))
        rho = torch.sigmoid(self.head_rho(r))                               # [B,N,1]
        iota = F.softplus(self.head_iota(r) - 2.0)                          # [B,N,1]
        delta = self.delta_scale * torch.tanh(self.head_delta(r))           # [B,N,H]
        yp = y
        if self.clip_k is not None:                                         # robust pressure: damp reporting dumps
            cap = self.clip_k * (y.median(1, keepdim=True).values + 1.0)
            yp = torch.minimum(y, cap)
        G = len(self.w); hist = list(yp.unbind(1))                          # list of [B,N]
        mus = []
        for t in range(self.H):
            P = sum(self.w[s] * hist[-1 - s] for s in range(min(G, len(hist))))     # [B,N]
            Q = (1 - rho[..., 0]) * P + rho[..., 0] * torch.einsum("bnm,bm->bn", A, P)
            m = (R[..., t] * Q + iota[..., 0]) * torch.exp(delta[..., t])
            mus.append(m); hist.append(m)
        mu = torch.stack(mus, 1)                                            # [B,H,N]
        out = {"mu": mu, "phi": F.softplus(self.log_phi) + 1e-3}
        if self.flags["gen"] and self.gen_kl > 0 and self._A_adp is not None and self.gen_mask.any():
            P_, Q_ = self.Agen[self.gen_mask], self._A_adp[self.gen_mask].clamp_min(1e-8)
            out["kl"] = self.gen_kl * (P_ * (torch.log(P_.clamp_min(1e-8)) - torch.log(Q_))).sum(-1).mean()
        if beta is not None: out["beta"] = beta
        if return_diag:
            out.update(R=R, rho=rho.squeeze(-1), iota=iota.squeeze(-1), A=A, pi=pi)
        return out


def nb_nll(y, mu, phi):
    """Negative-binomial NLL, mean mu, dispersion phi (var = mu + mu^2/phi... phi = size r)."""
    r = (1.0 / phi).clamp_min(1e-4)          # phi here is overdispersion alpha; size r = 1/alpha
    mu = mu.clamp_min(1e-6)
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
          + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))
    return -ll.mean()
