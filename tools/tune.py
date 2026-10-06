"""Texel-style tuning of PAX 2.0's tapered piece-square tables against the Stockfish-labelled positions.
Usage: python tune.py <data_dir with features.npz> <epochs> <out.json>
Eval (White's view) = sum over pieces of sign * (MG[t][i] * ph + EG[t][i] * (24 - ph)) / 24, ph = game phase 0..24.
Tables use the engine's index: rank 8 first, from the piece owner's side. Starts from the hand-written tables."""
import sys, json, numpy as np, torch
from train import VAL, PSTT
d, epochs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
z = np.load(d + '/features.npz'); fw, stm, cp, res = z['fw'].astype(np.int64), z['stm'], z['cp'], z['res']
LAM = 0.85
valid = fw < 768
white = fw < 384; f = np.where(white, fw, fw - 384); t = f // 64; sq = f % 64
idx = np.where(white, sq ^ 56, sq)                       # owner's view, rank 8 first
tab = np.where(valid, t * 64 + idx, 6 * 64)              # 384 = padding slot (always zero)
sign = np.where(valid, np.where(white, 1.0, -1.0), 0.0).astype(np.float32)
PHW = np.array([0, 1, 1, 2, 4, 0, 0])
ph = np.minimum((np.where(valid, PHW[np.minimum(t, 6)], 0)).sum(1), 24).astype(np.float32)
stms = np.where(stm == 0, 1.0, -1.0).astype(np.float32)
target = (LAM / (1 + np.exp(-cp / 400)) + (1 - LAM) * res).astype(np.float32)
dev = 'cuda'; T = lambda a: torch.from_numpy(a).to(dev)
TAB, SG, PH, STM, TG = T(tab), T(sign), T(ph), T(stms), T(target)
init = np.array([[VAL[k] + PSTT[k][i] for i in range(64)] for k in range(6)] + [[0] * 64], np.float32).reshape(-1)
MG = torch.tensor(init, device=dev, requires_grad=True); EG = torch.tensor(init, device=dev, requires_grad=True)
mask = torch.ones(7 * 64, device=dev); mask[6 * 64:] = 0
n = len(cp); perm = np.random.permutation(n); nval = 50000; va, tr = perm[:nval], perm[nval:]
opt = torch.optim.Adam([MG, EG], lr=2.0)
def ev(j):
    m = (MG * mask)[TAB[j]]; e = (EG * mask)[TAB[j]]; p = PH[j].unsqueeze(1) / 24
    return ((m * p + e * (1 - p)) * SG[j]).sum(1) * STM[j]
def loss(j): return ((torch.sigmoid(ev(j) / 400) - TG[j]) ** 2).mean()
with torch.no_grad(): print('start val', loss(T(va)).item())
for ep in range(epochs):
    np.random.shuffle(tr)
    for i in range(0, len(tr), 16384):
        j = T(tr[i:i + 16384]); opt.zero_grad(); l = loss(j); l.backward(); opt.step()
    if (ep + 1) % 5 == 0:
        with torch.no_grad(): print(f'epoch {ep + 1}: val {loss(T(va)).item():.5f}', flush=True)
    for g in opt.param_groups: g['lr'] = 2.0 * (1 - ep / epochs) + 0.1
r = lambda a: a.detach().cpu().numpy().reshape(7, 64)[:6].round(1).tolist()
json.dump({'mg': r(MG), 'eg': r(EG)}, open(out, 'w')); print('saved', out)
print('pawn mg/eg avg', MG[:64].mean().item(), EG[:64].mean().item(), 'queen', MG[256:320].mean().item(), EG[256:320].mean().item())
