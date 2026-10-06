"""Train PAX 2.0's evaluation network on the generated positions.
Usage: python train.py <data_dir> <epochs> <out.json>
Architecture (NNUE-style): two perspectives x 768 one-hot (side, piece, square) -> shared 256 (clipped ReLU)
-> concat [side to move, opponent] 512 -> 32 -> 32 -> 1. Output x 400 = centipawns for the side to move.
"""
import sys, glob, json, math, time
import numpy as np, torch, torch.nn as nn
import chess

import os
H = int(os.environ.get("PAX_H", "256")); PAD = 768
PT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

# The engine's hand-written evaluation (same tables as engine.js), so the network learns only the correction.
VAL = [100, 320, 330, 500, 900, 0]
PSTT = [
 [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
 [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
 [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
 [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
 [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
 [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20]]
def pst_stm(board):
    s = 0
    for sq, pc in board.piece_map().items():
        t = PT[pc.piece_type]; f = sq % 8; rank = sq // 8
        v = VAL[t] + PSTT[t][(7 - rank) * 8 + f if pc.color else rank * 8 + f]
        s += v if pc.color else -v
    return s if board.turn else -s

def feats(board):
    w, b = [], []
    for sq, pc in board.piece_map().items():
        t = PT[pc.piece_type]
        w.append((0 if pc.color else 1) * 384 + t * 64 + sq)          # White's view
        b.append((0 if not pc.color else 1) * 384 + t * 64 + (sq ^ 56))  # Black's view: own pieces first, board flipped
    return w, b

def load(data_dir, cache):
    try:
        z = np.load(cache); print('loaded cache', cache); return z['fw'], z['fb'], z['stm'], z['cp'], z['res'], z['pst']
    except FileNotFoundError: pass
    fens, cps, ress = [], [], []
    for f in glob.glob(data_dir + '/*.csv'):
        for line in open(f):
            p = line.rstrip('\n').split(',')
            if len(p) == 3: fens.append(p[0]); cps.append(float(p[1])); ress.append(float(p[2]))
    n = len(fens); print(n, 'positions'); t0 = time.time()
    fw = np.full((n, 32), PAD, np.int16); fb = np.full((n, 32), PAD, np.int16); stm = np.zeros(n, np.int8); pst = np.zeros(n, np.float32)
    for i, fen in enumerate(fens):
        bd = chess.Board(fen); w, b = feats(bd)
        fw[i, :len(w)] = w; fb[i, :len(b)] = b; stm[i] = 0 if bd.turn else 1; pst[i] = pst_stm(bd)
        if i % 500000 == 0: print(' featurised', i, f'{time.time() - t0:.0f}s', flush=True)
    cp = np.array(cps, np.float32); res = np.array(ress, np.float32)
    np.savez(cache, fw=fw, fb=fb, stm=stm, cp=cp, res=res, pst=pst)
    return fw, fb, stm, cp, res, pst

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(PAD + 1, H, mode='sum', padding_idx=PAD)
        self.ftb = nn.Parameter(torch.zeros(H))
        self.l1 = nn.Linear(2 * H, 32); self.l2 = nn.Linear(32, 32); self.out = nn.Linear(32, 1)
    def forward(self, fw, fb, stm):
        aw = torch.clamp(self.ft(fw) + self.ftb, 0, 1); ab = torch.clamp(self.ft(fb) + self.ftb, 0, 1)
        s = stm.unsqueeze(1).bool()
        x = torch.where(s, torch.cat([ab, aw], 1), torch.cat([aw, ab], 1))   # side to move first
        x = torch.clamp(self.l1(x), 0, 1); x = torch.clamp(self.l2(x), 0, 1)
        return self.out(x).squeeze(1)

if __name__ == '__main__':
    data_dir, epochs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fw, fb, stm, cp, res, pst = load(data_dir, data_dir + '/features.npz')
    LAM = float(os.environ.get('PAX_LAMBDA', '0.85'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev)
    n = len(cp); idx = np.random.permutation(n); nval = min(50000, n // 20); val, tr = idx[:nval], idx[nval:]
    T = lambda a: torch.from_numpy(a)
    FW, FB, STM = T(fw.astype(np.int64)).to(dev), T(fb.astype(np.int64)).to(dev), T(stm.astype(np.int64)).to(dev)
    target = T(LAM / (1 + np.exp(-cp / 400)) + (1 - LAM) * res).float().to(dev)   # Stockfish's view blended with the game result
    BASE = T(pst / 400).float().to(dev)                                               # the hand-written score; the network adds to it
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 8192
    for ep in range(epochs):
        net.train(); np.random.shuffle(tr); t0 = time.time(); tot = 0
        for i in range(0, len(tr), bs):
            j = torch.from_numpy(tr[i:i + bs]).to(dev)
            pred = torch.sigmoid(BASE[j] + net(FW[j], FB[j], STM[j]))
            loss = ((pred - target[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        sched.step(); net.eval()
        with torch.no_grad():
            j = torch.from_numpy(val).to(dev)
            vl = ((torch.sigmoid(BASE[j] + net(FW[j], FB[j], STM[j])) - target[j]) ** 2).mean().item()
        print(f'epoch {ep + 1}/{epochs}: train {tot / len(tr):.5f}  val {vl:.5f}  ({time.time() - t0:.0f}s)', flush=True)
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    w = {'H': H, 'residual': True, 'ft': sd['ft.weight'][:PAD].round(5).tolist(), 'ftb': sd['ftb'].round(5).tolist(),
         'l1w': sd['l1.weight'].round(5).tolist(), 'l1b': sd['l1.bias'].round(5).tolist(),
         'l2w': sd['l2.weight'].round(5).tolist(), 'l2b': sd['l2.bias'].round(5).tolist(),
         'ow': sd['out.weight'][0].round(5).tolist(), 'ob': float(sd['out.bias'][0])}
    json.dump(w, open(out, 'w')); torch.save(net.state_dict(), out.replace('.json', '.pt'))
    print('saved', out)
