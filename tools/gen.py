"""PAX 2.0 training data: Stockfish-labelled positions from games that start in Len Dorfman's MARK10 opening book.
Usage: python gen.py <positions_per_worker> <workers> <out_dir>
Each worker writes out_dir/data_<n>.csv with rows: fen,score_cp_side_to_move,result_side_to_move (1/0.5/0)
"""
import sys, os, random, glob, re, csv, time
import multiprocessing as mp
import chess, chess.engine

SF = os.environ.get('PAX_STOCKFISH', 'stockfish')   # path to a Stockfish binary
BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mark10w.txt')

def book_bytes():
    """Dorfman's MARK10.DAT, published with control bytes escaped as ~XX."""
    return re.sub(rb'~([0-9A-Fa-f]{2})', lambda m: bytes([int(m.group(1), 16)]), open(BOOK, 'rb').read())
REC = re.compile(rb'((?:[pnbrqkPNBRQK1-8]{1,8}/){8})-(..)', re.S)

def book_starts():
    """Positions from the book, with side to move taken from the stored move's piece colour."""
    out = []
    for m in REC.finditer(book_bytes()):
        place = m.group(1).decode()[:-1]
        frm = m.group(2)[0] - 1
        try: b = chess.Board(place + ' w - - 0 1')
        except ValueError: continue
        pc = b.piece_at(frm)
        if not pc: continue
        b.turn = pc.color
        # castling rights where king and rook still stand on their home squares
        cr = ''
        if b.piece_at(chess.E1) == chess.Piece(chess.KING, True):
            if b.piece_at(chess.H1) == chess.Piece(chess.ROOK, True): cr += 'K'
            if b.piece_at(chess.A1) == chess.Piece(chess.ROOK, True): cr += 'Q'
        if b.piece_at(chess.E8) == chess.Piece(chess.KING, False):
            if b.piece_at(chess.H8) == chess.Piece(chess.ROOK, False): cr += 'k'
            if b.piece_at(chess.A8) == chess.Piece(chess.ROOK, False): cr += 'q'
        b.set_castling_fen(cr or '-')
        if b.is_valid() and not b.is_game_over(): out.append(b.fen())
    return out

def score_cp(s, turn):
    v = s.pov(turn).score(mate_score=3000)
    return max(-3000, min(3000, v))

def worker(args):
    idx, target, out_dir, starts = args
    random.seed(idx * 7919 + int(time.time()))
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({'Threads': 1, 'Hash': 16})
    path = os.path.join(out_dir, f'data_{idx}.csv'); n = 0
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        while n < target:
            b = chess.Board(random.choice(starts))
            if b.is_game_over(): continue
            rows = []
            while not b.is_game_over(claim_draw=True) and b.ply() < 400:
                info = eng.analyse(b, chess.engine.Limit(depth=8), multipv=4)
                info = [i for i in info if 'pv' in i and i['pv']]
                if not info: break
                best = info[0]
                if not b.is_check() and not b.is_capture(best['pv'][0]) and abs(score_cp(best['score'], b.turn)) <= 1500:
                    rows.append((b.fen(), score_cp(best['score'], b.turn), b.turn))
                # choose among the top moves, softly favouring the best; occasionally a random one early on
                sc = [score_cp(i['score'], b.turn) for i in info]          # variety without blunders: soft choice among the top moves
                top = max(sc); wts = [2.718 ** ((s - top) / 30) for s in sc]
                mv = random.choices([i['pv'][0] for i in info], weights=wts)[0]
                b.push(mv)
                if abs(score_cp(best['score'], not b.turn)) >= 1500: break   # decided: stop the game here
            res = b.outcome(claim_draw=True)
            if res is None:                                          # stopped early or at the cap: use the eval
                last = rows[-1][1] if rows else 0
                last_turn = rows[-1][2] if rows else chess.WHITE
                white = 1.0 if (last if last_turn else -last) > 300 else 0.0 if (last if last_turn else -last) < -300 else 0.5
            else:
                white = 1.0 if res.winner is True else 0.0 if res.winner is False else 0.5
            for fen, sc, turn in rows:
                w.writerow((fen, sc, white if turn else 1 - white))
            n += len(rows); fh.flush()
    eng.quit()
    return n

if __name__ == '__main__':
    per, workers, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    starts = book_starts(); print(len(starts), 'starting positions from the book', flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        total = sum(pool.map(worker, [(i, per, out_dir, starts) for i in range(workers)]))
    print(f'{total} positions in {time.time() - t0:.0f}s', flush=True)
