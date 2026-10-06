// PAX 2.0 engine (../engine.js) vs Stockfish at a limited strength, alternating colours.
// Usage: node match_sf.js <net.json|none> <stockfish_elo> <games> <ms_per_move>
const vm = require('vm'), fs = require('fs'), { spawn } = require('child_process'), path = require('path');
const { Chess } = require('chess.js');                 // npm i chess.js@0.10.3
const E = require(path.join(__dirname, '..', 'engine.js'));
const [netPath, eloArg, gamesArg, msArg] = process.argv.slice(2);
if (netPath !== 'none') { const w = JSON.parse(fs.readFileSync(netPath, 'utf8')); w.mg ? E.setTables(w) : E.setNet(w); }
const SF = process.env.PAX_STOCKFISH || 'stockfish';   // path to a Stockfish binary

function uciEngine() {
  const p = spawn(SF); let buf = '', waiters = [];
  p.stdout.on('data', d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); waiters = waiters.filter(w => !w(line)); } });
  const send = s => p.stdin.write(s + '\n');
  const until = pred => new Promise(r => waiters.push(l => pred(l) && (r(l), true)));
  return { send, until, quit: () => p.kill() };
}
const uci = m => m.from + m.to + (m.promotion || '');

(async () => {
  const sf = uciEngine(); sf.send('uci'); await sf.until(l => l === 'uciok');
  sf.send('setoption name UCI_LimitStrength value true'); sf.send(`setoption name UCI_Elo value ${eloArg}`);
  sf.send('isready'); await sf.until(l => l === 'readyok');
  let score = 0, w = 0, d = 0, l = 0;
  for (let g = 0; g < +gamesArg; g++) {
    const paxWhite = g % 2 === 0; const c = new Chess(); const moves = [];
    sf.send('ucinewgame');
    while (!c.game_over() && c.history().length < 300) {
      const paxTurn = (c.turn() === 'w') === paxWhite;
      let mv;
      if (paxTurn) {
        const legal = c.moves({ verbose: true }).filter(m => !m.promotion || m.promotion === 'q');
        const r = E.search(c.fen(), legal.map(uci), +msArg, c.history().length < 8 ? 15 : 0);
        mv = r.uci;
      } else {
        sf.send(`position startpos moves ${moves.join(' ')}`); sf.send('go movetime 200');
        mv = (await sf.until(x => x.startsWith('bestmove'))).split(' ')[1];
      }
      const ok = c.move({ from: mv.slice(0, 2), to: mv.slice(2, 4), promotion: mv[4] || 'q' });
      if (!ok) { console.log('illegal move', mv, 'by', paxTurn ? 'PAX 2.0' : 'Stockfish'); break; }
      moves.push(mv);
    }
    let r = 0.5;
    if (c.in_checkmate()) r = ((c.turn() === 'w') === paxWhite) ? 0 : 1;
    score += r; r === 1 ? w++ : r === 0 ? l++ : d++;
    console.log(`game ${g + 1} (PAX 2.0 ${paxWhite ? 'White' : 'Black'}): ${r === 1 ? 'win' : r === 0 ? 'loss' : 'draw'} in ${c.history().length} plies`);
  }
  const n = +gamesArg, s = Math.min(Math.max(score / n, 0.5 / n), 1 - 0.5 / n);
  console.log(`vs Stockfish ${eloArg}: +${w} =${d} -${l}, score ${(score / n * 100).toFixed(0)}%, performance ≈ ${Math.round(+eloArg - 400 * Math.log10(1 / s - 1))}`);
  sf.quit();
})();
