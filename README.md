# AZ-Lite — Portfolio AI Chess Engine

**AZ-Lite** is a compact, AlphaZero-inspired chess engine implemented in Python.  
It combines **Monte-Carlo Tree Search (MCTS)** with a lightweight **neural network** (policy + value) and learns from **self-play**.  
The project is designed to be **readable, reproducible, and runnable locally**
---

## Highlights
- MCTS (PUCT) guided by neural priors and values.  
- On-the-fly move scoring via learned move embeddings (keeps the implementation compact).  
- End-to-end pipeline: self-play → JSONL replay data → training loop → checkpoints.  
- Interactive CLI to `selfplay`, `train`, and `play`.  
- Lightweight and modular so you can extend (GUI, distributed self-play, larger nets).

---

## Quickstart (minutes)

> Assumes Linux/macOS or WSL, Python 3.8+.

Create and activate a virtualenv, install deps:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Generate small self-play data (fast test):
```bash
./run_selfplay.sh 5 80 demo
```

Train on generated data:
```bash
./run_train.sh
```

Play against the engine:
```bash
./run_play.sh 200
```

During play, enter moves like e4, Nf3, e2 e4, or quit to exit.

---

## Example commands
```bash
# quick demo: create venv, install, run 3 selfplay games
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
./run_selfplay.sh 3 80 demo

# train (reads all JSONL in selfplay_data)
./run_train.sh

# play with 200 MCTS sims (load checkpoint with --checkpoint flag if desired)
./run_play.sh 200
```

---

## Design & architecture (short)

1. State encoder — small 3-layer CNN producing a compact state embedding from a 12×8×8 board tensor (6 piece types × 2 colors).
2. Policy head — learned embeddings for `from`, `to`, and `promotion` combined with state embedding to score legal moves on the fly.
3. Value head — scalar predicting game outcome (−1..1).
4. MCTS (PUCT) — neural priors bias search, Dirichlet noise added at root, value backups produce the training target `z`.
5. Self-play — saves `(state, π, z)` examples as JSONL in `selfplay_data/`.
6. Training loop — samples replay, optimizes `MSE(value,z) + CE(policy,π)`, saves checkpoints to az_checkpoints/.

For more details, `see docs/architecture.md` and `docs/hyperparams.md`.

---

Repository layout
```bash
azlite-portfolio/
├─ azlite_portfolio_clean.py     # main engine & CLI
├─ run_selfplay.sh               # wrapper for self-play
├─ run_train.sh                  # wrapper for training
├─ run_play.sh                   # wrapper to play interactively
├─ requirements.txt
├─ selfplay_data/                # generated JSONL examples
├─ az_checkpoints/               # saved PyTorch checkpoints
├─ docs/                         # architecture, hyperparams
├─ tests/                        # unit tests (parse + tensor)
├─ Makefile
├─ README.md
└─ LICENSE
```
---
