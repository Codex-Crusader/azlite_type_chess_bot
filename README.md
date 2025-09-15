# AZ-Lite — Portfolio AI Chess Engine

AZ-Lite is a compact, AlphaZero-inspired chess engine implemented in Python.  
It combines **Monte Carlo Tree Search (MCTS)** with a **neural network** evaluator to learn optimal play via self-play.  
This project demonstrates advanced AI concepts in a **portfolio-friendly, runnable Python project**.

---

## Features
- Lightweight **convolutional neural network** for policy + value evaluation.
- **MCTS** guided by neural priors with exploration noise.
- **Self-play data generation**, training loop, and checkpoint saving.
- CLI-based interface:
  - `selfplay` → generate training games
  - `train` → train neural network
  - `play` → play against the engine interactively
- Input moves using standard chess notation (`e4`, `Nf3`, etc.).
- Lightweight, modular, and easy to extend.

---

## Quick Start

1. **Setup virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Generate self-play data
```bash
./run_selfplay.sh 5 80 demo
```

3. Train the neural network
```bash
./run_train.sh
```

4. Play vs trained engine
```bash
./run_play.sh 200
```
Use quit during play to exit.

## Repository Structure
```bash
azlite-portfolio/
├─ azlite_portfolio_clean.py       # main engine
├─ selfplay_data/                  # generated training data (JSONL)
├─ az_checkpoints/                 # neural network checkpoints
├─ tests/                          # unit tests
├─ docs/                           # architecture & hyperparameters
├─ notebooks/                       # optional analysis/visualization
├─ run_selfplay.sh                 # self-play runner
├─ run_train.sh                     # training runner
├─ run_play.sh                      # interactive play runner
├─ Makefile                         # workflow shortcuts
├─ requirements.txt                 # dependencies
├─ README.md
└─ LICENSE
```
