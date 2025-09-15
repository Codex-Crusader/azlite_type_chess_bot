# AZ-Lite: Architecture Overview

AZ-Lite is a compact AlphaZero-inspired chess engine written in Python.  
It combines **Monte Carlo Tree Search (MCTS)** with a **neural network** evaluator for policy + value.

---

## 1. System Components

### Neural Network (Policy + Value Head)
- Input: 12×8×8 board tensor (piece planes).
- Convolutional backbone (3–4 layers).
- Policy head:
  - Produces a **move prior distribution** (logits).
  - Used by MCTS to bias search.
- Value head:
  - Predicts the expected game outcome (-1 = loss, 0 = draw, +1 = win).
  - Used for backpropagation in search.

### Monte Carlo Tree Search (MCTS)
- Uses **PUCT** (Predictor + Upper Confidence Trees).
- At each node:
  - Policy logits from NN guide move priors.
  - Value predictions are backed up through the tree.
  - Exploration noise (Dirichlet) added at root for variability.
- Produces a **move probability vector (π)** for training.

### Self-Play Module
- Plays games against itself using MCTS.
- Records training examples `(state, π, z)` where:
  - `state` = encoded board tensor
  - `π` = move probability distribution
  - `z` = game result from perspective of player at state
- Saves examples as JSONL.

### Training Loop
- Samples examples from `selfplay_data/`.
- Optimizes network parameters with loss:
  - `L = (z - v)^2 - π · log(p)`
  - where:
    - `z` = target outcome
    - `v` = predicted value
    - `π` = target move distribution
    - `p` = predicted move distribution
- Updates weights in `az_checkpoints/`.

### CLI Interface
- Supports three modes:
  1. `selfplay` → generate training data
  2. `train` → train network
  3. `play` → play against the engine

---

## 2. Data Flow

```text
+-----------+       +------------+       +------------+
|  Selfplay | ----> |  Training  | ----> | Checkpoint |
+-----------+       +------------+       +------------+
     |                   ^                       |
     v                   |                       v
  JSONL examples         |                Engine loads NN
                         |
                     CLI user plays

```
## 3. Differences from AlphaZero

1.  Lightweight CNN instead of ResNet.
2.  Move scoring done on-the-fly (legal moves only).
3. Simplified replay buffer (JSONL, not a DB).
4. Small-scale reinforcement learning, runnable on a laptop.

## 4. Strengths and Limitations

1. Easy to run locally (no cluster needed).
2. End-to-end reinforcement learning pipeline.
3. MCTS + NN design is portfolio-quality.
4. Plays weaker than Stockfish/Leela (much smaller net, fewer simulations).
5. Currently text-based (GUI could be added later).