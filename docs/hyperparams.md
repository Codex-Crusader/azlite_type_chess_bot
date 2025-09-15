# Hyperparameters — AZ-Lite

This file documents the key hyperparameters used in AZ-Lite.  
These can be tuned for speed vs playing strength.

---

## Neural Network
- Convolution filters: `[32, 64, 128]`
- Kernel size: 3
- Policy head:
  - 128 → 64 hidden units
- Value head:
  - 128 → 64 hidden units
- Activation: ReLU
- Optimizer: Adam
- Learning rate: `1e-3`

---

## Self-Play
- Episodes per run: 5 (default, adjustable)
- MCTS simulations per move: 80 (default)
- Exploration noise:
  - Dirichlet α = 0.3
  - ε = 0.25 (fraction mixing noise with priors)
- Temperature τ:
  - >30 moves: τ=1 (exploration)
  - ≤30 moves: τ=0 (greedy)

---

## Training
- Batch size: 64
- Loss:
  - Value loss = MSE
  - Policy loss = cross-entropy
  - Final loss = value + policy
- Epochs per training run: 5
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)

---

## MCTS
- Formula:  
  `Q + U = (W/N) + c_puct * P * sqrt(ΣN) / (1+N)`
- `c_puct`: 1.5
- Virtual loss: none (single-threaded)
- Maximum depth: limited by game length

---

## CLI Defaults
- Selfplay: `--episodes 5 --sims 80`
- Play: `--sims 200`
- Training: loads all JSONL in `selfplay_data/`

---

## Tuning Notes
- **More simulations per move** → stronger play, slower runtime.
- **Bigger CNN** → stronger, requires GPU or patience.
- **Replay buffer size**: currently unbounded JSONL; pruning old games may stabilize learning.
- **Learning rate**: small decay schedule improves stability.
- **Temperature**: adjust for more exploratory vs. sharp play.

---