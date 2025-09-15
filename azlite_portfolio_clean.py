"""
azlite_portfolio_clean.py

Cleaned AlphaZero-lite (lint fixes applied):
- removed unused parameter from self_play_episode
- ensured helper/function names use snake_case
- local variables in functions use snake_case
- preserved architecture and functionality
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- Config ---------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "az_checkpoints"
SELFPLAY_DIR = "selfplay_data"
PROFILE_FILE = "az_profiles.json"

# Hyperparams (tune for GPU / CPU)
BOARD_PLANES = 12  # 6 piece types x 2 colors
EMBED_DIM = 64
MOVE_EMB_DIM = 8
CNN_CHANNELS = 64
MCTS_SIMS = 200
DIRICHLET_ALPHA = 0.3
SELFPLAY_EPISODES = 10
SELFPLAY_MAX_MOVES = 400
TRAIN_BATCH_SIZE = 64
TRAIN_EPOCHS = 3
REPLAY_BUFFER_SIZE = 20000
LEARNING_RATE = 1e-3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)


# ----------------------------- Utilities ------------------------------------

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert python-chess board to a 12x8x8 binary tensor:
    planes 0-5: white pawn/knight/bishop/rook/queen/king
    planes 6-11: black ...
    """
    planes = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color
        plane_index = (0 if color == chess.WHITE else 6) + {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }[piece_type]
        # tensor row 0 at top (white perspective)
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        planes[plane_index, r, c] = 1.0
    return planes  # shape (12,8,8)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def legal_moves_list(board: chess.Board) -> List[chess.Move]:
    return list(board.legal_moves)


# ----------------------------- Neural Network -------------------------------

class AZNet(nn.Module):
    """
    Small CNN encoder that produces a state embedding.
    Policy head: on-the-fly scoring of legal moves via move embeddings
    combined with state embedding.
    Value head: scalar -1..1
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 move_emb_dim: int = MOVE_EMB_DIM, channels=CNN_CHANNELS):
        super().__init__()
        # encoder conv layers
        self.conv1 = nn.Conv2d(
            BOARD_PLANES,
            channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # final embedding
        self.fc_embed = nn.Linear(channels, embed_dim)

        # move embeddings (learned)
        self.from_emb = nn.Embedding(64, move_emb_dim)
        self.to_emb = nn.Embedding(64, move_emb_dim)
        # 0=no promo, 1=Q,2=R,3=B,4=N
        self.promotion_emb = nn.Embedding(5, move_emb_dim)

        # small MLP to score move given (state_embed + move_embs)
        policy_input_dim = embed_dim + 3 * move_emb_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)  # scalar logit per move
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh()
        )

    def forward_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 12,8,8)
        returns: state embedding (B, embed_dim)
        """
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.pool(h).view(h.size(0), -1)  # (B, channels)
        embed = self.fc_embed(h)
        return embed  # (B, embed_dim)

    def value(self, state_embed: torch.Tensor) -> torch.Tensor:
        return self.value_head(state_embed).squeeze(-1)  # (B,)

    def score_moves(self, state_embed: torch.Tensor,
                    from_idx: torch.Tensor, to_idx: torch.Tensor,
                    promo_idx: torch.Tensor) -> torch.Tensor:
        """
        Score many moves at once.

        state_embed: (B, E) or (E,) -> we broadcast appropriately
        from_idx: (K,) indices 0..63
        to_idx: (K,)
        promo_idx: (K,) 0..4
        returns logits: (K,)
        """
        device = state_embed.device
        fe = self.from_emb(from_idx.to(device))  # (K, move_emb_dim)
        te = self.to_emb(to_idx.to(device))
        pe = self.promotion_emb(promo_idx.to(device))
        if state_embed.dim() == 1:
            se = state_embed.unsqueeze(0).expand(from_idx.size(0), -1)
        else:
            se = state_embed.expand(from_idx.size(0), -1)
        # (K, E + 3*move_emb_dim)
        inp = torch.cat([se, fe, te, pe], dim=-1)
        logits = self.policy_mlp(inp).squeeze(-1)  # (K,)
        return logits


# ----------------------------- MCTS -----------------------------------------

@dataclass
class MCTSNode:
    prior: float
    N: int = 0
    W: float = 0.0
    children: Dict[str, "MCTSNode"] = None  # key: move.uci()
    move: Optional[chess.Move] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def Q(self) -> float:
        return (self.W / self.N) if self.N > 0 else 0.0


class MCTS:
    def __init__(
        self,
        net: AZNet,
        sims: int = MCTS_SIMS,
        c_puct: float = 1.2
    ):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct

    def run(self, board: chess.Board) -> Tuple[Dict[str, float], Optional[chess.Move]]:
        """
        Run MCTS from current board and return policy (visit count
        distribution over legal moves) and best move (most visited).
        best move may be None if no legal moves.
        """
        root = MCTSNode(prior=0.0)
        root.move = None
        root.children = {}

        legal = legal_moves_list(board)
        if not legal:
            return {}, None

        # NN evaluation for priors
        state_t = torch.tensor(
            board_to_tensor(board), dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            state_embed = self.net.forward_state(state_t)  # (1,E)

            from_idxs = []
            to_idxs = []
            promo_idxs = []
            for mv in legal:
                from_idxs.append(mv.from_square)
                to_idxs.append(mv.to_square)
                if mv.promotion is None:
                    promo_idxs.append(0)
                else:
                    promo_map = {chess.QUEEN: 1, chess.ROOK: 2,
                                 chess.BISHOP: 3, chess.KNIGHT: 4}
                    promo_idxs.append(promo_map.get(mv.promotion, 0))
            from_idx_t = torch.tensor(
                from_idxs, dtype=torch.long, device=DEVICE)
            to_idx_t = torch.tensor(to_idxs, dtype=torch.long, device=DEVICE)
            promo_idx_t = torch.tensor(
                promo_idxs, dtype=torch.long, device=DEVICE)
            logits = self.net.score_moves(
                state_embed.squeeze(0), from_idx_t, to_idx_t, promo_idx_t
            ).cpu().numpy()
            priors = softmax(logits)

        total_children = len(legal)
        for mv, p in zip(legal, priors):
            node = MCTSNode(prior=float(p))
            node.move = mv
            root.children[mv.uci()] = node

        # add Dirichlet noise at root for exploration
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * total_children)
        for i, mv in enumerate(legal):
            key = mv.uci()
            root.children[key].prior = (0.75 * root.children[key].prior
                                        + 0.25 * float(noise[i]))

        for _ in range(self.sims):
            self._simulate(board, root)

        visits = {k: n.N for k, n in root.children.items()}
        total = sum(visits.values()) or 1
        policy = {k: v / total for k, v in visits.items()}
        best_move_key = max(
            visits.items(), key=lambda x: x[1]
        )[0] if visits else None
        best_move = (root.children[best_move_key].move
                     if best_move_key is not None else None)
        return policy, best_move

    def _simulate(self, board: chess.Board, root: MCTSNode) -> float:
        """
        One simulation: selection -> expansion -> backup.
        Returns value from current player's perspective.
        """
        path = []
        node = root
        b = board.copy()

        # Selection
        while True:
            if not node.children:
                break
            total_n = sum(child.N for child in node.children.values())
            best_score = -1e9
            best_key = None
            for key, child in node.children.items():
                u = (self.c_puct * child.prior *
                     (np.sqrt(total_n) / (1 + child.N)))
                q = child.Q()
                score = q + u
                if score > best_score:
                    best_score = score
                    best_key = key
            chosen = node.children[best_key]
            path.append((node, chosen))
            b.push(chosen.move)
            node = chosen
            if not node.children:
                break

        # Expansion & evaluation at leaf
        if b.is_game_over():
            result = b.result(claim_draw=True)
            if result == "1-0":
                v = 1.0
            elif result == "0-1":
                v = -1.0
            else:
                v = 0.0
        else:
            state_t = torch.tensor(
                board_to_tensor(b), dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                state_embed = self.net.forward_state(state_t)
                v = float(self.net.value(state_embed).cpu().numpy()[0])
                legal_leaf = legal_moves_list(b)
                if legal_leaf:
                    from_idxs = []
                    to_idxs = []
                    promo_idxs = []
                    for mv in legal_leaf:
                        from_idxs.append(mv.from_square)
                        to_idxs.append(mv.to_square)
                        if mv.promotion is None:
                            promo_idxs.append(0)
                        else:
                            promo_map = {chess.QUEEN: 1, chess.ROOK: 2,
                                         chess.BISHOP: 3, chess.KNIGHT: 4}
                            promo_idxs.append(promo_map.get(mv.promotion, 0))
                    logits = self.net.score_moves(
                        state_embed.squeeze(0),
                        torch.tensor(from_idxs, dtype=torch.long,
                                     device=DEVICE),
                        torch.tensor(to_idxs, dtype=torch.long, device=DEVICE),
                        torch.tensor(promo_idxs, dtype=torch.long,
                                     device=DEVICE)).cpu().numpy()
                    priors = softmax(logits)
                else:
                    priors = []

            if legal_leaf:
                for mv, p in zip(legal_leaf, priors):
                    child = MCTSNode(prior=float(p))
                    child.move = mv
                    node.children[mv.uci()] = child

        # Backup value up the path
        value = v
        for parent, child in reversed(path):
            child.N += 1
            child.W += value
            value = -value

        return v


# ----------------------------- Replay Buffer --------------------------------

@dataclass
class SelfPlayExample:
    state: np.ndarray  # (12,8,8)
    legal_moves_uci: List[str]
    from_idxs: List[int]
    to_idxs: List[int]
    promo_idxs: List[int]
    pi: List[float]  # target distribution over legal_moves_uci (same order)
    # +1 white win, -1 black win, 0 draw
    # (from perspective of player to move at state)
    outcome: float


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, ex: SelfPlayExample) -> None:
        self.buffer.append(ex)

    def sample(self, batch_size: int) -> List[SelfPlayExample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ----------------------------- Self-Play ------------------------------------

def self_play_episode(mcts: MCTS, max_moves: int = SELFPLAY_MAX_MOVES,
                      temperature: float = 1.0) -> List[SelfPlayExample]:
    """
    Play one self-play game, returning list of training examples.
    The function no longer requires the net parameter because
    MCTS already holds the net.
    """
    board = chess.Board()
    examples: List[SelfPlayExample] = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        policy, best_move = mcts.run(board)
        legal = legal_moves_list(board)
        keys = [mv.uci() for mv in legal]
        visits = np.array([policy.get(k, 0.0) for k in keys],
                          dtype=np.float32)

        if temperature == 0:
            pi = np.zeros_like(visits)
            best_idx = int(np.argmax(visits)) if visits.size > 0 else 0
            if visits.size > 0:
                pi[best_idx] = 1.0
        else:
            if temperature != 1.0 and visits.sum() > 0:
                logits = np.log(np.clip(visits, 1e-12, 1.0))
                logits = logits / temperature
                exp = np.exp(logits - np.max(logits))
                pi = exp / exp.sum()
            else:
                pi = visits

        arr_state = board_to_tensor(board)
        from_idxs = [mv.from_square for mv in legal]
        to_idxs = [mv.to_square for mv in legal]
        promo_map = {
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
        }
        promo_idxs = [
            0 if mv.promotion is None else promo_map.get(mv.promotion, 0)
            for mv in legal
        ]
        examples.append(
            SelfPlayExample(
                state=arr_state,
                legal_moves_uci=keys,
                from_idxs=from_idxs,
                to_idxs=to_idxs,
                promo_idxs=promo_idxs,
                pi=pi.tolist() if isinstance(pi, np.ndarray) else list(pi),
                outcome=0.0,
            )
        )

        if len(keys) == 0:
            break
        # choose a move stochastically from pi to generate diverse games
        # (if pi sums to >0)
        probs = np.array(pi, dtype=np.float32)
        if probs.sum() <= 0:
            choice_idx = 0
        else:
            choice_idx = int(np.random.choice(len(keys), p=probs))
        chosen_move = legal[choice_idx]
        board.push(chosen_move)
        move_count += 1

    result = board.result(claim_draw=True)
    if result == "1-0":
        z_value = 1.0
    elif result == "0-1":
        z_value = -1.0
    else:
        z_value = 0.0

    for idx, ex in enumerate(examples):
        player_to_move_is_white = (idx % 2 == 0)
        ex.outcome = z_value if player_to_move_is_white else -z_value

    return examples


# ----------------------------- Training ------------------------------------

def train_from_buffer(net: AZNet, buffer: ReplayBuffer,
                      lr: float = LEARNING_RATE,
                      batch_size: int = TRAIN_BATCH_SIZE,
                      epochs: int = TRAIN_EPOCHS):
    """
    Sample minibatches and train network.
    """
    net.to(DEVICE)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        losses = []
        if len(buffer) == 0:
            print("Replay buffer empty; nothing to train.")
            return
        steps = max(1, len(buffer) // batch_size)
        for _ in range(steps):
            batch = buffer.sample(batch_size)
            optimizer.zero_grad()
            total_loss = 0.0
            total_examples = 0
            for ex in batch:
                state = torch.tensor(
                    ex.state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)
                state_embed = net.forward_state(state).squeeze(0)
                z = torch.tensor([ex.outcome], dtype=torch.float32,
                                 device=DEVICE)
                v_pred = net.value(state_embed.unsqueeze(0)).squeeze(0)
                loss_v = mse(v_pred, z)

                from_idx_t = torch.tensor(
                    ex.from_idxs, dtype=torch.long, device=DEVICE)
                to_idx_t = torch.tensor(
                    ex.to_idxs, dtype=torch.long, device=DEVICE)
                promo_idx_t = torch.tensor(
                    ex.promo_idxs, dtype=torch.long, device=DEVICE)
                logits = net.score_moves(
                    state_embed, from_idx_t, to_idx_t, promo_idx_t)
                pi = np.array(ex.pi, dtype=np.float32)
                if pi.sum() <= 0:
                    continue
                target_idx = int(np.argmax(pi))
                logits_2d = logits.unsqueeze(0)
                target_tensor = torch.tensor(
                    [target_idx], dtype=torch.long, device=DEVICE)
                loss_p = ce(logits_2d, target_tensor)

                loss = loss_v + loss_p
                total_loss = total_loss + loss
                total_examples += 1

            if total_examples == 0:
                continue
            avg_loss = total_loss / total_examples
            avg_loss.backward()
            optimizer.step()
            losses.append(avg_loss.item())
        print(f"Epoch {epoch+1}/{epochs}: avg loss {np.mean(losses):.4f}")

    ts = int(time.time())
    ckpt = os.path.join(CHECKPOINT_DIR, f"aznet_{ts}.pt")
    torch.save(net.state_dict(), ckpt)
    print("Saved checkpoint:", ckpt)


# ----------------------------- CLI Actions ----------------------------------

def do_selfplay(net: AZNet, episodes: int = SELFPLAY_EPISODES,
                sims: int = MCTS_SIMS, pid: str = "guest"):
    """
    Run several self-play episodes and store generated data in
    SELFPLAY_DIR/pid_#.jsonl
    """
    mcts = MCTS(net, sims=sims)
    for ep in range(episodes):
        print(f"Self-play episode {ep+1}/{episodes}")
        examples = self_play_episode(mcts)
        fn = os.path.join(
            SELFPLAY_DIR,
            f"{pid}_ep_{int(time.time())}_{ep}.jsonl",
        )
        with open(fn, "w", encoding="utf-8") as f:
            for ex in examples:
                rec = {
                    "state": ex.state.tolist(),
                    "legal_moves_uci": ex.legal_moves_uci,
                    "from_idxs": ex.from_idxs,
                    "to_idxs": ex.to_idxs,
                    "promo_idxs": ex.promo_idxs,
                    "pi": ex.pi,
                    "outcome": ex.outcome
                }
                f.write(json.dumps(rec) + "\n")
        print("Wrote", fn)
    print("Self-play finished.")


def load_selfplay_into_buffer(buffer: ReplayBuffer,
                              path_dir: str = SELFPLAY_DIR):
    """
    Read all jsonl files under directory and push into buffer.
    """
    files = [os.path.join(path_dir, f)
             for f in os.listdir(path_dir) if f.endswith(".jsonl")]
    files.sort()
    count = 0
    for fn in files:
        try:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    ex = SelfPlayExample(
                        state=np.array(rec["state"], dtype=np.float32),
                        legal_moves_uci=rec["legal_moves_uci"],
                        from_idxs=rec["from_idxs"],
                        to_idxs=rec["to_idxs"],
                        promo_idxs=rec["promo_idxs"],
                        pi=rec["pi"],
                        outcome=rec["outcome"])
                    buffer.push(ex)
                    count += 1
        except (OSError, json.JSONDecodeError):
            print("Skipping corrupt file:", fn)
    print(f"Loaded {count} examples into buffer from {len(files)} files.")


def human_vs_engine(net: AZNet, mcts_sims: int = 200):
    board = chess.Board()
    mcts = MCTS(net, sims=mcts_sims)
    print(
        "Human vs Engine. Enter moves like 'e4', 'e2 e4', 'Nf3'. "
        "Type 'quit' to exit."
    )
    while not board.is_game_over():
        print(board)
        raw = input("Your move: ").strip()
        if raw.lower() == "quit":
            print("Game aborted.")
            return
        mv = parse_input(board=board, raw=raw)
        if mv is None or mv not in board.legal_moves:
            print("Illegal or unparsable move. Try again.")
            continue
        board.push(mv)
        policy, best_move = mcts.run(board)
        if best_move is None:
            print("Engine has no moves; game over.")
            break
        board.push(best_move)
        print("Engine plays:", board.san(best_move))
    print("Result:", board.result())


def parse_input(board: chess.Board, raw: str) -> Optional[chess.Move]:
    """
    Accept many formats:
     - UCI: e2e4
     - SAN: Nf3, O-O
     - Grid 'from to': e2 e4 or e2,e4
     - Single target 'e4' -> disambiguation if multiple
    Returns a legal move or None.
    """
    s = raw.strip()
    # try UCI
    if len(s) == 4 and all(ch in "abcdefgh12345678" for ch in s):
        try:
            mv = chess.Move.from_uci(s)
            if mv in board.legal_moves:
                return mv
        except ValueError:
            pass

    # from-to with space or comma
    if " " in s or "," in s:
        parts = s.replace(",", " ").split()
        if len(parts) == 2 and all(len(p) == 2 for p in parts):
            try:
                mv = chess.Move.from_uci(parts[0] + parts[1])
                if mv in board.legal_moves:
                    return mv
            except ValueError:
                pass

    # single square like "e4" -> if exactly one candidate, return it; else None
    if len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678":
        try:
            target = chess.parse_square(s)
            candidates = [
                m for m in board.legal_moves if m.to_square == target
            ]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                pawns = [
                    m for m in candidates
                    if board.piece_at(m.from_square).piece_type == chess.PAWN
                ]
                if len(pawns) == 1:
                    return pawns[0]
                return None
        except ValueError:
            pass

    # SAN fallback
    try:
        mv = board.parse_san(s)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass

    return None


# ----------------------------- Main CLI ------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AZ-Lite portfolio engine (clean)"
    )
    parser.add_argument(
        "mode",
        choices=["selfplay", "train", "play"],
        help="Mode to run",
    )
    parser.add_argument("--episodes", type=int, default=SELFPLAY_EPISODES)
    parser.add_argument("--sims", type=int, default=MCTS_SIMS)
    parser.add_argument("--pid", type=str, default="guest")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    args = parser.parse_args()

    net = AZNet().to(DEVICE)
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        print("Loaded checkpoint", args.checkpoint)

    if args.mode == "selfplay":
        print("Starting self-play...")
        do_selfplay(net, episodes=args.episodes, sims=args.sims, pid=args.pid)
    elif args.mode == "train":
        buffer = ReplayBuffer()
        load_selfplay_into_buffer(buffer)
        if len(buffer) == 0:
            print("No self-play data; run selfplay first.")
            return
        train_from_buffer(net, buffer)
    elif args.mode == "play":
        human_vs_engine(net, mcts_sims=args.sims)
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()
