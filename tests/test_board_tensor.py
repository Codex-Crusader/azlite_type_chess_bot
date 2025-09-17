import sys
import os

# Add parent directory to Python path so azlite_portfolio_clean can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
import numpy as np
from azlite_portfolio_clean import board_to_tensor


def test_board_tensor_shape():
    board = chess.Board()
    t = board_to_tensor(board)
    assert isinstance(t, np.ndarray)
    assert t.shape == (12, 8, 8)
    # initial position: 16 white + 16 black pieces counted in planes
    assert t.sum() == 32


def test_empty_board():
    board = chess.Board.empty()
    t = board_to_tensor(board)
    assert t.sum() == 0  # No pieces on an empty board


def test_specific_pieces():
    """Test that pieces appear in the correct planes of the tensor"""
    board = chess.Board()
    t = board_to_tensor(board)
    
    # Check white pawns (typically on the second rank/row from bottom)
    white_pawns = t[0]  # Assuming first plane is white pawns
    assert white_pawns[6].sum() == 8  # 8 white pawns on the second rank
    
    # Check black pawns (typically on the seventh rank/row from bottom)
    black_pawns = t[6]  # Assuming seventh plane is black pawns
    assert black_pawns[1].sum() == 8  # 8 black pawns on the seventh rank
