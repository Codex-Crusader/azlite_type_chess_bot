import sys
import os

# Add parent directory to Python path so azlite_portfolio_clean can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
from azlite_portfolio_clean import parse_input


def test_parse_input_uci():
    board = chess.Board()
    mv = parse_input(board, "e2e4")
    assert mv is not None
    assert mv.uci() == "e2e4"


def test_parse_input_single_target():
    board = chess.Board()
    mv = parse_input(board, "e4")
    assert mv is not None
    assert mv.to_square == chess.parse_square("e4")


def test_parse_input_algebraic():
    board = chess.Board()
    mv = parse_input(board, "Nf3")  # Knight to f3
    assert mv is not None
    assert mv.uci() == "g1f3"  # Knight from g1 to f3


def test_parse_input_invalid():
    board = chess.Board()
    mv = parse_input(board, "invalid")
    assert mv is None
    
