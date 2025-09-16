import sys
import os

# Add parent directory to Python path so azlite_portfolio_clean can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
import numpy as np
from azlite_portfolio_clean import PROMOTION_MAP, self_play_episode

def test_promotion_map_constant():
    """Test that PROMOTION_MAP constant exists and has correct values"""
    # Test the constant exists and has expected structure
    assert PROMOTION_MAP is not None
    assert isinstance(PROMOTION_MAP, dict)
    assert len(PROMOTION_MAP) == 4
    
    # Test specific mappings
    assert PROMOTION_MAP[chess.QUEEN] == 1
    assert PROMOTION_MAP[chess.ROOK] == 2
    assert PROMOTION_MAP[chess.BISHOP] == 3
    assert PROMOTION_MAP[chess.KNIGHT] == 4

def test_promotion_map_values():
    """Test that PROMOTION_MAP has correct chess piece mappings"""
    expected = {
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 3,
        chess.KNIGHT: 4,
    }
    assert PROMOTION_MAP == expected

def test_promotion_map_usage():
    """Test that the promotion mapping logic works correctly"""
    
    assert PROMOTION_MAP.get(chess.QUEEN, 0) == 1
    assert PROMOTION_MAP.get(chess.ROOK, 0) == 2
    assert PROMOTION_MAP.get(chess.BISHOP, 0) == 3
    assert PROMOTION_MAP.get(chess.KNIGHT, 0) == 4
    assert PROMOTION_MAP.get(None, 0) == 0  

def test_promotion_indices_logic():
    """Test the promotion index calculation logic from self_play_episode"""
    
    mock_moves = [
        type('MockMove', (), {'promotion': chess.QUEEN})(),
        type('MockMove', (), {'promotion': chess.ROOK})(),
        type('MockMove', (), {'promotion': chess.BISHOP})(),
        type('MockMove', (), {'promotion': chess.KNIGHT})(),
        type('MockMove', (), {'promotion': None})(),  
    ]
    
   
    promo_idxs = [
        0 if mv.promotion is None else PROMOTION_MAP.get(mv.promotion, 0)
        for mv in mock_moves
    ]
    
    expected = [1, 2, 3, 4, 0]  # Queen, Rook, Bishop, Knight, None
    assert promo_idxs == expected