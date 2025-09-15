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
    

