from abc import ABC, abstractmethod
from typing import List, Protocol

import numpy as np


class Game(Protocol):
  ...


class Piece(ABC):
  """
  Either a rock or a knight (Chess like).
  """
  def __init__(self, game: Game, player: int) -> None:
    self.game = game
    self.player = player

  @abstractmethod
  def __repr__(self) -> str:
    pass
  
  @abstractmethod
  def valid_move(self, move) -> bool:
    pass


class Rock(Piece):
  """
  Rock
  """
  def __init__(self, game: Game, player: int) -> None:
    super().__init__(game, player)

  def __repr__(self) -> str:
    return f'{"w" if self.player == 0 else "b"}R'
  
  def valid_move(self, move) -> bool:
    current_location = self.game.locations[self.player]
    abs_move_row = abs(move[0] - current_location[0]) 
    abs_move_col = abs(move[1] - current_location[1]) 
    return (
      ((abs_move_row == 1) and (abs_move_col == 0))
      or
      ((abs_move_row == 0) and (abs_move_col == 1))
    )


class Knight(Piece):
  """
  Knight
  """
  def __init__(self, game: Game, player: int) -> None:
    super().__init__(game, player)

  def __repr__(self) -> str:
    return f'{"w" if self.player == 0 else "b"}K'
  
  def valid_move(self, move) -> bool:
    current_location = self.game.locations[self.player]
    abs_move_row = abs(move[0] - current_location[0]) 
    abs_move_col = abs(move[1] - current_location[1]) 
    return (
      ((abs_move_row == 1) and (abs_move_col == 2))
      or
      ((abs_move_row == 2) and (abs_move_col == 1))
    )


class CollectCoinsGame:
  """
  CollectCoinsGame
  """

  def __init__(self, pieces=['rock', 'rock']):
    assert len(pieces) == 2
    assert all(piece in ['rock', 'knight'] for piece in pieces)
    game = self
    self.pieces: List[Piece] = [
      Rock(game, i) if piece == 'rock' else Knight(game, i) for i, piece in enumerate(pieces)
    ]
    self.board = np.full((8, 8), True, dtype=bool)
    self.locations = [(0, 0), (7, 7)]
    for loc in self.locations:
      self.board[loc] = False
    self.coins = [0, 0]
    self.turn = 0

  def make_move(self, player, move) -> None:
    move = tuple(move)
    assert self.turn == player
    if move is not None:
      assert self.valid_move(player, move)
      self.coins[player] += int(self.board[move])
      self.board[move] = False
      self.locations[player] = move
    self.turn = 1 - self.turn

  def valid_move(self, player, move):
    move = tuple(move)
    if any(c < 0 or c > 7 for c in move):
      return False
    if any(location == move for location in self.locations):
      return False
    return self.pieces[player].valid_move(move)

  def render(self):
    horizontal = '-' * (self.board.shape[1] * 4 + 1)
    print(horizontal)
    for which_row, row in enumerate(self.board):
      row_copy = [f'{" "}{x}' for x in map({0: ' ', 1: '$'}.get, row)]
      for i, (player_location, player_piece) in enumerate(zip(self.locations, self.pieces)):
        if player_location[0] == which_row:
          row_copy[player_location[1]] = str(player_piece)
      print("|" + " |".join(row_copy) + " |")
      print(horizontal)
    print()
    print(f'{self.coins[0]}/{self.coins[1]}')

  def is_done(self) -> bool:
    return np.sum(self.board) < 1
