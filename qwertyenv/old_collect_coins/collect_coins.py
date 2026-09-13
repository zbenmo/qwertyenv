from itertools import product
import gymnasium as gym
import random

from qwertyenv.collect_coins_game import CollectCoinsGame


class CollectCoinsEnv(gym.Env):
  """
  CollectCoins Gym environment.

  This a Chess like, two players, turns, board game.
  The aim of the game is to collect more coins than the opponent.
  On each turn a the player's piece makes a move.
  If a coin is present in the destination the count is increased for the player.
  The game ends when there are not more coins on the board to collect. We then compare which player got the most.
  """

  def __init__(self, pieces=['rock', 'rock'], player=0):
    """
    player 0 is the first entry, AKA white
    player 1 is the second entry, AKA black
    """

    obs_space = dict(
      board = gym.spaces.Box(low=0, high=1, shape=(8 * 8,), dtype=bool),
      player = gym.spaces.MultiDiscrete([8, 8]),
      other_player = gym.spaces.MultiDiscrete([8, 8]),
    )    
    self.observation_space = gym.spaces.Dict(spaces=obs_space)
    self.action_space = gym.spaces.MultiDiscrete([8, 8])

    self.pieces = pieces
    self.player = player
    self.other_player = 1 - player
    self.game = None
    self.previous_coins = None
    self.reset()

  def reset(self, seed=None, options=None):
    self.game = CollectCoinsGame(self.pieces)
    self.previous_coins = 0
    return self._get_observation(), {} 

  def step(self, action):
    self.game.make_move(self.player, action)
    if not self.game.is_done():
      self._play_other()
    info = {}
    return self._get_observation(), self._calc_reward(), self.game.is_done(), False, info

  def _play_other(self):
    """
    a random move for now
    """

    move = self.provide_alternative_valid_action(None, self.other_player)
    self.game.make_move(self.other_player, move)

  def _get_observation(self):
    return dict(
      board=self.game.board.flatten(),
      player=self.game.locations[self.player],
      other_player=self.game.locations[self.other_player]
    )

  def _calc_reward(self):
    current_coins = self.game.coins[self.player]
    previous_coins = self.previous_coins
    self.previous_coins = current_coins
    done = self.game.is_done()
    if done:
      draw = self.game.coins[self.player] == self.game.coins[self.other_player]
      win = self.game.coins[self.player] > self.game.coins[self.other_player]
      return 0 if draw else (1 if win else -1)
    else:
      return (current_coins - previous_coins) * 0.01

  def render(self, *args, **argv):
    self.game.render()

  def check_action_valid(self, action, player=None) -> bool:
    """
    This helper function can be used when initializing EnsureValidAction wrapper as an example (see in examples).
    """

    return self.game.valid_move(
      (self.player if player is None else player),
      action
    )

  def provide_alternative_valid_action(self, action, player=None):
    """
    This helper function can be used when initializing EnsureValidAction wrapper as an example (see in examples).

    Select a random valid move.
    TOOD: maybe consult the provided action(move) and provide an action that is as close as possible.
    """

    all_moves = product(range(8), repeat=2)
    all_valid_moves = [move for move in all_moves if self.check_action_valid(move, player)]
    return None if len(all_valid_moves) < 1 else random.choice(all_valid_moves)
