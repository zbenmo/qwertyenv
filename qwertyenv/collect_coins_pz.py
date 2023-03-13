import functools
from itertools import product
import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import random

from qwertyenv.collect_coins_game import CollectCoinsGame


class CollectCoinsEnv(AECEnv):
  """
  CollectCoins PettingZoo environment.

  This a Chess like, two players turns board game.
  The aim of the game is to collect more coins than the opponent.
  On each turn a the player's piece makes a move.
  If a coin is present in the destination the count is increased for the player.
  The game ends when there are not more coins on the board to collect. We then compare which player got the most.

  Note: a 'rock' can make a 1 distance step only. In the future we'll have also the regular interpretation for rock.

  Note: pieces don't eat one another and cannot step on each other.
  Pieces can only move to a free square with a legal move, eather collecting a coin, or not collecting a conin if there is no coin in the destination. 
  """

  metadata = {"render_modes": ["human"], "name": "CollectCoins_v0"}

  def __init__(self, pieces=['rock', 'rock'], render_mode=None, with_mask=False):
    """
    player 0 is the first entry, AKA white
    player 1 is the second entry, AKA black
    """

    self.render_mode = render_mode
    self.with_mask = with_mask
    self.possible_agents = ["player_" + str(r) for r in range(2)]
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents))))
    )
    obs_space = dict(
      board = gym.spaces.MultiBinary([8, 8]),
      player = gym.spaces.Tuple([gym.spaces.Discrete(8)] * 2),
      other_player = gym.spaces.Tuple([gym.spaces.Discrete(8)] * 2)
    )    
    self._observation_space = gym.spaces.Dict(spaces=obs_space)
    self._action_space = gym.spaces.Tuple([gym.spaces.Discrete(8)] * 2)

    self.pieces = pieces
    self.game = None
    self.previous_coins = None
    self.reset()

  # this cache ensures that same space object is returned for the same agent
  # allows action space seeding to work as expected
  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
      # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
      return self._observation_space

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
      return self._action_space

  def reset(self, seed=None, options=None, return_info=False):
    """
    Reset needs to initialize the following attributes
    - agents
    - rewards
    - _cumulative_rewards
    - terminations
    - truncations
    - infos
    - agent_selection
    And must set up the environment so that render(), step(), and observe()
    can be called without issues.
    Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
    """
    self.game = CollectCoinsGame(self.pieces)

    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.terminations = {agent: False for agent in self.agents}
    self.truncations = {agent: False for agent in self.agents}
    self.infos = {agent: {} for agent in self.agents}
    self.state = {agent: None for agent in self.agents}
    self.observations = {agent: self._get_observation(self.agent_name_mapping[agent]) for agent in self.agents} # ??
    self.num_moves = 0
    """
    Our agent_selector utility allows easy cyclic stepping through the agents list.
    """
    self._agent_selector = agent_selector(self.agents)
    self.agent_selection = self._agent_selector.next()

    self.previous_coins = 0

  # def step(self, action):
  #   return self._get_observation(), self._calc_reward(), self.game.is_done(), info

  def step(self, action):
      """
      step(action) takes in an action for the current agent (specified by
      agent_selection) and needs to update
      - rewards
      - _cumulative_rewards (accumulating the rewards)
      - terminations
      - truncations
      - infos
      - agent_selection (to the next agent)
      And any internal state used by observe() or render()
      """

      if not isinstance(action, (tuple, np.ndarray, list)):
          action = (action // 8, action % 8)  # because for example, Tianshou may return np.int64 ?

      action = tuple(action) # because for example, Tianshou may return list
      assert isinstance(action, tuple)
      assert len(action) == 2

      if (
          self.terminations[self.agent_selection]
          or self.truncations[self.agent_selection]
      ):
          # handles stepping an agent which is already dead
          # accepts a None action for the one agent, and moves the agent_selection to
          # the next dead agent,  or if there are no more dead agents, to the next live agent
          self._was_dead_step(action)
          return

      agent = self.agent_selection
      self.game.make_move(self.agent_name_mapping[agent], action)

      # the agent which stepped last had its _cumulative_rewards accounted for
      # (because it was returned by last()), so the _cumulative_rewards for this
      # agent should start again at 0
      self._cumulative_rewards[agent] = 0

      # stores action of current agent
      self.state[self.agent_selection] = action

      # collect reward if it is the last agent to act
      if self._agent_selector.is_last():
          # rewards for all agents are placed in the .rewards dictionary
          self.rewards.update({
            self.agents[agent_idx]: self._calc_reward(agent_idx) for agent_idx in range(2) 
          })

          self.num_moves += 1
          # The truncations dictionary must be updated for all players.
          self.truncations = {
              agent: False for agent in self.agents # TODO: ?
          }

          # observe the current state
          for i in self.agents:
              self.observations[i] = self._get_observation(self.agent_name_mapping[i]) # self.state[
              #     self.agents[1 - self.agent_name_mapping[i]]
              # ]
      else:
          # necessary so that observe() returns a reasonable observation at all times.
          self.state[self.agents[1 - self.agent_name_mapping[agent]]] = 0 # TODO:??
          # no rewards are allocated until both players give an action
          self._clear_rewards()

      # selects the next agent.
      self.agent_selection = self._agent_selector.next()
      # Adds .rewards to ._cumulative_rewards
      self._accumulate_rewards()

      if self.render_mode == "human":
          self.render()

  # def _play_other(self):
  #   """
  #   a random move for now
  #   """

  #   move = self.provide_alternative_valid_action(None, self.other_player)
  #   self.game.make_move(self.other_player, move)

  def _get_observation(self, player):
    obs = dict(
      board=self.game.board,
      player=self.game.locations[player],
      other_player=self.game.locations[1 - player]
    )

    if self.with_mask:
       obs['mask'] = [True] * 8 * 8 # !!!

    return obs

  def _calc_reward(self, player):
    current_coins = self.game.coins[player]
    previous_coins = self.previous_coins
    self.previous_coins = current_coins
    done = self.game.is_done()
    if done:
      draw = self.game.coins[player] == self.game.coins[1 - player]
      win = self.game.coins[player] > self.game.coins[1 - player]
      return 0 if draw else (1 if win else -1)
    else:
      return (current_coins - previous_coins) * 0.01

  def render(self, *args, **argv):
    self.game.render()

  def observe(self, agent):
      """
      Observe should return the observation of the specified agent. This function
      should return a sane observation (though not necessarily the most up to date possible)
      at any time after reset() is called.
      """
      # observation of one agent is the previous state of the other
      return np.array(self.observations[agent])

  def close(self):
      """
      Close should release any graphical displays, subprocesses, network connections
      or any other environment data which should not be kept around after the
      user is no longer using the environment.
      """
      pass

  def check_action_valid(self, action, player=None) -> bool:
    """
    This helper function can be used when initializing EnsureValidAction wrapper as an example (see in examples).
    """

    if not isinstance(action, (tuple, np.ndarray, list)):
        action = (action // 8, action % 8)  # because for example, Tianshou may return np.int64 ?

    action = tuple(action)
    assert isinstance(action, tuple)
    assert len(action) == 2

    agent = self.agent_selection
    player = self.agent_name_mapping[agent] if player is None else player

    return self.game.valid_move(
      player,
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
