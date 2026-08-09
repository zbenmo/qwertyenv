from copy import copy
import functools
import random

import numpy as np
# from typing import Optional

from qwertyenv.take_5_game import Take5Game
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete, Tuple


class Take5Env(ParallelEnv):
    """Take5Env
    """

    metadata = {
        "name": "Take5_v0",
    }

    NUM_ROWS = 4
    NUM_COLS = 5

    def __init__(self, num_players: int, num_cards=104):
        super().__init__()
        self.possible_agents = list(range(num_players))
        self.agents = self.possible_agents.copy()
        self._num_cards = num_cards
        self._game = None # must reset first
        self._truncations = {a: False for a in self.possible_agents} # this will always be used

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._game = Take5Game(self.num_agents)
        self.agents = copy(self.possible_agents)
        obs, info = self._get_obs_and_info()
        return obs, info

    def step(self, actions):
        negative_points_before = [
            player.negative_points for player in self._game._players
        ]
        self._game.step([actions[a] for a in self.agents])
        observations, infos = self._get_obs_and_info()
        rewards = {
            a: before - after
            for a, before, after in zip(
                self.agents, negative_points_before, [
                    player.negative_points for player in self._game._players
                ])
        }
        done: bool = self._game.is_done()
        terminations = {a: done for a in self.agents}
        truncations = self._truncations.copy()

        if done:
            self.agents = []

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def render(self):
        self._game.render()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self._num_cards + 1)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Tuple((MultiDiscrete((Take5Env.NUM_ROWS, Take5Env.NUM_COLS)), MultiDiscrete(10)))

    def _get_obs_and_info(self):
        board = self._game._board
        cols = self._game._threshold - 1
        board_obs = np.full((len(board), cols), -1, dtype=np.int32)
        for i, row in enumerate(board):
            board_obs[i, :len(row)] = row

        obs = {
            a: {
                'observation': 
                    (board_obs, np.full(10, -1, dtype=np.int32)), # Tuple: board, players cards
            }
            for a in self.agents
        }
        for a, player in zip(self.agents, self._game._players):
            cards = player.cards
            obs[a]['observation'][1][:len(cards)] = cards
            obs[a]['action_mask'] = [1 if i in player.cards else 0 for i in range(104 + 1)]

        info = {
            a: {}
            for a in self.agents
        }

        return obs, info


if __name__ == "__main__":
#     from pettingzoo.test import parallel_api_test

#     env = Take5Env(3)
#     parallel_api_test(env, num_cycles=1_000_000)

    env = Take5Env(5)
    obs, info = env.reset()
    env.render()
    while True:
        actions = {
            a: o['action_mask'].index(1) for a, o in obs.items()
        }
        obs, reward, terminated, _, info = env.step(actions)
        env.render()
        if any(terminated.values()):
            break
