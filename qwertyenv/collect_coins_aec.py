from copy import copy
from functools import lru_cache
import random

import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from qwertyenv.collect_coins_game import (
    B_R,
    COIN,
    DIAMOND,
    EMPTY,
    RookRays,
    W_R,
    CollectCointsGame,
)


class CollectCoinsAEC(AECEnv):
    metadata = {"name": "collect_coins_aec_v0"}

    _positions = tuple(f"{col}{row}" for row in "12345678" for col in "abcdefgh")
    _position_to_action = {position: action for action, position in enumerate(_positions)}
    _piece_values = {EMPTY: 0, COIN: 1, DIAMOND: 2, W_R: 3, B_R: 4}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["white", "black"]
        self.agents = []
        self._game = None
        self._agent_selector = None
        self.terminations = {}
        self.truncations = {}
        self.rewards = {}
        self.infos = {}
        self._cumulative_rewards = {}

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(64)

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "board": MultiDiscrete(np.full((8, 8), 5, dtype=np.int64)),
            "action_mask": MultiBinary(64),
        })

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._game = CollectCointsGame()
        self._game._state.board["d4"] = DIAMOND
        self.agents = copy(self.possible_agents)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        board = self._game._state.board
        own_piece = W_R if agent == "white" else B_R
        other_piece = B_R if agent == "white" else W_R
        encoded_board = np.array(
            [
                3 if board[position] == own_piece else
                4 if board[position] == other_piece else
                self._piece_values[board[position]]
                for position in self._positions
            ],
            dtype=np.int64,
        ).reshape(8, 8)
        mask = np.zeros(64, dtype=np.int8)
        if not self.terminations[agent] and not self.truncations[agent]:
            for position in self._legal_targets(agent):
                mask[self._position_to_action[position]] = 1
        return {"board": encoded_board, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        legal_targets = self._legal_targets(agent)
        if not self.action_space(agent).contains(action):
            raise ValueError(f"Action must be an integer from 0 through 63, got {action!r}")
        target = self._positions[action]
        if target not in legal_targets:
            raise ValueError(f"Action {action} does not select a legal target square")

        score_before = self._game._scores["w" if agent == "white" else "b"]
        rook_position = self._rook_position(agent)
        self._game.step(f"{rook_position}{target}")
        score_after = self._game._scores["w" if agent == "white" else "b"]

        self._clear_rewards()
        self.rewards[agent] = score_after - score_before
        self._cumulative_rewards[agent] = 0

        done = self._game.is_done() or not self._legal_targets(self._next_agent(agent))
        self.terminations = {current_agent: done for current_agent in self.agents}
        if done:
            self._accumulate_rewards()
            return

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def close(self):
        self._game = None

    def render(self):
        if self._game is None:
            print("game is closed")
            return
        board = self._game._state.board
        print()
        for row in reversed("12345678"):
            print("".join("." if board[f"{col}{row}"] == EMPTY else board[f"{col}{row}"] for col in "abcdefgh"))
        print(f"turn={self.agent_selection}")
        print(f"scores={dict(self._game._scores)}")

    def _rook_position(self, agent):
        piece = W_R if agent == "white" else B_R
        return next(position for position, value in self._game._state.board.items() if value == piece)

    def _next_agent(self, agent):
        return "black" if agent == "white" else "white"

    def _legal_targets(self, agent):
        rook_position = self._rook_position(agent)
        rook = self._game._piece_for(
            W_R if agent == "white" else B_R, rook_position
        )
        board = self._game._state.board
        targets = []
        for ray in rook._rays:
            for position in ray:
                value = board[position]
                if value not in [EMPTY, COIN, DIAMOND]:
                    break
                targets.append(position)
                if value in [COIN, DIAMOND]:
                    break
        return targets


def test():
    import random

    env = CollectCoinsAEC()
    env.reset(1)
    env.render()
    for _ in range(100):
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        valid_actions = np.flatnonzero(env.observe(agent)["action_mask"])
        if len(valid_actions) == 0:
            break
        env.step(random.choice(valid_actions.tolist()))
        env.render()
    print(f"moves completed: {env._game._state.move_number}")
    env.close()
    print(f"game after close: {env._game}")


if __name__ == "__main__":
    test()