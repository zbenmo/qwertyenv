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
    W_R,
    CollectCointsGame,
)


class CollectCoinsAEC(AECEnv):
    metadata = {"name": "collect_coins_aec_v0"}

    _positions = tuple(f"{col}{row}" for row in "12345678" for col in "abcdefgh")
    _position_to_action = {position: action for action, position in enumerate(_positions)}
    _piece_values = {EMPTY: 0, COIN: 1, DIAMOND: 2, W_R: 3, B_R: 4}

    def __init__(self, fen=None):
        super().__init__()
        self._fen = fen
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
        return Discrete(4096)

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "board": MultiDiscrete(np.full((8, 8), 5, dtype=np.int64)),
            "action_mask": MultiBinary(4096),
        })

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        fen = options.get("fen", self._fen) if options else self._fen
        self._game = CollectCointsGame(fen)
        if fen is None:
            self._game._state.board["d4"] = DIAMOND
        self._assert_piece_per_agent()
        self.agents = copy(self.possible_agents)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        first_agent = "white" if self._game._state.turn == "w" else "black"
        agent_order = [first_agent, self._next_agent(first_agent)]
        self._agent_selector = agent_selector.agent_selector(agent_order)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        board = self._game._state.board
        own_positions = {position for _, position in self._piece_positions(agent)}
        other_positions = {
            position for _, position in self._piece_positions(self._next_agent(agent))
        }
        encoded_board = np.array(
            [
                3 if position in own_positions else
                4 if position in other_positions else
                self._piece_values[board[position]]
                for position in self._positions
            ],
            dtype=np.int64,
        ).reshape(8, 8)
        mask = np.zeros(4096, dtype=np.int8)
        if (
            agent == self.agent_selection
            and not self.terminations[agent]
            and not self.truncations[agent]
        ):
            for source, target in self._legal_moves(agent):
                source_index = self._position_to_action[source]
                target_index = self._position_to_action[target]
                mask[source_index * 64 + target_index] = 1
        return {"board": encoded_board, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        if not self.action_space(agent).contains(action):
            raise ValueError(f"Action must be an integer from 0 through 4095, got {action!r}")
        source_index, target_index = divmod(int(action), 64)
        source = self._positions[source_index]
        target = self._positions[target_index]
        if (source, target) not in self._legal_moves(agent):
            raise ValueError(f"Action {action} does not select a legal move")

        score_before = self._game._scores["w" if agent == "white" else "b"]
        self._game.step(f"{source}{target}")
        score_after = self._game._scores["w" if agent == "white" else "b"]

        self._clear_rewards()
        self.rewards[agent] = score_after - score_before
        self._cumulative_rewards[agent] = 0

        done = self._game.is_done() or not self._legal_moves(self._next_agent(agent))
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
        files = " ".join("abcdefgh")
        print(f"  {files}")
        for row in reversed("12345678"):
            squares = " ".join(
                "." if board[f"{col}{row}"] == EMPTY else board[f"{col}{row}"]
                for col in "abcdefgh"
            )
            print(f"{row} {squares} {row}")
        print(f"  {files}")
        print(f"turn={self.agent_selection}")
        print(f"scores={dict(self._game._scores)}")

    def _piece_positions(self, agent):
        is_white = agent == "white"
        pieces = [
            (piece, position)
            for position, piece in self._game._state.board.items()
            if piece not in [EMPTY, COIN, DIAMOND]
            and piece.isupper() == is_white
        ]
        assert pieces, (
            f"expected at least one piece for {agent}, found none"
        )
        return pieces

    def _assert_piece_per_agent(self):
        self._piece_positions("white")
        self._piece_positions("black")

    def _next_agent(self, agent):
        return "black" if agent == "white" else "white"

    def _legal_moves(self, agent):
        legal_moves = []
        for piece_str, piece_position in self._piece_positions(agent):
            piece = self._game._piece_for(piece_str, piece_position)
            legal_moves.extend(
                (move[:2], move[2:4])
                for move, _ in piece.possible_moves(
                    self._game._state, CollectCointsGame.is_checked
                )
            )
        return legal_moves


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