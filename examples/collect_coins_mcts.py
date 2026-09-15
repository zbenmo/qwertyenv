from copy import deepcopy
from dataclasses import dataclass, field
import math
import random

import numpy as np

from qwertyenv.collect_coins_aec import CollectCoinsAEC


GAME_OPTIONS = (
    (
        "Rook vs rook: coins",
        "r$6/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7R w",
    ),
    (
        "Rook vs rook: coins and diamond",
        "r$6/$$$$$$$$/$$$$$$$$/$$$*$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7R w",
    ),
    (
        "Knight vs knight: coins",
        "n$6/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7N w",
    ),
    (
        "Knight vs knight: coins and diamond",
        "n$6/$$$$$$$$/$$$$$$$$/$$$*$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7N w",
    ),
    (
        "Rook vs knight: coins",
        "r$6/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7N w",
    ),
    (
        "Rook vs knight: coins and diamonds",
        "r$6/$$$$$$$$/$$$*$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$$$$$$/7N w",
    ),
    (
        "Rook vs rook: coin islands",
        "r7/$$$5/8/8/3$$3/8/5$$$/7R w",
    ),
    (
        "Knight vs knight: coin islands and diamonds",
        "n7/$$$5/8/3*4/3$$3/8/5$$$/7N w",
    ),
    (
        "Two bishops vs two bishops: coins and diamond",
        "1b4b1/$$$$$$$$/$$$$$$$$/$$$$$$$$/$$$*$$$$/$$$$$$$$/$$$$$$$$/1B4B1 w",
    ),
)


@dataclass
class MCTSNode:
    visits: int = 0
    value: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)


class MCTSAgent:
    """UCT planner for one player in the Collect Coins AEC environment."""

    def __init__(self, player, simulations=256, rollout_depth=32, exploration=math.sqrt(2)):
        self.player = player
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.last_action_visits = {}

    @staticmethod
    def _actions(env):
        agent = env.agent_selection
        return np.flatnonzero(env.observe(agent)["action_mask"]).tolist()

    def choose_action(self, env):
        if env.agent_selection != self.player:
            raise ValueError(f"expected {self.player}'s turn, got {env.agent_selection}")

        actions = self._actions(env)
        if not actions:
            raise ValueError(f"no legal actions for {self.player}")
        if len(actions) == 1:
            self.last_action_visits = {actions[0]: 1}
            return actions[0]

        root = MCTSNode()
        for _ in range(self.simulations):
            simulation_env = deepcopy(env)
            node = root
            path = [node]
            value = self._score_difference(simulation_env)

            for _ in range(self.rollout_depth):
                available_actions = self._actions(simulation_env)
                if not available_actions or self._is_terminal(simulation_env):
                    break
                if simulation_env.agent_selection != self.player:
                    simulation_env.step(random.choice(available_actions))
                    continue

                unvisited = [action for action in available_actions if action not in node.children]
                if unvisited:
                    action = random.choice(unvisited)
                    child = MCTSNode()
                    node.children[action] = child
                    path.append(child)
                    simulation_env.step(action)
                    value = self._rollout(simulation_env)
                    break

                action = max(
                    available_actions,
                    key=lambda candidate: self._uct(node, node.children[candidate]),
                )
                node = node.children[action]
                path.append(node)
                simulation_env.step(action)
            else:
                value = self._score_difference(simulation_env)

            for visited_node in path:
                visited_node.visits += 1
                visited_node.value += value

        self.last_action_visits = {
            action: root.children.get(action, MCTSNode()).visits
            for action in actions
        }
        return max(actions, key=lambda action: self.last_action_visits[action])

    def _rollout(self, env):
        for _ in range(self.rollout_depth):
            if self._is_terminal(env):
                break
            actions = self._actions(env)
            if not actions:
                break
            env.step(random.choice(actions))
        return self._score_difference(env)

    def _score_difference(self, env):
        scores = env._game._scores
        opponent = "black" if self.player == "white" else "white"
        return scores[self.player[0]] - scores[opponent[0]]

    @staticmethod
    def _is_terminal(env):
        return any(env.terminations.values()) or any(env.truncations.values())

    def _uct(self, parent, child):
        if child.visits == 0:
            return float("inf")
        return child.value / child.visits + self.exploration * math.sqrt(
            math.log(parent.visits) / child.visits
        )


def action_label(env, action):
    source_index, target_index = divmod(action, 64)
    source = env._positions[source_index]
    target = env._positions[target_index]
    return f"{action}: {source}-{target}"


def choose_human_action(env):
    actions = MCTSAgent._actions(env)
    print("Legal moves: " + ", ".join(action_label(env, action) for action in actions))
    while True:
        answer = input("Choose a move (number or move, e.g. b1-d3): ").strip().lower()
        try:
            action = int(answer)
        except ValueError:
            matches = [
                action for action in actions
                if action_label(env, action).split(": ", 1)[1] == answer
            ]
            action = matches[0] if len(matches) == 1 else -1
        if action in actions:
            return action
        print("Please choose one of the listed legal moves.")


def choose_game_option():
    print("Choose a game:")
    for number, (description, _) in enumerate(GAME_OPTIONS, start=1):
        print(f"  {number}. {description}")

    while True:
        answer = input(f"Select 1-{len(GAME_OPTIONS)}: ").strip()
        try:
            option = int(answer)
        except ValueError:
            option = 0
        if 1 <= option <= len(GAME_OPTIONS):
            return GAME_OPTIONS[option - 1]
        print(f"Please select a number from 1 to {len(GAME_OPTIONS)}.")


def main_play(
    human_player="white",
    seed=None,
    simulations=256,
    rollout_depth=32,
    fen=None,
):
    if fen is None:
        description, fen = choose_game_option()
        print(f"Starting: {description}")
    planner_player = "black" if human_player == "white" else "white"
    planner = MCTSAgent(planner_player, simulations, rollout_depth)
    env = CollectCoinsAEC(fen=fen)
    env.reset(seed=seed)

    try:
        while not MCTSAgent._is_terminal(env):
            env.render()
            if env.agent_selection == human_player:
                action = choose_human_action(env)
            else:
                action = planner.choose_action(env)
                print(f"Planner chooses {action_label(env, action)}")
            env.step(action)
        env.render()
        print(f"Final scores: {dict(env._game._scores)}")
    finally:
        env.close()


if __name__ == "__main__":
    main_play()