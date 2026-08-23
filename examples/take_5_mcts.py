from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from tqdm import trange

from qwertyenv.take_5_pz import Take5Env
from qwertyenv.pz_to_gymnasium_wrappers import parallel_to_gymnasium


@dataclass
class MCTSNode:
    visits: int = 0
    value: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)


def possible_copy(env):
    """Return a determinization of the wrapped Take 5 environment.

    The board and the external player's hand are observable.  Deal every
    other remaining card randomly while keeping each hidden hand and the
    draw pile the same size as in the current game.
    """
    simulation_env = deepcopy(env)
    parallel_env = simulation_env._parallel_env
    game = parallel_env._game

    if game is None:
        raise RuntimeError("the environment must be reset before it can be copied")

    external_player = int(simulation_env._external_agent)
    known_cards = {
        card
        for row in game._board
        for card in row
    }
    known_cards.update(game._players[external_player].cards)

    unknown_cards = [
        card
        for card in range(1, parallel_env._num_cards + 1)
        if card not in known_cards
    ]
    random.shuffle(unknown_cards)

    offset = 0
    for player_index, player in enumerate(game._players):
        if player_index == external_player:
            continue
        hand_size = len(player.cards)
        player.cards = sorted(unknown_cards[offset:offset + hand_size])
        offset += hand_size

    game._cards = unknown_cards[offset:]

    simulation_env._observations, simulation_env._infos = (
        parallel_env._get_obs_and_info()
    )
    return simulation_env


class MCTSAgent:
    """A UCT agent for the single external player in the wrapped environment."""

    def __init__(self, simulations=16, rollout_depth=10, exploration=math.sqrt(2)):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration

    @staticmethod
    def _actions(observation):
        return [
            action
            for action, allowed in enumerate(observation["action_mask"])
            if allowed
        ]

    def choose_action(self, env, observation):
        actions = self._actions(observation)
        if len(actions) == 1:
            return actions[0]

        root = MCTSNode()
        for _ in range(self.simulations):
            simulation_env = possible_copy(env)
            simulation_observation = observation
            path = [root]

            for _ in range(self.rollout_depth):
                node = path[-1]
                available_actions = self._actions(simulation_observation)
                if not available_actions:
                    break

                unvisited = [
                    action for action in available_actions if action not in node.children
                ]
                if unvisited:
                    action = random.choice(unvisited)
                    child = MCTSNode()
                    node.children[action] = child
                    path.append(child)
                    value = self._advance(simulation_env, action)
                    if value is None:
                        value = self._rollout(simulation_env, self.rollout_depth)
                    break

                action = max(
                    available_actions,
                    key=lambda candidate: self._uct(node, node.children[candidate]),
                )
                path.append(node.children[action])
                value = self._advance(simulation_env, action)
                if value is not None:
                    break
                simulation_observation = simulation_env._last_observation
            else:
                value = 0.0

            for node in path:
                node.visits += 1
                node.value += value

        return max(actions, key=lambda action: root.children.get(action, MCTSNode()).visits)

    def _uct(self, parent, child):
        if child.visits == 0:
            return float("inf")
        return child.value / child.visits + self.exploration * math.sqrt(
            math.log(parent.visits) / child.visits
        )

    @staticmethod
    def _advance(env, action):
        observation, reward, terminated, truncated, info = env.step(action)
        env._last_observation = observation
        if terminated or truncated:
            if info.get("won") == 0:
                return 100.0 + reward
            return -100.0 + reward
        return None

    def _rollout(self, env, depth):
        total = 0.0
        for _ in range(depth):
            observation = env._last_observation
            actions = self._actions(observation)
            if not actions:
                return total
            result = self._advance(env, random.choice(actions))
            if result is not None:
                return total + result
        return total


def random_valid_action(observation):
    actions = [
        action
        for action, allowed in enumerate(observation["action_mask"])
        if allowed
    ]
    return random.choice(actions)


def main(eval_episodes=100):
    agent = MCTSAgent() # already better than 1 / 3
    # try also the following..
    # MCTSAgent(simulations=64, rollout_depth=10)
    # MCTSAgent(simulations=128, rollout_depth=15)
    # MCTSAgent(simulations=256, rollout_depth=20)
    env = parallel_to_gymnasium(
        Take5Env(num_players=3),
        external_agent=0,
        act_others=lambda agent, obs: random_valid_action(obs),
    )

    def evaluate():
        won = 0
        lost = 0
        for _ in trange(eval_episodes, desc="evaluation"):
            obs, info = env.reset()
            while True:
                action = agent.choose_action(env, obs)
                obs, reward, terminated, _, info = env.step(action)
                if terminated:
                    if info['won'] == 0:
                        won += 1
                    else:
                        lost += 1
                    break
        print(f'{won}/{won + lost}')


    obs, info = env.reset()
    env.render()
    while True:
        action = agent.choose_action(env, obs)
        obs, reward, terminated, _, info = env.step(action)
        env.render()
        if terminated:
            break

    evaluate()


if __name__ == "__main__":
    main()