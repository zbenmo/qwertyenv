from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from tqdm import trange

from qwertyenv.take_5_game import Take5Game
from qwertyenv.take_5_pz import Take5Env
from qwertyenv.pz_to_gymnasium_wrappers import parallel_to_gymnasium


@dataclass
class MCTSNode:
    visits: int = 0
    value: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)


def possible_copy(env):
    """Return a determinization of the wrapped Take 5 environment.

    The board, collected cards, and the external player's hand are observable.
    Deal every other remaining card randomly while keeping each hidden hand
    and the draw pile the same size as in the current game.
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
    known_cards.update(
        card
        for player in game._players
        for card in player.cards_collected
    )

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
        self._last_action_visits = {}

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
            self._last_action_visits = {actions[0]: 1}
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

        self._last_action_visits = {
            action: root.children.get(action, MCTSNode()).visits
            for action in actions
        }
        return max(actions, key=lambda action: self._last_action_visits[action])

    def action_goodness(self, action):
        """Return a symbol describing the latest search support for an action."""
        if action not in self._last_action_visits:
            raise RuntimeError("choose_action must be called before action_goodness")

        visits = self._last_action_visits
        ranked_actions = sorted(visits, key=visits.get, reverse=True)
        rank = ranked_actions.index(action)
        if len(ranked_actions) == 1:
            return "="

        percentile = rank / (len(ranked_actions) - 1)
        if percentile <= 0.2:
            return "++"
        if percentile <= 0.4:
            return "+"
        if percentile <= 0.6:
            return "="
        if percentile <= 0.8:
            return "-"
        return "--"

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
    if len(actions) < 1:
        return None
    else:
        return random.choice(actions)


def main(eval_episodes=100):
    agent = MCTSAgent(simulations=128, rollout_depth=15) # MCTSAgent() # already better than 1 / 3
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


def _format_card(card):
    return Take5Game._format_card(card)

def render_obs(obs):
    board, player_hand, played_cards = obs['observation']

    print("-" * 120)
    print("Board:")
    for _, cards in enumerate(board):
        real_cards = [c for c in cards.tolist() if c > -1]
        print(f"  " + " ".join([_format_card(card) for card in real_cards]) + "   _ " * (6 - len(real_cards) - 1) + "   * ")
    print("\nHand(s):")
    real_player_hand = [x for x in player_hand if x > -1]
    print(' '.join(map(_format_card, real_player_hand)))
    for _ in range(2):
        print()
        print(' '.join(map(lambda _: '?', real_player_hand)))
    print()


def render_action_goodness(obs, agent):
    actions = [card for card in obs['observation'][1] if card > -1]
    print("MCTS goodness:")
    print(' '.join(f'{_format_card(card)} ({agent.action_goodness(card)})' for card in actions))

def main_play():
    agent = MCTSAgent(simulations=128, rollout_depth=15)
    env = parallel_to_gymnasium(
        Take5Env(num_players=3),
        external_agent=0,
        act_others=lambda agent, obs: random_valid_action(obs),
    )

    obs, info = env.reset()
    render_obs(obs)
    while True:
        action = agent.choose_action(env, obs)
        render_action_goodness(obs, agent)
        user_action = int(input("what is your action ? "))
        valid_inputs = set(obs['observation'][1])
        valid_inputs.discard(-1)
        while user_action not in valid_inputs:
            print('Please try again')
            user_action = int(input("what is your action ? "))
        obs, reward, terminated, _, info = env.step(user_action)
        print()
        print(f"actions {info['actions']}")
        print(f'reward: {reward}')
        print()
        render_obs(obs)
        if terminated:
            env.render()
            break



if __name__ == "__main__":
    main()
    #main_play()