from gymnasium.envs.registration import register
from .ensure_valid_action import EnsureValidAction
from .up_down_left_right import UpDownLeftRight
from .pz_to_gymnasium_wrappers import aec_to_gymnasium, parallel_to_gymnasium

__version__ = "0.2.0"


register(
    id='qwertyenv/BlackJack-v0',
    entry_point='qwertyenv.black_jack:BJEnv',
    max_episode_steps=300
)

register(
    id='qwertyenv/CollectCoins-v0',
    entry_point='qwertyenv.collect_coins:CollectCoinsEnv',
    max_episode_steps=300
)

register(
    id='qwertyenv/Take5-v0',
    entry_point='qwertyenv.take_5_pz:Take5Env',
    max_episode_steps=300
)

__all__ = [
    EnsureValidAction,
    UpDownLeftRight,
    aec_to_gymnasium,
    parallel_to_gymnasium
]