from qwertyenv.take_5_pz import Take5Env
from qwertyenv.pz_to_gymnasium_wrappers import parallel_to_gymnasium


def main():
    env = parallel_to_gymnasium(Take5Env(num_players=3), external_agent=0, act_others=lambda agent, obs: obs['action_mask'].index(1))

    def evaluate():
        won = 0
        lost = 0
        for _ in range(1_000):
            env.reset()
            obs, info = env.reset()
            while True:
                action = obs['action_mask'].index(1)
                obs, reward, terminated, _, info = env.step(action)
                if terminated:
                    if info['won'] == 0:
                        won += 1
                    else:
                        lost += 1
                    break
        print(f'{won}/{won + lost}')

    def train():
        pass

    obs, info = env.reset()
    env.render()
    while True:
        action = obs['action_mask'].index(1)
        obs, reward, terminated, _, info = env.step(action)
        env.render()
        if terminated:
            break

    evaluate()
    train()
    evaluate()


if __name__ == "__main__":
    main()