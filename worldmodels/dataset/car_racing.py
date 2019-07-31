from gym.spaces.box import Box
from collections import defaultdict
from gym.envs.box2d.car_racing import CarRacing
import numpy as np

from PIL import Image
from gym.wrappers import Monitor

from worldmodels.params import results_dir


def process_frame(frame, screen_size=None, vertical_cut=-1, max_val=255, save_img=False):
    """ could be two functions  - resizing and saving"""
    frame = frame[:vertical_cut, :, :]
    frame = Image.fromarray(frame, mode='RGB')

    obs = frame.resize(screen_size)
    if save_img:
        frame.save(results_dir+'/debug/raw_frame{}.png'.format(save_img), 'PNG')
        obs.save(results_dir+'/debug/processed().png'.format(save_img), 'PNG')
    return np.array(obs) / max_val


def rollout(agent, env, max_length, results=None, debug=False):
    """ runs an episode """

    if results is None:
        results = defaultdict(list)

    env = env(debug=debug)
    agent = agent(env)

    done = False
    observation = env.reset()
    step = 0
    while not done:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action, save_img=debug)

        step += 1
        if step >= max_length:
            done = True

        transition = {
                'observation': observation,
                'action': action,
        }

        for key, data in transition.items():
            results[key].append(data)

        observation = next_observation

    env.close()

    return results


class CarRacingWrapper(CarRacing):

    screen_size = (64, 64)

    def __init__(self, debug=False):

        super().__init__()

        self.debug = debug

        #  new observation space to deal with resize
        self.observation_space = Box(
                low=0,
                high=255,
                shape=self.screen_size + (3,)
        )

    def step(self, action, save_img=False):

        frame, reward, done, info = super().step(action)
        self.viewer.window.dispatch_events()

        obs = process_frame(
            frame,
            self.screen_size,
            vertical_cut=84,
            max_val=255.0,
            save_img=save_img
        )

        return obs, reward, done, info

    def reset(self):
        raw = super().reset()
        self.viewer.window.dispatch_events()
        return process_frame(
            raw,
            self.screen_size,
            vertical_cut=84,
            max_val=255.0,
            save_img=False
        )


if __name__ == '__main__':
    env = CarRacingWrapper()

    obs = env.reset()
    d = False
    for step in range(1000):
        o, r, d, i = env.step(env.action_space.sample(), save_img=step)
