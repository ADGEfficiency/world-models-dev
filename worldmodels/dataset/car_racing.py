from PIL import Image

from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
import numpy as np

from worldmodels.params import results_dir


def process_frame(frame, screen_size=None, vertical_cut=-1, max_val=255, save_img=False):
    """ crops & convert to float """
    frame = frame[:vertical_cut, :, :]
    frame = Image.fromarray(frame, mode='RGB')

    obs = frame.resize(screen_size, Image.BILINEAR)
    return np.array(obs) / max_val


class CarRacingWrapper(CarRacing):
    screen_size = (64, 64)

    def __init__(self):
        super().__init__()

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

        if save_img:
            frame.save(
                results_dir+'/debug/raw_frame{}.png'.format(save_img),
                'PNG'
            )

            obs.save(
                results_dir+'/debug/processed().png'.format(save_img),
                'PNG'
            )

        return obs, reward, done, info

    def reset(self):
        raw = super().reset()

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        return process_frame(
            raw,
            self.screen_size,
            vertical_cut=84,
            max_val=255.0,
            save_img=False
        )
