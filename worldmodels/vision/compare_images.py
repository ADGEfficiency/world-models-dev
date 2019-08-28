
import os

import matplotlib.pyplot as plt
import numpy as np

from worldmodels.dataset.car_racing import CarRacingWrapper
from worldmodels.vision.vae import VAE
from worldmodels.params import vae_params, results_dir


def compare_images(model, sample_observations, image_dir):
    """ side by side comparison of image and reconstruction """
    reconstructed = model.forward(sample_observations)

    fig, axes = plt.subplots(
        nrows=sample_observations.shape[0],
        ncols=2,
        figsize=(5, 8)
    )

    for idx in range(sample_observations.shape[0]):
        actual_ax = axes[idx, 0]
        reconstructed_ax = axes[idx, 1]

        actual_ax.imshow(sample_observations[idx, :, :, :])
        reconstructed_ax.imshow(reconstructed[idx, :, :, :])
        actual_ax.set_axis_off()
        reconstructed_ax.set_axis_off()

        actual_ax.set_aspect('equal')
        reconstructed_ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(image_dir, 'compare.png'))


if __name__ == '__main__':

    vae_params['load_model'] = True
    model = VAE(**vae_params)

    def sample_observations():
        env = CarRacingWrapper()
        obs = env.reset()

        done = False
        observations = []
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            observations.append(obs)

        observations = np.array(observations)
        np.save(os.path.join(results_dir, 'sample-obs.npy'), observations)
        return observations

    observations = np.load(os.path.join(results_dir, 'sample-obs.npy'))
    sample = observations[np.random.randint(0, high=observations.shape[0], size=8)].astype(np.float32)

    compare_images(model, sample, os.path.join(results_dir))

