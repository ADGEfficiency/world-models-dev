"""
equivilant of series.py in original paper code

loads dataset (entire episode)

loads VAE

get mu and theta (from sampled obs)

this allows us to sample a different latent vector each time
"""

import argparse
from os.path import join

import tensorflow as tf

from worldmodels.params import vae_params, results_dir
from worldmodels.dataset.tf_records import encode_floats, batch_episodes, parse_random_rollouts
from worldmodels.vision.vae import VAE
from worldmodels.utils import list_records, make_directories


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_start', default=0, nargs='?', type=int)
    parser.add_argument('--episodes', default=10000, nargs='?', type=int)
    parser.add_argument('--data', default='local', nargs='?')
    args = parser.parse_args()
    print(args)

    records = list_records('random-rollouts', 'episode', args.data)
    make_directories('latent-stats')
    results_dir = join(results_dir, 'latent-stats')

    episode_start = args.episode_start
    episodes = args.episodes
    records = records[episode_start: episode_start + episodes]
    dataset = batch_episodes(parse_random_rollouts, records, episode_length=1000)

    model = VAE(**vae_params)

    for episode in range(episode_start, episode_start + episodes):
        print('processing episode {}'.format(episode))
        obs, act = next(dataset)
        assert obs.shape[0] == 1000
        mu, logvar = model.encode(obs)

        path = join(results_dir, 'episode{}.tfrecord'.format(episode))
        print('saving to {}'.format(path))
        with tf.io.TFRecordWriter(path) as writer:
            encoded = encode_floats({
                'action': act.numpy(),
                'mu': mu.numpy(),
                'logvar': logvar.numpy(),
            })
            writer.write(encoded)
