"""
equivilant of series.py in original paper code

loads dataset (entire episode)

loads VAE

get mu and theta (from sampled obs)

sample a different latent vector each time

used after VAE training, before memory training

"""
import argparse
from os.path import join
from os import makedirs, environ

import numpy as np
import tensorflow as tf

from worldmodels.dataset.upload_to_s3 import S3
from worldmodels.params import vae_params
from worldmodels.dataset.tf_records import encode_floats, batch_episodes, parse_random_rollouts
from worldmodels.vision.vae import VAE


home = environ['HOME']
results_dir = join(home, 'world-models-experiments', 'latent-stats')
makedirs(results_dir, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_start', default=0, nargs='?', type=int)
    parser.add_argument('--episodes', default=10000, nargs='?', type=int)
    args = parser.parse_args()
    print(args)

    episode_start = args.episode_start
    episodes = args.episodes

    model = VAE(**vae_params)

    print('setting env variables for AWS')
    environ["AWS_REGION"] = "eu-central-1"
    environ["AWS_LOG_LEVEL"] = "3"

    s3 = S3()
    records = s3.list_all_objects('random-rollouts')
    records = records[episode_start: episode_start + episodes]
    dataset = batch_episodes(parse_random_rollouts, records, episode_length=1000)

    for episode in range(episode_start, episode_start + episodes):
        print('processing episode {}'.format(episode))
        obs, act = next(dataset)
        assert obs.shape[0] == 1000
        mu, logvar = model.encode(obs)

        path = join(results_dir, 'episode{}.tfrecord'.format(episode))
        print('saving to {}'.format(path))
        with tf.io.TFRecordWriter(path) as writer:
            encoded = encode_floats({
                # 'observation': obs.numpy(),
                'action': act.numpy(),
                'mu': mu.numpy(),
                'logvar': logvar.numpy(),
            })
            writer.write(encoded)
