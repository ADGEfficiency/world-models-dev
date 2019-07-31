import argparse
from functools import partial
from multiprocessing import Pool
import os

import numpy as np
import tensorflow as tf

from worldmodels.dataset.car_racing import CarRacingWrapper, rollout
from worldmodels.dataset.tf_records import encode_floats


results_path = os.path.join(
    os.environ['HOME'],
    'world-models-experiments/random-rollouts'
)


def save_episode(results, process_id):
    """ results dictionary to .tfrecord """
    episode = np.random.randint(low=0, high=10000000)

    path = os.path.join(
        results_path,
        'process{}-episode{}.tfrecord'.format(process_id, episode)
    )

    print('saving to {}'.format(path))
    with tf.io.TFRecordWriter(path) as writer:
        for obs, act in zip(results['observation'], results['action']):
            encoded = encode_floats({'observation' : obs, 'action': act})
            writer.write(encoded)

    paths = os.listdir(results_path)
    episodes = [path for path in paths if 'episode' in path]
    print('{} episodes stored locally'.format(len(episodes)))


def rollouts(
    process_id,
    num_rollouts,
    agent,
    env,
    debug,
    max_length,
    results_path,
):
    for episode in range(num_rollouts):
        results = rollout(
            agent=agent,
            env=env,
            max_length=max_length,
            debug=debug
        )
        print('process {} episode {} length {}'.format(
            process_id, episode, len(results['observation'])
        ))

        save_episode(results, process_id)


class RandomAgent:

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', default=2, nargs='?', type=int)
    parser.add_argument('--total_episodes', default=10, nargs='?', type=int)
    parser.add_argument('--debug', default=0, nargs='?', type=int)
    parser.add_argument('--episode_length', default=1000, nargs='?', type=int)
    args = parser.parse_args()
    print(args)

    num_process = int(args.num_process)
    total_episodes = args.total_episodes
    episodes_per_process = int(total_episodes / num_process)
    debug = bool(args.debug)
    max_length = args.episode_length

    os.makedirs(results_path, exist_ok=True)

    env = CarRacingWrapper
    agent = RandomAgent

    with Pool(num_process) as p:
        p.map(
            partial(
                rollouts,
                num_rollouts=episodes_per_process,
                agent=agent,
                env=env,
                debug=debug,
                max_length=max_length,
                results_path=results_path,
            ),
            range(num_process)
        )
