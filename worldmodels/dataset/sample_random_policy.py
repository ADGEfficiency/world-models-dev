import argparse
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import os

import tensorflow as tf

from worldmodels.dataset.car_racing import CarRacingWrapper
from worldmodels.dataset.tf_records import encode_floats
from worldmodels.params import results_dir


results_dir = os.path.join(results_dir, 'random-rollouts')


def random_rollout(env, max_length, results=None):
    """ runs an episode with a random policy """

    if results is None:
        results = defaultdict(list)

    env = env()

    done = False
    observation = env.reset()
    step = 0
    while not done:
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)

        transition = {'observation': observation, 'action': action}

        for key, data in transition.items():
            results[key].append(data)

        observation = next_observation
        step += 1
        if step >= max_length:
            done = True

    env.close()

    return results


def save_episode(results, process_id, episode):
    """ results dictionary to .tfrecord """

    path = os.path.join(
        results_dir,
        'process{}-episode{}.tfrecord'.format(process_id, episode)
    )

    print('saving to {}'.format(path))
    with tf.io.TFRecordWriter(path) as writer:
        for obs, act in zip(results['observation'], results['action']):
            encoded = encode_floats({'observation': obs, 'action': act})
            writer.write(encoded)


def rollouts(
    process_id,
    num_rollouts,
    agent,
    env,
    max_length,
    results_dir,
):
    """ runs many episodes """
    for episode in range(num_rollouts):
        results = random_rollout(
            env=env,
            max_length=max_length,
        )
        print('process {} episode {} length {}'.format(
            process_id, episode, len(results['observation'])
        ))

        save_episode(results, process_id, episode)

        paths = os.listdir(results_dir)
        episodes = [path for path in paths if 'episode' in path]
        print('{} episodes stored locally'.format(len(episodes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', default=2, nargs='?', type=int)
    parser.add_argument('--total_episodes', default=10, nargs='?', type=int)
    parser.add_argument('--episode_length', default=1000, nargs='?', type=int)
    args = parser.parse_args()
    print(args)

    num_process = int(args.num_process)
    total_episodes = args.total_episodes
    episodes_per_process = int(total_episodes / num_process)
    max_length = args.episode_length

    os.makedirs(results_dir, exist_ok=True)

    env = CarRacingWrapper

    with Pool(num_process) as p:
        p.map(
            partial(
                rollouts,
                num_rollouts=episodes_per_process,
                env=env,
                max_length=max_length,
                results_dir=results_dir,
            ),
            range(num_process)
        )