import argparse
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import os

import numpy as np
import tensorflow as tf

from worldmodels.dataset.car_racing import CarRacingWrapper
from worldmodels.dataset.tf_records import encode_floats
from worldmodels.params import home
from worldmodels.control.train_controller import episode


def random_rollout(env, episode_length, results=None, seed=None):
    """ runs an episode with a random policy """

    if results is None:
        results = defaultdict(list)

    env = env(seed=seed)

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
        if step >= episode_length:
            done = True

    env.close()

    return results


def get_controller_params(generation):
    path = os.path.join(home, 'control', 'generations')
    gens = os.listdir(path)
    gens = [int(s.split('_')[-1]) for s in gens]
    max_gen = max(gens) 
    gen_best = []

    gens = list(range(max_gen - 5, max_gen))
    for gen in gens:
        rew = np.load(
            os.path.join(path, 'generation_{}'.format(gen), 'epoch-results.npy')
        )
        gen_best.append(max(rew))

    best_gen = gens[np.argmax(gen_best)]
    print('max gen {} best gen {} {}'.format(max_gen, best_gen, max(rew)))

    from time import sleep
    sleep(3)

    path = os.path.join(
        home,
        'control',
        'generations',
        'generation_{}'.format(best_gen),
        'best-params.npy'
    )
    print('loading controller from {}'.format(path))
    return np.load(path)


def controller_rollout(params, seed=42, episode_length=1000):
    """ runs an episode with pre-trained controller parameters"""
    results = episode(
        params,
        collect_data=True,
        episode_length=episode_length,
        seed=seed
    )
    return results[2]


def save_episode(results, process_id, episode, seed, dtype='tf-record'):

    if dtype == 'tf-record':
        save_episode_tf_record(results, process_id, episode)
    else:
        assert dtype == 'numpy'
        save_episode_numpy(results, seed)


def save_episode_tf_record(results, process_id, episode):
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


def save_episode_numpy(results, seed):
    """ results dictionary to .npy """
    path = os.path.join(
        home, 'controller-samples', str(seed)
    )
    os.makedirs(path, exist_ok=True)

    for name, data in results.items():
        results[name] = np.array([np.array(a) for a in data])
        print(name, results[name].shape)
        np.save(
            os.path.join(path, '{}.npy'.format(name)), data
        )


def rollouts(
    process_id,
    rollout_start,
    rollout_end,
    num_rollouts,
    env,
    episode_length,
    results_dir,
    policy='random-rollouts',
    controller_generation=0,
    dtype='tf-records'
):
    """ runs many episodes """

    #  seeds always the length of the total rollouts per process
    #  so that if we start midway we get a new seed
    np.random.seed(process_id)
    seeds = np.random.randint(0, high=2**32-1, size=num_rollouts)
    seeds = seeds[rollout_start: rollout_end]
    episodes = list(range(rollout_start, rollout_end))
    assert len(episodes) == len(seeds)

    for seed, episode in zip(seeds, episodes):
        if policy == 'controller-rollouts':
            params = get_controller_params(controller_generation)
            results = controller_rollout(
                params,
                seed=seed,
                episode_length=episode_length
            )

        else:
            assert policy == 'random-rollouts'
            results = random_rollout(
                env=env,
                episode_length=episode_length,
                seed=seed
            )

        print('process {} episode {} length {}'.format(
            process_id, episode, len(results['observation'])
        ))

        save_episode(results, process_id, episode, dtype=dtype, seed=seed)

        paths = os.listdir(results_dir)
        episodes = [path for path in paths if 'episode' in path]
        print('{} episodes stored locally'.format(len(episodes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_process', default=2, nargs='?', type=int)
    parser.add_argument('--total_episodes', default=10, nargs='?', type=int)
    parser.add_argument('--episode_length', default=1000, nargs='?', type=int)
    parser.add_argument('--start_episode', default=0, nargs='?', type=int)
    parser.add_argument('--policy', default='random-rollouts', nargs='?')
    parser.add_argument('--controller-generation', default=0, nargs='?', type=int)
    parser.add_argument('--dtype', default='tf-records', nargs='?')
    args = parser.parse_args()
    print(args)

    num_process = int(args.num_process)
    total_episodes = args.total_episodes
    episodes_per_process = int(total_episodes / num_process)
    episode_length = args.episode_length

    rollout_start = args.start_episode
    rollout_end = episodes_per_process
    assert rollout_end <= episodes_per_process

    results_dir = os.path.join(home, args.policy)
    os.makedirs(results_dir, exist_ok=True)

    env = CarRacingWrapper
    total_eps = num_process * episodes_per_process

    with Pool(num_process) as p:
        p.map(
            partial(
                rollouts,
                rollout_start=rollout_start,
                rollout_end=rollout_end,
                num_rollouts=episodes_per_process,
                env=env,
                episode_length=episode_length,
                results_dir=results_dir,
                policy=args.policy,
                controller_generation=args.controller_generation,
                dtype=args.dtype
            ),
            range(num_process)
        )
