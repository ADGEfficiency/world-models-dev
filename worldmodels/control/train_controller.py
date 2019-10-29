from collections import defaultdict
from multiprocessing import Pool
import os

import numpy as np
import logging

from worldmodels.dataset.car_racing import CarRacingWrapper
from worldmodels.params import vae_params, memory_params, env_params, results_dir


def make_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fldr = os.path.join(results_dir, 'control')
    os.makedirs(fldr, exist_ok=True)
    fh = logging.FileHandler(os.path.join(fldr, '{}.log'.format(name)))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def shape_controller_params(params, output_size=3):
    w = params[:-output_size].reshape(-1, output_size)
    b = params[-output_size:]
    return w, b


def get_action(z, state, params):
    w, b = shape_controller_params(params)

    net_input = np.concatenate([z, state], axis=None)

    action = np.tanh(net_input.dot(w) + b)

    action[1] = (action[1] + 1.0) / 2.0
    action[2] = np.clip(action[2], 0.0, 1.0)
    return action.astype(np.float32)


def episode(params, seed, collect_data=False):
    #  needs to be imported here for multiprocessing
    import tensorflow as tf
    from worldmodels.vision.vae import VAE
    from worldmodels.memory.memory import Memory

    vision = VAE(**vae_params)

    memory_params['num_timesteps'] = 1
    memory_params['batch_size'] = 1
    memory = Memory(**memory_params)

    state = memory.lstm.get_zero_hidden_state(
        np.zeros(35).reshape(1, 1, 35)
    )

    max_episode_length = 1000
    env = CarRacingWrapper()
    total_reward = 0

    data = defaultdict(list)

    env.seed(seed)
    np.random.seed(seed)

    obs = env.reset()
    for step in range(max_episode_length):
        obs = obs.reshape(1, 64, 64, 3).astype(np.float32)
        mu, logvar = vision.encode(obs)
        z = vision.reparameterize(mu, logvar)

        action = get_action(z, state[0], params)
        obs, reward, done, _ = env.step(action)

        x = tf.concat([
            tf.reshape(z, (1, 1, 32)),
            tf.reshape(action, (1, 1, 3))
        ], axis=2)

        _, h_state, c_state = memory(x, state, temperature=1.0)
        state = [h_state, c_state]

        total_reward += reward

        if done:
            step = max_episode_length

        if collect_data:
            data['observation'].append(obs)
            data['action'].append(action)
            data['mu'].append(mu)
            data['logvar'].append(logvar)

    env.close()
    logger.debug(total_reward)
    return total_reward, params, data


class CMAES:
    def __init__(self, x0, s0=0.5, opts={}):
        """
        x0 (OrderedDict) {'param_name': np.array}
        """
        self.num_parameters = len(x0)
        print('{} params in controller'.format(self.num_parameters))

        #  sigma init, weicht decay TODO
        self.solver = CMAEvolutionStrategy(x0, s0, opts)

    def __repr__(self):
        return '<pycma wrapper>'

    def ask(self):
        samples = self.solver.ask()
        return np.array(samples).reshape(-1, self.num_parameters)

    def tell(self, samples, fitness):
        return self.solver.tell(samples, -1 * fitness)

    @property
    def mean(self):
        return self.solver.mean


logger = make_logger('all-rewards')
global_logger = make_logger('rewards')


if __name__ == '__main__':
    generations = 500
    popsize = 64
    epochs = 16
    num_process = 16

    #popsize = 2
    #epochs = 2
    #generations = 2
    #num_process = 2

    #  need to open the Pool before importing from cma
    with Pool(popsize) as p:
        from cma import CMAEvolutionStrategy

        input_size = vae_params['latent_dim'] + memory_params['lstm_nodes']
        output_size = env_params['num_actions']

        weights = np.random.randn(input_size, output_size)
        biases = np.random.randn(output_size)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])

        es = CMAES(x0, opts={'popsize': popsize})

        for generation in range(generations):
            population = es.ask()

            epoch_results = np.zeros((popsize, epochs))
            for epoch in range(epochs):
                seeds = np.random.randint(
                    low=0, high=1000000,
                    size=population.shape[0]
                )

                results = p.starmap(episode, zip(population, seeds))
                rew, para = zip(*results)
                epoch_results[:, epoch] = rew

            epoch_results = np.mean(epoch_results, axis=1)
            assert epoch_results.shape[0] == popsize
            global_logger.debug(np.mean(epoch_results))

            es.tell(para, epoch_results)
