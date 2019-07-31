import numpy as np

import sys
from world_models.dataset.generate_dataset import rollout, rollouts, load_dataset

import gym


class EnvStub():

    """ a simple stub to use during tests """

    def __init__(self, **kwargs):
        pass

    def step(self, action, **kwargs):

        next_observation = action * 10
        reward = action / -2

        done = False
        if action == 10:
            done = True

        return next_observation, reward, done, {}

    def reset(self):
        return 0.0


class AgentStub:

    """ always does the same actions """

    def __init__(self, *args, **kwargs):
        self.actions = np.tile([1, 5, 3], 100)
        self.age = 0

    def act(self, observation):
        action = self.actions[self.age]
        self.age += 1
        return action



if __name__ == '__main__':
    agent = AgentStub
    env = EnvStub

    results = rollout(max_length=2, agent=agent, env=env)

    #  check obs, action, reward, next_obs, done
    #  check length

    #  def test_save ????
    #  just uses np.save


    #  def test_rollouts
    #
        #  combine with a test of load dataset???

    results_path = './test-data'

    rollouts(
        'alpha',
        num_rollouts=2,
        agent=agent,
        env=env,
        debug=False,
        max_length=10,
        results_path=results_path
    )

    dataset = load_dataset('./test-data')

    #  check observations, shape and data

    #  parametrize differnt actions?

