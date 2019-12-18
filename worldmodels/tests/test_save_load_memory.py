import os
from shutil import rmtree

import pytest
import numpy as np
import tensorflow as tf

from worldmodels.memory.memory import Memory
from worldmodels.params import memory_params


home = os.environ['HOME']
test_dir = os.path.join(home, 'world-models-experiments/test-save-mem')


@pytest.fixture
def setup():

    rmtree(test_dir)
    os.makedirs(test_dir)

    z = np.random.rand(10, 1000, 32)
    action = np.random.rand(10, 1000, 3)

    x = np.concatenate((z[:, :-1, :], action[:, :-1, :]), axis=2)
    y = z[:, 1:, :]
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == 35
    assert y.shape[2] == 32

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    sample = np.reshape(x[0][0], (1, 1, 35))

    return x, y, sample


def test_lstm(setup):
    x, y, sample = setup

    mem = Memory(**memory_params)
    train_zero_state = mem.lstm.get_zero_hidden_state(x)
    mem.train_op(x, y, train_zero_state)

    state = mem.lstm.get_zero_hidden_state(sample)
    old = mem.lstm(sample, state)
    mem.save(test_dir)

    del mem
    model = Memory(**memory_params)
    model.load(test_dir)
    state = model.lstm.get_zero_hidden_state(sample)
    new = model.lstm(sample, state)

    for ol, ne in zip(old, new):
        np.testing.assert_array_equal(ol, ne)

    model.train_op(x, y, train_zero_state)
    diff = model.lstm(sample, state)

    for ol, ne in zip(diff, new):
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, ol, ne)


def test_full_memory(setup):

    x, y, sample = setup

    mem = Memory(**memory_params)
    state = mem.lstm.get_zero_hidden_state(sample)

    old = mem(sample, state, temperature=0, threshold=0.5)
    mem.save(test_dir)
    del mem

    model = Memory(**memory_params)
    model.load(test_dir)
    new = model(sample, state, temperature=0, threshold=0.5)

    for ol, ne in zip(old, new):
        np.testing.assert_array_equal(ol, ne)

    train_zero_state = model.lstm.get_zero_hidden_state(x)
    model.train_op(x, y, train_zero_state)
    diff = model(sample, state, temperature=0, threshold=0.5)

    for ol, ne in zip(diff, new):
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, ol, ne)


def test_lstm_state(setup): 
    x, y, sample = setup
    model = Memory(**memory_params)
    state = model.lstm.get_zero_hidden_state(sample)

    old = model(sample, state, temperature=0, threshold=0.5)

    state = [tf.ones_like(state[0]), tf.ones_like(state[1])]
    new = model(sample, state, temperature=0, threshold=0.5)

    for ol, ne in zip(old, new):
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, ol, ne)
