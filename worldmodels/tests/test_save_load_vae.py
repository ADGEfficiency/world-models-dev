import os
import shutil

import numpy as np
import pytest

from world_models.vision.vae import VAE


@pytest.fixture()
def setup_test_data_dir():
    """ makes database, runs test, removes database """
    os.mkdir('./test_data')
    yield
    shutil.rmtree('./test_data')


def test_encode(setup_test_data_dir):
    model = VAE(7, results_dir='./test_data')
    batch = np.random.rand(10, 64, 64, 3).astype(np.float32)

    #  encode the same batch, check the same
    rec = model.encode(batch)
    rec_test = model.encode(batch)
    np.testing.assert_array_equal(rec, rec_test)

    #  apply gradients, check different
    model.backward(batch)
    new_req = model.forward(batch)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, new_req, rec)


def test_decode(setup_test_data_dir):
    model = VAE(7, './test_data')
    batch = np.random.rand(10, model.latent_dim).astype(np.float32)

    #  map from true image to reconstruction, check the same
    reconstructed = model.decode(batch)
    new_reconstructed = model.decode(batch)
    np.testing.assert_array_almost_equal(new_reconstructed, reconstructed)
    np.testing.assert_allclose(new_reconstructed, reconstructed)

    #  apply gradients, check different
    images = np.random.rand(10, 64, 64, 3).astype(np.float32)
    model.backward(images)
    new_reconstructed = model.decode(batch)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, new_reconstructed, reconstructed)


def test_save_load(setup_test_data_dir):
    model = VAE(7, './test_data')
    batch = np.random.rand(10, 64, 64, 3).astype(np.float32)

    latent = model.encode(batch)
    model.save('./test_data')

    new_model = VAE(7, './test_data')
    new_model.load('./test_data')
    test_latent = new_model.encode(batch)

    print(latent[0].numpy().sum())
    print(test_latent[0].numpy().sum())

    np.testing.assert_array_equal(latent, test_latent)
    np.testing.assert_allclose(latent, test_latent)
