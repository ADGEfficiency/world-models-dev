"""
The raw data we use to train the memory are previously generated
means and logvars

We use these rather than the reparameterized latent vector so that
we can sample a different latent vector for a given observation
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from worldmodels.dataset.upload_to_s3 import S3, list_local_records
from worldmodels.dataset.tf_records import shuffle_samples, parse_latent_stats
from worldmodels.memory.memory import Memory
from worldmodels.params import memory_params


def train(dataset, model, epochs, batch_per_epoch, save_every):

    epoch_loss = np.zeros(epochs)
    for epoch in range(epochs):

        batch_loss = np.zeros(batch_per_epoch)
        for batch_num in range(batch_per_epoch):
            batch = next(dataset)
            mu = batch['mu']
            logvars = batch['logvar']
            action = batch['action']

            epsilon = tf.random.normal(shape=mu.shape)
            z = mu + epsilon * tf.exp(logvars * .5)

            x = tf.concat(
                (z[:, :-1, :], action[:, :-1, :]),
                axis=2
            )

            y = z[:, 1:, :]

            assert x.shape[0] == y.shape[0]
            assert x.shape[2] == 35
            assert y.shape[2] == 32
            state = model.lstm.get_zero_hidden_state(x)

            batch_loss[batch_num] = model.train_op(x, y, state)
            assert 1==0 # fix logging message
            logger.info('epoch {} batch {}/{} loss {}'.format(
                epoch,
                batch_num,
                batch_per_epoch,
                batch_loss[batch_num]
            ))

            if batch_num % save_every == 0:
                model.save(results_dir)

        model.save(results_dir)

        epoch_loss[epoch] = np.mean(batch_loss)
        logger.info('epoch {} loss {}'.format(epoch, epoch_loss[epoch]))

    return epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3', dest='s3', action='store_true')
    parser.add_argument('--local', dest='s3', action='store_false')
    parser.set_defaults(s3=True)
    args = parser.parse_args()

    home = os.environ['HOME']
    results_dir = os.path.join(home, 'world-models-experiments', 'memory-training')
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir)

    if bool(args.s3):
        s3 = S3()
        records = s3.list_all_objects('latent-stats')
    else:
        records = list_local_records('latent-stats', 'episode')

    assert len(records) == 10000

    epochs = memory_params['epochs']
    batch_size = memory_params['batch_size']
    batch_per_epoch = int(len(records) / batch_size)
    print('starting training of {} epochs'.format(epochs))
    print('{} batches per epoch'.format(batch_per_epoch))

    memory_params['decay_steps'] = epochs * batch_per_epoch

    model = Memory(**memory_params)

    dataset = shuffle_samples(
        parse_latent_stats,
        records,
        batch_size=model.batch_size, shuffle_buffer=500, num_cpu=8
    )

    training_params = {
        'dataset': dataset,
        'model': model,
        'epochs': epochs,
        'batch_per_epoch': batch_per_epoch,
        'save_every': 20  # batches
    }

    print('setting env variables for AWS')
    os.environ["AWS_REGION"] = "eu-central-1"
    os.environ["AWS_LOG_LEVEL"] = "3"

    model = Memory(**memory_params)
    train(**training_params)
