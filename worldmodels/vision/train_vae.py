import argparse
from collections import defaultdict
import imageio
import os
import re
import shutil

import matplotlib.pyplot as plt
import tensorflow as tf

from world_models.vision import VAE
from world_models.dataset.upload_to_s3 import S3
from world_models.dataset.tf_records import load_and_shuffle_tf_records, parse_random_rollouts
from world_models.params import vae_params


def compare_images(model, sample_observations, image_dir):
    """ side by side comparison of image and reconstruction """
    reconstructed = model.forward(sample_observations)

    fig, axes = plt.subplots(
        nrows=sample_observations.shape[0],
        ncols=2,
        figsize=(5, 8)
    )

    for idx in range(sample_observations.shape[0]):
        actual_ax = axes[idx, 0]
        reconstructed_ax = axes[idx, 1]

        actual_ax.imshow(sample_observations[idx, :, :, :])
        reconstructed_ax.imshow(reconstructed[idx, :, :, :])
        actual_ax.set_axis_off()
        reconstructed_ax.set_axis_off()

        actual_ax.set_aspect('equal')
        reconstructed_ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(image_dir, 'compare.png'))


def generate_images(model, epoch, batch, sample_latent, image_dir):
    """ latent to reconstructed images """
    predictions = model.decode(sample_latent)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig('{}/epoch_{}_batch_{}.png'.format(image_dir, epoch, batch))


def sort_image_files(image_list):
    """ orders the images generated during training """
    epochs = defaultdict(list)

    #  group into epochs
    max_epoch = 0
    for image in image_list:
        epoch = re.search(r'epoch_([0-9]*)_(.*)', image).groups()[0]
        epochs[epoch].append(image)
        max_epoch = max(int(epoch), max_epoch)

    #  sort each of the lists
    sorted_batches = []
    for epoch in range(1, max_epoch+1):

        batch = epochs[str(epoch)]

        sort_array = []
        for image in batch:
            sort_array.append(int(re.search(r'batch_([0-9]+)', image).groups()[0]))

        sorted_batch = [image for idx, image in sorted(zip(sort_array, batch), reverse=False)]

        for batch in sorted_batch:
            sorted_batches.append(batch)

    return sorted_batches


def generate_gif(image_dir, output_dir):
    print('generating gif from images in {}'.format(image_dir))

    image_list = [x for x in os.listdir(image_dir) if '.png' in x]
    image_files = sort_image_files(image_list)
    image_files = [os.path.join(image_dir, x) for x in image_list]

    image_files = [imageio.imread(f) for f in image_files]

    anim_file = os.path.join(output_dir, 'training.gif')
    imageio.mimsave(anim_file, image_files)


def train(model, epochs, batch_size, log_every, save_every):

    s3 = S3()
    s3_records = s3.list_all_objects('random-rollouts')
    dataset = load_and_shuffle_tf_records(parse_random_rollouts, s3_records, batch_size)

    for sample_observations, _ in dataset.take(1):
        pass

    sample_observations = sample_observations.numpy()[:4]
    dataset = iter(dataset)
    sample_latent = tf.random.normal(shape=(16, model.latent_dim))

    batch_per_epoch = int(1000 * len(s3_records) / batch_size)
    print('starting training of {} epochs'.format(epochs))
    print('{} batches per epoch'.format(batch_per_epoch))

    batch_num = 1
    for epoch in range(epochs):
        generate_images(model, epoch, batch_num, sample_latent, image_dir)

        for batch_num in range(batch_per_epoch):

            batch, _ = next(dataset)
            losses = model.backward(batch)

            if batch_num % log_every == 0:
                print('epoch {}/{} - batch {}/{}'.format(epoch, epochs, batch_num, batch_per_epoch))
                for name, data in losses.items():
                    print(name, data.numpy())

            if batch_num % save_every == 0:
                model.save(results_dir)
                generate_images(model, epoch, batch_num, sample_latent, image_dir)
                compare_images(model, sample_observations, results_dir)
                generate_gif(image_dir, results_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default=1, nargs='?')
    parser.add_argument('--log_every', default=100, nargs='?')
    parser.add_argument('--save_every', default=1000, nargs='?')
    parser.add_argument('--fresh_start', default=0, nargs='?')
    args = parser.parse_args()

    vae_params['load_model'] = bool(args.load_model),
    results_dir = vae_params['results_dir']

    if bool(args.fresh_start):
        print('fresh start')
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    image_dir = os.path.join(results_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    model = VAE(**vae_params)
    training_params = {
        'model': model,
        'epochs': 10,
        'batch_size': 200,
        'log_every': int(args.log_every),  # batches
        'save_every': int(args.save_every)  # batches
    }

    print('cli')
    print('------')
    print(args)
    print('')

    print('training params')
    print('------')
    print(training_params)
    print('')

    print('vision params')
    print('------')
    print(vae_params)
    print('')

    print('setting env variables for AWS')
    os.environ["AWS_REGION"] = "eu-central-1"
    os.environ["AWS_LOG_LEVEL"] = "3"

    train(**training_params)
