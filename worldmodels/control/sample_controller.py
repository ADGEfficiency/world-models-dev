import imageio
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
import os

from worldmodels.control.train_controller import episode
from worldmodels.params import home


def plot_episode_data(da, rew, seed):
    labels = da['latent'][1:].reshape(-1, 32)
    preds = da['pred-latent'][:-1].reshape(-1, 32)
    error = np.mean(np.abs(labels - preds), axis=1)

    image_files = []
    for idx in range(labels.shape[0]):

        f, axes = plt.subplots(2, 3, figsize=(20, 8))
        im = da['observation'][idx]
        axes[0][0].imshow(im)
        axes[0][0].set_title('observation')
        re = da['reconstruct'][idx].reshape(64, 64, 3)
        axes[0][1].imshow(re)
        axes[0][1].set_title('reconstruction')

        pred_re = da['pred-reconstruct'][idx].reshape(64, 64, 3)
        axes[0][2].imshow(pred_re)
        axes[0][2].set_title('pred-reconstruction')

        axes[1][0].plot(da['vae-loss-reconstruct'][:idx+1], color='blue', label='vae-loss-reconstruct')
        axes[1][0].plot(da['vae-loss-kld'][:idx+1], color='black', label='vae-loss-kld')
        axes[1][0].set_xlim((0, len(labels)))
        axes[1][0].set_ylim((0, 100))
        axes[1][0].set_title('vae losses')
        axes[1][0].legend()

        axes[1][1].set_xlim((0, len(labels)))
        axes[1][1].plot(error[:idx], color='red', label='memory mae')
        axes[1][1].set_title('memory mean absolute error')

        axes[1][2].plot(da['total-reward'][:idx+1])
        axes[1][2].set_xlim((0, len(labels)))
        axes[1][2].set_title('total-reward')
        f.suptitle('step {} - rew {:3.1f} - seed {}'.format(idx, rew, seed))
        out_dir = os.path.join(home, 'debug', 'gif')
        os.makedirs(out_dir, exist_ok=True)
        f_name = os.path.join(out_dir, '{}.png'.format(idx))
        f.savefig(f_name)

        image_files.append(imageio.imread(f_name))
        print(f_name)

    anim_file = os.path.join(home, 'debug', 'training.gif')
    print('saving to gif')
    imageio.mimsave(anim_file, image_files, duration=0.2)


def main(seed):
    best = get_controller_params()
    rew, para, data = episode(
        best, seed, collect_data=True, max_episode_length=1000
    )
    da = process_episode_data(data, save=True)
    plot_episode_data(da, rew, seed)


if __name__ == '__main__':
    main(42)
