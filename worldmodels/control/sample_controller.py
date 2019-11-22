from functools import partial
from multiprocessing import Pool
import os
import numpy as np

from worldmodels.control.train_controller import episode


if __name__ == '__main__':

    from worldmodels.utils import list_records

    # local = list_records('control/generations', 'npz', data='local')

    # import re
    # import numpy as np
    # n = max([int(re.findall(r'\d+', p)[0]) for p in local]) - 2
    import os

    from worldmodels.params import results_dir

    gen = 452
    params = np.load(
        os.path.join(results_dir, 'control', 'generations', 'generation_{}'.format(gen), 'best-params.npy')
    )

    res = np.load(
        os.path.join(results_dir, 'control', 'generations', 'generation_{}'.format(gen), 'epoch-results.npy')
    )

    pop = np.load(
        os.path.join(results_dir, 'control', 'generations', 'generation_{}'.format(gen), 'population-params.npy')
    )

    best = params

    processes = 1
    with Pool(processes) as p:
        # seeds = np.random.randint(0, 2016, processes)
        seeds = [42]
        results = p.map(partial(episode, best, collect_data=True, max_episode_length=1000), seeds)
        rew, para, data = results[0]

    # seeds = np.random.randint(0, 2016, processes)

    da = data
    for name, arr in da.items():
        da[name] = np.array([np.array(a) for a in arr])

    for k, v in da.items():
        print(k, v.shape)

    labels = da['latent'][1:].reshape(-1, 32)
    preds = da['pred-latent'][:-1].reshape(-1, 32)
    error = np.mean(np.abs(labels - preds), axis=1)

    import imageio
    import matplotlib.pyplot as plt

    from PIL import Image

    for idx in range(labels.shape[0]):

        f, axes = plt.subplots(3, 2, figsize=(60, 20))
        im = da['observation'][idx]
        axes[0][0].imshow(im)
        re = da['reconstruct'][idx].reshape(64, 64, 3)
        axes[1][0].imshow(re)

        pred_re = da['pred-reconstruct'][idx].reshape(64, 64, 3)
        axes[2][0].imshow(pred_re)

        axes[0][1].plot(error[:idx], color='red')
        axes[0][1].set_xlim((0, len(labels)))

        axes[1][1].imshow(da['latent'][:idx+1].reshape(idx+1, 32).T)
        axes[1][1].set_xlim((0, len(labels)))

        axes[2][1].plot(da['total-reward'][:idx+1])
        axes[2][1].set_xlim((0, len(labels)))
        f.savefig('./debug/{}.png'.format(idx))

    from worldmodels.vision.train_vae import generate_gif

    image_files = os.listdir('./debug')
    image_files = [imageio.imread(os.path.join('./debug', f)) for f in image_files if '.png' in f]
    # sort!

    anim_file = os.path.join('training.gif')
    imageio.mimsave(anim_file, image_files)
