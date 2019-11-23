from functools import partial
from multiprocessing import Pool
import os
import numpy as np

from worldmodels.control.train_controller import episode
prom worldmodels.params import results_dir
from worldmodels.vision.compare_images import generate_gif


def get_controller_params(how='latest'):
    gens = os.listdir(os.path.join(results_dir, 'control', 'generations'))
    gens = [int(s.split('_')[-1]) for s in gens]
    gen = max(gens)
    path = os.path.join(results_dir, 'control', 'generations', 'generation_{}'.format(gen), 'best-params.npy')
    print('loading controller from {}'.format(path))
    return np.load(path)


if __name__ == '__main__':

    best = get_controller_params()

    processes = 1
    #  dont think i need parallelization here
    with Pool(processes) as p:
        # seeds = np.random.randint(0, 2016, processes)
        seeds = [42]
        results = p.map(partial(episode, best, collect_data=True, max_episode_length=1000), seeds)
        rew, para, data = results[0]

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
        axes[1][0].plot(error[:idx], color='red', label='memory mae')
        axes[1][0].set_xlim((0, len(labels)))
        axes[1][0].set_title('vae loss & memory mae')
        axes[1][0].legend()

        axes[1][1].imshow(da['latent'][:idx+1].reshape(idx+1, 32).T)
        axes[1][1].set_xlim((0, len(labels)))
        axes[1][1].set_title('latent')

        axes[1][2].plot(da['total-reward'][:idx+1])
        axes[1][2].set_xlim((0, len(labels)))
        axes[1][2].set_title('total-reward')
        f.suptitle('step {}'.format(idx))
        out_dir = os.path.join(results_dir, 'debug', 'gif')
        os.makedirs(out_dir, exist_ok=True)
        f_name = os.path.join(out_dir, '{}.png'.format(idx))
        f.savefig(f_name)

        image_files.append(imageio.imread(f_name))
        print(f_name)

    anim_file = os.path.join(results_dir, 'debug', 'training.gif')
    print('saving to gif')
    imageio.mimsave(anim_file, image_files, duration=0.2)
