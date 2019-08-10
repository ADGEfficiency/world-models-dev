from functools import partial
from multiprocessing import Pool
import os

from worldmodels.control.train_controller import episode


if __name__ == '__main__':

    from worldmodels.dataset.upload_to_s3 import list_local_records

    local = list_local_records('control/params', 'npz')

    import re
    import numpy as np
    n = max([int(re.findall(r'\d+', p)[0]) for p in local]) - 2
    import os

    from worldmodels.params import results_dir

    params = np.load(
        os.path.join(results_dir, 'control', 'params', 'gen{}.npz'.format(n))
    )

    res = params['epoch_results'].mean(axis=1)

    best = params['params'][np.argmax(res)+1, :]

    processes = 1
    with Pool(processes) as p:
        # seeds = np.random.randint(0, 2016, processes)
        seeds = [30]
        results = p.map(partial(episode, best, collect_data=True), seeds)
        rew, para, data = zip(*results)

        da = data[0]
        for name, arr in da.items():
            da[name] = np.array([np.array(a) for a in arr])

    for k, v in da.items():
        print(k, v.shape)
    # import pdb; pdb.set_trace()

    import imageio
    import matplotlib.pyplot as plt

    from PIL import Image

    images = [Image.fromarray(((255*arr).round()).astype(np.uint8)).resize((64*6, 64*6)) for arr in da['observation']]

    anim_file = os.path.join(results_dir, 'debug', 'replay.gif')
    imageio.mimsave(anim_file, images)
    np.save(os.path.join(results_dir, 'debug', 'obs.npy'), da['observation'])
