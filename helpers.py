import pickle
import numpy as np
import os
# import imageio


def save_obj(obj, name, path):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path, suffix='.pkl'):
    with open(path + '/' + name + suffix, 'rb') as f:
        return pickle.load(f)


def batch_data(input_list, n_batches):
    """
    Split the input list in to n_batches lists of approximately equal size (the last item will contain a remainder
    so is often smaller, but doesn't matter for multiprocessing purposes
    :param input_list:
    :param n_batches:
    :return:
    """
    out_list = []
    n_items = len(input_list)
    n_per_batch = int(np.ceil(n_items / n_batches))
    for ix in range(n_batches):
        if (ix + 1) * n_per_batch < n_items:
            ixs = input_list[ix * n_per_batch: (ix + 1) * n_per_batch]
        else:
            ixs = input_list[ix * n_per_batch:]
        out_list.append(ixs)
    return out_list


def col_batch(input_arr, n_batches):
    """
    Split the array input_arr into n_batches different arrays by column
    :param input_arr:
    :param n_batches:
    :return:
    """
    ix_lists = batch_data(list(range(input_arr.shape[1])), n_batches)
    return [input_arr[:, j] for j in ix_lists]


# def gif_from_savefigs(fname, dirname):
#     path = os.getcwd() + f'/outputs/{dirname}'
#     frames = []
#     n_imgs = 0
#     for file in os.listdir(path):
#         if file.endswith(".png"):
#             n_imgs += 1
#
#     for step_ix in range(n_imgs):
#         image = imageio.v2.imread(os.path.join(path, f'{step_ix}.png'))
#         frames.append(image)
#
#     imageio.mimsave(os.path.join(path, f'./{fname}.gif'),
#                     frames,
#                     fps=4)
