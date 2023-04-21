import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
from toric import cell_dicts_and_boundary_maps, adjacent_cells, torus, logical_x_toric, logical_z_toric
from scipy.sparse import csr_matrix
from matplotlib.colors import ListedColormap
import imageio
matplotlib.use('module://backend_interagg')


############################ These functions for simple ising model ########################

def nec_coords(ixs, size):
    north = (ixs[0] + 1) % size
    east = (ixs[1] + 1) % size
    return ixs, (north, ixs[1]), (ixs[0], east)


def tooms_rule(mat, p=0.5, q=0.5):
    new_mat = np.zeros(mat.shape)
    shape = mat.shape
    p_mat = np.random.choice([0, 1], size=mat.shape, p=[1-p, p])
    q_mat = np.random.choice([0, 1], size=mat.shape, p=[1-q, q])
    for i in range(shape[0]):
        for j in range(shape[1]):
            coords_set = nec_coords((i, j), mat.shape[0])
            if sum([mat[x] for x in coords_set]) < 2:
                val = 0
            else:
                val = 1
            if val:
                new_mat[i, j] = (val + p_mat[i, j]) % 2
            else:
                new_mat[i, j] = (val + q_mat[i, j]) % 2
    return new_mat


def run_tooms(size=50, p=0.1, q=0.1, length=10, show_every=1, initial_up=0.5):
    cells_2d_mat = np.random.choice([0, 1], size=(size, size), p=[initial_up, 1 - initial_up])
    print(np.sum(cells_2d_mat)/ size / size)
    plt.matshow(cells_2d_mat)
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    for _ in range(length):

        cells_2d_mat = tooms_rule(cells_2d_mat, p=p, q=q)
        if not (_ % 5):
            plt.matshow(cells_2d_mat)
            plt.title(_)
            plt.colorbar()
            plt.clim(0, 1)
            plt.show()
        # time.sleep(0.1)


##################################### These functions for error-correction #########################

def ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab, ne_dir=(1, 1)):
    ne_map = {}
    for q in cells[qubit_cell_dim]:
        q_ind = cells2i[qubit_cell_dim][q]
        embedded = [i for i in range(dimension) if q[i] % 2]
        check_inds = []
        for ix, dim in enumerate(embedded):
            nu_coord = list(q)
            nu_coord[dim] += ne_dir[ix]
            check_inds.append(cells2i[qubit_cell_dim - 1][torus(nu_coord, distance)])
        ne_map[q_ind] = check_inds
    ne_parity_mat = np.zeros(shape=(nq, n_stab), dtype=int)
    for q, s in ne_map.items():
        ne_parity_mat[q, s] = 1
    return ne_parity_mat


def toom_2d_phase_only(distance=20, dimension=2, error_rate=0.4, n_ca_iters=15):
    # Using dense matrices, sparse implementation below

    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_map = ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab)

    # parity check matrix
    h = np.zeros(shape=(n_stab, nq), dtype=int)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    # Sample errors
    e = error_rate
    q_errors = np.random.choice([0, 1], size=(nq, 1), p=[1 - e, e])
    syndrome = (h @ q_errors) % 2

    current_error = q_errors.copy()
    current_synd = syndrome.copy()
    prediction = np.zeros((nq, 1))
    plt.matshow(binary_error_to_mat_2d_tooms(q_errors, i2qcoord))
    plt.show()

    for _ in range(n_ca_iters):
        flips_this_round = (ne_map @ current_synd) == 2
        prediction += flips_this_round
        current_error = (current_error + flips_this_round) % 2
        current_synd = (h @ current_error) % 2
        plt.matshow(binary_error_to_mat_2d_tooms(current_error, i2qcoord))
        plt.title(_)
        plt.show()


    error = (q_errors + prediction) % 2
    plt.matshow(binary_error_to_mat_2d_tooms(error, i2qcoord))
    plt.show()


def tooms_5d_primal_decode(distance=3, dimension=5, error_rate=0.01, n_ca_iters=10, n_trials=100, dense=True):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    print(f'{nq=}')
    print(f'{n_stab=}')
    ne_map = ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab)

    # parity check matrix
    h = np.zeros(shape=(n_stab, nq), dtype=int)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[2])

    t0 = time()
    e = error_rate
    q_errors = np.random.choice([0, 1], size=(nq, n_trials), p=[1 - e, e])
    if dense:
        syndrome = (h @ q_errors) % 2

        current_error = q_errors.copy()
        current_synd = syndrome.copy()
        prediction = np.zeros(shape=q_errors.shape)

        for _ in range(n_ca_iters):
            flips_this_round = (ne_map @ current_synd) == 2
            prediction += flips_this_round
            current_error = (current_error + flips_this_round) % 2
            current_synd = (h @ current_error) % 2
        final_error = (q_errors + prediction) % 2
        error = (final_error.T @ correlation_surface) % 2
        e_rate = sum(error)/n_trials

    else:
        ne_parity_mat_sparse = csr_matrix(ne_map)
        q_err_sparse = csr_matrix(q_errors)
        h_sparse = csr_matrix(h)
        syndrome = h_sparse @ q_err_sparse
        syndrome.data %= 2

        prediction = csr_matrix(q_errors.shape, dtype=int)
        current_error = q_err_sparse.copy()
        for _ in range(n_ca_iters):
            flips_this_round = (ne_parity_mat_sparse @ syndrome) == 2
            prediction += flips_this_round
            current_error = current_error + flips_this_round
            current_error.data %= 2
            syndrome = h_sparse @ current_error
            syndrome.data %= 2
        final_error = q_err_sparse + prediction
        final_error.data %= 2
        error = (final_error.T @ csr_matrix(correlation_surface)).todense() % 2

        e_rate = sum(error) / n_trials

    t1 = time()
    print(f'{t1-t0=}')
    return e_rate


def binary_error_to_mat_2d_tooms(error, ix2coord_map):
    nq_err = len(error)
    distance = int(np.sqrt(nq_err))

    e_ixs = [i for i in range(len(error)) if error[i, 0] == 1]
    e_coords = [ix2coord_map[x] for x in e_ixs]
    e_coord_mat = [(e_[0] // 2, e_[1] // 2) for e_ in e_coords]
    e_mat = np.zeros((distance, distance))
    for e_ in e_coord_mat:
        e_mat[e_[0], e_[1]] = 1
    return e_mat


def get_tot_synd(loss_s, error_s):
    row_ixs, col_ixs = loss_s.nonzero()

    print(loss_s.nonzero())
    m2 = error_s * (loss_s == 0)
    return loss_s + m2


def get_tot_synd_sparse(loss_s, error_s):
    print(loss_s==0)
    m2 = error_s * (loss_s == 0)
    return loss_s + m2


def tooms_with_loss(distance, dimension=2, loss_rate=0.1, error_rate=0., n_ca_iters=10, change_dir_every=10, plot_fig=True, save_figs=False, save_dir=None, dense=True):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    ne_dir_ix = 0
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_maps = [ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab, ne_dir=ne_dir) for ne_dir in ne_dirs]
    ne_map = ne_maps[ne_dir_ix]

    h = np.zeros(shape=(n_stab, nq), dtype=int)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    losses = np.random.choice([0, 1], size=(nq, 1), p=[1 - loss_rate, loss_rate])
    errors = np.random.choice([0, 1], size=(nq, 1), p=[1 - error_rate, error_rate])
    # Find syndromes, they can have 3 values, 0: no error, 1: error, 2: erased outcome due to lost qubit
    syndromes = (h @ errors) % 2
    # erase all syndromes adjacent to lost qubits
    loss_synd = 3 * (h @ losses)
    tot_synd = get_tot_synd(loss_synd, syndromes)

    prediction = np.zeros(shape=errors.shape)
    current_error_synd = syndromes
    current_error = errors.copy()

    def ftr(nemap, st):
        m = nemap @ st
        return np.logical_or(m == 2, m == 4)

    if dense:
        for _ in range(n_ca_iters):
            if plot_fig:
                visualise_q_and_stab(losses, current_error, h, distance, i2qcoord, i2stabcoord, iterno=_, savefig=save_figs, dirname=save_dir)
            if _:
                if not _ % change_dir_every:
                    ne_dir_ix += 1
                    ne_dir_ix %= 4
                    ne_map = ne_maps[ne_dir_ix]
            current_tot_synd = get_tot_synd(loss_synd, current_error_synd)
            # flips_this_round = np.logical_or(((ne_map @ current_tot_synd) == 2), ((ne_map @ current_tot_synd) == 4))
            flips_this_round = ftr(ne_map, current_tot_synd)
            prediction += flips_this_round
            current_error = (current_error + flips_this_round) % 2
            current_error_synd = (h @ current_error) % 2
    else:
        ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]
        q_err_sparse = csr_matrix(current_error)
        h_sparse = csr_matrix(h)
        error_syndrome = h_sparse @ q_err_sparse
        error_syndrome.data %= 2
        loss_synd_sparse = csr_matrix(loss_synd)

        prediction = csr_matrix(current_error.shape, dtype=int)
        current_error = q_err_sparse.copy()
        for _ in range(n_ca_iters):
            if _:
                if not _ % change_dir_every:
                    ne_dir_ix += 1
                    ne_dir_ix %= 4
                    ne_map = ne_parity_mats_sparse[ne_dir_ix]
            current_tot_synd = get_tot_synd(loss_synd_sparse, error_syndrome)
            flips_this_round = ftr(ne_map, current_tot_synd)
            prediction += flips_this_round
            current_error = (current_error + flips_this_round)
            current_error.data %= 2
            error_syndrome = h @ current_error
            error_syndrome.data %= 2




def visualise_q_and_stab(losses, errors, h, distance, i2qcoord, i2stabcoord, iterno=None, savefig=False, dirname=None):
    syndromes = (h @ errors) % 2
    # erase all syndromes adjacent to lost qubits
    loss_synd = 2 * (h @ losses)
    n_stab = len(syndromes)
    n_q = len(errors)
    full_errors = np.ones(shape=errors.shape)
    for i in range(len(full_errors)):
        if losses[i]:
            full_errors[i] = 3
        elif errors[i]:
            full_errors[i] = 2

    full_synd = np.zeros(shape=loss_synd.shape)
    for i in range(n_stab):
        if loss_synd[i]:
            full_synd[i] = 6
        elif syndromes[i]:
            full_synd[i] = 5
        else:
            full_synd[i] = 4

    # expand to the full view including the syndromes
    full_mat = np.zeros((2 * distance, 2 * distance))
    for i in range(n_q):
        full_mat[i2qcoord[i]] = full_errors[i]
    for i in range(n_stab):
        full_mat[i2stabcoord[i]] = full_synd[i]
    cmap = ListedColormap(['w', 'palegreen', 'cyan', 'orange', 'g', 'b', 'red'])
    plt.matshow(full_mat, vmin=0, vmax=6, cmap=cmap)
    plt.colorbar()
    if iterno is not None:
        plt.title(iterno)
    if savefig:
        plt.savefig(fname=dirname + '/' + str(iterno))
        plt.close()
    else:
        plt.show()
        plt.close()


def gen_gif_tooms_2d_loss(distance=20, error_rate=0.4, loss_rate=0.1, n_ca_iters=100, change_dir_every=10):
    dirname = f'Tooms2Ddistance{distance}_error{error_rate}_loss{loss_rate}_{n_ca_iters}iters_chngdir{change_dir_every}'
    path = os.getcwd() + f'/outputs/{dirname}'
    if not os.path.exists(path):
        os.makedirs(path)

    tooms_with_loss(distance=distance, error_rate=error_rate, loss_rate=loss_rate, n_ca_iters=n_ca_iters, change_dir_every=change_dir_every, save_dir=path, save_figs=True, plot_fig=True)

    frames = []

    steps_list = []
    n_imgs = 0
    for file in os.listdir(path):
        if file.endswith(".png"):
            n_imgs += 1

    for step_ix in range(n_imgs):
        image = imageio.v2.imread(os.path.join(path, f'{step_ix}.png'))
        frames.append(image)

    imageio.mimsave(os.path.join(path, './ToomsWithLoss.gif'),
                    frames,
                    fps=5,
                    loop=1)



def main():
    errors = np.linspace(0.01, 0.015, 5)
    for distance in [3, 4, 5]:
        out = []
        for e in errors:
            log_error = tooms_5d_primal_decode(error_rate=e, n_trials=10000, n_ca_iters=30, dense=False,
                                               distance=distance)
            out.append(log_error[0, 0])
            print(out)
        plt.plot(errors, out, 'o-')
    plt.show()

    # toom_2d_phase_only()


if __name__ == '__main__':
    # gen_gif_tooms_2d_loss(loss_rate=0.2, error_rate=0.2, change_dir_every=5, n_ca_iters=50)
    t0 = time()
    tooms_with_loss(distance=50, error_rate=0.1, loss_rate=0.1, n_ca_iters=5, change_dir_every=5, plot_fig=False, dense=False)
    t1 = time()
    print(t1-t0)
    # main()
    # run_tooms(size=50, p=0.01, q=0.01, length=100, show_every=5, initial_up=0.5)




