from ca_decoder import gen_errors, ne_parity_map, binary_error_to_mat_2d_tooms
import numpy as np
import matplotlib.pyplot as plt
from toric import cell_dicts_and_boundary_maps, logical_x_toric
import os
from ca_decoder import tooms_with_loss_parallelized, lossy_tooms_sweeps, sparse_tooms_iters_eras_conv
from scipy.sparse import csr_matrix
from linear_algebra_inZ2 import loss_decoding_gausselim_fast_noordering_trackqbts


def local_loss_only_tests(distance=20, dimension=2, error_rate=0., loss_rate=0.1, max_ca_iters=20, custom_errors=False, custom_losses=False):
    if not (custom_errors or custom_losses):
        mat = np.random.choice([0, 1, 2], size=(distance, distance), p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    else:
        mat = np.zeros((distance, distance))
        for coord in custom_errors:
            mat[coord[0], coord[1]] = 1
        for coord in custom_losses:
            mat[coord[0], coord[1]] = 2
    plt.matshow(mat, vmin=0, vmax=2)
    plt.show()

    # Assume random outcomes for the lost qubits
    lost_qubit_mat = mat == 2
    lost_qubit_guesses = lost_qubit_mat * np.random.choice([0, 1], size=(distance, distance), p=[0.5, 0.5])
    tot_error = lost_qubit_guesses + (mat == 1)
    plt.matshow(tot_error, vmin=0, vmax=2)
    plt.show()
    # exit()
    sweep_dir = 0
    n_skips = 0
    for _ in range(max_ca_iters):
        plt.matshow(tot_error, vmin=0, vmax=2)
        plt.show()
        changes = False
        # Sweep lost qubits
        change_mat = np.zeros(tot_error.shape, dtype=int)
        for i in range(distance):
            for j in range(distance):
                # if lost_qubit_mat[i, j]:
                # if True:
                    x_nbs, y_nbs = get_neighbour_coords(i, j, distance)
                    this_val = tot_error[i, j]
                    x_incr = sweep_dir // 2
                    y_incr = sweep_dir % 2
                    n_val = tot_error[i, y_nbs[y_incr]]
                    e_val = tot_error[x_nbs[x_incr], j]
                    if n_val!= this_val and e_val != this_val:
                        change_mat[i, j] = 1
                        changes = True
        tot_error += change_mat
        tot_error %= 2
        if changes:
            n_skips = 0
        else:
            # break
            sweep_dir += 1
            sweep_dir %= 4
            n_skips += 1
            if n_skips == 4:
                break
    print(sum(tot_error * lost_qubit_mat))






def erasure_conversion_2d(distance=20, dimension=2, error_rate=0.1, loss_rate=0., n_ca_iters=15):

    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_map = ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab)
    print(ne_map.shape)
    print(nq, n_stab)

    # parity check matrix
    h = np.zeros(shape=(n_stab, nq), dtype=int)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    # Sample errors
    errors, losses = gen_errors(nq, 1, error_rate, loss_rate)
    syndrome = (h @ errors) % 2
    loss_synd = (h @ losses) % 2

    current_error = errors.copy()
    current_synd = syndrome.copy()
    prediction = np.zeros((nq, 1))
    plt.matshow(binary_error_to_mat_2d_tooms(errors + 2 * losses, i2qcoord))
    plt.show()

    for _ in range(n_ca_iters):
        flips_this_round = (ne_map @ current_synd) == 2
        prediction += flips_this_round
        current_error = (current_error + flips_this_round) % 2
        current_synd = (h @ current_error) % 2
        plt.matshow(binary_error_to_mat_2d_tooms(current_error, i2qcoord))
        plt.title(_)
        plt.show()

    error = (errors + prediction) % 2
    plt.matshow(binary_error_to_mat_2d_tooms(error, i2qcoord))
    plt.show()


def eras_conv_tests(ca_rule, distance=20, error_rate=0.1, loss_rate=0.1, n_ca_iters=5, second_func=None,
                    second_ca_iters=0, change_dir_every_f1=5, custom_losses=None, custom_errors=None, save_figs=False,
                    output_dirname=None, showfigs=False, change_sweep_every=None, flip_eras_sweep_dir=False):
    """
    Test function for different ca rules for an Ising model of spins, including erausure conversion
    Everything is just tracked with a single 2d matrix of spin values
    Frames can be saved
    :param ca_rule:
    :param distance:
    :param dimension:
    :param error_rate:
    :param loss_rate:
    :param n_ca_iters:
    :param second_func:
    :param second_ca_iters:
    :param change_dir_every_f1:
    :param custom_losses:
    :param custom_errors:
    :param save_figs:
    :param output_dirname:
    :param showfigs:
    :param change_sweep_every:
    :param flip_eras_sweep_dir:
    :return:
    """
    if not (custom_errors or custom_losses):
        mat = np.random.choice([0, 1, 2], size=(distance, distance), p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    else:
        mat = np.zeros((distance, distance))
        for coord in custom_errors:
            mat[coord[0], coord[1]] = 1
        for coord in custom_losses:
            mat[coord[0], coord[1]] = 2
    plt.matshow(mat, vmin=0, vmax=2)
    if showfigs:
        plt.show()
    if save_figs:
        path = os.getcwd() + f'/outputs/{output_dirname}'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            appendix = 0
            while os.path.exists(path + str(appendix)):
                appendix += 1
            path += str(appendix)
            os.makedirs(path)
            print(f'{output_dirname} already exists, created dir with appendix {appendix}')
        plt.savefig(fname=path + '/' + '0')
        plt.close()

    for _ in range(n_ca_iters):
        if change_sweep_every is not None:
            sweep_dir = (_ // change_sweep_every) % 4
            mat = ca_rule(mat, sweep_dir)
        else:
            mat = ca_rule(mat)
        plt.matshow(mat, vmin=0, vmax=2)
        if showfigs:
            plt.show()
        if save_figs:
            plt.savefig(fname=path + '/' + str(_+1))
            plt.close()
    if second_func is not None:
        sweep_dir = 0
        n_skips = 0
        # change_sweep_every = second_ca_iters // 4
        for _ in range(second_ca_iters):
            # if flip_eras_sweep_dir:
            #     sweep_dir = (_ // change_sweep_every) % 4
            mat, n_erased = second_func(mat, sweep_dir)
            if n_erased == 0:
                n_skips += 1
                if n_skips == 4:
                    break
                else:
                    sweep_dir += 1
                    sweep_dir %= 4
            else:
                n_skips = 0
            plt.matshow(mat, vmin=0, vmax=2)
            if showfigs:
                plt.show()
            if save_figs:
                plt.savefig(fname=path + '/' + str(_ + n_ca_iters + 1))
                plt.close()

################# The following are subroutines for what the cellular automata should do on 2D matrices ################

def do_nothing(m):
    return m


def eras_conv_simple(m, sweep_dir=0):
    n_eras = 0
    new_mat = np.zeros(m.shape, dtype=int)
    size = m.shape[0]
    x_incr = sweep_dir // 2
    y_incr = sweep_dir % 2
    for ix1 in range(size):
        for ix2 in range(size):
            x_nbs, y_nbs = get_neighbour_coords(ix1, ix2, size)
            # print(ix1, ix2, x_nbs)
            e_val = m[x_nbs[x_incr], ix2]
            n_val = m[ix1, y_nbs[y_incr]]
            val = m[ix1, ix2]
            if val != 2 and n_val != val and e_val != val:
                n_eras += 1
                new_mat[ix1, ix2] = 2
            else:
                new_mat[ix1, ix2] = m[ix1, ix2]
    return new_mat, n_eras


def og_tooms_with_loss(m, sweep_dir=0):
    """

    :param m:
    :param sweep_dir: int between 0 and 3
    :return:
    """
    new_mat = np.zeros(m.shape, dtype=int)
    size = m.shape[0]
    x_incr = sweep_dir // 2
    y_incr = sweep_dir % 2
    for ix1 in range(size):
        for ix2 in range(size):
            x_nbs, y_nbs = get_neighbour_coords(ix1, ix2, size)
            # print(ix1, ix2, x_nbs)
            e_val = m[x_nbs[x_incr], ix2]
            n_val = m[ix1, y_nbs[y_incr]]
            val = m[ix1, ix2]
            if val != 2:
                if n_val == 2 and e_val == 2:
                    new_mat[ix1, ix2] = m[ix1, ix2]
                elif n_val != val and e_val != val:
                    new_mat[ix1, ix2] = (val + 1) % 2
                else:
                    new_mat[ix1, ix2] = m[ix1, ix2]

            else:
                new_mat[ix1, ix2] = m[ix1, ix2]
    return new_mat


def tooms_ignore_loss(m):
    new_mat = np.zeros(m.shape, dtype=int)
    size = m.shape[0]
    for ix1 in range(size):
        for ix2 in range(size):
            x_nbs, y_nbs = get_neighbour_coords(ix1, ix2, size)
            # print(ix1, ix2, x_nbs)
            e_val = m[x_nbs[0], ix2]
            n_val = m[ix1, y_nbs[0]]
            val = m[ix1, ix2]
            if val != 2:
                diff = (val + 1) % 2
                if n_val == diff and e_val == diff:
                    new_mat[ix1, ix2] = (val + 1) % 2
                else:
                    new_mat[ix1, ix2] = m[ix1, ix2]

            else:
                new_mat[ix1, ix2] = m[ix1, ix2]
    return new_mat

###################### End of subroutines #######################


def get_neighbour_coords(i, j, l):
    """
    In 2D, get the neighbours of the coordinate (i,j) when incrementing in either +- x or +- y, for a lattice of size l
    :param i:
    :param j:
    :param l:
    :return:
    """
    i_increment_coords = [i+1, i-1]
    if i == l - 1:
        i_increment_coords[0] = 0
    elif i == 0:
        i_increment_coords[1] = l - 1

    j_increment_coords = [j+1, j-1]
    if j == l - 1:
        j_increment_coords[0] = 0
    elif j == 0:
        j_increment_coords[1] = l - 1

    return tuple(i_increment_coords), tuple(j_increment_coords)


if __name__ == '__main__':
    # test_cubic_erasure_conversion()
    # exit()
    #
    # local_loss_only_tests(loss_rate=0.4)
    # exit()

    Ls = [3, 5, 7]
    # L = 5
    dim = 3
    # loss_rates = np.linspace(0.01, 0.1, 4)
    loss_rates = [0.0]
    error_rates = np.linspace(0.005, 0.02, 9)

    out = {}
    for L in Ls:
        cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=L, dimension=dim)
        qubit_cell_dim = 2
        ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

        nq = len(cells[qubit_cell_dim])
        n_stab = len(cells[qubit_cell_dim - 1])
        ne_maps = [ne_parity_map(cells, qubit_cell_dim, cells2i, dim, L, nq, n_stab, ne_dir=ne_dir) for ne_dir in ne_dirs]
        ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]

        h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
        for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
            h[stab, qubits] = 1
        h_sparse = csr_matrix(h)
        stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
        qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)
        correlation_surface = logical_x_toric(cells, qubit_cell_dim, dim, L, cells2i[qubit_cell_dim])
        correlation_surface_sparse = csr_matrix(correlation_surface)  # Gets binary vector corresponding to the logical operator
        lattice_info = h_sparse, correlation_surface, qbt_syndr_mat, ne_parity_mats_sparse, nq
        i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
        i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
        hwithlogop = np.vstack([h, correlation_surface.T])

        n_shots = 5000

        out_this_L = {}
        for loss_rate in loss_rates:
            out_this_L_loss = {}
            for error_rate in error_rates:
                print(f'{L=}, {error_rate=}, {loss_rate=}')
                errors, losses = gen_errors(nq, n_shots, error_rate=error_rate, loss_rate=loss_rate)

                errors_after, losses_after = sparse_tooms_iters_eras_conv(num_iters=0, change_dir_freq=10,
                                                                          ne_mats=ne_parity_mats_sparse,
                                                                          q_err_sparse=csr_matrix(errors), h_sparse=h_sparse,
                                                                          loss_sparse=csr_matrix(losses), num_eras_iters=100)

                error_out = errors_after.toarray()
                losses_out = losses_after.toarray()

                log_loss = 0
                log_error = 0
                lost_qubit_ixs = [np.where(losses_out[:, ix])[0] for ix in range(n_shots)]

                for ix, es in enumerate(error_out.T):
                    this_error = error_out[:, ix]
                    error_ixs = np.where(this_error)[0]

                    lq = np.array(lost_qubit_ixs[ix], dtype=int)
                    loss_coords = [i2qcoord[x] for x in lq]
                    n_lost_q = len(lq)
                    if n_lost_q:
                        Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_noordering_trackqbts(hwithlogop, qbt_syndr_mat, lost_qbts=lq)
                    else:
                        Hwithlogop_ordered_rowechelon = hwithlogop
                    # Check if there is an error
                    logop_out = Hwithlogop_ordered_rowechelon[-1]
                    this_log_error = np.dot(logop_out, this_error) % 2
                    log_error += this_log_error
                    this_log_loss = 0
                    if np.any(logop_out[lq]):
                        this_log_loss = 1
                        log_loss += this_log_loss
                out_this_L_loss[error_rate] = (log_loss/n_shots, log_error/n_shots)
                # print(f'logical loss rate: {log_loss/n_shots}')
                # print(f'Logical error rate: {log_error/n_shots}')
            out_this_L[loss_rate] = out_this_L_loss
        out[L] = out_this_L

    for loss_rate in loss_rates:
        for L in Ls:
            print(out[L][loss_rate])
            plt.plot(out[L][loss_rate].keys(), [x[0] for x in list(out[L][loss_rate].values())])
        plt.ylabel('Logical loss rate')
        plt.xlabel('Physical error rate')
        plt.legend(Ls)
        plt.title(f'logical loss rate for physical loss rate {loss_rate}, erasure conversion only')
        plt.show()
        for L in Ls:
            print(out[L][loss_rate])
            plt.plot(out[L][loss_rate].keys(), [x[1] for x in list(out[L][loss_rate].values())])
        plt.ylabel('Logical error rate')
        plt.xlabel('Physical error rate')
        plt.legend(Ls)
        plt.title(f'logical error rate for physical loss rate {loss_rate}, erasure conversion only')
        plt.show()


