import os
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
from toric import cell_dicts_and_boundary_maps, adjacent_cells, torus, logical_x_toric, logical_z_toric
from scipy.sparse import csr_matrix, csc_matrix
from matplotlib.colors import ListedColormap
import imageio
from linear_algebra_inZ2 import loss_decoding_gausselim_fast_trackqbts, loss_decoding_gausselim_fast_noordering_trackqbts
from itertools import repeat
from helpers import save_obj
# matplotlib.use('module://backend_interagg')


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
        error = (final_error.T @ csr_matrix(correlation_surface)).toarray() % 2

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


def ftr(nemap, st):
    m = nemap @ st
    return np.logical_or(m == 2, m == 4)


def ftr_sparse(nemap, st):
    m = nemap @ st
    a = m == 2
    b = m == 4
    return a + b


def tooms_with_loss_parallelized(distance, dimension=5, loss_rate=0.1, error_rate=0., n_ca_iters=10, change_dir_every=10, n_shots=1,
                        qubit_cell_dim=None, parallelise=True):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    if qubit_cell_dim is None:
        qubit_cell_dim = 2
    if qubit_cell_dim != dimension // 2:
        print(f"Warning, qubit cell dimension is {qubit_cell_dim} for a {dimension}-dimensional toric code")
    ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_maps = [ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab, ne_dir=ne_dir) for ne_dir
               in ne_dirs]
    ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]

    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    h_sparse = csr_matrix(h)
    stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
    qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)  # is this the same as the boundary maps?
    correlation_surface = csr_matrix(logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])) # Gets binary vector corresponding to the logical operator

    errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots),
                                   p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    es = errors_full == 1
    ls = errors_full == 2
    errors = np.array(es, dtype=int)
    losses = np.array(ls, dtype=int)
    lost_qubit_ixs = [np.where(losses[:, ix])[0].astype(int) for ix in range(n_shots)]

    n_cpu = multiprocessing.cpu_count() - 1
    # batch the n_shots to run on different cores - this is the error and loss matrices
    errors_batched = col_batch(errors, n_cpu)
    losses_batched = col_batch(losses, n_cpu)
    lq_ix_batched = batch_data(lost_qubit_ixs, n_cpu)
    q_errs_sparse = [csr_matrix(es) for es in errors_batched]
    q_loss_sparse = [csr_matrix(ls) for ls in losses_batched]
    # print(n_cpu, )

    tot_errors = 0
    tot_losses = 0
    t0 = time()
    if parallelise:
        lossy = loss_rate > 1e-10
        errory = error_rate > 1e-10
        pool = multiprocessing.Pool(n_cpu)
        out = pool.starmap(run_tooms_and_ge_on_batch, zip(q_errs_sparse, q_loss_sparse, repeat(h_sparse), repeat(ne_parity_mats_sparse), repeat(n_ca_iters),
                                                          repeat(change_dir_every), lq_ix_batched, repeat(correlation_surface), repeat(qbt_syndr_mat),
                                                          repeat(lossy), repeat(errory)))
        tot_errors = sum([x[0] for x in out])
        tot_losses = sum([x[1] for x in out])
    else:
        for i in range(n_cpu):
            error_sparse = q_errs_sparse[i]
            loss_sparse = q_loss_sparse[i]
            n_errors, n_losses = run_tooms_and_ge_on_batch(error_sparse, loss_sparse, h_sparse, ne_parity_mats_sparse, n_ca_iters,
                                                           change_dir_every, lq_ix_batched[i], correlation_surface, qbt_syndr_mat,
                                                           lossy=loss_rate > 1e-10, errory=error_rate > 1e-10)
            tot_errors += n_errors
            tot_losses += n_losses
    print(f'total time taken: {time() - t0}')
    return tot_errors/n_shots, tot_losses/n_shots


def run_tooms_and_ge_on_batch(error_mat, loss_mat, h, ne_map, n_ca_iters, chng_freq, lost_q_ixs, log_op, qub_synd_mat,
                              lossy=True, errory=True):
    num_in_batch = error_mat.shape[1]
    if errory:
        current_error = sparse_tooms_iters(n_ca_iters, chng_freq, ne_map, error_mat, h, loss_mat)
    else:
        current_error = error_mat
    if lossy:
        current_error = current_error.toarray()
        h_with_log_op = np.vstack([h.toarray(), log_op.toarray().T])
        n_errors, n_log_losses = loss_decode_check_error(lost_q_ixs, h_with_log_op, qub_synd_mat, error_rate=int(errory),
                                                         error_mat=current_error, batch_size=num_in_batch)

        pass
    else:
        res_errors = log_op.multiply(current_error)
        n_errs_in_support = np.array(np.sum(res_errors, axis=0))
        n_errs_in_support %= 2
        n_errors = np.sum(np.array(n_errs_in_support))
        n_log_losses = 0
    return n_errors, n_log_losses



def tooms_with_loss(distance, dimension=5, loss_rate=0.1, error_rate=0., n_ca_iters=10, change_dir_every=10, n_shots=1,
                    plot_fig=True, save_figs=False, save_dir=None, dense=True, num_loss_ca_iters=0, ca_losses=False,
                    qubit_cell_dim=None, printing=False, parallelise=False, new_gauss_elim=True, n_processes=None):
    if n_shots > 1:
        assert not plot_fig
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    if qubit_cell_dim is None:
        qubit_cell_dim = 2
    if qubit_cell_dim != dimension // 2:
        print(f"Warning, qubit cell dimension is {qubit_cell_dim} for a {dimension}-dimensional toric code")
    ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    ne_dir_ix = 0
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_maps = [ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab, ne_dir=ne_dir) for ne_dir in ne_dirs]
    ne_map = ne_maps[ne_dir_ix]

    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    # Ensure no qubit can be lost and flipped
    errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots), p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    es = errors_full == 1
    ls = errors_full == 2
    errors = np.array(es, dtype=int)
    losses = np.array(ls, dtype=int)


    t0 = time()
    if error_rate > 1e-10:
        if dense:
            syndromes = (h @ errors) % 2
            loss_synd = (h @ losses) != 0

            prediction = np.zeros(shape=errors.shape)
            current_error_synd = syndromes
            current_error = errors.copy()

            for _ in range(n_ca_iters):
                if plot_fig:
                    visualise_q_and_stab(losses, current_error, h, distance, i2qcoord, i2stabcoord, iterno=_, savefig=save_figs, dirname=save_dir)
                if _:
                    if not _ % change_dir_every:
                        ne_dir_ix += 1
                        ne_dir_ix %= 4
                        ne_map = ne_maps[ne_dir_ix]
                # current_tot_synd = get_tot_synd(loss_synd, current_error_synd)
                current_tot_synd = 3 * loss_synd + current_error_synd - loss_synd * current_error_synd
                # flips_this_round = np.logical_or(((ne_map @ current_tot_synd) == 2), ((ne_map @ current_tot_synd) == 4))
                flips_this_round = ftr(ne_map, current_tot_synd)
                prediction += flips_this_round
                current_error = (current_error + flips_this_round) % 2
                current_error_synd = (h @ current_error) % 2

        else:
            ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]
            q_err_sparse = csr_matrix(errors)
            h_sparse = csr_matrix(h)
            loss_sparse = csr_matrix(losses)

            current_error = sparse_tooms_iters(n_ca_iters, change_dir_every, ne_parity_mats_sparse, q_err_sparse,
                                               h_sparse, loss_sparse)


            # The final errors after ca decoding are in current_error, rows are different qubits, columns are different shots
            # The sum of the dot product of the tiled correlation surface mat with this can be used to get the parity of the
            # intersection, and the logical error
    else:
        current_error = errors
    t1 = time()
    if printing:
        print(f'{n_shots} of lossy Tooms took {t1-t0}s')

    correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])  # Gets binary vector corresponding to the logical operator

    if loss_rate < 1e-10:
        logop_sparse = csr_matrix(correlation_surface)
        intersection = logop_sparse.multiply(current_error)
        num_intersections = np.array(np.sum(intersection, axis=0))
        num_intersections %= 2
        # print(type(num_intersections), num_intersections.shape)
        n_errors = np.sum(num_intersections)
        n_log_losses = 0
        log_loss_rate = 0
        log_error_rate = n_errors / n_shots
    else:
        current_error = current_error.toarray()

        # Do gaussian elimination
        stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
        qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32) # is this the same as the boundary maps?
        # qbt_syndr_mat_alt = np.array([sorted(b_maps[qubit_cell_dim][k]) for k in range(nq)], dtype=int)
        # assert np.allclose(qbt_syndr_mat_alt, qbt_syndr_mat)

        h_with_log_op = np.vstack([h, correlation_surface.T])
        lost_qubit_ixs = [np.where(losses[:, ix])[0].astype(int) for ix in range(n_shots)]

        if parallelise:
            if n_processes is None:
                n_processes = multiprocessing.cpu_count() - 1
            log_error_rate, log_loss_rate = parallelised_gauss_elim_batched(h_with_log_op, lost_qubit_ixs,
                                                                    qbt_syndr_mat, n_shots,
                                                                    current_error,
                                                                    error_rate, n_batches=n_processes)
        else:
            if new_gauss_elim:
                n_errors, n_log_losses = loss_decode_check_error(lost_qubit_ixs, h_with_log_op, qbt_syndr_mat, error_rate, current_error, batch_size=n_shots)
            else:
                lostfirst_qbts_orders = [np.flip(np.argsort(x)) for x in losses.T]
                h_with_log_op_sparse = csr_matrix(h_with_log_op)
                num_lost_qbts = sum(losses)
                n_log_losses = 0
                n_errors = 0
                for ix in range(n_shots):
                    lostfirst_qbts_order = lostfirst_qbts_orders[ix]
                    Hwithlogop_ordered = reorder_perm_mat(h_with_log_op_sparse, lostfirst_qbts_order).toarray()
                    qbt_syndr_mat_dec = reorder_qbt_synd_adj(qbt_syndr_mat, lostfirst_qbts_order)
                    n_lost_q = num_lost_qbts[ix]
                    if n_lost_q:
                        Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_trackqbts(
                                                                            Hwithlogop_ordered, qbt_syndr_mat_dec, n_lost_q)
                        new_logop = Hwithlogop_ordered_rowechelon[-1]

                    else:
                        Hwithlogop_ordered_rowechelon = h_with_log_op
                        qbt_syndr_mat_dec = qbt_syndr_mat
                        new_logop = Hwithlogop_ordered_rowechelon[-1]

                    # print(new_logop[:, :n_lost_q])
                    try:
                        a = np.any(new_logop[:n_lost_q])
                    except IndexError:
                        print(new_logop)
                        print(f'{n_lost_q=}')
                    if a:
                        n_log_losses += 1
                    else:
                        # determine if there has been a logical error
                        error = is_err(new_logop, current_error,  ix)
                        n_errors += error

            log_loss_rate = n_log_losses / n_shots
            log_error_rate = n_errors / n_shots
    num_errors = np.array(current_error.sum(axis=0))

    # plt.plot(list(range(n_shots)), num_errors)
    # plt.show()
    # print(f'Average number of resultant errors {num_errors.sum()/n_shots}')
    if printing:
        print(f'{n_shots} of gaussian elimination and error testing took {time() - t1}s')
        print(f'logical loss rate: {log_loss_rate}, logical error rate: {log_error_rate}')
        print(f'physical loss rate: {loss_rate}, physical error rate: {error_rate}')
    return log_loss_rate, log_error_rate


def sparse_tooms_iters(num_iters, change_dir_freq, ne_mats, q_err_sparse, h_sparse, loss_sparse):
    ne_map = ne_mats[0]
    error_syndrome = h_sparse @ q_err_sparse
    error_syndrome.data %= 2
    loss_synd_sparse = (h_sparse @ loss_sparse) != 0
    prediction = csr_matrix(q_err_sparse.shape, dtype=int)
    current_error = q_err_sparse.copy()
    ne_dir_ix = 0
    for _ in range(num_iters):
        if _ and (not _ % change_dir_freq):
            ne_dir_ix += 1
            ne_dir_ix %= 4
            ne_map = ne_mats[ne_dir_ix]

        # Taking off their product takes away 1 where they intersect
        current_tot_synd = 3 * loss_synd_sparse + error_syndrome - loss_synd_sparse.multiply(error_syndrome)

        flips_this_round = ftr_sparse(ne_map, current_tot_synd)
        # print(flips_this_round)
        prediction += flips_this_round
        current_error += flips_this_round
        # print(current_error)
        current_error.data %= 2
        error_syndrome = h_sparse @ current_error
        error_syndrome.data %= 2
    return current_error


def loss_decode_check_error(lost_qubits, h_with_log_op, qbt_syndr_mat, error_rate, error_mat, batch_size=1):
    """

    :param lost_qubits: list of lists
    :param h_with_log_op: numpy array
    :param qbt_syndr_mat:
    :param error_rate:
    :param error_mat: array of ints
    :param batch_size: int
    :return:
    """
    log_loss = 0
    log_error = 0
    for ix in range(batch_size):
        lq = np.array(lost_qubits[ix], dtype=int)
        n_lost_q = len(lost_qubits)
        if n_lost_q:
            Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_noordering_trackqbts(
                h_with_log_op, qbt_syndr_mat, lost_qbts=lq)
        else:
            Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = h_with_log_op, qbt_syndr_mat
        new_logop = Hwithlogop_ordered_rowechelon[-1]
        if np.any(new_logop[lq]):
            log_loss += 1
        elif error_rate > 1e-10:
            error_vec = error_mat[:, ix]
            # determine if there has been a logical error
            if not isinstance(error_vec, np.ndarray):
                print(type(error_vec))

            log_error += np.dot(new_logop, error_vec) % 2

    return log_error, log_loss


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


def parallelised_gauss_elim_batched(h_with_log_op, lost_qubit_ixs, qbt_syndr_mat, n_shots, error_mat, error_rate, n_batches):
    assert multiprocessing.cpu_count() >= n_batches
    ixs_batched = batch_data(list(range(n_shots)), n_batches)
    num_in_batch = [len(x) for x in ixs_batched]
    errors_batched = [error_mat[:, ixs_batched[i]] for i in range(n_batches)]
    lost_qubit_ix_batched = [[lost_qubit_ixs[y] for y in x] for x in ixs_batched]

    pool = multiprocessing.Pool(n_batches)

    out = pool.starmap(loss_decode_check_error, zip(lost_qubit_ix_batched, repeat(h_with_log_op), repeat(qbt_syndr_mat),
                                                    repeat(error_rate), errors_batched, num_in_batch))
    log_err_rate = sum([x[0] for x in out]) / n_shots
    log_l_rate = sum([x[1] for x in out]) / n_shots

    return log_err_rate, log_l_rate


def is_err(logop, error_mat, i):
    if isinstance(error_mat, csr_matrix):
        return np.dot(logop, error_mat[:, i].toarray()) % 2
    else:
        return np.dot(logop, error_mat[:, i]) % 2


def reorder_h(h, ordering):
    idx = np.empty_like(ordering)
    idx[ordering] = np.arange(len(ordering))
    return h[:, idx]


def reorder_perm_mat(h, ordering):
    """
    Use sparse matrices to permute the columns of h
    :param h: csr_matrix of the parity check matrix and logical operator
    :param ordering:
    :return:
    """
    nq = len(ordering)
    permutation_matrix = csr_matrix((np.ones(nq, dtype=np.uint8), (list(range(nq)), ordering)), shape=(nq, nq))
    return (permutation_matrix @ h.transpose()).transpose()



def reorder_qbt_synd_adj(mat, ordering):
    return mat[ordering]


def ca_gauss_elim(distance, dimension=2, loss_rate=0.1, error_rate=0., n_ca_iters=10, n_shots=1, plot_fig=True, save_figs=False, save_dir=None, num_loss_ca_iters=0):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])

    errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots),
                                   p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    es = errors_full == 1
    ls = errors_full == 2
    errors = np.array(es, dtype=int)
    losses = np.array(ls, dtype=int)
    loss_sparse = csr_matrix(losses)

    logical_operator = logical_x_toric(cells, qubit_cell_dim, dimension, distance,
                                          cells2i[2])  # Gets binary vector corresponding to the logical operator
    # print(correlation_surface.shape)
    if num_loss_ca_iters > 1:
        logical_operator_tiled = np.tile(logical_operator, n_shots)
        logical_operator_sparse = csr_matrix(logical_operator_tiled)
    else:
        logical_operator_sparse = csr_matrix(logical_operator)

    # Find the first dimension that each qubit is embedded in, this is the direction that we will move the logical operator
    # TODO construct this at the same time as the NE map?
    qbts_to_change_in_support = qbts_to_change_support_loss(cells, qubit_cell_dim, cells2i, dimension, distance, nq,
                                                            cob_maps)
    qbt_support_mat_sparse = csr_matrix(qbts_to_change_in_support)

    for _ in range(num_loss_ca_iters):
        problem_q = logical_operator_sparse.multiply(
            loss_sparse)  # this gives 1s where qubits in the support of the logical operator are lost
        q_to_flip = qbt_support_mat_sparse @ problem_q
        q_to_flip.data %= 2  # tells us which qubits to add or remove from the support of the logical operator
        logical_operator_sparse += q_to_flip
        logical_operator_sparse.data %= 2  # qubits can either be in or not in the moved correlation surface

    # we should end up with n_shots different correlation surfaces that avoid losses

    #
    #
    #
    # if ca_losses:
    #     if num_loss_ca_iters > 1:
    #         correlation_surface_tiled = np.tile(correlation_surface, n_shots)
    #         correlation_surface_sparse = csr_matrix(correlation_surface_tiled)
    #     else:
    #         correlation_surface_sparse = csr_matrix(correlation_surface)
    #
    #     # Find the first dimension that each qubit is embedded in, this is the direction that we will move the logical operator
    #     # TODO construct this at the same time as the NE map?
    #     qbts_to_change_in_support = qbts_to_change_support_loss(cells, qubit_cell_dim, cells2i, dimension, distance, nq, cob_maps)
    #     qbt_support_mat_sparse = csr_matrix(qbts_to_change_in_support)
    #
    #     for _ in range(num_loss_ca_iters):
    #         problem_q = correlation_surface_sparse.multiply(loss_sparse)  # this gives 1s where qubits in the support of the logical operator are lost
    #         q_to_flip = qbt_support_mat_sparse @ problem_q
    #         q_to_flip.data %= 2  # tells us which qubits to add or remove from the support of the logical operator
    #         correlation_surface_sparse += q_to_flip
    #         correlation_surface_sparse.data %= 2  # qubits can either be in or not in the moved correlation surface
    #
    #     # we should end up with n_shots different correlation surfaces that avoid losses


def qbts_to_change_support_loss(cells, qubit_cell_dim, cells2i, dimension, distance, nq, co_boundary_map, i2qcoord=None):
    qbts_to_change_in_support = np.zeros(shape=(nq, nq), dtype=np.uint8)
    for q in cells[qubit_cell_dim]:
        q_ind = cells2i[qubit_cell_dim][q]
        embedded = tuple([i for i in range(dimension) if q[i] % 2])
        increment_dir = embedded[0]
        stab_coord = list(q)
        stab_coord[increment_dir] += 1
        stab_coord = torus(stab_coord, distance)
        north_stab = cells2i[qubit_cell_dim - 1][stab_coord]
        adj_qubit_ix = co_boundary_map[qubit_cell_dim - 1][north_stab]
        qbts_to_change_in_support[adj_qubit_ix, q_ind] = 1
        # print(q, stab_coord, [i2qcoord[nu_q] for nu_q in adj_qubit_ix])
    return qbts_to_change_in_support


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


def test_north_stab_sweep(dimension=3, distance=5):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
    nq = len(cells[qubit_cell_dim])

    qbts_to_change_in_support = qbts_to_change_support_loss(cells, qubit_cell_dim, cells2i, dimension, distance, nq, cob_maps, i2qcoord)

    show_mat = np.zeros(shape=(nq, nq))

    test_qbt = random.randint(0, nq)
    test_vec = np.zeros((nq, 1))
    test_vec[test_qbt, 0] = 1
    chng_qbt = qbts_to_change_in_support @ test_vec
    print(test_qbt)
    print(chng_qbt.nonzero())
    test_qbt_coords = i2qcoord[test_qbt]
    chng_qbt_coords = [i2qcoord[x] for x in chng_qbt.nonzero()[0]]
    print(f'{test_qbt_coords=}')
    print(f'{chng_qbt_coords=}')




def main():
    errors = np.linspace(0.016, 0.019, 6)
    for distance in [3, 4, 5]:
        out = []
        for e in errors:
            log_error = tooms_5d_primal_decode(error_rate=e, n_trials=10000, n_ca_iters=30, dense=False,
                                               distance=distance)
            out.append(log_error)
            print(out)
        plt.plot(errors, out, 'o-')
    plt.show()

    # toom_2d_phase_only()


def dim3_loss_tests():
    losses = np.linspace(0.1, 0.3, 11)
    # losses = [0.2]
    dists = (3,4, 5, 6, 7)
    # dists = [11]

    for d in dists:
        losses_out = []
        for l in losses:
            print(f'Analysing distance {d}, physical loss rate: {l}')
            log_loss_rate, log_error_rate = tooms_with_loss(dimension=3, distance=d, error_rate=0.0, qubit_cell_dim=1,
                                                            loss_rate=l, n_ca_iters=200, change_dir_every=100,
                                                            plot_fig=False, dense=False, n_shots=1000, printing=True,
                                                            parallelise=d > 5, new_gauss_elim=True, n_processes=10)
            losses_out.append(log_loss_rate)
        plt.plot(losses, losses_out)
    plt.show()


def loss_threshold():
    dim = 5
    Ls = [3, 4]
    losses = np.linspace(0.07, 0.15, 5)
    # Ls = [5]
    # losses = [0.1]
    err = 0.0
    # eras = 0.35
    n_shots = 100
    n_cores = 10

    def parallel_condition(l, dim):
        if dim == 3 and l > 8:
            return True
        elif dim == 4 and l > 3:
            return True
        elif dim == 5:
            return True
        else:
            return False

    for L in Ls:
        out = []
        for eras in losses:
            log_loss_rate, log_error_rate = tooms_with_loss(dimension=dim, distance=L, error_rate=err, qubit_cell_dim=2,
                                                            loss_rate=eras, n_ca_iters=200, change_dir_every=50,
                                                            plot_fig=False, dense=False, n_shots=n_shots, printing=True,
                                                            parallelise=parallel_condition(L, dim), new_gauss_elim=True,
                                                            n_processes=n_cores)
            out.append(log_loss_rate)
        plt.plot(losses, out)
    plt.show()
    print(f'{log_loss_rate=}, {log_error_rate=}')
    print(f'{n_shots=}')


def lossy_tooms_sweeps(Ls, error_rates, n_shots=100, loss=0., save_data=False, savefig=False, outdir=None, dim=5):
    # error_rates = np.linspace(0.01, 0.02, 7)
    # Ls = [3, 5]
    # Ls = [3]
    # error_rates = [0.05]
    if isinstance(loss, float) or loss == 0:
        loss = [loss]
    for l in loss:
        if savefig or save_data:
            assert outdir is not None
            path = os.getcwd() + f'/outputs/{outdir}'
            if not os.path.exists(path):
                os.makedirs(path)
            fnames = f'dim5_toric_error_sweep_loss{l}_maxL_{Ls[-1]}_{n_shots}shots'
        out_dict = {}
        for L in Ls:
            out = []
            print(f'Calculating distance {L}, loss rate: {l}')
            for e in error_rates:
                print(f'error rate: {e}')
                log_loss_rate, log_error_rate = tooms_with_loss(dimension=dim, distance=L, error_rate=e, qubit_cell_dim=2,
                                                                loss_rate=l, n_ca_iters=100, change_dir_every=20,
                                                                plot_fig=False, dense=False, n_shots=n_shots, printing=True,
                                                                parallelise=L>3, new_gauss_elim=True,
                                                                n_processes=5)
                out.append(log_error_rate)
                # print(f'{log_loss_rate=}, {log_error_rate=}')
            out_dict[L] = (error_rates, out)
        if savefig:
            for L in Ls:
                plt.plot(out_dict[L][0], out_dict[L][1])
            plt.legend(Ls)
            plt.xlabel('Phenom. noise')
            plt.ylabel('Logical error rate')
            plt.title(f'loss rate: {l}, dimension {dim} clusterized toric code')
            plt.savefig(path + '/' + fnames + '.png')
        if save_data:
            save_obj(out_dict, fnames, path)



if __name__ == '__main__':
    for e in [0.00]:
        # tooms_with_loss(distance=3, dimension=5, loss_rate=0.001, error_rate=e, n_ca_iters=100, change_dir_every=20, n_shots=1000, qubit_cell_dim=2, parallelise=False, new_gauss_elim=True, printing=True, plot_fig=False, dense=False)
        print(tooms_with_loss_parallelized(4, 5, error_rate=e, loss_rate=0.01, n_ca_iters=100, change_dir_every=20, n_shots=10000, qubit_cell_dim=2, parallelise=True))
    exit()
    lossy_tooms_sweeps([3, 4], error_rates=np.linspace(0.001, 0.01, 5), loss=[0.001], n_shots=1000, savefig=True, outdir='test_sweep_2')




