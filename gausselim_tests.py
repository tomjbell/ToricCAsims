import numpy as np
from toric import cell_dicts_and_boundary_maps, logical_x_toric, toric_parity_check_matrix
from linear_algebra_inZ2 import loss_decoding_gausselim_fast_noordering_trackqbts, loss_decoding_gausselim_fast_trackqbts
import os
import matplotlib.pyplot as plt
from FusionRaussendorf_6Rings_tb import FusionLatticeRaussendorf
from ca_decoder import gen_errors, loss_decode_check_error
from helpers import col_batch, batch_data, save_obj
import multiprocessing
from itertools import repeat

cwd = os.getcwd()


def lossy_toric_code_gausselim(dimension=2, distance=5, error_rate=0., loss_rate=0.1, n_shots=1, qubit_cell_dim=1):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    # print(nq)
    ps = [(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate]
    errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots), p=ps)
    es = errors_full == 1
    ls = errors_full == 2
    errors = np.array(es)
    losses = np.array(ls, dtype=int)

    num_lost_qbts = sum(losses)
    print(f'{distance=}, {loss_rate=}, num_lost={sum(num_lost_qbts)/n_shots/nq}, {nq=}, {n_shots=}, {ps=}')
    lostfirst_qbts_orders = [np.flip(np.argsort(x)) for x in losses.T]

    correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])  # Gets binary vector corresponding to the logical operator


    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
    qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)

    h_with_log_op = np.vstack([h, correlation_surface.T])
    # print(h_with_log_op.dtype)

    n_log_losses = 0
    for ix in range(n_shots):
        lostfirst_qbts_order = lostfirst_qbts_orders[ix]
        # print(lostfirst_qbts_order)
        Hwithlogop_ordered = h_with_log_op[:, lostfirst_qbts_order]
        qbt_syndr_mat_dec = qbt_syndr_mat[lostfirst_qbts_order]
        n_lost_q = num_lost_qbts[ix]

        if n_lost_q:
            Hwithlogop_ordered_rowechelon, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_trackqbts(Hwithlogop_ordered,
                                                                                                      qbt_syndr_mat_dec,
                                                                                                      n_lost_q)
            new_logop = Hwithlogop_ordered_rowechelon[-1]
        else:
            Hwithlogop_ordered_rowechelon = h_with_log_op
            qbt_syndr_mat_dec = qbt_syndr_mat
            new_logop = Hwithlogop_ordered_rowechelon[-1]

        if np.any(new_logop[:n_lost_q]):
            n_log_losses += 1

        # ##### Readjust matrices to keep only not lost qubits and rows
        # new_logop = new_logop[n_lost_q:]
        # new_qbt_syndr_mat_dec = qbt_syndr_mat_dec[n_lost_q:]
        #
        # if n_lost_q:
        #     first_nonlostsyndr = (int(n_lost_q) - np.argmax(
        #         (np.any(Hwithlogop_ordered_rowechelon[:n_lost_q, :n_lost_q], axis=1))[::-1]))
        #     NewH_unfiltered = Hwithlogop_ordered_rowechelon[first_nonlostsyndr:-1, n_lost_q:]
        #     new_qbt_syndr_mat_dec = new_qbt_syndr_mat_dec - first_nonlostsyndr
        # else:
        #     NewH_unfiltered = Hwithlogop_ordered_rowechelon[:-1]
        #
        # new_qbt_syndr_mat_dec, new_ixs, inverse_ixs, occ_counts, has_no_zeroed_qbt = merge_multiedges_in_Hmat_faster(
        #     new_qbt_syndr_mat_dec)
        # NewH = NewH_unfiltered[:, new_ixs]
        # new_logop = new_logop[new_ixs]
        # print(f'{num_lost_qbts=}')
        # print(new_logop)
        # print(sum(new_logop))
        # print(new_logop.shape)
        #
        # if NewH.shape[0] < 2:
        #     n_log_losses += 1
    print(n_log_losses/n_shots)
    return n_log_losses/n_shots


def rauss_lat_loss_tests():
    L = [3, 5, 7]
    losses = np.linspace(0.15, 0.3, 8)
    for l in L:
        print(f'Calculating distance {l}')
        data = [lossy_toric_code_gausselim(3, l, error_rate=0., loss_rate=loss, n_shots=1000, qubit_cell_dim=1) for loss
                in losses]
        plt.plot(losses, data)
    plt.show()


def test_face_qubit_gausselim(outdir=None, num_points=9):
    Ls = [3, 5, 7, 9, 11]
    # L = 5
    dim = 3
    # loss_rates = np.linspace(0.01, 0.1, 4)
    loss_rates = np.linspace(0.7, 0.8, num_points)
    error_rate = 0.0

    out = {}
    for L in Ls:
        cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=L, dimension=dim)
        qubit_cell_dim = 2

        nq = len(cells[qubit_cell_dim])
        n_stab = len(cells[qubit_cell_dim - 1])

        h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
        for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
            h[stab, qubits] = 1
        stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
        qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)
        correlation_surface = logical_x_toric(cells, qubit_cell_dim, dim, L, cells2i[qubit_cell_dim])
        i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
        i2stabcoord = {v: k for k, v in cells2i[qubit_cell_dim - 1].items()}
        hwithlogop = np.vstack([h, correlation_surface.T])

        n_shots = 1000

        out_this_L = {}
        for loss_rate in loss_rates:
            print(f'{L=}, {error_rate=}, {loss_rate=}')
            errors, losses = gen_errors(nq, n_shots, error_rate=error_rate, loss_rate=loss_rate)
            lost_qubit_ixs = [np.where(losses[:, ix])[0] for ix in range(n_shots)]

            if L > 5:
                n_cpu = multiprocessing.cpu_count() - 1
                errors_batched = col_batch(errors, n_cpu)
                losses_batched = col_batch(losses, n_cpu)
                lq_ix_batched = batch_data(lost_qubit_ixs, n_cpu)
                batch_sizes = [len(lq_ix_batched[j]) for j in range(n_cpu)]
                pool = multiprocessing.Pool(n_cpu)
                result = pool.starmap(loss_decode_check_error, zip(lq_ix_batched, repeat(hwithlogop), repeat(qbt_syndr_mat), repeat(error_rate), errors_batched, batch_sizes))
                log_loss = sum([x[1] for x in result])
            else:
                log_error, log_loss = loss_decode_check_error(lost_qubit_ixs, hwithlogop, qbt_syndr_mat, error_rate, errors, batch_size=n_shots)
            out_this_L[loss_rate] = log_loss / n_shots
        out[L] = out_this_L
    fname = f"Gauss_elim_loss_thresh_qubitcell{qubit_cell_dim}_dim{dim}_L{Ls[0]}-{Ls[-1]}_{n_shots}shots_{num_points}points"
    appendix = 1
    while f"{fname}_{appendix}.pkl" in os.listdir(outdir):
        appendix += 1
    save_obj((loss_rates, out), f"{fname}_{appendix}", outdir)

    for L in Ls:
        plt.plot([k for k in out[L].keys()], [v for v in out[L].values()], 'o-')
    plt.legend(Ls)
    plt.xlabel('Physical loss rate')
    plt.ylabel('Logical loss rate')
    plt.show()





if __name__ == '__main__':
    outdir = os.path.join(os.getcwd(), 'outputs', '24_05_22', 'gausselim')
    for _ in range(5):
        test_face_qubit_gausselim(outdir=outdir)
    exit()

    old_style = False
    rauss_lat = True

    if rauss_lat:
        h_primal, h_dual, xlog, zlog = toric_parity_check_matrix(2, 3)
        h = np.array(h_primal, dtype=np.uint8)
        log_op = xlog
        log_op = log_op[:, 0]
        num_meas = h.shape[1]
    else:
        lattice = FusionLatticeRaussendorf(2, 2, 2, arch_type='6q_ring')
        h = lattice.get_matching_matrix()
        log_op = lattice.log_ops_fus[0]
        num_meas = h.shape[1]
        log_op = np.array([1 if x in log_op else 0 for x in range(num_meas)], dtype=np.uint8)

    stabs_per_qubit = 2
    qbt_syndr_mat = np.where(h.T)[1].reshape((num_meas, stabs_per_qubit)).astype(dtype=np.int32)
    print(log_op)
    h_with_logop = np.vstack([h, log_op])
    h = h_with_logop.copy()

    plt.matshow(h)
    plt.title('h0')
    plt.show()

    lost_qubits = np.array([0, 1], dtype=np.int64)
    if old_style:
        losses_binary = np.zeros(num_meas, dtype=np.uint8)
        num_lost_qbts = len(lost_qubits)
        losses_binary[lost_qubits] = np.ones(num_lost_qbts, dtype=np.uint8)
        # lostfirst_qbts_order = np.flip(np.argsort(losses_binary))
        lostfirst_qbts_order = list(range(num_meas))
        print(lostfirst_qbts_order)
        h_reoordered = h_with_logop[:, lostfirst_qbts_order]
        qbt_syndr_mat_dec = qbt_syndr_mat[lostfirst_qbts_order]

        h_after, qbt_syndr_mat_dec = loss_decoding_gausselim_fast_trackqbts(h_reoordered, qbt_syndr_mat_dec, num_lost_qbts)
    else:
        h_after, qbt_syndr_mat2 = loss_decoding_gausselim_fast_noordering_trackqbts(h, qbt_syndr_mat, lost_qbts=lost_qubits)
    print(h.shape)
    plt.matshow(h_after)
    plt.title('h1')
    plt.show()
    # print(qbt_syndr_mat_dec)
    # print(lossy_toric_code_gausselim(3, 5, loss_rate=0.25, n_shots=1000))


