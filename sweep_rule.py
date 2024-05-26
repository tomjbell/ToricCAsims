import os
from linear_algebra_inZ2 import loss_decoding_gausselim_fast_noordering_trackqbts, loss_decoding_gausselim_fast_noordering_trackstabs
import numpy as np
from toric import cell_dicts_and_boundary_maps, torus, logical_x_toric
from itertools import chain, combinations, product, repeat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import multiprocessing
from helpers import save_obj
from helpers import batch_data, col_batch
from ca_decoder import gen_errors
from time import perf_counter
from math import ceil


def ge_on_hypercubic(distance=2, dimension=2, qubit_cell_dim=1, error_rate=0, loss_rate=0.2, n_shots=1):
    """
    TODO Unfinished function -> can we update the future faces and future cells map after gaussian elimination to
    TODO apply the sweep rule to the inhomogeneous gaussian-eliminated lattice
    :param distance:
    :param dimension:
    :param qubit_cell_dim:
    :param error_rate:
    :param loss_rate:
    :param n_shots:
    :return:
    """
    future_dir = tuple([1] * dimension)
    cells, cells2i, b_maps, cob_maps, b_map_mats, cob_map_mats = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension, get_matrices=True)
    ix2cell =[{v: k for k, v in cell_dict.items()} for cell_dict in cells2i]

    future_faces_map = {}
    future_edges_map = {}
    for v in cells[0]:
        future_faces_map[cells2i[0][v]] = [cells2i[2][face] for face in future_cells_toric(+2, v, dimension, distance, future_dir)]
        future_edges_map[cells2i[0][v]] = [cells2i[1][edge] for edge in future_cells_toric(+1, v, dimension, distance, future_dir)]
    if qubit_cell_dim is None:
        qubit_cell_dim = 2
    if qubit_cell_dim != dimension // 2:
        print(f"Warning, qubit cell dimension is {qubit_cell_dim} for a {dimension}-dimensional toric code")


    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])

    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    print(f'{np.allclose(h, b_map_mats[qubit_cell_dim])=}')
    stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
    qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)

    errors, losses = gen_errors(nq, n_shots, error_rate, loss_rate)
    lost_qubit_ixs = [np.where(losses[:, ix])[0].astype(int) for ix in range(n_shots)]
    error_qubit_ixs_all = [np.where(errors[:, ix])[0].astype(int) for ix in range(n_shots)]

    log_op = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])
    h_with_log_op = np.vstack([h, log_op.T])
    plt.matshow(h_with_log_op)
    plt.show()

    for ix in range(n_shots):
        lq = np.array(lost_qubit_ixs[ix], dtype=int)
        # lq = np.array([0, 6, 2], dtype=int)

        print(lq)
        n_lost_q = len(lq)
        Hwithlogop_ordered_rowechelon, new_stabs_ix = loss_decoding_gausselim_fast_noordering_trackstabs(h_with_log_op, lost_qbts=lq, n_stabs=n_stab)
        print(new_stabs_ix)



        plt.matshow(Hwithlogop_ordered_rowechelon)
        plt.show()

        h_out = Hwithlogop_ordered_rowechelon[:-1, :]

        # TODO Get eliminated qubits and stabilizers
        eliminated_stabs = new_stabs_ix[:n_lost_q]
        print(f'{eliminated_stabs=}')

        # The new qubits are the other ones in the support of the stabilizer, so can directly get their indices from the

        # Update the coords2ix map to include the new qubits in the positions of the eliminated ones
        for j, q in enumerate(lq):
            qcoord = np.array(ix2cell[2][q])
            elimstabcoord = np.array(ix2cell[1][eliminated_stabs[j]])
            print(f'{qcoord=}, {elimstabcoord=}')
            merged_q_coord = torus((elimstabcoord - qcoord) * 2 + qcoord, distance=distance)
            print(f'{merged_q_coord=}')
            merged_q_ix = cells2i[2][merged_q_coord]
            cells2i[2][tuple(qcoord)] = merged_q_ix
        print(cells2i[2])


            #TODO Find new boundary map matrices
        # remove stabilizers that have been eliminated from the vertex -> edge map
        ve_map = cob_map_mats[0]
        # plt.matshow(ve_map)
        # plt.show()
        ve_map[eliminated_stabs, :] = np.zeros(len(cells[0]), dtype=int)
        # plt.matshow(ve_map)
        # plt.show()

        # TODO Find new future map matrices
        future_edges_matrix = map_dict_to_array(future_edges_map)
        # Set eliminated stabilizers to zero
        future_edges_matrix[:, eliminated_stabs] = 0


def map_dict_to_array(dict_to_change):
    indim = max(dict_to_change.keys()) + 1
    outdim = max([x for y in dict_to_change.values() for x in y]) + 1
    out = np.zeros(shape=(outdim, indim), dtype=int)
    for key, value in dict_to_change.items():
        out[value, key] = 1
    return out


def get_incident_vertices(edge_boundary, edge_vertex_map=None):
    incident_vertices = set()
    for e in edge_boundary:
        new_v = set(edge_vertex_map[e])
        incident_vertices.update(new_v)
    return incident_vertices


def powerset(iterable, include_empty=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if include_empty:
        return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def future_cells_toric(dim_shift, cell_coords, dimension, distance, future_dir):
    """
    TODO ensure that this function always gives future faces and edges in the same orders relative to the vertex location
    it should do
    :param dim_shift: go to cells in +dim_shift dimensions (can be negative)
    :param cell_coords:
    :param dimension:
    :param distance:
    :param future_dir:
    :return:
    """
    neighbours = []
    embedded_in = [i for i in range(dimension) if cell_coords[i] % 2]
    orthogonal = [i for i in range(dimension) if i not in embedded_in]
    if np.sign(dim_shift) == -1:
        move_in = embedded_in
    elif np.sign(dim_shift) == 1:
        move_in = orthogonal
    else:
        raise ValueError
    dim_change = dim_shift * np.sign(dim_shift)
    for increment_dims in combinations(move_in, r=dim_change):
        move_vec = np.zeros(dimension)
        for move_dirs in product((1, -1), repeat=dim_change):
            move_vec[list(increment_dims)] = move_dirs
            if np.dot(move_vec, future_dir) > 0:
                nu_vec = cell_coords + move_vec
                neighbours.append(torus(tuple(nu_vec), distance=distance))
    return neighbours


def erasure_only_sweep_rule(qubit_cell_dim=2, n_shots=1, loss_rate=0.1, n_sweeps=5, printing=False, error_q=None,
                            erasure_q=None, lattice_info=None, parallelize=False, local_map=None, final_full_swp=False):
    """
    Deal with erasures by reporting random outcomes for the erased qubits and then performing a sweep rule restricted
    to the lost regions. Need to change direction due to the presence of boundaries
    :param qubit_cell_dim:
    :param n_shots:
    :param loss_rate:
    :param n_sweeps:
    :param printing:
    :param error_q:
    :param erasure_q:
    :param lattice_info:
    :return:
    """
    if lattice_info is not None:
        cells, cells2i, b_maps, cob_maps, future_faces_maps, future_edges_maps, h, correlation_surface = lattice_info
        i2qubit_coord = {v: k for k, v in cells2i[qubit_cell_dim].items()}

    else:
        raise ValueError('No lattice information provided')
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim-1])
    #initialise erasures and corresponding errors:


    if h is None:
        h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
        for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
            h[stab, qubits] = 1
    h_sparse = csr_matrix(h)

    num_failures = 0
    if parallelize:
        # initialize errors inside each process
        n_cpu = multiprocessing.cpu_count() - 1
        if n_shots % n_cpu:
            shots_per_batch = ceil(n_shots / n_cpu)
            last_batch_size = n_shots - (n_cpu - 1) * shots_per_batch
            batch_sizes = [shots_per_batch for _ in range(n_cpu - 1)] + [last_batch_size]
        else:
            batch_sizes = [n_shots / n_cpu for _ in range(n_cpu)]

        pool = multiprocessing.Pool(n_cpu)
        out = pool.starmap(sweep_eras_batched, zip(batch_sizes, repeat(loss_rate), repeat(h), repeat(future_faces_maps),
                                                   repeat(future_edges_maps), repeat(nq), repeat(b_maps), repeat(n_sweeps),
                                                   repeat(final_full_swp), repeat(local_map)))
        final_error = np.hstack(out)
    else:
        if error_q is None and erasure_q is None:
            errors, erasures = gen_errors(nq, n_shots, 0, loss_rate)
            errors = erasures * np.random.choice((0, 1), p=(0.5, 0.5), size=(nq, n_shots))
        else:
            errors, erasures = np.zeros((nq, 1)), np.zeros((nq, 1))
            errors[error_q, 0] = 1
            erasures[erasure_q, 0] = 1
            n_shots = 1
        final_error = np.zeros((nq, n_shots))
        for shot_ix in range(n_shots):
            errors_this_shot = errors[:, shot_ix]
            error_ixs = set(np.where(errors_this_shot)[0])
            # print(error_ixs)
            erasures_this_shot = erasures[:, shot_ix]
            erasure_ixs = np.where(erasures_this_shot)[0]
            syndrome = (h @ errors_this_shot) % 2
            nt_edges = np.where(syndrome)[0]
            # print(nt_edges)
            nt_vertices = get_incident_vertices(nt_edges, b_maps[1])
            # print(erasure_ixs)
            for j in range(len(future_faces_maps)):
                for sweep_ix in range(n_sweeps):
                    if printing:
                        error_coords = [i2qubit_coord[ix] for ix in error_ixs]
                        erasure_coords = [i2qubit_coord[ix] for ix in erasure_q]
                        print(f'{error_coords=}')
                        print(f'{erasure_coords=}')
                        print(f'sweep direction index: {j}')
                    tot_flips_this_sweep = []
                    for v in nt_vertices:
                        ff = future_faces_maps[j][v]
                        fe = future_edges_maps[j][v]
                        error_restricted = set(nt_edges).intersection(fe)
                        if error_restricted:
                            ff_in_erasure = list(set(ff).intersection(erasure_ixs))
                            for face_set in powerset(ff_in_erasure):
                                # get boundary
                                tot_boundary = set()
                                for f in face_set:
                                    boundary = b_maps[2][f]
                                    tot_boundary = tot_boundary.symmetric_difference(boundary)
                                local_boundary = tot_boundary.intersection(fe)

                                if error_restricted == local_boundary:
                                    if printing:
                                        print(f'{local_boundary=}')
                                        print(face_set)
                                        print(
                                            f'identified face set = {face_set}, coords: {[i2qubit_coord[q] for q in face_set]}')
                                        print('\n')
                                    tot_flips_this_sweep += face_set
                                    break
                        # print(tot_flips_this_sweep)
                    flipped_syndromes = set()
                    for face in tot_flips_this_sweep:
                        flipped_syndromes = flipped_syndromes.symmetric_difference(set(b_maps[2][face]))
                    nt_edges = list(set(nt_edges).symmetric_difference(flipped_syndromes))

                    error_ixs = set(tot_flips_this_sweep).symmetric_difference(error_ixs)
                    if not tot_flips_this_sweep:
                        # If no flips, skip
                        break
                    # print(error_ixs)
                    # error = np.zeros((nq, 1))
                    # error[list(error_ixs), 0] = 1
                    # synd = (h @ error) % 2
                    # print(np.sum(synd))
            num_errors_remaining = len(error_ixs)
            assert len(error_ixs.difference(erasure_ixs)) == 0
            final_error[list(error_ixs), shot_ix] = 1

    final_error_rate = np.sum(final_error) / (n_shots * nq)
    print(f"{final_error_rate=}, initial error rate {loss_rate / 2}")
    intersection_with_logop = correlation_surface * final_error
    intersection_parity = np.sum(intersection_with_logop, axis=0)
    intersection_parity %= 2
    num_errors = np.sum(intersection_parity)
    #         # if num_errors_remaining:
        #     num_failures += 1
        #     print(flipped_syndromes)
        #     print([i2qubit_coord[ix] for ix in error_ixs], error_ixs)
        #     print([i2qubit_coord[ix] for ix in erasure_ixs])
        #     exit()

    print(num_errors/n_shots)
    return num_errors/n_shots


def sweep_eras_batched(batch_size, eras_rate, h, ff_maps, fe_maps, nq, boundary_maps, num_sweeps,
                       final_full_sweeps=False, local_map=None):
    """
    This function samples erasures on a 3D lattice and performs sweep rule on the regions of lost qubits
    :param batch_size:
    :param eras_rate:
    :param h:
    :param ff_maps:
    :param fe_maps:
    :param nq:
    :param boundary_maps:
    :param num_sweeps:
    :return:
    """
    # sample erasures
    local_sampler = np.random.RandomState()
    erasures = local_sampler.choice((0, 1), p=(1 - eras_rate, eras_rate), size=(nq, batch_size)).astype(np.uint8)
    errors = erasures * local_sampler.choice((0, 1), p=(0.5, 0.5), size=(nq, batch_size))

    batch_final_errs = np.zeros((nq, batch_size), dtype=np.uint8)
    for shot_ix in range(batch_size):
        errors_this_shot = errors[:, shot_ix]
        erasures_this_shot = erasures[:, shot_ix]
        errors_ix_after_sweeps = sweep_per_shot_erasure(errors_this_shot, erasures_this_shot, boundary_maps, ff_maps,
                                                        fe_maps, h, num_sweeps)
        batch_final_errs[list(errors_ix_after_sweeps), shot_ix] = 1

    if final_full_sweeps:
        syndrome = (h @ batch_final_errs) % 2
        error_qubit_ixs = [np.where(batch_final_errs[:, ix])[0].astype(np.uint32) for ix in range(batch_size)]
        non_triv_synd_ixs = [np.where(syndrome[:, ix])[0].astype(np.uint32) for ix in range(batch_size)]

        out = sweep_rule_per_shot(n_sweeps=num_sweeps, non_triv_synd_ixs=non_triv_synd_ixs, boundary_maps=boundary_maps, ff_map=ff_maps[0],
                                  fe_map=fe_maps[0], error_qubit_ixs=error_qubit_ixs, local_map=local_map, use_local_map=local_map is not None)
        error_ixs_out = [list(x) for x in out]

        error_out_mat = np.zeros(shape=(nq, batch_size), dtype=np.uint8)

        for shot, q_er_ixs in enumerate(error_ixs_out):
            error_out_mat[q_er_ixs, shot] = np.ones(len(q_er_ixs), dtype=np.uint8)
        batch_final_errs = error_out_mat

    return batch_final_errs

def sweep_per_shot_erasure(error, erasure, bm, ff_maps, fe_maps, h_mat, number_sweeps):
    erasure_ixs = np.where(erasure)[0]
    error_ixs = np.where(error)[0]
    syndrome = (h_mat @ error) % 2
    nt_edges = np.where(syndrome)[0]
    for _ in range(2):
        for j in range(len(ff_maps)):
            for sweep_ix in range(number_sweeps):
                nt_vertices = get_incident_vertices(nt_edges, bm[1])
                tot_flips_this_sweep = []
                for v in nt_vertices:
                    ff = ff_maps[j][v]
                    fe = fe_maps[j][v]
                    error_restricted = set(nt_edges).intersection(fe)
                    if error_restricted:
                        ff_in_erasure = list(set(ff).intersection(erasure_ixs))
                        for face_set in powerset(ff_in_erasure):
                            # get boundary
                            tot_boundary = set()
                            for f in face_set:
                                boundary = bm[2][f]
                                tot_boundary = tot_boundary.symmetric_difference(boundary)
                            local_boundary = tot_boundary.intersection(fe)
                            if error_restricted == local_boundary:
                                tot_flips_this_sweep += face_set
                                break
                    # print(tot_flips_this_sweep)
                flipped_syndromes = set()
                for face in tot_flips_this_sweep:
                    flipped_syndromes = flipped_syndromes.symmetric_difference(set(bm[2][face]))
                nt_edges = list(set(nt_edges).symmetric_difference(flipped_syndromes))

                error_ixs = set(tot_flips_this_sweep).symmetric_difference(error_ixs)
                if not tot_flips_this_sweep:
                    # If no flips, skip
                    break
    num_errors_remaining = len(error_ixs)
    assert len(error_ixs.difference(erasure_ixs)) == 0
    return error_ixs
    # batch_final_errs[list(error_ixs), shot_ix] = 1


def run_sweep_rule_decoder(dimension, distance, future_dir, qubit_cell_dim=2, n_shots=1, error_rate=0.01, loss_rate=0., n_sweeps=5,
         printing=False, error_q=None, serial=True, lattice_info=None, use_local_map=False):
    """
    TODO implement the optional lattice info for Tooms as well
    :param dimension:
    :param distance:
    :param future_dir:
    :param qubit_cell_dim:
    :param n_shots:
    :param error_rate:
    :param loss_rate:
    :param n_sweeps:
    :param printing:
    :param error_q: for using custom errors
    :param serial:
    :param lattice_info:
    :return:
    """

    if lattice_info is not None:
        cells, cells2i, b_maps, cob_maps, future_faces_map, future_edges_map, h, correlation_surface, local_boundary_face_map = lattice_info
        future_edges_map = future_edges_map[0]
        future_faces_map = future_faces_map[0]
        h_sparse = csr_matrix(h)
        nq = len(cells[qubit_cell_dim])
        n_stab = len(cells[qubit_cell_dim - 1])

    else:
        cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
        local_boundary_face_map = None
        use_local_map = None

        future_faces_map = {}
        future_edges_map = {}
        for v in cells[0]:
            future_faces_map[cells2i[0][v]] = [cells2i[2][face] for face in future_cells_toric(+2, v, dimension, distance, future_dir)]
            future_edges_map[cells2i[0][v]] = [cells2i[1][edge] for edge in future_cells_toric(+1, v, dimension, distance, future_dir)]
        nq = len(cells[qubit_cell_dim])
        n_stab = len(cells[qubit_cell_dim - 1])
        h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
        for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
            h[stab, qubits] = 1
        h_sparse = csr_matrix(h)
        stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
        correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])  # Gets binary vector corresponding to the logical operator

    if error_q:
        errors = np.zeros((nq, n_shots), dtype=int)
        errors[error_q] = np.ones((len(error_q), n_shots))
    else:
        errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots), p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
        es = errors_full == 1
        ls = errors_full == 2
        errors = np.array(es, dtype=int)
        e_sparse = csr_matrix(errors)
        losses = np.array(ls, dtype=int)
        lost_qubit_ixs = [np.where(losses[:, ix])[0].astype(int) for ix in range(n_shots)]
        error_qubit_ixs_all = [np.where(errors[:, ix])[0].astype(int) for ix in range(n_shots)]

    tot_logical_errors = 0

    error_synd_sparse = h_sparse @ e_sparse
    error_synd_sparse.data %= 2
    error_syndromes = error_synd_sparse.toarray()
    non_triv_synd_ixs_all = [np.where(error_syndromes[:, ix])[0].astype(int) for ix in range(n_shots)]

    if serial:
        error_ixs_out = []
        for ix in range(n_shots):
            error_ixs_out = sweep_rule_per_shot(n_sweeps, non_triv_synd_ixs_all, b_maps, future_faces_map,
                                                     future_edges_map, error_qubit_ixs=error_qubit_ixs_all,
                                                     use_local_map=use_local_map, local_map=local_boundary_face_map, batch=True)


            print(f'total errors: {tot_logical_errors}')
    else:
        n_processes = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(n_processes)
        non_triv_synd_ixs_batched = batch_data(non_triv_synd_ixs_all, n_processes)
        error_qubit_ixs_batched = batch_data(error_qubit_ixs_all, n_processes)


        out = pool.starmap(sweep_rule_per_shot, zip(repeat(n_sweeps), non_triv_synd_ixs_batched, repeat(b_maps),
                                                    repeat(future_faces_map), repeat(future_edges_map), error_qubit_ixs_batched,
                                                    repeat(use_local_map), repeat(local_boundary_face_map)))
        error_ixs_out = [list(x) for y in out for x in y]

    def get_tot_errors(error_ixs, corr):
        error_out_mat = np.zeros(shape=(nq, n_shots), dtype=np.uint8)
        for shot, q_er_ixs in enumerate(error_ixs):
            error_out_mat[q_er_ixs, shot] = np.ones(len(q_er_ixs), dtype=np.uint8)
        logical_errors = np.sum(corr * error_out_mat, axis=0) % 2
        return np.sum(logical_errors)
    tot_logical_errors = get_tot_errors(error_ixs_out, correlation_surface)

    return tot_logical_errors / n_shots


def do_sweeps(nt_synd_ix, err_ixs, n_swp, fem, ffm, bm, lm):
    for jx in range(n_swp):
        tot_flips_this_sweep = []
        nt_edges = nt_synd_ix
        nt_vertices = get_incident_vertices(nt_edges, bm[1])

        for v in nt_vertices:
            ff = ffm[v]
            fe = fem[v]
            error_restricted = set(nt_edges).intersection(set(fe))
            if error_restricted and (
            not len(error_restricted) % 2):  # is there any error boundary in the vertex future?
                fe_inds = sorted([fe.index(e) for e in list(error_restricted)])
                # print([face_ix for face_ix in local_map[tuple(fe_inds)]])
                try:
                    tot_flips_this_sweep += [ff[face_ix] for face_ix in lm[tuple(fe_inds)]]
                except TypeError:
                    print(f"{ff=}, {lm[tuple(fe_inds)]}")
                    raise ValueError
        flipped_syndromes = set()
        for face in tot_flips_this_sweep:
            flipped_syndromes = flipped_syndromes.symmetric_difference(set(bm[2][face]))
        nt_synd_ix = list(set(nt_synd_ix).symmetric_difference(flipped_syndromes))

        err_ixs = set(tot_flips_this_sweep).symmetric_difference(err_ixs)
    return err_ixs


def do_sweeps_no_local_map(nt_synd_ix, err_ixs, n_swp, fem, ffm, bm):
    """

    :param nt_synd_ix: non-trivial syndrome indices
    :param err_ixs: error indices
    :param n_swp: number of sweeps
    :param fem: future edges map
    :param ffm: future faces map
    :param bm: boundary maps
    :return:
    """
    for jx in range(n_swp):
        tot_flips_this_sweep = []
        nt_edges = nt_synd_ix
        nt_vertices = get_incident_vertices(nt_edges, bm[1])

        for v in nt_vertices:
            ff = ffm[v]
            fe = set(fem[v])
            error_restricted = set(nt_edges).intersection(fe)

            if error_restricted and (not len(error_restricted) % 2):  # is there an error boundary in the vertex future?
                for face_set in powerset(ff):
                    # get boundary
                    tot_boundary = set()
                    for f in face_set:
                        boundary = bm[2][f]
                        tot_boundary = tot_boundary.symmetric_difference(boundary)
                    local_boundary = tot_boundary.intersection(fe)
                    # print(f'{local_boundary=}')
                    # print(f'identified face set = {face_set}, coords: {[i2coord[2][q] for q in face_set]}')
                    # print('\n')
                    if error_restricted == local_boundary:
                        tot_flips_this_sweep += face_set
                        break
        flipped_syndromes = set()
        for face in tot_flips_this_sweep:
            flipped_syndromes = flipped_syndromes.symmetric_difference(set(bm[2][face]))
        nt_synd_ix = list(set(nt_synd_ix).symmetric_difference(flipped_syndromes))

        err_ixs = set(tot_flips_this_sweep).symmetric_difference(err_ixs)
    return err_ixs


def sweep_rule_per_shot(n_sweeps, non_triv_synd_ixs, boundary_maps, ff_map, fe_map, error_qubit_ixs,
                       use_local_map=False, local_map=None, batch=True):
    out_list = []
    if batch:
        # print(type(error_qubit_ixs))
        for shot in range(len(non_triv_synd_ixs)):
            non_triv_synd_ixs_in = non_triv_synd_ixs[shot]
            error_qubit_ixs_in = error_qubit_ixs[shot]

            if use_local_map:
                # print('using local map')
                e_out = do_sweeps(non_triv_synd_ixs_in, error_qubit_ixs_in, n_sweeps, fe_map, ff_map, boundary_maps, local_map)
                out_list.append(e_out)
            else:
                raise NotImplementedError
    else:
        if use_local_map:
            # print('using local map')
            e_out = do_sweeps(non_triv_synd_ixs, error_qubit_ixs, n_sweeps, fe_map, ff_map,
                                        boundary_maps, local_map)
            out_list.append(e_out)
        else:
            raise NotImplementedError

    return out_list


def local_edge_to_faces_hypercubic(dimension):
    """
    Assuming future direction is (1, 1, 1, ...)
    This function finds the map of flipped edges in the local future of a vertex to the faces that should be flipped
    to correct it, so that this can be done by lookup in constant time per iteration
    It assumes the same ordering of cells in the boundary maps, i.e. a homogenous lattice, in this case hypercubic
    :param dimension:
    :param local_edge:
    :return:
    """
    boundary2face_ix = {}
    # find the local future cells
    move_in = list(range(dimension))
    local_future_cells = [[], []]
    for future_cell_dim in (1, 2):
        for increment_dims in combinations(move_in, r=future_cell_dim):
            new_vec = np.zeros(dimension, dtype=int)
            new_vec[list(increment_dims)] = 1
            local_future_cells[future_cell_dim-1].append(new_vec)

    # index the future edges and faces
    future_cells_2_ix = [{tuple(coord): ix for ix, coord in enumerate(cell_list)} for cell_list in local_future_cells]

    # find all possible local boundaries
    poss_local_boundaries = powerset(range(dimension))
    # restrict to those with even number of edges
    poss_local_boundaries_restr = [x for x in poss_local_boundaries if not (len(x) % 2)]

    # find all the boundaries of the future faces when restricted to the future edges
    f_boundary_dict = {}
    for face in local_future_cells[1]:
        dims_embedded_in = np.where(face)[0]
        face_boundary = []
        for d in dims_embedded_in:
            e_vec = face.copy()
            e_vec[d] -= 1
            face_boundary.append(e_vec)
        f_boundary_dict[future_cells_2_ix[1][tuple(face)]] = set([future_cells_2_ix[0][tuple(e)] for e in face_boundary])
    # print(f'{f_boundary_dict=}')

    # find boundaries of sets of future faces
    ff_boundaries = {}
    for face_set in powerset(range(len(local_future_cells[1]))):
        boundary = set()
        f_ixs = []
        for f_ix in face_set:
            f_ixs.append(f_ix)
            boundary = boundary.symmetric_difference(f_boundary_dict[f_ix])
        new_key = tuple(sorted(boundary))
        if (new_key not in ff_boundaries.keys()) or (len(f_ixs) < len(ff_boundaries[new_key])):  # only replace the future faces if this has lower weight
            ff_boundaries[tuple(sorted(boundary))] = f_ixs
    # print(f'{ff_boundaries=}')

    # iterate through possible future boundaries and identify future face sets
    for b in poss_local_boundaries_restr:
        if b in ff_boundaries.keys():
            boundary2face_ix[b] = ff_boundaries[b]
        else:
            print(b)

    # print(boundary2face_ix)
    # print(ix2cell)
    # print([f'edges: {[ix2cell[0][y] for y in k]}, faces: {[ix2cell[1][x] for x in v]}' for k, v in boundary2face_ix.items()])
    return boundary2face_ix


def init_lattice_sweep_rule(dimension, distance, future_dirs, qubit_cell_dim=2, logop=False, get_local_map=False):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim-1])
    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    local_boundary_face_map = local_edge_to_faces_hypercubic(dimension)
    if type(future_dirs[0]) is int:
        future_dir = [future_dirs]
    ff_maps = []
    fe_maps = []
    for fd in future_dirs:
        future_faces_map = {}
        future_edges_map = {}
        for v in cells[0]:
            future_faces_map[cells2i[0][v]] = [cells2i[2][face] for face in
                                               future_cells_toric(+2, v, dimension, distance, fd)]
            future_edges_map[cells2i[0][v]] = [cells2i[1][edge] for edge in
                                               future_cells_toric(+1, v, dimension, distance, fd)]
        ff_maps.append(future_faces_map)
        fe_maps.append(future_edges_map)

    if logop:
        corr_surf = logical_x_toric(cells, q_cell_dim=2, distance=distance, dimension=dimension, q2i_dict=cells2i[2])
        if get_local_map:
            return cells, cells2i, b_maps, cob_maps, ff_maps, fe_maps, h, corr_surf, local_boundary_face_map

        return cells, cells2i, b_maps, cob_maps, ff_maps, fe_maps, h, corr_surf
    return cells, cells2i, b_maps, cob_maps, ff_maps, fe_maps, h


def run_erasure_only_sweep(dim, Ls, n_sweeps=20, n_shots=1000, savedata=False, final_full_sweep=False):
    future_dirs = [(1, 1, 1), (-1, -1, -1)]
    future_dirs = [x for x in product((1, -1), repeat=3)]
    # future_dirs = [(1, 1, 1)]
    loss_rates = np.linspace(0.5, 0.8, 11)
    # for L in (3, 4, 5):
    outfile = f"sweep_erasure_dim{dim}_L{Ls[0]}-{Ls[-1]}_{n_sweeps}sweeps_{n_shots}shots_final_sweep{final_full_sweep}"
    outdir = os.path.join(os.getcwd(), 'outputs', '24_05_26', 'sweep_erasure_only')
    data = (loss_rates, {})
    for L in Ls:
        this_fname = f"{outfile}_upto{L}"
        t0 = perf_counter()
        print(f"running {L=}")
        lattice_info = init_lattice_sweep_rule(dim, L, future_dirs=future_dirs, logop=True)
        local_boundary_face_map = local_edge_to_faces_hypercubic(dim)
        y = []
        for l in loss_rates:
            out = erasure_only_sweep_rule(lattice_info=lattice_info, n_sweeps=n_sweeps, loss_rate=l, n_shots=n_shots, parallelize=True,
                                          local_map=local_boundary_face_map, final_full_swp=final_full_sweep)
            y.append(out)
            print(f"cumulative time this size: {perf_counter() - t0}")
        data[1][L] = y

        if savedata:
            appendix = 1
            while f"{this_fname}_{appendix}.pkl" in os.listdir(outdir):
                appendix += 1
            save_obj(data, f"{this_fname}_{appendix}", outdir)

    for k, v in data[1].items():
        plt.plot(data[0], v, label=k)
    plt.legend()
    plt.show()



    # exit()
    #
    # future_dir = tuple([1] * 5)
    # dim = 5
    # errors = np.linspace(0.01, 0.035, 7)
    # path = os.getcwd()
    # Ls = [3, 4]
    # for distance in Ls:
    #     lattice_info = init_lattice(dim, L, future_dir)
    #
    #     out = []
    #
    #     for error_rate in errors:
    #         print(f'{distance=}, {error_rate=}')
    #         error_rate = run_sweep_rule_decoder(dimension=dim, distance=distance, future_dir=tuple([1] * dim),
    #                                             error_rate=error_rate, n_sweeps=10, printing=False, n_shots=1000,
    #                                             serial=False, lattice_info=lattice_info)
    #         out.append(error_rate)
    #     plt.plot(errors, out)
    #     save_obj(out, f'distance{distance}_data', path)
    # plt.xlabel('Phenomenological noise')
    # plt.ylabel('logical error rate')
    # plt.title('10 sweeps, 1000 shots')
    # plt.legend(Ls)
    # plt.savefig('Sweep_rule_5D_toric.png')
    # plt.show()


if __name__ == '__main__':

    for _ in range(5):
        run_erasure_only_sweep(dim=3, Ls=[3, 4, 5, 6, 7], savedata=True, n_sweeps=20)
    exit()



    dimension = 5
    distances = [3, 4, 5]
    future_dir = [1] * dimension
    qubit_cell_dim = 2
    n_shots = 1000
    loss_rate = 0.
    outdir = os.path.join(os.getcwd(), 'outputs', '24_05_22', 'sweep')
    savedata = True
    n_sweepses = [10, 20, 50, 100]
    for n_sweeps in n_sweepses:
        for _ in range(10):
            results = {k: [] for k in distances}
            error_rates = np.linspace(0.014, 0.027, 7)

            for distance in distances:
                print(f"Running distance {distance}")
                lattice_info = init_lattice_sweep_rule(dimension, distance, [future_dir], logop=True, get_local_map=True)

                for e in error_rates:
                    print(f"Doing error rate {e}")
                    t0 = perf_counter()
                    out = run_sweep_rule_decoder(dimension, distance, future_dir, qubit_cell_dim=2, n_shots=n_shots, error_rate=e, loss_rate=loss_rate,
                                       n_sweeps=n_sweeps, printing=False, error_q=None, serial=False, lattice_info=lattice_info, use_local_map=True)
                    results[distance].append(out)
                    print(f"time this shot: {perf_counter() - t0}")
                if savedata:
                    fname = f"sweep_rule_L{distances[0]}-{distances[-1]}uptoL{distance}_dim{dimension}_{n_sweeps}sweeps_{n_shots}shots"
                    appendix = 1
                    while f"{fname}_{appendix}.pkl" in os.listdir(outdir):
                        appendix += 1
                    save_obj(results, f"{fname}_{appendix}", path=outdir)
                print("\n")
            for l in distances:
                plt.plot(error_rates, results[l], label=l)
            plt.legend()
            plt.show()
    print(out)
    exit()

    main()
    exit()



    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dim)
    local_boundary_face_map = local_edge_to_faces_hypercubic(distance)
    future_faces_map = {}
    future_edges_map = {}
    for v in cells[0]:
        future_faces_map[cells2i[0][v]] = [cells2i[2][face] for face in
                                           future_cells_toric(+2, v, dim, distance, future_dir)]
        future_edges_map[cells2i[0][v]] = [cells2i[1][edge] for edge in
                                           future_cells_toric(+1, v, dim, distance, future_dir)]

    erasure_only_sweep_rule(dimension=3, distance=5, future_dir=[1,1,1])

    ge_on_hypercubic(distance=3, qubit_cell_dim=2)
    exit()

    main()


