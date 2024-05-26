import numpy as np
import pyvista as pv
from toric import cell_dicts_and_boundary_maps, build_coords, logical_x_toric
from ca_decoder import ne_parity_map, visualise_q_and_stab, gen_errors, sparse_tooms_iters_eras_conv, lossy_tooms_sweeps, loss_decode_check_error, tooms_no_loss
from pyvista import examples
from scipy.sparse import csr_matrix
from sweep_rule import init_lattice_sweep_rule, sweep_per_shot_erasure, sweep_rule_per_shot
from itertools import product


def error_mats_from_coords(error_coords, loss_coords, nq, coord2ix):
    error_mat = np.zeros((nq, 1))
    loss_mat = np.zeros((nq, 1))
    error_ixs = [coord2ix[2][c] for c in error_coords]
    loss_ixs = [coord2ix[2][c] for c in loss_coords]
    error_mat[error_ixs, 0] = 1
    loss_mat[loss_ixs, 0] = 1
    return error_mat, loss_mat


def qubit_ix_to_cells(nq, b_maps, i2coord_dicts, dim, distance, cells2i_vis):
    """
    Build a dictionary for every face
    :param faces:
    :return:
    """
    out = {}
    for face_ix in range(nq):
        # face_coords = i2coord[2][face_ix]
        edgeset = b_maps[2][face_ix]
        vertex_set = set([x for e in edgeset for x in b_maps[1][e]])
        # find the indices of the vertices
        vertex_coords = [i2coord_dicts[0][v] for v in vertex_set]
        # Change the coords so they can be displayed on the visualisation with wrap-around

        new_ixs = []
        new_v_set_coords = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(dim):
            dim_coords = [v[i] for v in vertex_coords]
            if min(dim_coords) == 0 and max(dim_coords) == ((distance - 1) * 2):
                new_dim_coords = []
                for c in dim_coords:
                    if c == 0:
                        new_dim_coords.append(distance * 2)
                    else:
                        new_dim_coords.append(c)
            else:
                new_dim_coords = dim_coords
            for ix in range(4):
                new_v_set_coords[ix][i] = new_dim_coords[ix]

        # Get the coords in the correct order
        new_v_set_coords = np.array(new_v_set_coords)
        ordered = False
        passed = 0
        while not ordered:
            for i in range(4):
                diff = new_v_set_coords[i] - new_v_set_coords[(i + 1) % 4]
                if np.abs(sum(diff)) != 2:
                    new_v_set_coords[[(i+1) % 4, (i+2) % 4]] = new_v_set_coords[[(i+2) % 4, (i+1) % 4]]
                    passed = 0
                    break
                else:
                    passed += 1
            if passed == 4:
                ordered = True

        # find the indices of these new coordinates in the visualisation lattice
        ixs = [cells2i_vis[0][tuple(v)] for v in new_v_set_coords]
        # ixs = [ixs[0], ixs[2], ixs[3], ixs[1]]
        cells_mesh = [4] + ixs
        out[face_ix] = cells_mesh
    return out


def plot_3D_lattice(lattice_mesh, face_ix_cell_dict, error_ixs, loss_ixs, other_ixs=None, show_loss_edges=True):
    p2 = pv.Plotter()
    p2.add_mesh(lattice_mesh, show_edges=True)
    p2.add_points(lattice_mesh.points, color='red', point_size=10)

    if len(error_ixs) > 0:
        error_cells_mesh = np.array([face_ix_cell_dict[f] for f in error_ixs])
        error_cells_mesh.flatten()
        error_mesh = pv.PolyData(lattice_mesh.points, error_cells_mesh)
        p2.add_mesh(error_mesh, color='pink', edge_color='red', line_width=5, show_edges=True)
    if len(loss_ixs) > 0:
        loss_cells_mesh = np.array([face_ix_cell_dict[f] for f in loss_ixs])
        loss_cells_mesh.flatten()
        loss_mesh = pv.PolyData(lattice_mesh.points, loss_cells_mesh)
        p2.add_mesh(loss_mesh, color='blue', edge_color='green', line_width=5, show_edges=show_loss_edges)
    if other_ixs is not None:
        other_cells_mesh = np.array([face_ix_cell_dict[f] for f in other_ixs])
        other_cells_mesh.flatten()
        other_cells_mesh = pv.PolyData(lattice_mesh.points, other_cells_mesh)
        p2.add_mesh(other_cells_mesh, color='purple', edge_color='red', line_width=5, show_edges=False)
    marker = pv.create_axes_marker(label_size=(0.05, 0.05))
    p2.add_actor(marker)
    p2.show()


def get_lattice_mesh(distance, dimension, nq, b_maps, i2coord, qubit_cell_dim=2):
    # build the cells for the vis
    # Create an extra layer of vertices at the edge to associate with the coordinates at 0, so that we can
    # visualise the periodic boundary conditions
    cells_vis, cells2i_vis, b_maps_vis, cob_maps_vis = cell_dicts_and_boundary_maps(distance=distance + 1, dimension=dimension, periodic_bcs=False)
    if qubit_cell_dim == 2:
        q_ix_to_cell_dict = qubit_ix_to_cells(nq, b_maps, i2coord, dimension, distance, cells2i_vis)
    else:
        raise NotImplementedError

    all_vertices = np.array(cells_vis[0], dtype=float)
    edges = b_maps_vis[1]
    print(edges)
    lines = np.zeros((len(cells_vis[1]), 3), dtype=int)

    for key, value in edges.items():
        row = [2] + sorted(value)
        lines[key] = row
    lines = lines.flatten()
    print(lines)
    mymesh = pv.PolyData(all_vertices, lines=lines)
    return mymesh, q_ix_to_cell_dict


def visualise_from_coords(distance, dimension, error_coords=None, loss_coords=None, error_mat=None, loss_mat=None,
                          from_mats=False, n_iters_to_show=20, other_mat_to_show=None, qubit_cell_dim=2, erasure_conversion=False):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension, periodic_bcs=True)
    i2coord = [{v: k for k, v in cells2i[ix].items()} for ix in range(dimension + 1)]
    nq = len(cells[2])
    n_stab = len(cells[1])

    ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    ne_maps = [ne_parity_map(cells, 2, cells2i, dimension, distance, nq, n_stab, ne_dir=ne_dir) for ne_dir
               in ne_dirs]
    ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]
    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[1].items():
        h[stab, qubits] = 1
    h_sparse = csr_matrix(h)
    stabs_per_qubit = len(b_maps[2][0])
    correlation_surface = csr_matrix(logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim]))

    if from_mats:
        custom_errors, custom_losses = error_mat, loss_mat
    else:
        custom_errors, custom_losses = error_mats_from_coords(error_coords, loss_coords, nq, coord2ix=cells2i)

    error_ixs = np.where(custom_errors)[0]
    loss_ixs = np.where(custom_losses)[0]
    if other_mat_to_show is not None:
        other_ixs = np.where(other_mat_to_show)[0]
    else:
        other_ixs = None

    lattice_mesh, face_ix_to_cell_dict = get_lattice_mesh(distance, dimension, nq, b_maps, i2coord, qubit_cell_dim=qubit_cell_dim)
    plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=loss_ixs, error_ixs=error_ixs, other_ixs=other_ixs)

    current_error = csr_matrix(custom_errors)
    current_losses = csr_matrix(custom_losses)
    error_syndrome = h_sparse @ current_error
    error_syndrome.data %= 2

    for _ in range(n_iters_to_show):
        # print(f'{error_ixs=}')
        # print(f'{loss_ixs=}')
        print([i2coord[2][e] for e in error_ixs])
        print([i2coord[2][l] for l in loss_ixs])

        current_error, current_losses = csr_matrix(current_error), csr_matrix(current_losses)
        if _ % 10:
            tmp_list = ne_parity_mats_sparse[1:]
            tmp_list.append(ne_parity_mats_sparse[0])
            ne_parity_mats_sparse = tmp_list
        esp_out, loss_out = sparse_tooms_iters_eras_conv(num_iters=0, change_dir_freq=10,
                                                         ne_mats=ne_parity_mats_sparse,
                                                         q_err_sparse=current_error, h_sparse=h_sparse,
                                                         loss_sparse=current_losses, num_eras_iters=1)
        current_error, current_losses = esp_out.toarray(), loss_out.toarray()
        error_ixs = np.where(current_error)[0]
        loss_ixs = np.where(current_losses)[0]

        plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=loss_ixs,
                        error_ixs=error_ixs)




def visualise_sweep_from_coords(distance, dimension, error_coords=None, loss_coords=None, error_mat=None, loss_mat=None,
                          from_mats=False, n_iters_to_show=20, other_mat_to_show=None, qubit_cell_dim=2, eras_rate=0.5):

    future_dirs = [(-1, 1, -1)]
    future_dirs = [x for x in product((1, -1), repeat=3)]
    print(future_dirs)

    lattice_info = init_lattice_sweep_rule(dimension, distance, future_dirs=future_dirs, logop=True)
    cells, cells2i, b_maps, cob_maps, future_faces_maps, future_edges_maps, h, correlation_surface = lattice_info

    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    # initialise erasures and corresponding errors:

    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1

    i2coord = [{v: k for k, v in cells2i[ix].items()} for ix in range(dimension + 1)]

    h_sparse = csr_matrix(h)
    stabs_per_qubit = len(b_maps[2][0])


    # if from_mats:
    #     custom_errors, custom_losses = error_mat, loss_mat
    # else:
    #     # get random erasures and errors
    #
    loss_coords = [(1, 0, 1), (3, 0, 1), (1, 0, 3), (1, 2, 1), (3, 2, 1), (1, 2, 3), (1, 1, 0), (3, 1, 0), (1, 1, 4), (4, 1, 1), (0, 1, 1), (0, 1, 3)]
    error_coords = loss_coords
    errors, erasures = error_mats_from_coords(error_coords, loss_coords, nq, coord2ix=cells2i)
    # local_sampler = np.random.RandomState()
    # erasures = local_sampler.choice((0, 1), p=(1 - eras_rate, eras_rate), size=(nq, 1)).astype(np.uint8)
    # errors = erasures * local_sampler.choice((0, 1), p=(0.5, 0.5), size=(nq, 1))

    error_ixs = np.where(errors)[0]
    loss_ixs = np.where(erasures)[0]
    if other_mat_to_show is not None:
        other_ixs = np.where(other_mat_to_show)[0]
    else:
        other_ixs = None

    lattice_mesh, face_ix_to_cell_dict = get_lattice_mesh(distance, dimension, nq, b_maps, i2coord, qubit_cell_dim=qubit_cell_dim)
    plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=loss_ixs, error_ixs=error_ixs, other_ixs=other_ixs)
    plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=[],
                        error_ixs=error_ixs, show_loss_edges=False)


    for _ in range(n_iters_to_show):
        # print(len(future_faces_maps))
        # print(future_faces_maps[0])
        # print(future_faces_maps[1])
        error_ixs = list(sweep_per_shot_erasure(errors, erasures, b_maps, future_faces_maps, future_edges_maps, h, number_sweeps=20))
        # print(error_ixs)
        errors = np.zeros((nq, 1), dtype=np.uint8)
        errors[list(error_ixs), 0] = 1
        plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=[],
                        error_ixs=error_ixs, show_loss_edges=False)


def main():

    qubit_cell_dim = 2
    distance = 5
    dimension = 3


    # build the cells for error correction stuff
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension, periodic_bcs=True)
    i2coord = [{v: k for k, v in cells2i[ix].items()} for ix in range(dimension+1)]
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])

    ne_dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    ne_maps = [ne_parity_map(cells, qubit_cell_dim, cells2i, dimension, distance, nq, n_stab, ne_dir=ne_dir) for ne_dir in ne_dirs]
    ne_parity_mats_sparse = [csr_matrix(ne_m) for ne_m in ne_maps]
    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    h_sparse = csr_matrix(h)
    stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
    qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)
    correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim])
    correlation_surface_sparse = csr_matrix(correlation_surface)  # Gets binary vector corresponding to the logical operator
    h_with_logop = np.vstack([h, correlation_surface.T], dtype=np.uint8)

    # Run a decoder and see what happens to the errors

    n_shots = 10000
    error_rate, loss_rate = 0.005, 0.005
    errors, losses = gen_errors(nq, n_shots, error_rate=error_rate, loss_rate=loss_rate)
    custom_errors, custom_losses = errors, losses
    print(custom_errors)

    errors_sparse, loss_sp = csr_matrix(custom_errors, shape=(nq, n_shots)), csr_matrix(custom_losses, shape=(nq, n_shots))

    print('predecoder')

    esp_out, loss_out = sparse_tooms_iters_eras_conv(num_iters=0, change_dir_freq=10, ne_mats=ne_parity_mats_sparse,
                                                     q_err_sparse=errors_sparse, h_sparse=h_sparse,
                                                     loss_sparse=loss_sp, num_eras_iters=100)

    print('postdecoder')

    lattice_mesh, face_ix_to_cell_dict = get_lattice_mesh(distance, dimension, nq, b_maps, i2coord)
    errors_out, losses_out = esp_out.toarray(), loss_out.toarray()

    lq = [np.where(losses_out[:, ix])[0] for ix in range(n_shots)]

    log_error, log_loss, log_error_ixs, log_loss_ixs = loss_decode_check_error(lq, h_with_logop, qbt_syndr_mat, error_rate, errors_out,
                                                  batch_size=n_shots, get_ixs=True)
    print(log_error, log_loss)

    for log_loss_ix in log_loss_ixs:
        print(log_loss_ix)
        original_error, original_loss = errors[:, log_loss_ix], losses[:, log_loss_ix]
        error_ixs = np.where(original_error)[0]
        loss_ixs = np.where(original_loss)[0]

        plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=loss_ixs, error_ixs=error_ixs)

        custom_errors = original_error.reshape(nq, 1)
        custom_losses = original_loss.reshape(nq, 1)
        for _ in range(20):
            # print(f'{error_ixs=}')
            # print(f'{loss_ixs=}')
            print([i2coord[2][e] for e in error_ixs])
            print([i2coord[2][l] for l in loss_ixs])

            custom_errors, loss_sp = csr_matrix(custom_errors), csr_matrix(custom_losses)
            if _ % 5:
                tmp_list = ne_parity_mats_sparse[1:]
                tmp_list.append(ne_parity_mats_sparse[0])
                ne_parity_mats_sparse = tmp_list
                # print(ne_parity_mats_sparse)
            esp_out, loss_out = sparse_tooms_iters_eras_conv(num_iters=0, change_dir_freq=10, ne_mats=ne_parity_mats_sparse,
                                                             q_err_sparse=custom_errors, h_sparse=h_sparse,
                                                             loss_sparse=loss_sp, num_eras_iters=1)
            custom_errors, custom_losses = esp_out.toarray(), loss_out.toarray()
            error_ixs = np.where(custom_errors)[0]
            loss_ixs = np.where(custom_losses)[0]
            lq = np.where(custom_losses[:, 0])[0]

            log_error, log_loss = loss_decode_check_error([lq], h_with_logop, qbt_syndr_mat, error_rate, custom_errors, batch_size=1)
            print(f'{log_error=}')
            print(f'{log_loss=}')
            plot_3D_lattice(lattice_mesh, face_ix_cell_dict=face_ix_to_cell_dict, loss_ixs=loss_ixs, error_ixs=error_ixs)


def sweep_rule_erasure_example():
    from sweep_rule import init_lattice_sweep_rule, local_edge_to_faces_hypercubic
    L=5
    dim=3
    erasure_rate = 0.55
    future_dirs = (1, 1, 1)
    lattice_info = init_lattice_sweep_rule(dim, L, future_dirs=future_dirs, logop=True)
    local_boundary_face_map = local_edge_to_faces_hypercubic(dim)



if __name__ == '__main__':
    visualise_sweep_from_coords(dimension=3, distance=5, eras_rate=0.5, n_iters_to_show=30)
    exit()
    # main()
    # exit()
    # error_coords = [(4, 5, 3), (7, 8, 7)]

    error_ixs = [7, 69, 70, 75, 115, 125, 134, 180, 186, 199, 227, 262, 292 ,343 ,360 ,364]
    error_ixs = [ 54 , 74  ,90 , 91,  98, 126, 127, 141, 143, 157, 183, 234, 242, 293, 295, 336, 355]
    error_mat = np.zeros((3 * 5 ** 3, 1))
    error_mat[error_ixs, 0] = 1
    loss_mat = np.zeros((3 * 5 ** 3, 1))
    error_coords = []
    # loss_coords = [(1, 9, 2), (2, 5, 9), (3, 3, 4), (5, 0, 1), (6, 7, 7), (7, 2, 3), (9, 1, 2)]
    visualise_from_coords(distance=5, dimension=3, error_coords=None, loss_coords=None, error_mat=error_mat, loss_mat=loss_mat, from_mats=True)
    exit()


    main()
