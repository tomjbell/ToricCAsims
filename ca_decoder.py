import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
from toric import cell_dicts_and_boundary_maps, adjacent_cells, torus, logical_x_toric, logical_z_toric
from scipy.sparse import csr_matrix
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

def toom_2d_phase_only(distance=20, dimension=2, error_rate=0.4, n_ca_iters=15):
    # Using dense matrices, sparse implementation below

    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)
    qubit_cell_dim = 2
    i2qcoord = {v: k for k, v in cells2i[qubit_cell_dim].items()}
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])
    ne_map = {}
    for q in cells[qubit_cell_dim]:
        q_ind = cells2i[qubit_cell_dim][q]
        embedded = [i for i in range(dimension) if q[i] % 2]
        check_inds = []
        for dim in embedded:
            nu_coord = list(q)
            nu_coord[dim] += 1
            check_inds.append(cells2i[qubit_cell_dim - 1][torus(nu_coord, distance)])
        ne_map[q_ind] = check_inds
    ne_parity_mat = np.zeros(shape=(nq, n_stab), dtype=int)
    for q, s in ne_map.items():
        ne_parity_mat[q, s] = 1

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
        flips_this_round = (ne_parity_mat @ current_synd) == 2
        prediction += flips_this_round
        current_error = (current_error + flips_this_round) % 2
        current_synd = (h @ current_error) % 2
        plt.matshow(binary_error_to_mat_2d_tooms(current_error, i2qcoord))
        plt.title(_)
        plt.show()

    # for _ in range(n_ca_iters):
    #     for q, synds in ne_map.items():
    #         # print(q, [current_synd[s][0] for s in synds])
    #         if sum([current_synd[s][0] for s in synds]) // 2:
    #             prediction[q] += 1
    #             current_error[q] += 1
    #     current_error %= 2
    #     current_synd = (h @ current_error) % 2
    #     plt.matshow(binary_error_to_mat_2d_tooms(current_error, i2qcoord))
    #     plt.title(_)
    #     plt.show()

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
    ne_map = {}
    for q in cells[qubit_cell_dim]:
        q_ind = cells2i[qubit_cell_dim][q]
        embedded = [i for i in range(dimension) if q[i] % 2]
        check_inds = []
        for dim in embedded:
            nu_coord = list(q)
            nu_coord[dim] += 1
            check_inds.append(cells2i[qubit_cell_dim - 1][torus(nu_coord, distance)])
        ne_map[q_ind] = check_inds
    ne_parity_mat = np.zeros(shape=(nq, n_stab), dtype=int)
    for q, s in ne_map.items():
        ne_parity_mat[q, s] = 1
    # print(ne_map)

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
            flips_this_round = (ne_parity_mat @ current_synd) == 2
            prediction += flips_this_round
            current_error = (current_error + flips_this_round) % 2
            current_synd = (h @ current_error) % 2
        final_error = (q_errors + prediction) % 2
        error = (final_error.T @ correlation_surface) % 2
        e_rate = sum(error)/n_trials

    else:
        ne_parity_mat_sparse = csr_matrix(ne_parity_mat)
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


def main():
    errors = np.linspace(0.01, 0.015, 5)
    for distance in [3, 4, 5]:
        out = []
        for e in errors:
            log_error = tooms_5d_primal_decode(error_rate=e, n_trials=10000, n_ca_iters=10, dense=False,
                                               distance=distance)
            out.append(log_error[0, 0])
            print(out)
        plt.plot(errors, out, 'o-')
    plt.show()

    # toom_2d_phase_only()


if __name__ == '__main__':
    main()
    run_tooms(size=50, p=0.01, q=0.01, length=100, show_every=5, initial_up=0.5)




