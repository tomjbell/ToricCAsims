import numpy as np
from itertools import product, combinations
from pymatching.matching import Matching
import matplotlib.pyplot as plt


def torus(coords, distance):
    # Wrap around the coordinate system, modding by the distance in each dim
    coords_out = []
    for c in coords:
        coords_out.append(c % (2 * distance))
    return tuple(coords_out)


def build_coords(distance: int, dimension=2) -> np.array:
    """
    Find the locations of cells in a d-dim square lattice - will be used to build the toric codes
    TODO create a class cell that keeps the information regarding the dimension in which the cell is embedded, it's
    TODO coordinates, it's index and any other information we need.
    :param distance:
    :param dimension:
    :return:
    """
    cells = [[] for _ in range(dimension + 1)]

    # find positions of 0-cells
    for coord in product(range(0, 2 * distance, 2), repeat=dimension):
        cells[0].append(coord)
        for cell_dim in range(1, dimension + 1):
            # To go from the vertex to the n-dim cell center, increase n of the coordinates by 1. dim choose n options
            for increment_directions in combinations(list(range(dimension)), cell_dim):
                cell_coords = tuple([coord[i] + int(i in increment_directions) for i in range(dimension)])
                cells[cell_dim].append(cell_coords)
    return cells


def adjacent_cells(dim_shift, cell_coords, dimension, distance):
    # Get the adjacent d+1 or d-1 cells for a d-dim cell at cell_coords
    neighbours = []
    embedded_in = [i for i in range(dimension) if cell_coords[i] % 2]
    orthogonal = [i for i in range(dimension) if i not in embedded_in]
    if dim_shift == -1:
        move_in = embedded_in
    elif dim_shift == 1:
        move_in = orthogonal
    else:
        raise ValueError
    for dim in move_in:
        for direction in (+1, -1):
            coord = cell_coords[:dim] + tuple([direction + cell_coords[dim]]) + cell_coords[dim+1:]
            neighbours.append(torus(tuple(coord), distance=distance))
    return neighbours


def cell_dicts_and_boundary_maps(distance, dimension):
    cells = build_coords(distance, dimension=dimension)

    # Index all cells
    cells2i = [{q: i for i, q in enumerate(sorted(cells[d]))} for d in range(dimension + 1)]

    # get all boundary maps
    b_maps = [{} for _ in range(dimension + 1)]
    co_b_maps = [{} for _ in range(dimension + 1)]
    for d in range(dimension + 1):
        if d > 0:
            for c in cells[d]:
                boundary = adjacent_cells(dim_shift=-1, cell_coords=c, dimension=dimension, distance=distance)
                b_maps[d][cells2i[d][c]] = [cells2i[d - 1][b] for b in boundary]
        if d < dimension + 1:
            for c in cells[d]:
                co_boundary = adjacent_cells(dim_shift=+1, cell_coords=c, dimension=dimension, distance=distance)
                co_b_maps[d][cells2i[d][c]] = [cells2i[d + 1][b] for b in co_boundary]
    return cells, cells2i, b_maps, co_b_maps


def non_triv_loops(starter_coord, dims_to_explore, distance):
    if type(starter_coord) is list:
        curr_cells = starter_coord
    else:
        curr_cells = [starter_coord]
    coords_set = set(curr_cells)
    for move_dir in dims_to_explore:
        first_move = True

        while first_move or (curr_cells[0] not in coords_set):
            if not first_move:
                for c in curr_cells:
                    coords_set.add(c)
            # increment current cells
            next_cells = []
            for c in curr_cells:
                next_coords = list(c)
                next_coords[move_dir] += 2
                next_cells.append(torus(next_coords, distance))
            curr_cells = next_cells
            first_move = False
        curr_cells = list(coords_set)
    return curr_cells


def logical_x_toric(cells, q_cell_dim, dimension, distance, q2i_dict):
    # Logical X. This is a collection of m-cells, where m=qubit_cell_dim, that have no m+1 dim boundary, and are not the
    # boundary of a m-1 cell.
    #TODO this currently only returns one of the logical operators, to get the others would need to pick other starting
    # qubits that are embedded in the other dimensions
    # Pick a m-cell
    first_cell = cells[q_cell_dim][0]
    # get the set of directions to move in
    orthogonal = [i for i in range(dimension) if not first_cell[i] % 2]

    x_logical_coords = non_triv_loops(first_cell, orthogonal, distance)
    x_logical_inds = [q2i_dict[q] for q in x_logical_coords]
    x_logical_bin = np.zeros((len(cells[q_cell_dim]), 1))
    for x_ind in x_logical_inds:
        x_logical_bin[x_ind, 0] = 1
    return x_logical_bin


def logical_z_toric(cells, q_cell_dim, dimension, distance, z2i_dict):
    # This function gets the loical Z operator of the toric code and then propogates it in a loop
    # to get the dual correlation surface on the m+1 cells

    # Logical Z - these are chains of m-cells with no m-1 boundary
    # pick a cell
    first_cell = cells[q_cell_dim][0]
    # get the set of directions to move in
    orthogonal = [i for i in range(dimension) if not first_cell[i] % 2]
    current_cells = [first_cell]
    # get the direction to move in
    embedded = [i for i in range(dimension) if current_cells[0][i] % 2]
    z_logical_coords = non_triv_loops(first_cell, embedded, distance)
    # Now propogate in an orthogonal direction until you get back to where you started
    propogate_dir = orthogonal[0]
    new_qubits = []
    for q in z_logical_coords:
        q2 = list(q)
        q2[propogate_dir] += 1
        new_qubits.append(torus(q2, distance))
    z_correlation_surface = non_triv_loops(new_qubits, [orthogonal[0]], distance)
    z_inds = [z2i_dict[q] for q in z_correlation_surface]
    z_logical_bin = np.zeros((len(cells[q_cell_dim + 1]), 1))
    for z_ind in z_inds:
        z_logical_bin[z_ind, 0] = 1
    return z_logical_bin


def toric_parity_check_matrix(distance, dimension=2):
    cells, cells2i, b_maps, co_b_maps = cell_dicts_and_boundary_maps(distance, dimension)

    cells = build_coords(distance, dimension=dimension)
    qubit_cell_dim = dimension // 2

    q2i = cells2i[qubit_cell_dim]
    z2i = cells2i[qubit_cell_dim + 1]
    x2i = cells2i[qubit_cell_dim - 1]
    cc2i = cells2i[qubit_cell_dim + 2]

    x_stab_dict = co_b_maps[qubit_cell_dim-1]
    z_stab_dict = b_maps[qubit_cell_dim + 1]
    clusterized_z_checks = b_maps[qubit_cell_dim + 2]

    # Find the logical operators of the code.
    x_logical_bin = logical_x_toric(cells, qubit_cell_dim, dimension, distance, q2i)
    z_logical_bin = logical_z_toric(cells, qubit_cell_dim, dimension, distance, z2i)

    # Get the full check matrix
    shape_primal = (len(cells[qubit_cell_dim-1]), len(cells[qubit_cell_dim]))
    shape_dual = (len(cells[qubit_cell_dim+2]), len(cells[qubit_cell_dim+1]))
    # H = np.zeros(shape)
    H_primal = np.zeros(shape_primal)
    H_dual = np.zeros(shape_dual)

    for cc, qubits in clusterized_z_checks.items():
        # H[cc, qubits] = 1
        H_dual[cc, qubits] = 1
    for x_checks, qubits in x_stab_dict.items():
        # H[x_checks, qubits] = 1
        H_primal[x_checks, qubits] = 1
    # print([sum(x) for x in H.T])
    return H_primal, H_dual, x_logical_bin, z_logical_bin


def sample_errors(hp, hd, x, z, noise=0.01, repetitions=1000):
    def get_error_mat(h, correlation_surface, e, reps):
        matching = Matching(h, e)
        nstab, nq = h.shape
        # Generate syndromes
        q_errors = np.random.choice([0, 1], size=(nq, reps), p=[1 - e, e])
        syndromes = (h @ q_errors) % 2
        predictions = matching.decode_batch(syndromes.T)

        # Add predictions to physical errors
        error_mat = (q_errors + predictions.T) % 2
        # Decide whether a logical error happened
        # If the support of a logical operator and the error intersect only once, there has been a logical error
        error = (error_mat.T @ correlation_surface) % 2
        return error
    primal_error_mat = get_error_mat(hp, x, noise, repetitions)
    dual_error_mat = get_error_mat(hd, z, noise, repetitions)

    overall_error = np.logical_or(primal_error_mat, dual_error_mat)
    error_rate = sum(overall_error)/repetitions
    return error_rate


def sweep_error_prob(distance, dimension=3):
    hp, hd, x, z = toric_parity_check_matrix(distance=distance, dimension=dimension)
    # print(h.T[0])
    error_rates = np.linspace(0.01, 0.04, 8)
    out = []
    for e in error_rates:
        out.append(sample_errors(hp, hd, x, z, e, repetitions=10000))
    plt.plot(error_rates, out, 'o-')


def plot_RHG_threshold():
    legend = []
    for distance in (3, 5, 7, 9, 11):
        legend.append(distance)
        print(f'{distance=}')
        sweep_error_prob(distance, dimension=3)
    plt.legend(legend)
    # plt.savefig('3d_toric_clusterized_threshold')
    plt.show()



def main():
    pass


if __name__ == '__main__':
    main()
