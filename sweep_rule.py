import numpy as np
from toric import cell_dicts_and_boundary_maps, torus, logical_x_toric
from itertools import chain, combinations, product


def sweep_rule(cells, sweep_direction, error_boundary):
    # Apply this rule simultaneously to every vertex in the lattice

    # 1 find the edges in the future of the vertex
    # 1b Find the boundary of the error restricted to the future edges, if none, pass

    # 2 Find the faces in the future of the vertex

    # 3 For each subset of faces in the future of the vertex:
    # a. Find the boundary of the faces restricted to the future edges
    # if the boundary matches that found in 1b, mark the set of faces, exit

    pass


def get_incident_vertices(edge_boundary, edge_vertex_map=None):
    incident_vertices = set()
    for e in edge_boundary:
        new_v = set(edge_vertex_map[e])
        incident_vertices.update(new_v)
    return incident_vertices


def get_local_boundary(v, boundary):
    pass


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))



def implement_sweep(current_boundary, cells_to_ix, boundary_map2):
    vertices_to_inspect = get_incident_vertices(current_boundary)
    faces_to_flip = []

    for vertex in vertices_to_inspect:
        local_boundary = get_local_boundary(vertex, current_boundary)

        faces = future_faces(vertex)
        for face_set in powerset(faces):
            boundary = boundary_map2(face_set)
            if boundary == local_boundary:
                faces_to_flip.append([cells_to_ix[2][f] for f in faces])
                break
    return faces_to_flip


def future_cells_toric(dim_shift, cell_coords, dimension, distance, future_dir):
    """
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


def future_faces(vertex):
    pass


def main(dimension, distance, future_dir, qubit_cell_dim=2, n_shots=1, error_rate=0.01, loss_rate=0., n_sweeps=5, printing=False):
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=distance, dimension=dimension)

    future_faces_map = {}
    future_edges_map = {}
    for v in cells[0]:
        future_faces_map[cells2i[0][v]] = [cells2i[2][face] for face in future_cells_toric(+2, v, dimension, distance, future_dir)]
        future_edges_map[cells2i[0][v]] = [cells2i[1][edge] for edge in future_cells_toric(+1, v, dimension, distance, future_dir)]

    # Generate some errors
    nq = len(cells[qubit_cell_dim])
    n_stab = len(cells[qubit_cell_dim - 1])

    h = np.zeros(shape=(n_stab, nq), dtype=np.uint8)
    for stab, qubits in cob_maps[qubit_cell_dim - 1].items():
        h[stab, qubits] = 1
    stabs_per_qubit = len(b_maps[qubit_cell_dim][0])
    qbt_syndr_mat = np.where(h.T)[1].reshape((nq, stabs_per_qubit)).astype(dtype=np.int32)  # is this the same as the boundary maps?
    correlation_surface = logical_x_toric(cells, qubit_cell_dim, dimension, distance, cells2i[qubit_cell_dim]) # Gets binary vector corresponding to the logical operator

    errors_full = np.random.choice([0, 1, 2], size=(nq, n_shots), p=[(1 - error_rate) * (1 - loss_rate), (1 - loss_rate) * error_rate, loss_rate])
    es = errors_full == 1
    ls = errors_full == 2
    errors = np.array(es, dtype=int)
    losses = np.array(ls, dtype=int)
    lost_qubit_ixs = [np.where(losses[:, ix])[0].astype(int) for ix in range(n_shots)]
    error_qubit_ixs = [np.where(errors[:, ix])[0].astype(int) for ix in range(n_shots)]
    print(error_qubit_ixs[0])
    for jx in range(n_sweeps):
        error_syndrome = h @ errors
        non_triv_synd_ixs = [np.where(error_syndrome[:, ix])[0].astype(int) for ix in range(n_shots)]
        for ix in range(n_shots):

            tot_flips = []
            nt_edges = non_triv_synd_ixs[ix]
            nt_vertices = get_incident_vertices(nt_edges, b_maps[1])
            if printing:
                print(f'{nt_edges=}')
                print(f'{nt_vertices=}')
            for v in nt_vertices:
                ff = future_faces_map[v]
                fe = set(future_edges_map[v])
                if printing:
                    print(f'Future face indices: {ff}')
                    print(f'Future edge indices: {fe}')
                for face_set in powerset(ff):
                    # get boundary
                    tot_boundary = set()
                    for f in face_set:
                        boundary = b_maps[qubit_cell_dim][f]
                        tot_boundary = tot_boundary.symmetric_difference(boundary)
                    local_boundary = tot_boundary.intersection(fe)
                    error_restricted = set(nt_edges).intersection(fe)
                    if error_restricted == local_boundary:
                        tot_flips += face_set
                        break
            print(f'{tot_flips=}')
            errors[tot_flips, :] += 1
            errors %= 2

        print(f'{sum(errors)=}')






if __name__ == '__main__':
    main(dimension=2, distance=10, future_dir=(1, 1), error_rate=0.3, n_sweeps=10)
