import numpy as np
from toric import cell_dicts_and_boundary_maps, torus
from itertools import chain, combinations


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
        incident_vertices += new_v
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
    #TODO complete this function
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


def future_faces(vertex):
    pass


def main():
    cells, cells2i, b_maps, cob_maps = cell_dicts_and_boundary_maps(distance=3, dimension=2)
    sweep_direction = (1, 1)
    print(b_maps[1])




if __name__ == '__main__':
    main()
