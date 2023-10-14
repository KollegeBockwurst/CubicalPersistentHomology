from sage.all import *

import graph
from singular_cube import SingularCube


def create_chain_complex(g: graph.Graph, max_dim: int):
    # Find all singular cubes:
    singular_cubes = []
    n = len(g.vertex_labels)

    for d in range(max_dim + 1):
        singular_cubes.append([])
        for singular_cube in g.get_sndc(d):
            try:
                singular_cubes[-1].append(SingularCube(d,g,singular_cube))
            except ValueError:
                pass

    data = dict()

    for d in range(0, max_dim+1):
        boundary_matrix = []
        if len(singular_cubes[d]) == 0:
            break
        for singular_cube in singular_cubes[d]:
            if d == 0:
                boundary_matrix.append([0])
                continue
            column = [0] * len(singular_cubes[d - 1])
            face_plus, face_minus = singular_cube.face_map()
            for i in range(len(singular_cubes[d - 1])):
                for face in face_plus:
                    if singular_cubes[d - 1][i] == face:
                        column[i] += 1
                        okay = true

                for face in face_minus:
                    if singular_cubes[d - 1][i] == face:
                        column[i] -= 1
                        okay = true

            boundary_matrix.append(column)
        data[d] = Matrix(ZZ, boundary_matrix).transpose()
    return ChainComplex(data, degree_of_differential=-1)
