from multiprocessing import Pool
from sage.all import Matrix, ZZ
from sage.matrix.special import block_matrix


def generate_face_maps(singular_cubes, start, stop):
    result = [None] * stop[0]
    if stop[1] > 0:
        result.append(None)

    for cube_dim in range(start[0], stop[0] + 1):
        start_index = start[1] if cube_dim == start[0] else 0
        stop_index = stop[1] if cube_dim == stop[0] else len(singular_cubes[cube_dim])

        if cube_dim == 0:
            continue

        face_matrix = [None for _ in range(start_index, stop_index)]
        for j in range(start_index, stop_index):
            # we need to generate the associated row in the result matrix, which is a column of face_matrix
            matrix_column = [0] * len(singular_cubes[cube_dim - 1])  # initialize coöumn
            summand = -1  # summand fluctuating between -1 and 1, representing (-1)^k in the boundary map formula
            for k in range(cube_dim):  # loop through all dimensions of the chosen cube
                mask = 1 << k  # mask to differentiate between the positive and negative face in this dim
                face_0 = []  # negative face
                face_1 = []  # positive face
                for h in range(len(singular_cubes[cube_dim][j])):  # loop through all vertices of the cube
                    # using bit operations, we append the vertex either to the positive or negative face
                    if h & mask == 0:
                        face_0.append(singular_cubes[cube_dim][j][h])
                    else:
                        face_1.append(singular_cubes[cube_dim][j][h])

                for cube_index in range(len(singular_cubes[cube_dim - 1])):  # loop through all cube of dim cube_dim - 1
                    # when we find the positive (or negative) face, we can write it into the current matrix column
                    # see cubical boundary formula for more information
                    # it can happen, that a positive/negative face is NOT found. this happens, when it is degenerate
                    # in this case, we just ignore it, since it is 0 in our chain complex (see definition)
                    if face_0 == singular_cubes[cube_dim - 1][cube_index]:
                        matrix_column[cube_index] += summand
                    if face_1 == singular_cubes[cube_dim - 1][cube_index]:
                        matrix_column[cube_index] -= summand
                summand = -summand  # change summand sign
            face_matrix[j-start_index] = matrix_column  # add column to matrix
        result[cube_dim] = face_matrix

    return result


class FaceMapGeneratorScheduler:
    def __init__(self, singular_cubes, thread_number: int):
        self.singular_cubes = singular_cubes
        self.thread_number = thread_number

    def run(self):
        total_number = 0
        for dim in range(len(self.singular_cubes)):
            total_number += len(self.singular_cubes[dim])

        numbers_per_thread = (total_number // self.thread_number) + 1
        index = 0
        dim = 0
        args = []
        for i in range(self.thread_number):
            start_index = index
            start_dim = dim
            number = numbers_per_thread
            while index + number > len(self.singular_cubes[dim]):
                number += index - len(self.singular_cubes[dim])
                index = 0
                dim += 1
                if dim >= len(self.singular_cubes):
                    break

            stop_dim = dim if dim < len(self.singular_cubes) else len(self.singular_cubes) - 1
            stop_index = index + number if dim < len(self.singular_cubes) else len(self.singular_cubes[-1])
            index = stop_index
            args.append([self.singular_cubes, [start_dim, start_index], [stop_dim, stop_index]])
            if dim >= len(self.singular_cubes):
                break

        with Pool(self.thread_number) as p:
            p_results = p.starmap(generate_face_maps, args)

        face_maps = dict()  # result dictionary
        for p_result in p_results:
            for dim in range(len(p_result)):
                if p_result[dim] is not None:
                    if dim in face_maps.keys():
                        face_maps[dim].extend(p_result[dim])
                    else:
                        face_maps[dim] = p_result[dim]

        for dim in face_maps.keys():
            face_maps[dim] = Matrix(ZZ, face_maps[dim]).transpose()

        face_maps[0] = Matrix(0, len(self.singular_cubes[0]))
        return face_maps
