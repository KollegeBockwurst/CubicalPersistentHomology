from multiprocessing import Pool
import numpy as np
from sage.all import Graph, identity_matrix


def generate_singular_cubes(adjacency_graph, max_dim: int, filtrate_function, max_filtration: int, start_option, end_option):
    """
    Takes a sage graph and computes all singular, non-degenerate cubes up to dimension max_dim
    :param start_option:
    :param max_filtration:
    :param filtrate_function:
    :param adjacency_graph: An undirected sage graph
    :param max_dim: Maximum dimension to take into account.
    :return: A list of length max_dim + 1 containing singular cubes, where the i-th entry contains a list aof all
    non-degenerate singular cubes of dimension i. A singular cube of dimension i is represented as list of length 2^i,
    containing the image indices of a i-dimensional cube graph, sorted in lexiographical order
    """
    singular_cubes = [[] for _ in range(max_dim + 1)]  # create list of independent lists
    filtration_values = [[] for _ in range(max_dim + 1)]

    order_graph = adjacency_graph.nrows()
    order_max_cube = pow(2, max_dim)  # number of vertices in a cube graph of dimension max_dim

    # options[i] saves the unhandled, but possible (as graph map) options for mapping the i-th cube vertex to
    # my_graph "possible" always relates to the mapping saved in cube_mapping[0:i]

    options = [[] for _ in range(order_max_cube)]  # create list of independent lists
    options[0] = list(range(order_graph))  # initialize options[0], since this is always possible
    cube_mapping = [None] * order_max_cube  # saves the current state of the mapping

    np_adjacency = np.matrix(adjacency_graph)

    loop_flag = True  # will be set to false once it is clear that there are no singular cubes left
    while loop_flag:  # main loop

        # on this point there is at least one more option for at least one vertex saved in options
        # goal in this iteration is to pick one option, apply it to cube_mapping, then generating the possible
        # options for the next vertex. If the applied vertex is the last vertex of a singular cube, we will check
        # this singular cube for degeneracy, then add it to the list of found singular cubes

        # we generate the options "left to right", starting with vertex 0 (see initialization of options)
        # therefore, we will look for the vertex with existing options the furthest on the right

        filtration_flag = False
        # saves if there was a singular cube found with too hight filtration value to prevent generating of new options

        for change_index in range(order_max_cube - 1, -1, -1):  # loop cube vertices from right to left
            if len(options[change_index]) > 0:  # check for existing options
                cube_mapping[change_index] = options[change_index].pop(0)  # apply the option to cube_mapping

                # check if cube_mapping is in own range
                own_range = start_option[0:change_index + 1] <= cube_mapping[0:change_index + 1] < end_option
                if not own_range:
                    if cube_mapping[0:change_index + 1] >= end_option:
                        loop_flag = False
                    break
                # ----------
                # check if change_index+1 == 2^x, i.e. change_index is last vertex of a cube (of dim x):
                if (change_index + 1) & change_index == 0 and start_option <= cube_mapping[0:change_index + 1]:

                    # to find the cube_dim x, we compute log_2(change_index+1) by using bit operations:
                    cube_dim = 0
                    i = 1
                    while (i & (change_index + 1)) == 0:
                        i = i << 1
                        cube_dim += 1

                    # now we have found a singular cube of dimension cube_dim
                    singular_cube = cube_mapping[0:change_index + 1]

                    # check the singular cube for degeneracy:
                    degeneracy_flag = False  # will be set to True if a degeneracy is found
                    for k in range(cube_dim):  # loop through all dimensions of the cube and have a look at the
                        # faces
                        mask = (1 << k) - 1  # bit mask, used to split integers
                        degeneracy_flag = True  # set flag to True, just for the moment
                        for h in range(pow(2, cube_dim - 1)):  # loop through all vertices of one **face** of the
                            # cube vertex1/vertex2 are the h-th vertices of the positive (1) resp. negative (0)
                            # face in dim k
                            vertex1 = ((h & ~mask) << 1) | 1 << k | (h & mask)  # create vertex1 by bit manipulation
                            vertex2 = ((h & ~mask) << 1) | (h & mask)  # create vertex2 by bit manipulation
                            # now check if the positive and negative face are different in at least one vertex
                            # then set degeneracy_flag to false again. Otherwise it will stay True and indicate a
                            # degen.
                            if singular_cube[vertex1] != singular_cube[vertex2]:
                                degeneracy_flag = False
                                break
                        # check if the cube was degenerated in dimension k:
                        if degeneracy_flag:
                            break
                    # check if the cube was degenerated in any dimension. If not, add it to the output list:
                    if not degeneracy_flag:
                        # compute the cube's filtration
                        unique_image = np.unique(singular_cube)
                        subgraph_adjacency = np_adjacency[np.ix_(unique_image, unique_image)]
                        filtration_value = filtrate_function(subgraph_adjacency)
                        filtration_flag = filtration_value >= max_filtration
                        # check if filtration value is less than infinity
                        if not filtration_flag:
                            filtration_values[cube_dim].append(filtration_value)
                            singular_cubes[cube_dim].append(singular_cube)

                # ----------
                # generate the options for the next vertex (change_index + 1), if there is a next vertex:
                if change_index < order_max_cube - 1 and not filtration_flag:
                    next_options = []  # saves the found options for vertex change_index+1
                    for possible_option in range(order_graph):  # loop through all possible options for the mapping
                        # we now need to check if we still have a graph map if vertex change_index+1 gets mapped to
                        # the vertex possible_option of the given graph, i.e. if the mapping
                        # cube_mapping[0:change_index+1].append(possible_option) is a graph mapping
                        # Since we know the structure of cubes very well, we only need to check very few edges
                        # to other vertices. E.g. a vertex in a cube of dimension i has a maximum of i adjacents
                        # we need to consider.
                        possible_flag = True  # will be set to False if the option is not possible
                        for j in range(max_dim):  # loop through all adjacent vertices of (cube) vertex
                            # change_index+1
                            # we can simply generate the needed vector by flipping one bit of change_index+1
                            # see lexicographic ordering for more information
                            vertex3 = (change_index + 1) ^ (1 << j)  # flips jth bit
                            if vertex3 > change_index + 1:
                                # the found vertex is further to the right, so we do not consider it
                                continue

                            # Finally we can check if the edge in the cube remains unchanged in the image
                            if not adjacency_graph[possible_option][cube_mapping[vertex3]]:
                                possible_flag = False
                                break

                        if possible_flag:  # add option if we still have a graph map
                            next_options.append(possible_option)
                    options[change_index + 1] = next_options  # add all possible options to the outer options list
                # break the loop moving our main pointer (change_index) from right to left,
                # since we modified the option list or need to re-visit the actual change_index:
                break

            # if there aren't any options left, we must have found them all:
            if change_index == 0:
                loop_flag = False  # no more singular cubes possible, leave main loop

    return singular_cubes, filtration_values


class SingularCubeGeneratorScheduler:
    def __init__(self, graph: Graph, max_dim: int, filtration_function, max_filtration: int, thread_number: int):
        self.max_dim = max_dim
        self.graph = graph
        self.thread_number = thread_number
        self.filtration_function = filtration_function
        self.max_filtration = max_filtration + 1
        # manual adjustment in cae of inf values, so that later comparisons still work

    def run(self):
        adjacency_graph = self.graph.adjacency_matrix() + identity_matrix(self.graph.order())  # adjacency matrix of
        # the given graph
        singular_cubes = [[] for _ in range(self.max_dim + 1)]
        filtration_values = [[] for _ in range(self.max_dim + 1)]
        start_options = []

        def decimal_to_base_fixed_size(number, base, size):
            digits = []
            while number:
                digits.append(int(number % base))
                number //= base
            while len(digits) < size:
                digits.append(0)
            return digits[::-1]

        max_vertex_number = pow(2, self.max_dim)
        number_of_options = pow(self.graph.order(), max_vertex_number)
        numbers_per_thread = (number_of_options // self.thread_number) + 1
        last_stop = [0]*max_vertex_number
        for i in range(0, number_of_options, numbers_per_thread):
            start_index = last_stop
            last_stop = [float("inf")] if i+numbers_per_thread >= number_of_options else \
                decimal_to_base_fixed_size(i+numbers_per_thread, self.graph.order(), max_vertex_number)
            stop_index = last_stop
            start_options.append((start_index, stop_index))

        args = [(adjacency_graph, self.max_dim, self.filtration_function, self.max_filtration) + start_option for
                start_option in start_options]

        with Pool(len(start_options)) as p:
            p_results = p.starmap(generate_singular_cubes, args)

        for p_result in p_results:
            p_cubes, p_filtrations = p_result
            for i in range(self.max_dim + 1):
                singular_cubes[i].extend(p_cubes[i])
                filtration_values[i].extend(p_filtrations[i])

        return singular_cubes, filtration_values
