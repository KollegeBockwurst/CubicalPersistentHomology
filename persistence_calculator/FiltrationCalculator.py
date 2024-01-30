from multiprocessing import Pool
import numpy as np


def calculate_filtration_maps(filtrate_function, graph_adjacency, singular_cubes, start, stop):
    filtration_values = [None] * stop[0]
    inf_counter = 0
    if stop[1] > 0:
        filtration_values.append(None)

    for cube_dim in range(start[0], stop[0] + 1):
        start_index = start[1] if cube_dim == start[0] else 0
        stop_index = stop[1] if cube_dim == stop[0] else len(singular_cubes[cube_dim])
        filtration_values[cube_dim] = []
        for j in range(start_index, stop_index):
            singular_cube = singular_cubes[cube_dim][j]
            unique_image = np.unique(singular_cube)
            subgraph_adjacency = graph_adjacency[np.ix_(unique_image, unique_image)]
            filtration_values[cube_dim].append(filtrate_function(subgraph_adjacency))
            if filtration_values[cube_dim][-1] == float("inf"):
                inf_counter += 1

    return filtration_values


class FiltrationCalculatorScheduler:
    def __init__(self, filtration_function, graph, singular_cubes, thread_number: int):
        self.graph = graph
        self.filtration_function = filtration_function
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
        graph_adjacency = np.matrix(self.graph.adjacency_matrix())
        for i in range(self.thread_number):
            start_index = index
            start_dim = dim
            number = numbers_per_thread
            while index + number >= len(self.singular_cubes[dim]):
                number += index - len(self.singular_cubes[dim])
                index = 0
                dim += 1
                if dim >= len(self.singular_cubes):
                    break

            stop_dim = dim if dim < len(self.singular_cubes) else len(self.singular_cubes) - 1
            stop_index = index + number if dim < len(self.singular_cubes) else len(self.singular_cubes[-1])
            index = stop_index
            args.append([self.filtration_function, graph_adjacency, self.singular_cubes, [start_dim, start_index],
                         [stop_dim, stop_index]])
            if dim >= len(self.singular_cubes):
                break

        with Pool(self.thread_number) as p:
            p_results = p.starmap(calculate_filtration_maps, args)

        filtration_values = []
        for p_result in p_results:
            for dim in range(len(p_result)):
                if p_result[dim] is not None:
                    while dim >= len(filtration_values):
                        filtration_values.append([])
                    filtration_values[dim].extend(p_result[dim])

        return filtration_values
