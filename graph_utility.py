from itertools import product
from graph import Graph
from map import Map
import numpy as np


def contraction_length(g: Graph, max_contraction_length: int, **kwargs):
    """
    Computes the minimum contraction length of this graph.
    Supports extended output by using "extended_output = True"

    :param g: Graph
    :param max_contraction_length: int
        maximal considerable contraction length (to reduce workload). Algorithm will terminate if the graph is not
        contractible in max_contraction_length steps

    :return: int
        Contraction length of graph, -1 if graph is not contractible in max_contractible_length.
        If extended_output: Tupel of contraction_length and a list of maps representing the contraction steps
    """

    num_vertices = g.adjacency_matrix.shape[0]
    distance_matrix = g.get_distance_matrix()

    # find vertex with the smallest maximum distance to other vertices:
    row_max_values = distance_matrix.max(axis=1, initial=None)
    pointed_vertex_ids = np.where(row_max_values == row_max_values.min())[0]

    mapping_steps = [Map(g, g, list(range(num_vertices)))]

    # trivial case:
    if num_vertices < 2:
        if "extended_output" in kwargs.keys() and kwargs["extended_output"]:
            return 0, mapping_steps
        else:
            return 0

    for pointed_vertex_id in pointed_vertex_ids:
        for step in range(max_contraction_length):
            promoted_options = []
            last_mapping_step = mapping_steps[-1].mapping
            for vertex in range(num_vertices):
                options = np.append([last_mapping_step[vertex]],
                                    np.where(g.adjacency_matrix[last_mapping_step[vertex]])[0])
                option_distances = distance_matrix[pointed_vertex_id, options]
                promoted_options.append(options[option_distances == option_distances.min()])

            for combined_option in product(*promoted_options):
                new_mapping_step = Map(g, g, combined_option)
                if new_mapping_step.is_homomorphism():
                    mapping_steps.append(new_mapping_step)
                    break

            if step + 2 != len(mapping_steps):  # not contractible to this vertex
                break

            if all(item == pointed_vertex_id for item in mapping_steps[-1].mapping):
                if "extended_output" in kwargs.keys() and kwargs["extended_output"]:
                    return step + 1, mapping_steps
                else:
                    return step + 1

    # not contractible in max_contraction_length steps:
    if "extended_output" in kwargs.keys() and kwargs["extended_output"]:
        return -1, None
    else:
        return -1
