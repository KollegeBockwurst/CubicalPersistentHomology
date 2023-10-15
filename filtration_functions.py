import numpy as np
from itertools import product


def filtrate_by_homotopy(distance_matrix, **kwargs):
    """
    Computes the minimum contraction length of this graph.
    """

    num_vertices = distance_matrix.shape[0]

    # find vertex with the smallest maximum distance to other vertices:
    row_max_values = distance_matrix.max(axis=1, initial=None)
    pointed_vertex_ids = np.where(row_max_values == row_max_values.min())[0]

    if "max_contraction_length" in kwargs.keys():
        max_contraction_length = kwargs["max_contraction_length"]
    else:
        max_contraction_length = int(max(val if val < float("inf") else 0 for sublist in distance_matrix for val in sublist))

    mapping_steps = [list(range(num_vertices))]

    # trivial case:
    if num_vertices < 2:
        return 0

    for pointed_vertex_id in pointed_vertex_ids:
        for step in range(max_contraction_length):
            promoted_options = []
            last_mapping_step = mapping_steps[-1]
            for vertex in range(num_vertices):
                options = np.append([last_mapping_step[vertex]],
                                    np.where(distance_matrix[last_mapping_step[vertex]] == 1)[0])
                option_distances = distance_matrix[pointed_vertex_id, options]
                promoted_options.append(options[option_distances == option_distances.min()])

            for combined_option in product(*promoted_options):
                # check if combined_option is a graph map:
                for k in range(num_vertices):
                    not_hom_flag = True
                    for l in range(k):
                        if distance_matrix[k][l] == 1:
                            if distance_matrix[combined_option[k]][combined_option[l]] > 1:
                                break
                    else:
                        not_hom_flag = False
                    if not_hom_flag:
                        break
                else:
                    mapping_steps.append(combined_option)
                    break
            else:
                break  # no possible option found

            if all(item == pointed_vertex_id for item in mapping_steps[-1]):
                return step + 1

    # not contractible in max_contraction_length steps:
    return float("inf")


def filtrate_by_diameter(adjacency_matrix):
    # Initialize the distance matrix with the input graph adjacency matrix
    distance_matrix = np.full(adjacency_matrix.shape, np.inf)
    distance_matrix[adjacency_matrix == 1] = 1
    np.fill_diagonal(distance_matrix, 0)

    # Number of vertices in the graph
    num_vertices = distance_matrix.shape[0]

    # Main loop: try all possible intermediate vertices
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                # Update the distance matrix by considering the path through vertex k
                if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return int(np.max(distance_matrix))


def filtrate_by_number_of_vertices(adjacency_matrix):
    return adjacency_matrix.shape[0]
