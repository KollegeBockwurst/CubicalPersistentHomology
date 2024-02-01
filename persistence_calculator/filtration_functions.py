import numpy as np
from itertools import product


def floyd_warshall(adjacency_matrix):
    """
        Compute the shortest path distances between all pairs of vertices in a graph using the Floyd-Warshall algorithm.

        :return: np.ndarray
            A 2D NumPy array where the element at (i, j) is the shortest path distance from vertex i to vertex j.
            If there is no path between vertex i and vertex j, the value will be np.inf.
        """
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

    return distance_matrix


def filtrate_by_homotopy(adjacency_matrix, **kwargs):
    """
    Computes the minimum contraction length of this graph.
    """
    distance_matrix = floyd_warshall(adjacency_matrix)
    num_vertices = distance_matrix.shape[0]

    # find vertex with the smallest maximum distance to other vertices:
    row_max_values = distance_matrix.max(axis=1, initial=None)
    pointed_vertex_ids = np.where(row_max_values == row_max_values.min())[0]

    if "max_contraction_length" in kwargs.keys():
        max_contraction_length = kwargs["max_contraction_length"]
    else:
        max_contraction_length = int(
            max(val if val < float("inf") else 0 for sublist in distance_matrix for val in sublist))

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
                print("needed to move to another vertex")
                break  # no possible option found

            if all(item == pointed_vertex_id for item in mapping_steps[-1]):
                if step + 1 != row_max_values.min():
                    print(f"{step + 1}--{row_max_values.min()}")
                    print(distance_matrix)
                middle_point_filtration = filtrate_by_middle_point_distance(adjacency_matrix)
                if step + 1 != middle_point_filtration:
                    print(f"homotopy:{step+1},  middle_point:{middle_point_filtration}")
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


def filtrate_by_middle_point_distance(adjacency_matrix):
    distance_matrix = floyd_warshall(adjacency_matrix)
    return np.min(np.max(distance_matrix, axis=0))


def filtrate_by_number_of_edges(adjacency_matrix):
    return int(np.sum(adjacency_matrix == 1) / 2)


def filtrate_by_max_branches(adjacency_matrix):
    return int(np.max(np.sum(adjacency_matrix == 1, axis=1)))
