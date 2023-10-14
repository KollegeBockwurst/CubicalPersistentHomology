from sage.all import *
import graph
import graph_utility
import sage_tools
import numpy as np

import sage_utility
import time


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


G = Graph([(0, 1), (1, 2),(2,3),(3,4),(4,0)])
max_dim = 3

print("Calculating singular, non-degenerated cubes...")
start_time = time.time()
singular_cubes = sage_utility.find_singular_non_degenerate_cubes(G, max_dim)
print(f'{time.time() - start_time} s')

print("Calculating face maps...")
start_time = time.time()
face_maps = sage_utility.create_face_maps(singular_cubes)
print(f'{time.time() - start_time} s')

print("Filtrating face maps...")
start_time = time.time()
filtration = sage_utility.filtrate_cubes(singular_cubes, G, filtrate_by_diameter)
print(f'{time.time() - start_time} s')

print("Calculating persistence diagram...")
start_time = time.time()
diagram = sage_utility.compute_persistence_diagram2(face_maps, filtration)
print(f'{time.time() - start_time} s')

print("Drawing persistence diagram...")
start_time = time.time()
max_steps = max_value = max(val for sublist in filtration for val in sublist)
sage_utility.draw_diagram(diagram, "test", max_steps)
print(f'{time.time() - start_time} s')

print("Compute betti numbers using SageMath")
start_time = time.time()
chain_complexes = []
betti_numbers = []

for i in range(max_steps):
    filtered_face_maps = dict()
    for k in range(np.max(list(face_maps.keys()))+1):
        col_indices = [index for index, value in enumerate(filtration[k]) if value <= i]
        row_indices = []
        if k > 0:
            row_indices = [index for index, value in enumerate(filtration[k - 1]) if value <= i]
        filtered_face_maps[k] = \
            Matrix([[face_maps[k][i, j] for j in col_indices] for i in row_indices]).change_ring(FiniteField(2))
    chain_complexes.append(ChainComplex(filtered_face_maps, degree_of_differential=-1))
    betti_numbers.append(chain_complexes[i].betti())
print(f'{time.time() - start_time} s')

print("Consistency check")
start_time = time.time()
for i in range(len(betti_numbers)):
    for j in betti_numbers[i].keys():
        if j >= len(diagram):
            continue
        counter = 0
        for tupel in diagram[j]:
            if tupel[0] <= i < tupel[1]:
                counter += 1
        if counter != betti_numbers[i][j]:
            print(f'Consistency check failed in persistence step {i}')
            break
print(f'{time.time() - start_time} s')
exit(0)

my_graph = graph.Graph.from_edge_list(['A', 'B', 'C', 'D', 'E', 'F'],
                                      [['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E'], ['E', 'F'],
                                       ['F', 'A']])

chain_complex = sage_tools.create_chain_complex(my_graph, 3)
print(chain_complex.betti())
exit(0)
print(graph_utility.contraction_length(my_graph, 5))

face_map_xy = Matrix(ZZ, [[1, 2], [1, 2]])
face_map_yz = Matrix(ZZ, [[]])

C = ChainComplex([face_map_xy])
print(C.betti())
