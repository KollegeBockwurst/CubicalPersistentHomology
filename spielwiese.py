from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
import filtration_functions
import numpy as np
import sage_utility

G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(3)
singular_cubes = sage_utility.find_singular_non_degenerate_cubes(G, 3)
face_maps = sage_utility.create_face_maps(singular_cubes)
with open('matrix_3.txt', 'w') as f:
    # Write the matrix to the file
    f.write(str(face_maps[3]))

with open('matrix_2.txt', 'w') as f:
    # Write the matrix to the file
    f.write(str(face_maps[2]))

with open('3_cubes.txt', 'w') as f:
    # Write the matrix to the file
    for cube in singular_cubes[3]:
        f.write(str(cube)+"\n")

with open('2_cubes.txt', 'w') as f:
    # Write the matrix to the file
    for cube in singular_cubes[2]:
        f.write(str(cube)+"\n")

number = []
for cube in singular_cubes[3]:
    number.append(len(set(cube)))

index_list = np.argsort(number)
M_reordered = face_maps[3].matrix_from_columns(index_list)

with open('matrix_3_ordered.txt', 'w') as f:
    # Write the matrix to the file
    f.write(str(M_reordered))

with open('3_cubes_ordered.txt', 'w') as f:
    # Write the matrix to the file
    for i in index_list:
        f.write(str(singular_cubes[3][i])+"\n")

print(face_maps[2].rank())
print(face_maps[3].rank())
print(face_maps[2].ncols()-face_maps[2].rank())
exit(7)
face_maps_F2 = dict()
for key in face_maps.keys():
    face_maps_F2[key] = face_maps[key].change_ring(FiniteField(2))
C = ChainComplex(face_maps_F2, degree_of_differential=-1)

chain = C.free_module(2).gen(15)
print(C.differential(2)*chain)
# preimage = C.differential(3).solve_right(chain)
preimage = face_maps_F2[3].solve_right(chain)
preimage_indices = np.where(np.array(preimage) == 1)[0]
for j in preimage_indices:
    print(f"{j}: {singular_cubes[3][j]}")
print(face_maps[2].column(15))
