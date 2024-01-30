import time

from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
import sage_utility
from persistence_calculator.FaceMapGenerator import FaceMapGeneratorScheduler

from persistence_calculator.SingularCubeGenerator import SingularCubeGeneratorScheduler

G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(7)
stamp = time.time()
cubes1 = sage_utility.find_singular_non_degenerate_cubes(G, 2)
print(time.time() - stamp)
stamp = time.time()
scheduler = SingularCubeGeneratorScheduler(G, 2, 4)
cubes2 = scheduler.run()
print(time.time() - stamp)
print("---")
stamp = time.time()
face_maps1 = sage_utility.create_face_maps(cubes1)
print(time.time() - stamp)
stamp = time.time()
scheduler = FaceMapGeneratorScheduler(cubes2, 4)
face_maps2 = scheduler.run()
print(time.time() - stamp)
