import time

from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
import sage_utility

from persistence_calculator.SingularCubeGenerator import SingularCubeGeneratorScheduler

G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(7)
stamp = time.time()
cubes1 = sage_utility.find_singular_non_degenerate_cubes(G, 2)
print(time.time() - stamp)
stamp = time.time()
scheduler = SingularCubeGeneratorScheduler(G, 2, 16)
cubes2 = scheduler.run()
print(time.time() - stamp)
