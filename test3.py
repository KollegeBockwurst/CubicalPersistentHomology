import time

from sage.all import *
from sage.graphs.graph_generators import GraphGenerators

import filtration_functions
import sage_utility
from persistence_calculator.FaceMapGenerator import FaceMapGeneratorScheduler
from persistence_calculator.FiltrationCalculator import FiltrationCalculatorScheduler

from persistence_calculator.SingularCubeGenerator import SingularCubeGeneratorScheduler

G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(4)
max_dim = 2
filtration_function = filtration_functions.filtrate_by_diameter

stamp = time.time()
scheduler = SingularCubeGeneratorScheduler(G, max_dim + 1, 8)
cubes2 = scheduler.run()
print(time.time() - stamp)

stamp = time.time()
scheduler = FaceMapGeneratorScheduler(cubes2, 8)
face_maps2 = scheduler.run()
print("---")
stamp = time.time()
scheduler = FiltrationCalculatorScheduler(filtration_function, G, cubes2, 8)
filtration2 = scheduler.run()
print(time.time() - stamp)
stamp = time.time()
diagram = sage_utility.compute_persistence_diagram2(face_maps2, filtration2)
print(time.time() - stamp)
stamp = time.time()
G.plot().save(filename="temp.png")
max_steps = max_value = max(0 if val == float("inf") else val for sublist in filtration2 for val in sublist)
sage_utility.draw_diagram(diagram, f'{G.name()}'
                                   f'\n {filtration_function.__name__}, max_dim: {max_dim}', max_steps,
                          show_plot=True)
