import time

from sage.all import *
from sage.graphs.graph_generators import GraphGenerators

from persistence_calculator import filtration_functions
import sage_utility
from persistence_calculator.FaceMapGenerator import FaceMapGeneratorScheduler
from persistence_calculator.FiltrationCalculator import FiltrationCalculatorScheduler

from persistence_calculator.SingularCubeGenerator import SingularCubeGeneratorScheduler

G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(4)
stamp = time.time()
cubes1 = sage_utility.find_singular_non_degenerate_cubes(G, 2)
print(time.time() - stamp)
stamp = time.time()
scheduler = SingularCubeGeneratorScheduler(G, 2, 8)
cubes2 = scheduler.run()
print(time.time() - stamp)
if str(cubes1) != str(cubes2):
    print("Error1")
print("---")
stamp = time.time()
face_maps1 = sage_utility.create_face_maps(cubes1)
print(time.time() - stamp)
stamp = time.time()
scheduler = FaceMapGeneratorScheduler(cubes2, 8)
face_maps2 = scheduler.run()
print(time.time() - stamp)
if str(face_maps1[2]) != str(face_maps2[2]):
    print("Error2")
    print(face_maps1)
    print(face_maps2)
print("---")
stamp = time.time()
filtration1 = sage_utility.filtrate_cubes(cubes1, G, filtration_functions.filtrate_by_diameter)
print(time.time() - stamp)
stamp = time.time()
scheduler = FiltrationCalculatorScheduler(filtration_functions.filtrate_by_diameter, G, cubes2, 8)
filtration2 = scheduler.run()
print(time.time() - stamp)
if str(filtration1) != str(filtration2):
    print("Error3")
