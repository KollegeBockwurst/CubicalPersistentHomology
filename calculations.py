from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
from persistence_calculator import filtration_functions
from persistence_calculator.algorithm import calc_persistence_diagram

graph = sage.graphs.graph_generators.GraphGenerators.CubeGraph(3)
calc_persistence_diagram(graph, 2, filtration_functions.filtrate_by_diameter)
