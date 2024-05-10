from sage.all import *
from persistence_calculator import filtration_functions
from persistence_calculator.algorithm import calc_persistence_diagram
from sage.graphs.graph_generators import GraphGenerators

post_octaeder = Graph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3),
                       (0, 4), (1, 4), (2, 4),
                       (0, 5), (2, 5), (3, 5),
                       (1, 6), (2, 6), (3, 6),
                       (0, 7), (1, 7), (3, 7),
                       (4, 5), (5, 6), (6, 4), (4, 7), (5, 7), (6, 7)])

post_octaeder.name("Post_Octaeder")
G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(3)
calc_persistence_diagram(G, 2, filtration_functions.filtrate_by_diameter, float("inf"), "")
#asdasd