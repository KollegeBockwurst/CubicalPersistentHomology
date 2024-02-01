from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
from persistence_calculator import filtration_functions
import sage_utility

G = GraphGenerators.smallgraphs.BrouwerHaemersGraph()
print(G.order())
exit(2)

# G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(3)
# G = Graph([(0,1),(1,2),(2,3),(3,4), (4,5), (5,6), (6,7), (7,0),(0,4),(2,6), (1,5)])
# G = Graph([(0,1),(1,3),(0,2),(2,3),(0,4),(1,5),(2,6),(4,5),(4,6)])
# G = Graph([(0,1),(1,2),(2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (8,9), (9,10),(10,11), (11, 8), (0,4),(4,8), (1,5),(5,9), (2,6), (6,10), (3,7) , (7,11)])
# G = sage.graphs.graph_generators.GraphGenerators.CubeGraph(6)

# G = Graph([(0,1),(1,2),(2,3),(3,0)])
# G = G.cartesian_product(sage.graphs.graph_generators.GraphGenerators.CubeGraph(1))

G = Graph([(0,1),(0,2),(0,3),(0,4),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5),(4,5)])

sage_utility.persistence(G, filtration_functions.filtrate_by_diameter, 3, time_measurement=True)
sage_utility.persistence(G, filtration_functions.filtrate_by_number_of_vertices, 3, time_measurement=True)
sage_utility.persistence(G, filtration_functions.filtrate_by_number_of_edges, 3, time_measurement=True)
exit(8)
counter = 0
while True:
    # Parameters for the random graph
    n = 7  # Number of vertices
    p = 0.2  # Probability for each pair of vertices to be connected

    # Create the random graph
    G = GraphGenerators.RandomGNP(n, p)

    # Plot the graph
    sage_utility.persistence(G, filtration_functions.filtrate_by_max_branches, 3, time_measurement=True,
                             title=f"id:{counter}")
    counter += 1
