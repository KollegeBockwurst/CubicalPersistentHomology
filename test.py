from sage.all import *
import filtration_functions
import sage_utility

G = Graph([(0, 1),(1,2),(2,3),(3,4),(4,5),(5,0)])
max_dim = 3

sage_utility.persistence(G, filtration_functions.filtrate_by_homotopy, 3, use_distances=True, title="test1")
