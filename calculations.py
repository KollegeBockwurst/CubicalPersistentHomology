import os
import shutil
import time

import requests
from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
from persistence_calculator import filtration_functions
from persistence_calculator.algorithm import compute_multiple_persistence
import inspect


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def inform(message):
    requests.post(f"https://wirepusher.com/send?id=FSAympGDs&title=PythonMessage&message={message}&type=Python")


def analyze_list(graph_list, max_dim, relative_path):
    clear_dir(relative_path)
    compute_multiple_persistence(graph_list, max_dim, filtration_functions.filtrate_by_diameter, relative_path)
    compute_multiple_persistence(graph_list, max_dim, filtration_functions.filtrate_by_number_of_vertices,
                                 relative_path)
    compute_multiple_persistence(graph_list, max_dim, filtration_functions.filtrate_by_number_of_edges, relative_path)
    shutil.make_archive(f"results/cube_graphs", 'zip', "results/cube_graphs")
    shutil.rmtree("results/cube_graphs")


inform("StartedProgram")

cube_graph_list = []
for i in range(1, 6):
    cube_graph_list.append(GraphGenerators.CubeGraph(i))
analyze_list(cube_graph_list, 2, "results/cube_graphs/")
inform("FinishedCubes")

platonic_graph_list = [GraphGenerators.TetrahedralGraph(), GraphGenerators.HexahedralGraph(),
                       GraphGenerators.DodecahedralGraph(), GraphGenerators.OctahedralGraph(),
                       GraphGenerators.IcosahedralGraph()]
analyze_list(platonic_graph_list, 2, "results/platonic_graphs/")
inform("FinishedPlatonic")

small_graph_list = []
for name, obj in inspect.getmembers(GraphGenerators.smallgraphs):
    if inspect.isfunction(obj):
        try:
            graph = obj()
            if not isinstance(graph, Graph):
                continue
            if graph.order() > 15:
                continue
            small_graph_list.append(graph)  # Call the function
            print(f"Function {name} executed successfully.")
        except Exception as e:
            print(f"Error executing function {name}: {e}")
analyze_list(small_graph_list, 2, "results/small_graphs/")
inform("FinishedSmallGraphs")

shutil.make_archive(f"results_{time.time()}", 'zip', "results/")
inform("FinishedProgram")
