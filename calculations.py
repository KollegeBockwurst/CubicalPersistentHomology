import os
import shutil
import time

import requests
from sage.all import *
from sage.graphs.graph_generators import GraphGenerators
from persistence_calculator import filtration_functions
from persistence_calculator.algorithm import calc_persistence_diagram
import inspect
import threading

small_graph_list = []


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def inform(message):
    requests.post(f"https://wirepusher.com/send?id=FSAympGDs&title=PythonMessage&message={message}&type=Python")


inform("StartedProgram")
if not os.path.exists("results/"):
    os.makedirs("results/")

cube_graph_list = []
for i in range(1, 6):
    cube_graph_list.append(GraphGenerators.CubeGraph(i))
if not os.path.exists("results/cube_graphs/"):
    os.makedirs("results/cube_graphs/")
for graph in cube_graph_list:
    thread = threading.Thread(
        target=calc_persistence_diagram,
        args=(graph, 2, filtration_functions.filtrate_by_diameter, "results/cube_graphs/"))
    thread.start()
    thread.join()
    thread = threading.Thread(
        target=calc_persistence_diagram, args=(graph, 2, filtration_functions.filtrate_by_number_of_vertices,
                                               "results/cube_graphs/"))
    thread.start()
    thread.join()
shutil.make_archive(f"results/cube_graphs", 'zip', "results/cube_graphs")
shutil.rmtree("results/cube_graphs")
inform("FinishedCubes")

platonic_graph_list = [GraphGenerators.TetrahedralGraph(), GraphGenerators.HexahedralGraph(),
                       GraphGenerators.DodecahedralGraph(), GraphGenerators.OctahedralGraph(),
                       GraphGenerators.IcosahedralGraph()]
clear_dir("results/platonic_graphs/")
for graph in platonic_graph_list:
    thread = threading.Thread(
        target=calc_persistence_diagram,
        args=(graph, 2, filtration_functions.filtrate_by_diameter, "results/platonic_graphs/"))
    thread.start()
    thread.join()
    thread = threading.Thread(
        target=calc_persistence_diagram,
        args=(graph, 2, filtration_functions.filtrate_by_number_of_vertices, "results/platonic_graphs/"))
    thread.start()
    thread.join()
shutil.make_archive(f"results/platonic_graphs", 'zip', "results/platonic_graphs")
shutil.rmtree("results/platonic_graphs")
inform("FinishedPlatonic")

for name, obj in inspect.getmembers(GraphGenerators.smallgraphs):
    if inspect.isfunction(obj):
        try:
            graph = obj()
            if not isinstance(graph, Graph):
                continue
            small_graph_list.append(graph)  # Call the function
            print(f"Function {name} executed successfully.")
        except Exception as e:
            print(f"Error executing function {name}: {e}")
clear_dir("results/small_graphs/")
for graph in small_graph_list:
    thread = threading.Thread(
        target=calc_persistence_diagram,
        args=(graph, 2, filtration_functions.filtrate_by_diameter, "results/small_graphs/"))
    thread.start()
    thread.join()
    thread = threading.Thread(
        target=calc_persistence_diagram,
        args=(graph, 2, filtration_functions.filtrate_by_number_of_vertices, "results/small_graphs/"))
    thread.start()
    thread.join()
shutil.make_archive(f"results/small_graphs", 'zip', "results/small_graphs")
shutil.rmtree("results/small_graphs")
inform("FinishedSmallGraphs")

shutil.make_archive(f"results_{time.time()}", 'zip', "results/")
inform("FinishedProgram")
