import multiprocessing
import time

from persistence_calculator.FaceMapGenerator import FaceMapGeneratorScheduler
from persistence_calculator.FiltrationCalculator import FiltrationCalculatorScheduler
from persistence_calculator.GaussianCalculator import GaussianCalculator
from persistence_calculator.GlobalOrderingCalculator import GlobalOrderingCalculator
from persistence_calculator.ImageDrawer import ImageDrawer
from persistence_calculator.PersistenceReader import PersistenceReader

from persistence_calculator.SingularCubeGenerator import SingularCubeGeneratorScheduler


def calc_persistence_diagram(graph, max_dim, filtration_function, relative_path):
    print(f"Calculating persistence of {graph.name()}")
    cpu_cores = multiprocessing.cpu_count()
    title = f"{graph.name()}-{filtration_function.__name__}-max_dim:{max_dim}"
    stamp = time.time()
    start_stamp = stamp
    scheduler = SingularCubeGeneratorScheduler(graph, max_dim + 1, cpu_cores * 2)
    cubes = scheduler.run()
    print(f"Generated Cubes: {time.time() - stamp}")

    stamp = time.time()
    scheduler = FaceMapGeneratorScheduler(cubes, cpu_cores * 2)
    face_maps = scheduler.run()
    print(f"Generated Face Maps: {time.time() - stamp}")

    stamp = time.time()
    scheduler = FiltrationCalculatorScheduler(filtration_function, graph, cubes, cpu_cores*2)
    filtration = scheduler.run()
    print(f"Calculated filtrations: {time.time() - stamp}")

    stamp = time.time()
    scheduler = GlobalOrderingCalculator(face_maps, filtration)
    global_face_maps, global_filtration, global_dimension = scheduler.run()
    print(f"Calculated global ordering: {time.time() - stamp}")
    stamp = time.time()
    scheduler = GaussianCalculator(global_face_maps)
    result_matrix = scheduler.run()
    print(f"Performed gaussian algorithm: {time.time() - stamp}")
    stamp = time.time()
    scheduler = PersistenceReader(result_matrix, global_filtration, global_dimension, max_dim)
    ordered_diagram = scheduler.run()
    print(f"Performed persistence reader: {time.time() - stamp}")

    stamp = time.time()
    scheduler = ImageDrawer(ordered_diagram, graph, title, relative_path)
    path = scheduler.run()
    print(f"Performed image creation: {time.time() - stamp}")
    print(f"Completed. Total time: {time.time() - start_stamp}")
    return path
