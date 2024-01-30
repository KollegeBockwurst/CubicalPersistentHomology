from concurrent.futures import ThreadPoolExecutor
from sage.all import Graph, identity_matrix


def generate_face_map(adjacency_graph, max_dim: int, start_options: list):

    return None


class FaceMapGeneratorScheduler:
    def __init__(self, singular_cubes, thread_number: int):
        self.singular_cubes = singular_cubes
        self.thread_number = thread_number

    def run(self):

        with ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            futures = []
            numbers_per_thread = (self.graph.order() // self.thread_number)+1

            for i in range(self.thread_number):
                start_index = i*numbers_per_thread
                stop_index = max((i+1)*numbers_per_thread, self.graph.order())
                futures.append(
                    executor.submit(generate_singular_cubes, adjacency_graph=adjacency_graph, max_dim=self.max_dim,
                                    start_options=list(range(start_index, stop_index))))
                if stop_index == self.graph.order():
                    break

            for future in futures:
                future_result = future.result()
                for i in range(self.max_dim + 1):
                    result[i] = result[i] + future_result[i]
        return result
