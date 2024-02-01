from multiprocessing import Pool
from sage.all import Matrix
from collections import Counter


class GlobalOrderingCalculator:
    def __init__(self, face_maps, filtration):
        self.face_maps = face_maps
        self.filtration = filtration

    def run(self):
        global_filtration = [item for sublist in self.filtration for item in sublist]
        cumulative_lengths = [0]
        for i in range(len(self.face_maps)):
            cumulative_lengths.append(len(self.face_maps[i]))
            if i > 0:
                cumulative_lengths[i] += cumulative_lengths[i-1]

        global_face_maps = [[x + cumulative_lengths[i-1] for x in item] for i in range(len(self.face_maps)) for item in self.face_maps[i]]
        global_dimension = [i for i in range(len(self.face_maps)) for _ in self.face_maps[i]]
        sorted_indices = sorted(range(len(global_filtration)), key=lambda i: (global_filtration[i], i))
        sorted_positions = [sorted_indices.index(i) for i in range(len(global_filtration))]
        global_filtration = [global_filtration[i] for i in sorted_indices]
        global_face_maps = [[sorted_positions[x] for x in global_face_maps[i]] for i in sorted_indices]
        global_dimension = [global_dimension[i] for i in sorted_indices]
        return global_face_maps, global_filtration, global_dimension
