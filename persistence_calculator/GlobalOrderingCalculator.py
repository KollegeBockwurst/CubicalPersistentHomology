from multiprocessing import Pool
from sage.all import Matrix
from collections import Counter


def sort_index(my_list):
    return sorted(my_list)


def merge_sorted_lists(lists):
    merged_list = []
    indexes = [0] * len(lists)  # Track current index for each list

    while True:
        min_value = None
        min_list = -1
        for i in range(len(lists)):
            if indexes[i] < len(lists[i]):
                if min_value is None or lists[i][indexes[i]] < min_value:
                    min_value = lists[i][indexes[i]]
                    min_list = i
        if min_list == -1:  # All lists have been fully traversed
            break
        merged_list.append(min_value)
        indexes[min_list] += 1

    return merged_list


class GlobalOrderingCalculator:
    def __init__(self, face_maps, filtration, thread_number):
        self.face_maps = face_maps
        self.filtration = filtration
        self.thread_number = thread_number

    def run(self):
        global_filtration = [item for sublist in self.filtration for item in sublist]
        cumulative_lengths = [0]
        for i in range(len(self.face_maps)):
            cumulative_lengths.append(len(self.face_maps[i]))
            if i > 0:
                cumulative_lengths[i] += cumulative_lengths[i - 1]

        global_face_maps = [[x + cumulative_lengths[i - 1] for x in item] for i in range(len(self.face_maps)) for item
                            in self.face_maps[i]]
        global_dimension = [i for i in range(len(self.face_maps)) for _ in self.face_maps[i]]

        args = []
        enumerated_filtration = list(map(lambda x: (x[1], x[0]), enumerate(global_filtration)))
        chunk_size = (len(enumerated_filtration) // self.thread_number) + 1
        for i in range(0, len(enumerated_filtration), chunk_size):
            chunk = enumerated_filtration[i:i+chunk_size]
            args.append([chunk])

        with Pool(len(args)) as p:
            p_results = p.starmap(sort_index, args)
        sorted_enumerate = merge_sorted_lists(p_results)

        sorted_indices = list(map(lambda x: x[1], sorted_enumerate))
        sorted_positions = [sorted_indices.index(i) for i in range(len(global_filtration))]
        global_filtration = [global_filtration[i] for i in sorted_indices]
        global_face_maps = [[sorted_positions[x] for x in global_face_maps[i]] for i in sorted_indices]
        global_dimension = [global_dimension[i] for i in sorted_indices]
        return global_face_maps, global_filtration, global_dimension
