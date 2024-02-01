from multiprocessing import Pool
from sage.all import Matrix
from collections import Counter


class PersistenceReader:
    def __init__(self, result_matrix, global_filtration, global_dimension, max_dim):
        self.result_matrix = result_matrix
        self.global_filtration = global_filtration
        self.global_dimension = global_dimension
        self.max_dim = max_dim

    def run(self):
        def top(my_column):
            """
            Computes the greatest index where my_column is not 0
            uses sparse matrix representation
            :param my_column: A column of a matrix
            :return: index top(my_column)
            """
            # Only iterate over non-zero entries
            if len(my_column) == 0:
                return None
            return max(my_column)

        persistence_diagram = []
        R = self.result_matrix
        for k in range(len(R)):
            persistence_diagram.append([None, None, None])  # initialization
            top_k = top(R[k])  # compute top(R[k])
            if top_k is None:  # i.e. column is zero, a new persistent feature is born
                if self.global_filtration[k] == float("inf"):  # the persistence never gets born
                    break
                # a new persistence is born here, since the kernel of our matrix got bigger
                persistence_diagram[k][0] = self.global_dimension[k]
                persistence_diagram[k][1] = self.global_filtration[k]
                persistence_diagram[k][2] = float('inf')
            else:
                # a persistence must have died here, since the image got bigger (thanks to the reduced form)
                if len(persistence_diagram) <= top_k:
                    print(f"{k} {top_k} {len(persistence_diagram)}")
                    print(R)
                persistence_diagram[top_k][2] = self.global_filtration[k]

        # filter persistence diagram for unnecessary entries
        ordered_diagram = dict()
        for j in range(self.max_dim + 1):
            ordered_diagram[j] = list(tuple(entry[1:]) for entry in persistence_diagram
                                      if entry[0] is not None and entry[0] == j and entry[1] != entry[2])

        return ordered_diagram
