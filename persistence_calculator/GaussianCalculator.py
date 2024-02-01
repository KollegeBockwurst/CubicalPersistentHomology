

class GaussianCalculator:
    def __init__(self, global_face_maps):
        self.global_face_maps = global_face_maps

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

        R = self.global_face_maps
        # top_indices[i] saves the column index of the column with top(column) == i
        top_indices = [None] * len(R)

        for k in range(len(R)):
            col_k = R[k]
            top_k = top(col_k)

            while top_k is not None and top_indices[top_k] is not None:
                # add columns w.r.t. sparse matrix representation in F_2:
                for row_index in R[top_indices[top_k]]:
                    if row_index in col_k:
                        col_k.remove(row_index)
                    else:
                        col_k.append(row_index)
                top_k = top(col_k)

            R[k] = col_k  # probably unneccessary, since R[k] is already col_k
            if top_k is not None:
                top_indices[top_k] = k

        return R
