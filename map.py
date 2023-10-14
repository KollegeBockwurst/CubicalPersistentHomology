from graph import Graph


class Map:
    def __init__(self, domain: Graph, codomain: Graph, mapping: list):
        """
        Represents a map between to graphs

        :param domain: Graph
            Domain of the map.

        :param codomain: Graph
            Codomain of the map.

        :param mapping: list
            1D list containing the image_ids of the image of vertices in domain in codomain
        """
        self.domain = domain
        self.codomain = codomain
        self.mapping = mapping

        # Consistency checks:
        if len(mapping) != domain.adjacency_matrix.shape[0]:
            raise ValueError("Mapping not well defined.")

        if not all([x >= 0 for x in mapping]):
            raise ValueError("Mapping not well defined.")

        if not all([x < codomain.adjacency_matrix.shape[0] for x in mapping]):
            raise ValueError("Mapping is not well defined.")

    def __str__(self):
        """
        Generates a string representation of the map.
        """
        result = f'<Graph map:\n'
        for i in range(len(self.domain.vertex_labels)):
            result += f'{self.domain.vertex_labels[i]} -> {self.codomain.vertex_labels[self.mapping[i]]}\n'
        result += ">"
        return result

    def is_homomorphism(self):
        """
       Computes if the map is a graph homomorphism or not.

        :return: bool
            True if map is graph homomorphism, False if not
        """
        for i in range(self.domain.adjacency_matrix.shape[0]):
            for j in range(self.domain.adjacency_matrix.shape[1]):
                if self.domain.adjacency_matrix[i, j]:
                    if not self.codomain.adjacency_matrix[self.mapping[i], self.mapping[j]] \
                            and self.mapping[i] != self.mapping[j]:
                        return False
        return True
