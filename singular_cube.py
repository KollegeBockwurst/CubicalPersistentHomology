import graph_utility
from graph import Graph
from map import Map


class SingularCube(Map):
    def __init__(self, d, codomain, mapping, **kwargs):
        super().__init__(Graph.hypercube(1, d), codomain, mapping)  # not clear if d and 1 is right here
        self.dimension = d

        if "disable_consistent_check" in kwargs.keys() and kwargs["disable_consistent_check"]:
            return

        if not self.is_homomorphism():
            raise ValueError("Singular Cube is not a graph homomorphism.")
        for i in range(d):
            plus_cube_mapping = []
            minus_cube_mapping = []
            for j in range(len(self.mapping)):
                if (j & (1 << i)) != 0:
                    plus_cube_mapping.append(self.mapping[j])
                else:
                    minus_cube_mapping.append(self.mapping[j])
            if d > 0 and plus_cube_mapping == minus_cube_mapping:
                raise ValueError("Singular Cube is degenerated.")

    def __eq__(self, other):
        return self.mapping == other.mapping and self.dimension == other.dimension

    def face_map(self):
        add = []
        subtract = []
        for i in range(self.dimension):
            plus_cube_mapping = []
            minus_cube_mapping = []
            for j in range(len(self.mapping)):
                if (j & (1 << i)) != 0:
                    plus_cube_mapping.append(self.mapping[j])
                else:
                    minus_cube_mapping.append(self.mapping[j])
            if i % 2 == 0:
                try:
                    add.append(SingularCube(self.dimension - 1, self.codomain, minus_cube_mapping))
                except ValueError:
                    pass
                try:
                    subtract.append(SingularCube(self.dimension - 1, self.codomain, plus_cube_mapping))
                except ValueError:
                    pass
            else:
                try:
                    subtract.append(SingularCube(self.dimension - 1, self.codomain, minus_cube_mapping))
                except ValueError:
                    pass
                try:
                    add.append(SingularCube(self.dimension - 1, self.codomain, plus_cube_mapping))
                except ValueError:
                    pass
        return add, subtract

    def contraction_length(self):
        image = self.codomain.get_subgraph(self.mapping)
        return graph_utility.contraction_length(image, self.dimension)

    def diameter(self):
        image = self.codomain.get_subgraph(self.mapping)
        return image.compute_diameter()

    def vertex_number(self):
        return len(set(self.mapping))
