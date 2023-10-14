import numpy as np


class Graph:
    # Initialize instance directly from edge matrix:
    def __init__(self, vertex_labels: list, adjacency_matrix: np.ndarray, **kwargs):
        """
        Represents a simple, non-directional graph without weights

        :param vertex_labels: list
            A 1D list representing labels of vertices. Used mostly only for output format.

        :param adjacency_matrix: np.ndarray
            A 2D numpy array representing the adjacency relations between all vertices. Per convention, the main
            diagonal is False
        """
        self.vertex_labels = vertex_labels
        # Convention: main diagonal is False
        np.fill_diagonal(adjacency_matrix, False)
        self.adjacency_matrix = adjacency_matrix

        if "distance_matrix" in kwargs.keys():
            self.distance_matrix = kwargs['distance_matrix']
        else:
            self.distance_matrix = None

        # Consistency checks:
        if adjacency_matrix.shape != (len(vertex_labels), len(vertex_labels)):
            raise IndexError("Edge_matrix shape is incompatible with vertex list.")

    # Initialize instance from edge set:
    @classmethod
    def from_edge_list(cls, vertex_labels: list, edges: list):
        """
        Creates a new a simple, non-directional graph without weights

        :param vertex_labels: list
            A 1D list representing labels of vertices. Used mostly only for output format.

        :param edges: list
            A 1D list with lists of size 2, which represent an edge between two vertices by using the vertex_labels.
        """
        # Create edge matrix:
        edge_matrix = np.full((len(vertex_labels), len(vertex_labels)), False)
        for edge in edges:
            vertex_id0 = vertex_labels.index(edge[0])
            vertex_id1 = vertex_labels.index(edge[1])
            edge_matrix[vertex_id0, vertex_id1] = True
            edge_matrix[vertex_id1, vertex_id0] = True
        return cls(vertex_labels, edge_matrix)

    # Create line graph
    @classmethod
    def line_graph(cls, n):
        """
        Creates a new a simple, non-directional line graph of size n without weights. A line graph consists of
        n+1 vertices and n edges, connecting these vertices in a line.

        :param n: int
            Length of line graph (non-negative integer).
        """
        edges = []
        for i in range(n):
            edges = edges + [[i, i + 1]]
        return cls.from_edge_list(list(range(n + 1)), edges)

    # Create hypercubes:
    @classmethod
    def hypercube(cls, n, d):
        """
        Creates a new a simple, non-directional hypercube graph of size n without weights. A hypercube of length n
        and dimension d is the n-times cartesion product of the line graph of length n with itself

        :param n: int
            Edge-length of the hypercube (non-negative integer)

        :param d: int
            Dimension of the hypercube (non-negative integer)
        """
        if d <= 0:
            return cls.line_graph(0)

        cube = cls.line_graph(n)
        for i in range(d - 1):
            cube = cube.cartesian_product(cls.line_graph(n))
        return cube

    # Create empty graph:
    @classmethod
    def empty_graph(cls):
        """
        Creates a new a simple, non-directional hypercube graph without any vertices or edges
        """
        return cls.from_edge_list([], [])

    # print object:
    def __str__(self):
        """
        Generates a short representation of the graph.
        """
        return f'<Graph object with {len(self.vertex_labels)} vertices and ' \
               f'{int(np.count_nonzero(self.adjacency_matrix) / 2)} edges>'

    def cartesian_product(self, other: 'Graph') -> 'Graph':
        """
        Computes the cartesian product of this graph and another graph.
        Order of vertices: The new vertices a ordered like in the following example:
        self.vertex_labels: ['A','B']
        other.vertex_labels: [0,1]
        self.cartesian_product(other).vertex_labels: ['0,A','1,A','0,B','1,B']

        :param other: Graph
            the graph to compute the cartesian product with.

        :return: Graph
            Cartesian product of self and other.
        """
        vertex_labels_1 = self.vertex_labels
        vertex_labels_2 = other.vertex_labels
        vertex_labels_3 = []

        # name the new vertices:
        for j in range(len(vertex_labels_2)):
            for i in range(len(vertex_labels_1)):
                vertex_labels_3.append(f'{vertex_labels_1[i]},{vertex_labels_2[j]}')

        # Create adjacency matrix by merging the adjacency matrices of self and other:
        adjacency_matrix1 = self.adjacency_matrix
        adjacency_matrix2 = other.adjacency_matrix
        adjacency_matrix = []
        for i in range(adjacency_matrix2.shape[0]):
            line = []
            for j in range(adjacency_matrix2.shape[1]):
                if i == j:
                    line.append(adjacency_matrix1)
                else:
                    if adjacency_matrix2[i, j]:
                        line.append(np.eye(adjacency_matrix1.shape[0], dtype=bool))
                    else:
                        line.append(np.full(adjacency_matrix1.shape, False))
            adjacency_matrix.append(line)

        adjacency_matrix = np.block(adjacency_matrix)

        return Graph(vertex_labels_3, adjacency_matrix)

    def compute_diameter(self):
        distance_matrix = self.get_distance_matrix()
        return distance_matrix.max(initial=None)

    def get_distance_matrix(self):
        if self.distance_matrix is None:
            self.distance_matrix = self.floyd_warshall()
        return self.distance_matrix

    def floyd_warshall(self):
        """
        Compute the shortest path distances between all pairs of vertices in a graph using the Floyd-Warshall algorithm.

        :return: np.ndarray
            A 2D NumPy array where the element at (i, j) is the shortest path distance from vertex i to vertex j.
            If there is no path between vertex i and vertex j, the value will be np.inf.
        """
        # Initialize the distance matrix with the input graph adjacency matrix
        distance_matrix = np.full(self.adjacency_matrix.shape, np.inf)
        distance_matrix[self.adjacency_matrix] = 1
        np.fill_diagonal(distance_matrix, 0)

        # Number of vertices in the graph
        num_vertices = distance_matrix.shape[0]

        # Main loop: try all possible intermediate vertices
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    # Update the distance matrix by considering the path through vertex k
                    if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

        return distance_matrix

    def get_subgraph(self, vertex_indices: list[int]):
        """
       Returns the subgraph containing only the vertices in vertex_indices and all edges between those vertices

       :return: Graph
           Subgraph
       """
        vertex_indices = sorted(set(vertex_indices))
        vertex_labels = self.vertex_labels[vertex_indices]
        adjacency_matrix = self.adjacency_matrix[np.ix_(vertex_indices, vertex_indices)]
        if self.distance_matrix is None:
            return Graph(vertex_labels, adjacency_matrix)
        distance_matrix = self.distance_matrix[np.ix_(vertex_indices, vertex_indices)]
        return Graph(vertex_labels, adjacency_matrix, distance_matrix=distance_matrix)

    """
    def find_singular_cubes(self, d: int):
        # TODO, very inefficient
        singular_cubes = []
        n = len(self.vertex_labels)
        possibilities = pow(n, pow(2, d))
        for possibility in range(possibilities):
            number = possibility
            mapping = []
            for i in range(pow(2, d)):
                mapping.append(number % n)
                number = number // n
            try:
                singular_cubes.append(SingularCube(d, self, mapping))
            except ValueError:
                # print(f"Mapping {mapping} was not a good mapping.")
                pass
        return singular_cubes
    """

    def get_sndc(self, d: int):
        cube_mappings = []
        n = self.adjacency_matrix.shape[0]

        cube_adjacency = self.hypercube(1,d).adjacency_matrix

        # options initialization:
        cube_mapping = [None] * cube_adjacency.shape[0]
        num_vertices_cube = cube_adjacency.shape[0]
        options = [None] * num_vertices_cube
        options[0] = list(range(n))
        loop_marker = True
        while loop_marker:
            for change_index in range(num_vertices_cube - 1, -1, -1):
                if options[change_index] is not None and len(options[change_index]) > 0:
                    cube_mapping[change_index] = options[change_index].pop(0)
                    if change_index == num_vertices_cube-1: # erweitern, um auch kleiner singuläre komplexe herauszufinden
                        cube_mappings.append(cube_mapping.copy())
                    if change_index < num_vertices_cube-1:
                        # find options for change_index+1:
                        possibility = [True]*n
                        adjacant = cube_adjacency[change_index+1]
                        for i in range(change_index+1):
                            if adjacant[i]:
                                for j in range(n):
                                    possibility[j] = possibility[j] and (self.adjacency_matrix[cube_mapping[i]][j] or cube_mapping[i] == j)
                        options[change_index + 1] = [i for i, x in enumerate(possibility) if x]
                    break;
                if change_index == 0:
                    loop_marker = False  # no more singular cubes possible

        return cube_mappings
