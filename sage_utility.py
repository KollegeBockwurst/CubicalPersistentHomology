from matplotlib import image as mpimg
from sage.all import *
import numpy as np
import matplotlib.pyplot as plt
import time


def find_singular_non_degenerate_cubes(my_graph: Graph, max_dim: int):
    """
    Takes a sage graph and computes all singular, non-degenerate cubes up to dimension max_dim
    :param my_graph: An undirected sage graph
    :param max_dim: Maximum dimension to take into account.
    :return: A list of length max_dim + 1 containing singular cubes, where the i-th entry contains a list aof all
    non-degenerate singular cubes of dimension i. A singular cube of dimension i is represented as list of length 2^i,
    containing the image indices of a i-dimensional cube graph, sorted in lexiographical order
    """
    singular_cubes = [[] for _ in range(max_dim + 1)]  # create list of independent lists
    order_graph = my_graph.order()  # number of vertices in the given graph
    # sage adjacency is with zeros on main diagonal:
    adjacency_graph = my_graph.adjacency_matrix() + identity_matrix(order_graph)  # adjacency matrix of the given graph

    order_max_cube = pow(2, max_dim)  # number of vertices in a cube graph of dimension max_dim

    # options[i] saves the unhandled, but possible (as graph map) options for mapping the i-th cube vertex to my_graph
    # "possible" always relates to the mapping saved in cube_mapping[0:i]

    options = [[] for _ in range(order_max_cube)]  # create list of independent lists
    options[0] = list(range(order_graph))  # initialize options[0], since this is always possible
    cube_mapping = [None] * order_max_cube  # saves the current state of the mapping

    loop_flag = True  # will be set to false once it is clear that there are no singular cubes left
    while loop_flag:  # main loop

        # on this point there is at least one more option for at least one vertex saved in options
        # goal in this iteration is to pick one option, apply it to cube_mapping, then generating the possible
        # options for the next vertex. If the applied vertex is the last vertex of a singular cube, we will check
        # this singular cube for degeneracy, then add it to the list of found singular cubes

        # we generate the options "left to right", starting with vertex 0 (see initialization of options)
        # therefore, we will look for the vertex with existing options the furthest on the right

        for change_index in range(order_max_cube - 1, -1, -1):  # loop cube vertices from right to left
            if len(options[change_index]) > 0:  # check for existing options
                cube_mapping[change_index] = options[change_index].pop(0)  # apply the option to cube_mapping

                # ----------
                # check if change_index+1 == 2^x, i.e. change_index is last vertex of a cube (of dim x):
                if (change_index + 1) & change_index == 0:

                    # to find the cube_dim x, we compute log_2(change_index+1) by using bit operations:
                    cube_dim = 0
                    i = 1
                    while (i & (change_index + 1)) == 0:
                        i = i << 1
                        cube_dim += 1

                    # now we have found a singular cube of dimension cube_dim
                    singular_cube = cube_mapping[0:change_index + 1]

                    # check the singular cube for degeneracy:
                    degeneracy_flag = False  # will be set to True if a degeneracy is found
                    for k in range(cube_dim):  # loop through all dimensions of the cube and have a look at the faces
                        mask = (1 << k) - 1  # bit mask, used to split integers
                        degeneracy_flag = True  # set flag to True, just for the moment
                        for h in range(pow(2, cube_dim - 1)):  # loop through all vertices of one **face** of the cube
                            # vertex1/vertex2 are the h-th vertices of the positive (1) resp. negative (0) face in dim k
                            vertex1 = ((h & ~mask) << 1) | 1 << k | (h & mask)  # create vertex1 by bit manipulation
                            vertex2 = ((h & ~mask) << 1) | (h & mask)  # create vertex2 by bit manipulation
                            # now check if the positive and negative face are different in at least one vertex
                            # then set degeneracy_flag to false again. Otherwise it will stay True and indicate a degen.
                            if singular_cube[vertex1] != singular_cube[vertex2]:
                                degeneracy_flag = False
                                break
                        # check if the cube was degenerated in dimension k:
                        if degeneracy_flag:
                            break
                    # check if the cube was degenerated in any dimension. If not, add it to the output list:
                    if not degeneracy_flag:
                        singular_cubes[cube_dim].append(singular_cube)

                # ----------
                # generate the options for the next vertex (change_index + 1), if there is a next vertex:
                if change_index < order_max_cube - 1:
                    next_options = []  # saves the found options for vertex change_index+1
                    for possible_option in range(order_graph):  # loop through all possible options for the mapping
                        # we now need to check if we still have a graph map if vertex change_index+1 gets mapped to
                        # the vertex possible_option of the given graph, i.e. if the mapping
                        # cube_mapping[0:change_index+1].append(possible_option) is a graph mapping
                        # Since we know the structure of cubes very well, we only need to check very few edges
                        # to other vertices. E.g. a vertex in a cube of dimension i has a maximum of i adjacants
                        # we need to consider.
                        possible_flag = True  # will be set to False if the option is not possible
                        for j in range(max_dim):  # loop through all adjacant vertices of (cube) vertex change_index+1
                            # we can simply generate the needed vector by flipping one bit of change_index+1
                            # see lexiographic ordering for more information
                            vertex3 = (change_index + 1) ^ (1 << j)  # flips jth bit
                            if vertex3 > change_index + 1:
                                # the found vertex is further to the right, so we do not consider it
                                continue

                            # Finally we can check if the edge in the cube remains unchanged in the image
                            if not adjacency_graph[possible_option][cube_mapping[vertex3]]:
                                possible_flag = False
                                break

                        if possible_flag:  # add option if we still have a graph map
                            next_options.append(possible_option)
                    options[change_index + 1] = next_options  # add all possible options to the outer options list
                # break the loop moving our main pointer (change_index) from right to left,
                # since we modified the option list or need to re-visit the actual change_index:
                break

            # if there aren't any options left, we must have found them all:
            if change_index == 0:
                loop_flag = False  # no more singular cubes possible, leave main loop

    return singular_cubes


def create_face_maps(singular_cubes: list):
    """
    Takes a list of singular, non-degenerate cubes in different dimension. Computes the cubical boundary maps and returns
    them in a dictionary, refering to the dimension of the complex.
    :param singular_cubes: List (index = dimension) with lists of singular cubes. A singular cube is represented by a
    list, where the elements represent the image of the cube's vertices, ordered lexiographically
    :return: A dictionary wich maps a dimension to the associated boundary map, represented as sage matrix
    """
    face_maps = dict()  # result dictionary
    for cube_dim in range(len(singular_cubes)):  # loop through all provided dimensions
        # if cube_dim is zero, there are only trivial face maps, and sagemath handles this implicitly, but we can
        # do it expilictly as well::
        if cube_dim == 0:
            face_maps[cube_dim] = Matrix(0, len(singular_cubes[cube_dim]))
            continue

        # generate a matrix containing the boundary map cube_dim --> cube_dim - 1
        # note that this matrix will be transposed later
        face_matrix = [None for _ in range(len(singular_cubes[cube_dim]))]
        for j in range(len(singular_cubes[cube_dim])):  # loop through all cubes of dim cube_dim
            # we need to generate the associated row in the result matrix, which is a column of face_matrix
            matrix_column = [0] * len(singular_cubes[cube_dim - 1])  # initialize coöumn
            summand = -1  # summand fluctuating between -1 and 1, representing (-1)^k in the boundary map formula
            for k in range(cube_dim):  # loop through all dimensions of the chosen cube
                mask = 1 << k  # mask to differentiate between the positive and negative face in this dim
                face_0 = []  # negative face
                face_1 = []  # positive face
                for h in range(len(singular_cubes[cube_dim][j])):  # loop through all vertices of the cube
                    # using bit operations, we append the vertex either to the positive or negative face
                    if h & mask == 0:
                        face_0.append(singular_cubes[cube_dim][j][h])
                    else:
                        face_1.append(singular_cubes[cube_dim][j][h])

                for cube_index in range(len(singular_cubes[cube_dim - 1])):  # loop through all cube of dim cube_dim - 1
                    # when we find the positive (or negative) face, we can write it into the current matrix column
                    # see cubical boundary formula for more information
                    # it can happen, that a positive/negative face is NOT found. this happens, when it is degenerate
                    # in this case, we just ignore it, since it is 0 in our chain complex (see definition)
                    if face_0 == singular_cubes[cube_dim - 1][cube_index]:
                        matrix_column[cube_index] += summand
                    if face_1 == singular_cubes[cube_dim - 1][cube_index]:
                        matrix_column[cube_index] -= summand
                summand = -summand  # change summand sign
            face_matrix[j] = matrix_column  # add column to matrix
        face_maps[cube_dim] = Matrix(ZZ, face_matrix).transpose()  # create sage matrix, transpose it
    return face_maps


def filtrate_cubes(singular_cubes: list, my_graph: Graph, filtrate_function, **kwargs):

    graph_adjacency = np.matrix(my_graph.adjacency_matrix())  # adjacency matrix with zeros on main diagonal

    filtration_values = []
    inf_counter = 0
    for cube_dim in range(len(singular_cubes)):
        filtration_values.append([])
        for singular_cube in singular_cubes[cube_dim]:
            unique_image = np.unique(singular_cube)
            subgraph_adjacency = graph_adjacency[np.ix_(unique_image, unique_image)]
            filtration_values[cube_dim].append(filtrate_function(subgraph_adjacency))
            if filtration_values[cube_dim][-1] == float("inf"):
                inf_counter += 1
    if inf_counter > 0:
        print(f'   Warning: Lost {inf_counter} cubes to infinity filtration values.')
    return filtration_values


def compute_persistence_diagram2(face_maps, filtration):
    """
    Computes a persistent diagram
    :param face_maps: List containing face_maps. the i-th item is the i-th face map, i.e. the face map from dim i to i-1
    :param filtration: List of lists of integers. filtration[i][j] is the filtration value of the jth singular cube of
    in the ith dimension
    :return: persistence diagram
    """
    start_time = time.time()  # time measurement
    # max_step is the maximum filtration value except infinity
    max_step = max_value = max(val if val < infinity else 0 for sublist in filtration for val in sublist)
    max_dimension = len(filtration) - 1  # maximal dimension of singular cubes in the provided data

    global_ordering = []  # global ordering with respect to the (original) face_map and filtration variables
    global_filtration = []  # filtration values of the i-th singular cube, w.r.t. the global ordering
    global_dimension = []  # dimension of the i-th singular cube, w.r.t. the global ordering
    ''' *** fetch general information about provided arguments *** '''
    # number_of_singular_cubes[i] saves number of cubes of dimension <= i
    number_of_singular_cubes = [0] * (max_dimension + 1)
    # count rows of face_maps:
    for k in range(max_dimension + 1):
        number_of_singular_cubes[k] = face_maps[k].ncols()
        if k > 0:
            number_of_singular_cubes[k] += number_of_singular_cubes[k - 1]
    # global_reversed_ordering[i] stores the new position of the former i-th cube, resp. to the ordering in face_maps
    global_reversed_ordering = [None] * number_of_singular_cubes[-1]  # initialization
    ''' *** find global ordering: *** '''
    # pre-sort each dimension w.r.t. their filtration values:
    pre_sort = []
    for k in range(max_dimension + 1):
        pre_sort.append(np.argsort(filtration[k], axis=0))

    actual_step = 0  # saves the lowest filtration value where more cubes might need to be added to global ordering
    # index_marker saves which singular cubes have already been added to global ordering w.r.t. dimensions:
    index_marker = [0] * len(filtration)
    cube_counter = 0  # saves the total number of already oredered cubes
    while True:
        for k in range(max_dimension + 1):
            if index_marker[k] < len(filtration[k]) and filtration[k][pre_sort[k][index_marker[k]]] == actual_step:
                if len(global_ordering) == 302:
                    print(f"{k} {pre_sort[k][index_marker[k]]}!")
                global_ordering.append((k, pre_sort[k][index_marker[k]]))
                global_filtration.append(actual_step)
                global_dimension.append(k)
                reversed_index = (number_of_singular_cubes[k - 1] if k > 0 else 0) + pre_sort[k][index_marker[k]]
                global_reversed_ordering[reversed_index] = cube_counter
                cube_counter += 1
                index_marker[k] += 1
                break
        else:
            actual_step += 1
            if float("inf") > actual_step > max_step:  # i.e. all singular cubes should have been handled
                actual_step = float("inf")
            elif actual_step == float("inf"):
                break

    ''' *** create sparse matrix representation R by using all face_maps and the global ordering *** '''
    R = []  # array initialization. R[i] saves the row indices of the i-th columnn, where the value is 1 (in F_2)
    for k in range(number_of_singular_cubes[-1]):
        R.append([])
        cube_dimension = global_dimension[k]
        if cube_dimension > 0:
            start_index = number_of_singular_cubes[cube_dimension - 2] if cube_dimension > 1 else 0
            column = face_maps[cube_dimension].columns()[global_ordering[k][1]]
            for j in range(face_maps[cube_dimension].nrows()):
                if column[j] % 2 == 1:  # implicit conversion to Ring F_2
                    R[k].append(global_reversed_ordering[start_index + j])

    print(len(R))
    exit(7)
    print(f'   Create representation R w.r.t. global ordering: {time.time() - start_time}s')
    start_time = time.time()

    ''' *** Compute reduced from of D *** '''

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

    # top_indices[i] saves the column index of the column with top(column) == i
    top_indices = [None] * len(R)

    # move through matrix and perform gauss elimination, i.e. persistent algorithm
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

    print(f'   Compute reduced form of D: {time.time() - start_time}s')
    start_time = time.time()

    ''' *** read persistence diagram from reduces matrix D *** '''
    # persistence_diagram[i] consists a list [a,b,c] of length three showing if a persistent feature in dim. a is
    # born at step b, dies in step c and is born since the i-th column of the matrix is zero
    persistence_diagram = []
    for k in range(len(R)):
        persistence_diagram.append([None, None, None])  # initialization
        top_k = top(R[k])  # compute top(R[k])
        if top_k is None:  # i.e. column is zero, a new persistent feature is born
            if global_filtration[k] == float("inf"):  # the persistence never gets born
                break
            # a new persistence is born here, since the kernel of our matrix got bigger
            persistence_diagram[k][0] = global_dimension[k]
            persistence_diagram[k][1] = global_filtration[k]
            persistence_diagram[k][2] = float('inf')
        else:
            # a persistence must have died here, since the image got bigger (thanks to the reduced form)
            if len(persistence_diagram) <= top_k:
                print(f"{k} {top_k} {len(persistence_diagram)}")
                print(R)
            persistence_diagram[top_k][2] = global_filtration[k]

    # filter persistence diagram for unnecessary entries
    ordered_diagram = dict()
    for j in range(max_dimension):  # CAVE: ignoring highest dim, since cubes one dim higher are missing
        ordered_diagram[j] = list(tuple(entry[1:]) for entry in persistence_diagram
                                  if entry[0] is not None and entry[0] == j and entry[1] != entry[2])

    print(f'   Read persistence from reduced form: {time.time() - start_time}s')
    start_time = time.time()
    return ordered_diagram


def draw_diagram(diagram, title, max_step, **kwargs):
    """
    Draws a persistent diagram. This code was written by ChatGPT
    :param max_step:
    :param diagram: k
    :param title: ojn
    :return: on
    """

    data_by_dimension = diagram
    colors = ['blue', 'orange', 'green', 'red', 'pink']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    idx = 0
    for dim in sorted(data_by_dimension.keys()):
        data = data_by_dimension[dim]
        # Sort data for better visualization
        data = sorted(data, key=lambda x: x[1] - x[0], reverse=True)

        for birth, death in data:
            ax2.plot([birth, death if death < infinity else max_step], [idx, idx], color=colors[dim], lw=2,
                     label=f'Dimension {dim}' if 'Dimension ' + str(dim) not in [l.get_label()
                                                                                 for l in
                                                                                 ax2.get_lines()] else "")

            ax2.scatter([birth], [idx], color=colors[dim], s=50)  # Highlighting start

            if death != float('inf'):
                ax2.scatter([death], [idx], color="black", s=50)  # Highlighting end if not infinite
            idx += 1

    ax2.set_yticks(range(idx))
    ax2.set_yticklabels([f"Feature {i + 1}" for i in range(idx)])
    ax2.set_xlabel('Scale (Birth-Death)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax2.legend(loc="upper left")

    file_name_title = title.replace("\n", "_").replace(" ", "_")

    image_path = "temp.png"
    img = mpimg.imread(image_path)
    ax1.imshow(img)
    ax1.axis('off')

    result = ""
    for dim in sorted(data_by_dimension.keys()):
        data = data_by_dimension[dim]
        result += "# persistence dim " + str(dim) + ": " + str(len(data)) + "\n"

    # Adding a textbox
    ax3.text(0, 1, result, fontsize=12, color='blue', ha='left', va='top')

    # Removing axes
    ax3.axis('off')

    plt.title(f'Barcode Diagram - {title}')
    plt.tight_layout()
    plt.savefig(title.replace("\n", "_").replace(" ", "_") + "_" + str(time.time()) + ".png")
    return plt


def persistence(G, filtration_function, max_dim, **kwargs):
    """
    Performs the whole persistent algorithm on G
    :param G: A sagemath graph
    :param filtration_function: A filtration function
    :param max_dim: cubes of dim >max_dim will not be considered
    :param kwargs: use_distances=True: a distance matrix rather than an adjacency matrix will be given
    to the filtration function
    :return: side-effect only, draws and saves a persistent diagram
    """
    time_measurement = False
    if "time_measurement" in kwargs.keys() and kwargs["time_measurement"]:
        time_measurement = True

    if time_measurement:
        print("Calculating singular, non-degenerated cubes...")
    start_time = time.time()
    singular_cubes = find_singular_non_degenerate_cubes(G, max_dim)
    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Calculating face maps...")

    start_time = time.time()
    face_maps = create_face_maps(singular_cubes)

    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Filtrating face maps...")

    start_time = time.time()
    filtration = filtrate_cubes(singular_cubes, G, filtration_function)

    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Calculating persistence diagram...")

    start_time = time.time()
    diagram = compute_persistence_diagram2(face_maps, filtration)

    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Drawing persistence diagram...")

    start_time = time.time()
    G.plot().save(filename="temp.png")
    max_steps = max_value = max(0 if val == float("inf") else val for sublist in filtration for val in sublist)
    draw_diagram(diagram, f'{G.name()}'
                          f'\n {filtration_function.__name__}, max_dim: {max_dim}', max_steps,
                 show_plot=kwargs["show_plot"] if "show_plot" in kwargs.keys() else True)
    # Enable sage math consistency check (RAM intensive):
    return;
    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Compute betti numbers using SageMath")

    start_time = time.time()
    chain_complexes = []
    betti_numbers = []

    for i in range(max_steps):
        filtered_face_maps = dict()
        for k in range(np.max(list(face_maps.keys())) + 1):
            col_indices = [index for index, value in enumerate(filtration[k]) if value <= i]
            row_indices = []
            if k > 0:
                row_indices = [index for index, value in enumerate(filtration[k - 1]) if value <= i]
            filtered_face_maps[k] = \
                Matrix([[face_maps[k][i, j] for j in col_indices] for i in row_indices]).change_ring(FiniteField(2))
        chain_complexes.append(ChainComplex(filtered_face_maps, degree_of_differential=-1))
        betti_numbers.append(chain_complexes[i].betti())

    if time_measurement:
        print(f'{time.time() - start_time} s')
        print("Consistency check")

    start_time = time.time()
    for i in range(len(betti_numbers)):
        for j in betti_numbers[i].keys():
            if j >= len(diagram):
                continue
            counter = 0
            for tupel in diagram[j]:
                if tupel[0] <= i < tupel[1]:
                    counter += 1
            if counter != betti_numbers[i][j]:
                print(f'Consistency check failed in persistence step {i}')
                break

    if time_measurement:
        print(f'{time.time() - start_time} s')
