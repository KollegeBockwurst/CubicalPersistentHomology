# Cubical persistent homology calculator


## Overview

This project was created as part of my master thesis at Philipps university Marburg. It provides a framework to compute persistent homology of cubical complexes, in particular of cubical homology on graphs. Cubical homology on graphs is a singular homology. The singular cubes are graph maps from graph cubes onto a given graph, which is represented as $\texttt{SageMath}$ object. First, all possible singular cubes are generated using a custom multithreading algorithm. Note, that this step might need **a lot** memory, since the number of singular cubes grows exponentially with the calculated dimension of homology. Afterwards, custom *filtrations functions* are applied to filter the singular cubes and generate a proper boundary matrix, which is then feeded into a commonly used persistence algorithm based on gaussian elimination. To improve performance when working with big matrices, only computations over the field  $\mathbb{F}_2$.  

## Important methods

Basic syntax:
```
from sage.all import *  
from persistence_calculator import filtration_functions  
from persistence_calculator.algorithm import calc_persistence_diagram  

graph = Graph() # Define SageMath graph here
max_homology_dimension = 0 # Set maximum dimension
filtration_function = filtration_functions.filtrate_by_diameter # Choose predefined or custom, valid filtration function
max_filtration_value = 0 # Choose maximum filtration value, all other singular cubes will be disregarded
path = "" # Choose relative path for output file

calc_persistence_diagram(graph, max_homology_dimension, filtration_function, max_filtration_value, path)
```
One can also use
```
compute_multiple_persistence(graph_list, max_dim, filtration_function, relative_path)
```
to compute the persistent homologies of multiple graphs in parallel. However, memory limitations will become more crucial when using this method obviously.
## Filtration functions

The following filtration functions are included in ```persistence_calculator.filtration_functions ```:

 - ``` filtrate_by_homotopy ```: Uses minimal relative contraction length of a singular cube's image.
 - ``` filtrate_by_diameter ```: Uses relative diameter of a singular cube's image.
 - ``` filtrate_by_number_of_vertices ```: Uses the number of vertices in the image of a singular cube.
 - ``` filtrate_by_number_of_edges ```: Uses the number of edges in the image of a singular cube.

Note, that any filtration function  must be compatible with the homology's boundary map.
## Results

In "results" some computed persistences have been uploaded. 

## References
1. Barcelo, H., Greene, C., Jarrah, A. & Welker, V. Discrete Cubical and Path Ho-
mologies of Graphs. *Algebraic Combinatorics* **2**, 417–437 (2019).
2. Otter, N., Porter, M., Tillmann, U., Grindrod, P. & Harrington, H. A roadmap for
the computation of persistent homology. *EPJ data science* **6** (2017).
3. The Sage Developers. SageMath, the Sage Mathematics Software System (Version 10.0). [https://www.sagemath.org](https://www.sagemath.org) (2024)

