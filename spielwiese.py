list_of_lists = [[1, 2], [3, 4, 5], [6]]
indices = [i for i, sublist in enumerate(list_of_lists) for _ in sublist]

print(indices)