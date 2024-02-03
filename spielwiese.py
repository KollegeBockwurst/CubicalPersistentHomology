my_list = [4,2,7,4,5,6,2]
print(my_list[3:18])
sorted_positions = [i[0] for i in sorted(enumerate(my_list), key=lambda x: x[1])]
print(sorted_positions)

