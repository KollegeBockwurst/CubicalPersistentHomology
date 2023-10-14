import matplotlib.pyplot as plt

data_by_dimension = [[(1,2)], [(0,3)], [(5,6)]]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 6))

idx = 0
for dim, data in enumerate(data_by_dimension):
    # Sort data for better visualization
    data = sorted(data, key=lambda x: x[1]-x[0], reverse=True)

    for birth, death in data:
        plt.plot([birth, death], [idx, idx], color=colors[dim], lw=2, label=f'Dimension {dim}' if 'Dimension '+str(dim) not in [l.get_label() for l in plt.gca().get_lines()] else "")
        plt.scatter([birth, death], [idx, idx], color=colors[dim], s=50)  # Highlighting start and end
        idx += 1

plt.yticks(range(idx), [f"Feature {i+1}" for i in range(idx)])
plt.xlabel('Scale (Birth-Death)')
plt.title('Barcode Diagram')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
