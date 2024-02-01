import time
from multiprocessing import Pool

from matplotlib import pyplot as plt, image as mpimg
from sage.all import Matrix
from collections import Counter

from sage.rings import infinity


class ImageDrawer:
    def __init__(self, ordered_persistence, graph, title, path):
        self.ordered_persistence = ordered_persistence
        self.graph = graph
        self.title = title
        self.path = path

    def run(self):

        colors = ['blue', 'orange', 'green', 'red', 'pink']
        max_step = max(
            0 if val[1] == float("inf") else val[1] for sublist in self.ordered_persistence.values() for val in sublist)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

        idx = 0
        for dim in sorted(self.ordered_persistence.keys()):
            data = self.ordered_persistence[dim]
            # Sort data for better visualization
            data = sorted(data, key=lambda x: x[1] - x[0], reverse=True)

            for birth, death in data:
                ax2.plot([birth, death if death < float('inf') else max_step], [idx, idx], color=colors[dim], lw=2,
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

        file_name_title = self.title.replace("\n", "_").replace(" ", "_")

        self.graph.plot().save(filename="temp.png")
        image_path = "temp.png"
        img = mpimg.imread(image_path)
        ax1.imshow(img)
        ax1.axis('off')

        result = ""
        for dim in sorted(self.ordered_persistence.keys()):
            data = self.ordered_persistence[dim]
            result += "# persistence dim " + str(dim) + ": " + str(len(data)) + "\n"

        # Adding a textbox
        ax3.text(0, 1, result, fontsize=12, color='blue', ha='left', va='top')

        # Removing axes
        ax3.axis('off')

        plt.title(f'Barcode Diagram - {self.title}')
        plt.tight_layout()
        path = self.path + self.title.replace("\n", "_").replace(" ", "_") + "_" + str(time.time()) + ".png"
        plt.savefig(path)
        return path