import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASET_NAME = "100_000_n2/train"
DATASET_DIR = "datasets"

def get_category_counts(dataset_path) -> dict[str, int]:
    categories_counter = {}
    if not os.path.isdir(dataset_path):
        print(f"No dataset with this name exitsts at this path {dataset_path}\n")
        return
    labels_path = os.path.join(dataset_path, "labels")
    if not os.path.isdir(labels_path):
        print(f"No directory with this name exitsts at this path {labels_path}\n")
        return
    files = os.listdir(labels_path)
    background_count = 0
    total_instances = 0
    print(f'Num of images: {len(files)}')
    for filename in files:
        background = True
        with open(os.path.join(labels_path, filename)) as f:
            for line in f.readlines():
                background = False
                category = line.split()[0]  # First in yolo annot is a category
                if category not in categories_counter:
                    categories_counter[category] = 0
                categories_counter[category] += 1
                total_instances += 1
        if background:
            background_count += 1

    print(f'Number of background images: {background_count}\n')
    print(f'Number of instances in total: {total_instances}')
    return categories_counter


def graph_category_counts(categories_counter: dict[str, int]):
    categories = sorted(categories_counter, key=int)
    values = [categories_counter[id] for id in categories]
    plt.xlabel("Kategorie")
    plt.xticks(rotation=45)
    plt.ylabel("Počet výskytů")
    plt.title("Rozložení kategorii v trénovacím datasetu CATSD")
    data = {'cat': categories, 'val' : values}
    sns.barplot(x='cat', y='val',data=data, palette="muted")
    ax = sns.barplot(x='cat', y='val', data=data, palette="muted")

    # Select only 4 evenly spaced tick positions
    tick_positions = np.linspace(0, len(categories) - 1, 4, dtype=int)  # Get 4 indices
    tick_labels = [categories[i] for i in tick_positions]  # Get corresponding category names

    ax.set_xticks(tick_positions)  # Set tick positions
    ax. set_xticklabels(tick_labels, rotation=45)  # Set tick labels with rotation
    plt.tight_layout()
    plt.savefig("100_000_categories_train", format='pdf')
    plt.show()


def analyze_dataset(dataset_path):
    categories_counter = get_category_counts(dataset_path)
    print(categories_counter)
    graph_category_counts(categories_counter)


if __name__ == "__main__":
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    print(dataset_path)
    analyze_dataset(dataset_path)
