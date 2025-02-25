import os
import matplotlib.pyplot as plt

DATASET_NAME = "100_000/train"
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
    for filename in files:
        with open(os.path.join(labels_path, filename)) as f:
            for line in f.readlines():
                category = line.split()[0]  # First in yolo annot is a category
                if category not in categories_counter:
                    categories_counter[category] = 0
                categories_counter[category] += 1
    return categories_counter


def graph_category_counts(categories_counter: dict[str, int]):
    categories = sorted(categories_counter, key=int)
    values = [categories_counter[id] for id in categories]
    plt.xlabel("Kategorie")
    plt.xticks(rotation=45)
    plt.ylabel("Počet výskytů")
    plt.title("Počet instancí na kategorií")
    plt.bar(categories, values, color="blue")
    plt.show()
    plt.savefig("100_000_categories")


def analyze_dataset(dataset_path):
    categories_counter = get_category_counts(dataset_path)
    graph_category_counts(categories_counter)


if __name__ == "__main__":
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    analyze_dataset(dataset_path)
