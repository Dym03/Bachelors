import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASET_NAME = "Mapillary/val"
DATASET_DIR = "datasets"
ONLY_CATSD = True
ids_of_interest = [271,272,374,375,376,377,162,109,352,353,354,355,356,357,274,275,247,248,249,250,276,277,251,252,253
,254,255,262,238,239,240,230,231,232,233,136,137,139,140,141,145,146,148,151,155,152,128,129,153,130,131,227,107,234,235
,283,284,386,387,388,348,349,285,286,321,322,323,324,325,326,327,110,111,112,381,19,382,20,21,383,342,343,176,177
,174,175,180,181,182,183,71,185,186,187,188,189,190,198,199,200,201,202,344,345,346,347]

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
    instances_of_interest = 0
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
                if int(category) in ids_of_interest:
                    instances_of_interest += 1
        if background:
            background_count += 1
    if DATASET_NAME[:DATASET_NAME.index('/')] == 'Mapillary':
        categories_counter.pop('89')
        categories_counter.pop('85')
        categories_counter.pop('399')
    print(max(categories_counter.values()))
    print(f'Number of background images: {background_count}\n')
    print(f'Number of instances in total: {total_instances}')
    print(f'Number of instances of interest: {instances_of_interest}')
    return categories_counter


def graph_category_counts(categories_counter: dict[str, int]):
    categories = sorted(categories_counter, key=int)
    if ONLY_CATSD:
        values = [categories_counter[str(id)] if (str(id) in categories_counter.keys()) else 0 for id in ids_of_interest]
        categories = sorted(ids_of_interest)
    else:
        values = [categories_counter[id] for id in categories]
    
    plt.xlabel("Kategorie")
    plt.xticks(rotation=45)
    plt.ylabel("Počet výskytů")
    plt.title("Rozložení kategorii v validačním datasetu Mapillary")
    data = {'cat': categories, 'val' : values}
    sns.barplot(x='cat', y='val',data=data, palette="muted")
    ax = sns.barplot(x='cat', y='val', data=data, palette="muted")

    # Select only 4 evenly spaced tick positions
    tick_positions = np.linspace(0, len(categories) - 1, 4, dtype=int)  # Get 4 indices
    tick_labels = [categories[i] for i in tick_positions]  # Get corresponding category names

    ax.set_xticks(tick_positions)  # Set tick positions
    ax. set_xticklabels(tick_labels, rotation=45)  # Set tick labels with rotation
    plt.tight_layout()
    plt.savefig(f"{DATASET_NAME[:DATASET_NAME.index('/')]}_categories_{DATASET_NAME[DATASET_NAME.index('/')+1:]}", format='pdf')
    plt.show()


def analyze_dataset(dataset_path):
    categories_counter = get_category_counts(dataset_path)
    print(categories_counter)
    graph_category_counts(categories_counter)


if __name__ == "__main__":
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    print(dataset_path)
    analyze_dataset(dataset_path)
