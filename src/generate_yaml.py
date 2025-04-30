import os

DATASET_PATH = "datasets/yolo_dataset_2"


def create_yaml(dataset_path):
    """
    Create a yaml conf file for yolo training.
    """
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("This yaml file already exists")
        return
    with open(yaml_path, mode="w+") as f:
        f.write(f"path: ../{dataset_path}\n")
        f.write("train: train\n")
        f.write("val: val\n")
        f.write("names:\n")
        signs = ["pozadí"] + [""] * len(os.listdir("data/signs"))
        for sign_path in os.listdir("data/signs"):
            id, sign_name = (
                int(sign_path[0 : sign_path.find("_")]),
                sign_path[sign_path.find("_") + 1 : sign_path.find(".")],
            )
            signs[id + 1] = sign_name
            # f.write(f"{id}: {sign_name}\n")
        for i, name in enumerate(signs):
            f.write(f"    {i}: {name}\n")


if __name__ == "__main__":
    create_yaml(DATASET_PATH)
