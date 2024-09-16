from PIL import Image
import random
import torch
from torchvision.transforms import v2
import os

SIGN_MIN_SIZE = 20
SIGN_MAX_SIZE = 100
IMAGE_SIZE = 640

SIGN_DIR = "data/signs"
BACKGROUND_IMG_DIR = "datasets/base_img"
DATASET_ROOT_DIR = "datasets/test_dataset/test"
OUTPUT_IMG_DIR = os.path.join(DATASET_ROOT_DIR, "img")
OUTPUT_LABEL_DIR = os.path.join(DATASET_ROOT_DIR, "labels")


class Yolo_annotation:
    def __init__(self, id: int, center_x: float, center_y: float, width: float, height: float):
        self.id = id
        self.x = center_x
        self.y = center_y
        self.w = width
        self.h = height

    def __repr__(self):
        return f"{self.id} {self.x} {self.y} {self.w} {self.h}\n"


def load_signs() -> dict:
    sign_dict = {}
    for filename in os.listdir(SIGN_DIR):
        idx = filename[: filename.find("_")]
        print(idx, filename)
        sign_dict[idx] = filename

    return sign_dict


def get_merged_background_sign(background: Image, sign: Image, position: tuple):
    # sign2 = Image.composite(sign, Image.new("RGB", sign.size, "white"), sign)

    augmentor = v2.RandAugment()
    sign_converted = sign.convert("RGB")
    sign3 = augmentor(sign_converted)
    background.paste(sign3, position, sign)

    return background


def write_annotation(image_name: str, yolo_annots: list[Yolo_annotation]):
    with open(f"{OUTPUT_LABEL_DIR}/{image_name}.txt", "w") as file:
        for annot in yolo_annots:
            file.write(repr(annot))


def prepare_dirs():
    if not os.path.isdir(DATASET_ROOT_DIR):
        os.mkdir(DATASET_ROOT_DIR)
        os.mkdir(OUTPUT_IMG_DIR)
        os.mkdir(OUTPUT_LABEL_DIR)
    elif not os.path.isdir(OUTPUT_IMG_DIR):
        os.mkdir(OUTPUT_IMG_DIR)
        if not os.path.isdir(OUTPUT_LABEL_DIR):
            os.mkdir(OUTPUT_LABEL_DIR)
    elif not os.path.isdir(OUTPUT_LABEL_DIR):
        os.mkdir(OUTPUT_LABEL_DIR)
    else:
        return -1


if __name__ == "__main__":
    if prepare_dirs() == -1:
        print("Dataset already exists")
        exit()
    sign_dict = load_signs()

    with open(os.path.join(DATASET_ROOT_DIR, "annotation.csv"), "x") as annot_file:
        for filename in os.listdir(BACKGROUND_IMG_DIR):
            annot_file.write(filename + "\n")
            signs = random.choices(
                list(sign_dict.items()), k=random.randint(0, 5)
            )  # Chooses random 5 signs to put into a picture
            annot_list = []
            background = Image.open(f"{BACKGROUND_IMG_DIR}/{filename}")
            background.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
            for idx, sign_path in signs:
                sign = Image.open(f"{SIGN_DIR}/{sign_path}").convert("RGBA")
                new_width, new_height = (
                    random.randint(SIGN_MIN_SIZE, SIGN_MAX_SIZE),
                    random.randint(SIGN_MIN_SIZE, SIGN_MAX_SIZE),
                )
                sign.thumbnail((new_width, new_height))
                pos_x, pos_y = (
                    random.randint(0, background.size[0] - sign.size[0]),
                    random.randint(0, background.size[1] - sign.size[1]),
                )
                sign_annot = Yolo_annotation(
                    idx,
                    (pos_x + (sign.size[0] / 2)) / background.size[0],
                    (pos_y + (sign.size[1] / 2)) / background.size[1],
                    sign.size[0] / background.size[0],
                    sign.size[1] / background.size[1],
                )  # The annotations have to be normalized
                background = get_merged_background_sign(
                    background, sign, (pos_x, pos_y)
                )
                annot_list.append(sign_annot)
            write_annotation(filename, annot_list)
            background.save(f"{OUTPUT_IMG_DIR}/{filename}")
            # background.show()

            # input("Press Enter to plot the next point...")
