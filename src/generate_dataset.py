from PIL import Image
import random
from torchvision.transforms import v2
import os
from enum import Enum
from generate_yaml import create_yaml
from tqdm import tqdm

SIGN_MIN_SIZE = 60
SIGN_MAX_SIZE = 120
IMAGE_SIZE = 512

SIGN_DIR = "data/signs"
BACKGROUND_IMG_DIR = "background_photos"
DATASET_ROOT_DIR = "datasets/10_000_n3"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT_DIR, "train", "labels")
VALIDATION_IMG_DIR = os.path.join(DATASET_ROOT_DIR, "val", "images")
VALIDATION_LABEL_DIR = os.path.join(DATASET_ROOT_DIR, "val", "labels")


class SAVE_MODE(Enum):
    TRAIN = 1
    VALIDATION = 2


class Yolo_annotation:
    def __init__(
        self, id: int, center_x: float, center_y: float, width: float, height: float
    ):
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
        idx = int(filename[: filename.find("_")]) + 1
        print(idx, filename)
        sign_dict[idx] = filename

    return sign_dict


def do_overlap(img_1, img_2):
    x1, y1, width, height = img_1
    x2, y2, width_2, height_2 = img_2

    x1_br, y1_br = x1 + width, y1 + height
    x2_br, y2_br = x2 + width_2, y2 + height_2

    x_intersect_left = max(x1, x2)
    y_intersect_top = max(y1, y2)
    x_intersect_right = min(x1_br, x2_br)
    y_intersect_bottom = min(y1_br, y2_br)

    if x_intersect_left < x_intersect_right and y_intersect_top < y_intersect_bottom:
        return True
    return False


def get_merged_background_sign(background: Image, sign: Image, position: tuple):
    # sign2 = Image.composite(sign, Image.new("RGB", sign.size, "white"), sign)

    # augmentor = v2.AutoAugment() # Old method, not maybe that useful
    sign_converted = sign.convert("RGBA")
    # sign3 = augmentor(sign_converted)
    transforms = v2.RandomApply(
        transforms=[
            v2.RandomAffine(degrees=(5, 10), fill=(0, 0, 0, 0)),
            v2.RandomPerspective(distortion_scale=0.3, p=1, fill=(0, 0, 0, 0)),
            v2.RandomRotation(degrees=(5, 10)),
        ]
    )

    sign3 = transforms(sign_converted)
    background.paste(sign3, position, sign3)

    return background


def write_annotation(
    image_name: str, yolo_annots: list[Yolo_annotation], mode: SAVE_MODE
):
    annot_file_path = ""
    if mode == SAVE_MODE.TRAIN:
        annot_file_path = TRAIN_LABEL_DIR
    elif mode == SAVE_MODE.VALIDATION:
        annot_file_path = VALIDATION_LABEL_DIR

    with open(
        f"{annot_file_path}/{image_name[0 : image_name.find('.')]}.txt", "w"
    ) as file:
        for annot in yolo_annots:
            file.write(repr(annot))


def prepare_dirs():
    if not os.path.isdir(DATASET_ROOT_DIR):
        os.mkdir(DATASET_ROOT_DIR)
        os.mkdir(os.path.join(DATASET_ROOT_DIR, "train"))
        os.mkdir(os.path.join(DATASET_ROOT_DIR, "val"))
        os.mkdir(TRAIN_IMG_DIR)
        os.mkdir(TRAIN_LABEL_DIR)
        os.mkdir(VALIDATION_IMG_DIR)
        os.mkdir(VALIDATION_LABEL_DIR)
    else:
        return -1


if __name__ == "__main__":
    if prepare_dirs() == -1:
        print("Dataset already exists")
        exit()
    sign_dict = load_signs()

    with open(os.path.join(DATASET_ROOT_DIR, "annotation.csv"), "x") as annot_file:
        current_save_mode = SAVE_MODE.TRAIN
        dataset_size = len(os.listdir(BACKGROUND_IMG_DIR))
        val_split = dataset_size * 0.8  # 80% train 20% validation
        for i, filename in enumerate(
            tqdm(os.listdir(BACKGROUND_IMG_DIR), desc="Generating Dataset")
        ):
            if i > val_split:
                current_save_mode = SAVE_MODE.VALIDATION
            annot_file.write(filename + "\n")
            signs = random.choices(
                list(sign_dict.items()),
                k=random.choices([0, 1, 2, 3, 4, 5], weights=[0.05, 0.19, 0.19, 0.19, 0.19, 0.19])[0]
            )  # Chooses upto 5 random signs to put into a picture
            annot_list = []
            background = Image.open(f"{BACKGROUND_IMG_DIR}/{filename}")
            background = background.resize((IMAGE_SIZE, IMAGE_SIZE))
            inserted_signs = []
            for idx, sign_path in signs:
                sign = Image.open(f"{SIGN_DIR}/{sign_path}").convert("RGBA")
                new_width, new_height = (
                    random.randint(SIGN_MIN_SIZE, SIGN_MAX_SIZE),
                    random.randint(SIGN_MIN_SIZE, SIGN_MAX_SIZE),
                )
                sign.thumbnail((new_width, new_height))
                while 1:
                    pos_x, pos_y = (
                        random.randint(0, background.size[0] - sign.size[0]),
                        random.randint(0, background.size[1] - sign.size[1]),
                    )
                    overlaps = False
                    overlaps = [
                        do_overlap(i_sign, (pos_x, pos_y, sign.size[0], sign.size[1]))
                        for i_sign in inserted_signs
                    ]  # Check for overlaping sings, we now dont want that, so we eliminate them by finding new coords
                    if overlaps.count(True) == 0:
                        break

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
                inserted_signs.append((pos_x, pos_y, sign.size[0], sign.size[1]))
                annot_list.append(sign_annot)
            write_annotation(filename, annot_list, current_save_mode)
            if current_save_mode == SAVE_MODE.TRAIN:
                background.save(f"{TRAIN_IMG_DIR}/{filename}")
            elif current_save_mode == SAVE_MODE.VALIDATION:
                background.save(f"{VALIDATION_IMG_DIR}/{filename}")
            # background.show()
    create_yaml(DATASET_ROOT_DIR)
