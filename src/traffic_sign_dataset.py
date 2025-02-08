import os
import pandas as pd
from PIL import Image
from torch import tensor, float32, int64, zeros
from torch.utils.data import Dataset


def yolo_to_coco(box, img_width, img_height):
    """
    Convert YOLO format (x_center, y_center, width, height)
    to COCO format (x_min, y_min, x_max, y_max).

    Args:
    - box: List of bounding box in YOLO format [x_center, y_center, width, height] (relative values).
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - List of bounding box in COCO format [x_min, y_min, x_max, y_max] (absolute values in pixels).
    """
    x_center, y_center, width, height = box

    # Convert relative coordinates to absolute pixel values
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate the top-left and bottom-right coordinates
    x_min = float(x_center - (width / 2))
    y_min = float(y_center - (height / 2))
    x_max = float(x_center + (width / 2))
    y_max = float(y_center + (height / 2))

    return [x_min, y_min, x_max, y_max]


class TrafficSignDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotations_path: str,
        img_dir: str,
        label_dir: str,
        transform,
    ):
        self.root_dir = root_dir
        self.annotations =  #pd.read_csv(os.path.join(root_dir, annotations_path))
        self.img_dir = os.path.join(root_dir, img_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_name = self.annotations.iloc[index].iloc[0]
        img_file_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_file_path)
        img = self.transform(img)
        # if self.transform:
        #     img = self.transform(img)
        # tensor_img = torch.tensor(img)

        label_file_path = os.path.join(self.label_dir, file_name + ".txt") # TODO The file name had a .jpg before now it is without it, so i have to remove it
        with open(label_file_path, "r") as f:
            ids = []
            boxes = []
            # areas = []
            for line in f.readlines():
                tokens = [
                    int(float(i)) if int(float(i)) == float(i) else float(i)
                    for i in line.split(" ")
                ]
                ids.append(tokens[0])
                yolo_box = yolo_to_coco(tokens[1:], img.shape[1], img.shape[2])
                boxes.append(yolo_box)
        if len(boxes) == 0:
            boxes_tensor = zeros((0, 4), dtype=float32)  # Empty tensor of shape [0, 4]
            ids_tensor = tensor([], dtype=int64)
        else:
            boxes_tensor = tensor(boxes, dtype=float32)
            ids_tensor = tensor(ids, dtype=int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = ids_tensor
        return (img, target)
