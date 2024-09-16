import torch
import os

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io.image import decode_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


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
        self.annotations = pd.read_csv(os.path.join(root_dir, annotations_path))
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
        print(img.shape)
        # if self.transform:
        #     img = self.transform(img)
        # tensor_img = torch.tensor(img)

        label_file_path = os.path.join(self.label_dir, file_name + ".txt")
        tensor_labels = []
        with open(label_file_path, "r") as f:
            labels = []
            for line in f.readlines():
                tokens = [
                    int(float(i)) if int(float(i)) == float(i) else float(i)
                    for i in line.split(" ")
                ]
                labels.append(tokens)
            tensor_labels = torch.tensor(labels)
        return (img, tensor_labels)


if __name__ == "__main__":
    train_dataset = TrafficSignDataset(
        "datasets/test_dataset/train", "annotation.csv", "img", "labels", ToTensor()
    )
    test_dataset = TrafficSignDataset(
        "datasets/test_dataset/test", "annotation.csv", "img", "labels", ToTensor()
    )
    print(len(train_dataset), len(test_dataset))

    training_loader = DataLoader(train_dataset, shuffle=True)
    validation_loader = DataLoader(test_dataset, shuffle=False)

    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    images = images[0]

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    preprocess = weights.transforms()

    batch = [preprocess(images)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(
        images,
        boxes=prediction["boxes"],
        labels=labels,
        colors="red",
        width=4,
        font_size=30,
    )
    im = to_pil_image(box.detach())
    im.show()
