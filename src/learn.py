import torch
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# import torchvision


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir: str, annotations_path: str, img_dir: str, label_dir: str):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(root_dir, annotations_path))
        self.img_dir = os.path.join(root_dir, img_dir)
        self.label_dir = os.path.join(root_dir, label_dir)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_name = self.annotations.iloc[index].iloc[0]
        img_file_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_file_path)
        # tensor_img = torch.tensor(img)
        
        label_file_path = os.path.join(self.label_dir, file_name + ".txt")
        tensor_labels = []
        with open(label_file_path, 'r') as f:
            labels = []
            for line in f.readlines():
                tokens = [int(float(i)) if int(float(i)) == float(i) else float(i) for i in line.split(" ")]
                labels.append(tokens)
            tensor_labels = torch.tensor(labels)
        return (img, tensor_labels)


if __name__ == "__main__":
    train_dataset = TrafficSignDataset("datasets/test_dataset/train", "annotation.csv", "imgs", "labels")
    test_dataset = TrafficSignDataset("datasets/test_dataset/test", "annotation.csv", "imgs", "labels")

