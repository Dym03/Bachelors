import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import os

from learn import load_model, custom_collate_fn, create_mapping_dict
from traffic_sign_dataset import TrafficSignDataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

MODEL_BASE_DIR = "models/"
MODEL_NAME = "0.04058232057011673.pt"
DATASET_BASE_DIR = "datasets"
DATASET_NAME = ""

if __name__ == "__main__":
    mapping_dict = create_mapping_dict("data/signs")
    model_path = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, opt, sch = load_model(model_path, device, box_score_thresh=0.90)

    model.to(device)
    model.eval()
    image_path = "data/img/50.jpg"

    image = Image.open(image_path)
    transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    images = [transforms(image)]
    images = [image.to(device) for image in images]

    predictions = model(images)
    print(predictions)
    # print(type(predictions[0]["labels"]))
    labels = [mapping_dict[int(id) + 1] for id in predictions[0]["labels"]]
    print(labels)
    box = draw_bounding_boxes(
        images[0],
        boxes=predictions[0]["boxes"],
        labels=labels,
        colors="green",
        width=4,
        font_size=40,
    )
    im = to_pil_image(box.detach())
    im.show()
