import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import os

from traffic_sign_dataset import TrafficSignDataset
from torch.utils.data import DataLoader
from learn import load_model, create_mapping_dict
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Initialize metric
metric = MeanAveragePrecision(iou_type="bbox")

MODEL_BASE_DIR = "models/"
MODEL_NAME = "0.04058232057011673.pt"
DATASET_BASE_DIR = "datasets"
DATASET_NAME = "yolo_dataset_2"


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


if __name__ == "__main__":
    transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    val_dataset = TrafficSignDataset(
        os.path.join(DATASET_BASE_DIR, DATASET_NAME),
        "val/images",
        "val/labels",
        ToTensor(),
        transform=transforms,
    )
    mapping_dict = create_mapping_dict("data/signs")
    model_path = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model, opt, sch = load_model(model_path, device, box_score_thresh=0.70)

    model.to(device)
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    metric = MeanAveragePrecision(iou_type="bbox")
    for img, targets in val_loader:
        img = torch.stack(img).to(device)
        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]
        for image in img:
            predictions = model(img)
            predictions = predictions
            predictions[0]["labels"] = torch.tensor(
                list(map(lambda x: x + 1, predictions[0]["labels"])), dtype=torch.int64
            )
            print(predictions)
            print(targets)
            metric.update(predictions, targets)

    mAP_results = metric.compute()
    print(mAP_results)
    # dataset_path = os.path.join(DATASET_BASE_DIR, DATASET_NAME)
    # for image_path in files(dataset_path):
    #     image_path = os.path.join(dataset_path, image_path)
    #     image = Image.open(image_path)
    #     images = [transforms(image)]
    #     images = [image.to(device) for image in images]

    #     predictions = model(images)
    #     print(predictions)
    #     # print(type(predictions[0]["labels"]))
    #     labels = [mapping_dict[int(id) + 1] for id in predictions[0]["labels"]]
    #     print(labels)
    #     box = draw_bounding_boxes(
    #         images[0],
    #         boxes=predictions[0]["boxes"],
    #         labels=labels,
    #         colors="black",
    #         width=4,
    #         font_size=40,
    #     )
    #     im = to_pil_image(box.detach())
    #     im.show()
    #     input("Press Enter to continue...")
    #     im.close()
