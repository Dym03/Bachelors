import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import os

from traffic_sign_dataset import TrafficSignDataset, convert_yolo_to_torch_outputs, apply_nms, COCO_to_My
from torch.utils.data import DataLoader
from learn import load_model, create_mapping_dict
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from enum import Enum
from ultralytics import YOLO
import numpy as np


# Initialize metric
class Eval_Type(Enum):
    FASTER_RCNN = 1
    YOLO_MY = 2
    YOLO_BASE = 3



MODEL_BASE_DIR = ""
MODEL_NAME = "yolo11l.pt"
DATASET_BASE_DIR = "datasets"
DATASET_NAME = "100_000_n2"
EVAL_TYPE = Eval_Type.YOLO_BASE


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def print_metrics(metric, mapping_dict):
    print(
        f"Overall mAP50-95 : {metric['map']:<.4f} mAP_small : {metric['map_small']:<.4f} mAP_med : {metric['map_medium']:<.4f} mAP_large : {metric['map_large']:<.4f}"
    )
    print(f"{'Class':<30}  :{'mAP50-95':<10}{'mAR':<10}")
    for i, c_id in enumerate(metric["classes"]):
        if c_id in mapping_dict:
            print(
                f"{mapping_dict[c_id.item()]:<30} : {metric['map_per_class'][i]:<10.4f}{metric['mar_100_per_class'][i]:<10.4f}"
            )
    print(metric)


def evaluate_faster_RCNN(model, data_loader, metric, device):
    with torch.no_grad():
        for i, (img, targets) in enumerate(data_loader):
            #        img = torch.stack(img).to(device)
            img = [image.to(device) for image in img]
            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in targets
            ]
            predictions = model(img)
            predictions[0]["labels"].to(dtype=torch.int64, device=device)
            metric.update(predictions, targets)
            if i % 100 == 0:
                print(f"Img {i} out of {len(data_loader)}")
    result = metric.compute()
    return result

def translate_predictions(predictions, device):
    translation_dict = None
    if EVAL_TYPE == Eval_Type.YOLO_BASE:
        translation_dict = COCO_to_My
    predictions[0]['labels'] = torch.tensor(list(map(lambda x: translation_dict[x] if x in translation_dict else -1, predictions[0]['labels']))).to(device)
    return predictions

def evaluate_YOLO(model, data_loader, metric, device):
    for i, (img, targets) in enumerate(data_loader):
        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]
        predictions = model(img, verbose=False)
        predictions = convert_yolo_to_torch_outputs(predictions, device)
        if EVAL_TYPE in [Eval_Type.YOLO_BASE]:
            predictions = translate_predictions(predictions, device)
        #predictions = apply_nms(predictions, device)
        if 11 in predictions[0]["labels"]:
            print(targets)
            print(predictions)
        metric.update(predictions, targets)
        if i % 100 == 0:
            print(f"Img {i} out of {len(data_loader)}")
    result = metric.compute()
    return result


def get_val_dataset():
    val_dataset = None
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        val_dataset = TrafficSignDataset(
            os.path.join(DATASET_BASE_DIR, DATASET_NAME),
            "val/images",
            "val/labels",
            transform=transforms,
        )
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE]:
        val_dataset = TrafficSignDataset(
            os.path.join(DATASET_BASE_DIR, DATASET_NAME),
            "val/images",
            "val/labels",
            None,
        )
    return val_dataset


def get_model(model_path, device):
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        model, opt, sch = load_model(model_path, device, box_score_thresh=0.70)
        print(model)
        model.to(device)
        model.eval()
        return model
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE]:
        return YOLO(model_path)


if __name__ == "__main__":
    val_dataset = get_val_dataset()
    mapping_dict = create_mapping_dict("data/signs")
    model_path = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    model = get_model(model_path, device)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, average='micro')
    metric.to(device)
    result = None
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        result = evaluate_faster_RCNN(model, val_loader, metric, device)
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE]:
        result = evaluate_YOLO(model, val_loader, metric, device)
    
    for k in result.keys():
        print(f"{k}: {result[k]}")
    print_metrics(result, mapping_dict)
