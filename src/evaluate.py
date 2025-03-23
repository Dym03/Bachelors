import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import os

from traffic_sign_dataset import TrafficSignDataset, convert_yolo_to_torch_outputs, apply_nms, COCO_to_My, Mapillary_to_My, My_to_Mapillary
from torch.utils.data import DataLoader
from learn import load_model, create_mapping_dict
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from enum import Enum
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ultralytics.models.yolo.detect.val import DetectionValidator
import numpy as np
from Mapillary_utils import Mapillary_mapping_dict 


# Initialize metric
class Eval_Type(Enum):
    FASTER_RCNN = 1
    YOLO_MY = 2
    YOLO_BASE = 3
    YOLO_MAPILLARY = 4



MODEL_BASE_DIR = "torch_runs/run_2025-02-28_100_100_000_n2/models"
MODEL_NAME = "0.014049725461147336.pt"
DATASET_BASE_DIR = "datasets"
DATASET_NAME = "100_000_n2"
EVAL_TYPE = Eval_Type.FASTER_RCNN
num_of_appearences = {}
predicted_signs = {}

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
        if c_id >= 0 and c_id <= 396:
            print(
                f"{mapping_dict[c_id.item()]:<30} : {metric['map_per_class'][i]:<10.4f}{metric['mar_100_per_class'][i]:<10.4f}"
            )


def evaluate_faster_RCNN(model, data_loader, metric, device, yolo_metrics, yolo_val):
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
            tp = yolo_val._process_batch(predictions[0]['boxes'], targets[0]['boxes'], targets[0]['labels']).int()
            yolo_metrics.process(tp, predictions[0]['scores'], predictions[0]['labels'], targets[0]['labels'])
            if i % 100 == 0:
                print(f"Img {i} out of {len(data_loader)}")
    result = metric.compute()
    print(f'YOLO Metrics: {yolo_metrics.results_dict}')
    return result

def translate_predictions(predictions, device):
    translation_dict = None
    if EVAL_TYPE == Eval_Type.YOLO_BASE:
        translation_dict = COCO_to_My
    elif EVAL_TYPE == Eval_Type.YOLO_MAPILLARY:
        translation_dict = Mapillary_to_My
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.FASTER_RCNN] and DATASET_NAME == 'Mapillary':
        translation_dict = My_to_Mapillary
    labels = predictions[0]["labels"].tolist()  # Convert tensor to list
    translated_labels = [
        translation_dict[x] if x in translation_dict else -1 for x in labels
    ]
    for x in labels:
        act_predicted = predicted_signs.setdefault(x, 0)
        predicted_signs[x] = act_predicted + 1
        if x in translation_dict:
            curr_count = num_of_appearences.setdefault(x, 0)
            num_of_appearences[x] = curr_count + 1

    predictions[0]["labels"] = torch.tensor(translated_labels, dtype=torch.int64, device=device)
    return predictions

def evaluate_YOLO(model, data_loader, metric, device, yolo_metrics, yolo_val):
    for i, (img, targets) in enumerate(data_loader):
        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]
        predictions = model.predict(img, verbose=False, imgsz=512)
        predictions = convert_yolo_to_torch_outputs(predictions, device)
        if (EVAL_TYPE in [Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY] and DATASET_NAME != 'Mapillary') or (EVAL_TYPE == Eval_Type.YOLO_MY and DATASET_NAME == 'Mapillary'):
            predictions = translate_predictions(predictions, device)
        #predictions = apply_nms(predictions, device)
        metric.update(predictions, targets)
        tp = yolo_val._process_batch(predictions[0]['boxes'], targets[0]['boxes'], targets[0]['labels']).int()
        yolo_metrics.process(tp, predictions[0]['scores'], predictions[0]['labels'], targets[0]['labels'])
        if i % 100 == 0:
            print(f"Img {i} out of {len(data_loader)}")
    result = metric.compute()
    print(f'YOLO Metrics: {yolo_metrics.results_dict}')
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
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY]:
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
        model.to(device)
        model.eval()
        return model
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY]:
        return YOLO(model_path)


if __name__ == "__main__":
    val_dataset = get_val_dataset()
    mapping_dict = None

    yolo_metrics = DetMetrics()
    yolo_val = DetectionValidator()
    if DATASET_NAME == 'Mapillary':
        mapping_dict = Mapillary_mapping_dict
    else:
        mapping_dict = create_mapping_dict("data/signs")
    yolo_metrics.names = mapping_dict  # should be the dictionary of classes
    model_path = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = get_model(model_path, device)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, average='macro')
    metric.to(device)
    result = None
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        result = evaluate_faster_RCNN(model, val_loader, metric, device, yolo_metrics, yolo_val)
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY]:
        result = evaluate_YOLO(model, val_loader, metric, device, yolo_metrics, yolo_val)
    
    for k in result.keys():
        print(f"{k}: {result[k]}")
    print_metrics(result, mapping_dict)
    print(num_of_appearences)
    print(f'Predicted dict: {predicted_signs}')
