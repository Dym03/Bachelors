import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import os

from traffic_sign_dataset import TrafficSignDataset, convert_yolo_to_torch_outputs, COCO_to_My, Mapillary_to_My, My_to_Mapillary, CATSD_to_GTSDB
from torch.utils.data import DataLoader
from train import load_model, create_mapping_dict
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from enum import Enum
from ultralytics import YOLO
import numpy as np
from Mapillary_utils import Mapillary_mapping_dict, GTSDB_mapping_dict 


# Specifies the eval type that should be used, which is connected to steps are are needed
class Eval_Type(Enum):
    FASTER_RCNN = 1
    YOLO_MY = 2
    YOLO_BASE = 3
    YOLO_MAPILLARY = 4



#MODEL_BASE_DIR = "torch_runs/run_2025-02-28_100_100_000_n2/models"
#MODEL_NAME = "0.014048114550448642.pt"
MODEL_BASE_DIR = "runs/detect/yolo11l.pt_100_000_n2_50_2025-03-23/weights"
MODEL_NAME = "best.pt"
DATASET_BASE_DIR = "datasets"
DATASET_NAME = "Mapillary"
EVAL_TYPE = Eval_Type.YOLO_MY
num_of_appearences = {}
predicted_signs = {}

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def print_metrics(macro_metric, micro_metric, mapping_dict):
    """
    Prints metrics in YOLO like style

    Args:
        macro_metrics: Macro Mean Average Precision metric.
        micro_metrics: Micro Mean Average Precision metric.
        mapping_dict (dict): Dictionary to map ids to sign names

    """
    filtered_mAP = list(filter(lambda x: x > 0, macro_metric['map_per_class']))
    my_mAP = 0
    if (len(filtered_mAP) > 0):
        my_mAP = sum(filtered_mAP) / len(filtered_mAP)
    print(
            f'''{'Overall':<30} |{'Macro':<10}|{'Micro':<10}|{'My':<10}|
{'mAP50-95':<30}|{macro_metric['map']:<10.4f}|{micro_metric['map']:<10.4f}|{my_mAP:<10.4f}|
{'mAP50':<30}|{macro_metric['map_50']:<10.4f}|{micro_metric['map_50']:<10.4f}|
{'mAR100':<30}|{macro_metric['mar_100']:<10.4f}|{micro_metric['mar_100']:<10.4f}| 
{'mAP_small':<30}|{macro_metric['map_small']:<10.4f}|{micro_metric['map_small']:<10.4f}|
{'mAP_med':<30}|{macro_metric['map_medium']:<10.4f}|{micro_metric['map_medium']:<10.4f}|
{'mAP_large':<30}|{macro_metric['map_large']:<10.4f}|{micro_metric['map_large']:<10.4f}|'''
    )
    print(f"{'Class':<30}|{'mAP50-95':<10}|{'mAR':<10}|")
    for i, c_id in enumerate(macro_metric["classes"]):
        if c_id >= 0 and c_id <= 396:
            print(
                f"{mapping_dict[c_id.item()]:<30}|{macro_metric['map_per_class'][i]:<10.4f}|{macro_metric['mar_100_per_class'][i]:<10.4f}|"
            )


def evaluate_faster_RCNN(model, data_loader, macro_metric, micro_metric, device):
    """
    Evaluates Faster RCNN model
    
    Args:
        model: Model that will be evaluated.
        data_loader (DataLoader): Dataloader with validation data.
        macro_metrics: Macro Mean Average Precision metric.
        micro_metrics: Micro Mean Average Precision metric.
        device: Device to move data to.
    
    Returns:
        macro_metrics: Updated Macro Mean Average Precision metric.
        micro_metrics: Updated Micro Mean Average Precision metric.
    """
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
            
            if (DATASET_NAME in ['Mapillary', 'GTSDB']):
                predictions = translate_predictions(predictions, device)
            macro_metric.update(predictions, targets)
            micro_metric.update(predictions, targets)
            if i % 100 == 0:
                print(f"Img {i} out of {len(data_loader)}")
    macro_result = macro_metric.compute()
    micro_result = micro_metric.compute()
    return macro_result, micro_result

def translate_predictions(predictions, device):
    """
    Translate predictions from one dataset to another using the translation dictionary. Used when cross evaluating models.\n
    Also collects data of signs that are not in the translation dict, so the ones that need to be looked at if they were not overlooked.

    Args:
        predictions: Outputs of the models.
        device: Device to move the predicitions to.

    """
    translation_dict = None
    if EVAL_TYPE == Eval_Type.YOLO_BASE:
        translation_dict = COCO_to_My
    elif EVAL_TYPE == Eval_Type.YOLO_MAPILLARY:
        translation_dict = Mapillary_to_My
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.FASTER_RCNN] and DATASET_NAME == 'Mapillary':
        translation_dict = My_to_Mapillary
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.FASTER_RCNN] and DATASET_NAME == 'GTSDB':
        translation_dict = CATSD_to_GTSDB
    labels = predictions[0]["labels"].tolist()
    translated_labels = [
        translation_dict[x] if x in translation_dict else -1 for x in labels
    ]
    for x in labels:
        if x in translation_dict:
            curr_count = num_of_appearences.setdefault(x, 0)
            num_of_appearences[x] = curr_count + 1
        else:
            act_predicted = predicted_signs.setdefault(x, 0)
            predicted_signs[x] = act_predicted + 1

    predictions[0]["labels"] = torch.tensor(translated_labels, dtype=torch.int64, device=device)
    return predictions

def evaluate_YOLO(model, data_loader, macro_metric, micro_metric, device):
    """
    Evaluates YOLO model
    
    Args:
        model: Model that will be evaluated.
        data_loader (DataLoader): Dataloader with validation data.
        macro_metrics: Macro Mean Average Precision metric.
        micro_metrics: Micro Mean Average Precision metric.
        device: Device to move data to.

    Returns:
        macro_metrics: Updated Macro Mean Average Precision metric.
        micro_metrics: Updated Micro Mean Average Precision metric.

    """
    for i, (img, targets) in enumerate(data_loader):
        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]
        predictions = model.predict(img, verbose=False, imgsz=512, conf=0.001, iou=0.6)
        predictions = convert_yolo_to_torch_outputs(predictions, device)
        if (EVAL_TYPE in [Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY] and DATASET_NAME != 'Mapillary') or (EVAL_TYPE == Eval_Type.YOLO_MY and DATASET_NAME == 'Mapillary'):
            predictions = translate_predictions(predictions, device)
        if (DATASET_NAME == 'GTSDB'):
            predictions = translate_predictions(predictions, device)
        
        macro_metric.update(predictions, targets)
        micro_metric.update(predictions, targets)
        if i % 100 == 0:
            print(f"Img {i} out of {len(data_loader)}")
    macro_result = macro_metric.compute()
    micro_result = micro_metric.compute()
    return macro_result, micro_result


def get_val_dataset():
    """
    Gets correct validation dataset for the model.

    Returns:
        val_dataset (TrafficSignDataset): Retuned dataset with correct transforms. 

    """
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
    """
    Loads correct model based on the EvalType.

    Args:
        model_path (str): Path to the model.
        device (str): Device that the model will be loaded to.

    """
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        model, opt, sch = load_model(model_path, device, box_score_thresh=0.60)
        model.to(device)
        model.eval()
        return model
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY]:
        return YOLO(model_path)


if __name__ == "__main__":
    val_dataset = get_val_dataset()
    mapping_dict = None

    if DATASET_NAME == 'Mapillary':
        mapping_dict = Mapillary_mapping_dict
    elif DATASET_NAME == 'GTSDB':
        mapping_dict = GTSDB_mapping_dict
    else:
        mapping_dict = create_mapping_dict("data/signs")
    model_path = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = get_model(model_path, device)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    macro_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, average='macro', iou_thresholds=None)
    micro_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, average='micro', iou_thresholds=None)
    macro_metric.to(device)
    micro_metric.to(device)
    result = None
    if EVAL_TYPE == Eval_Type.FASTER_RCNN:
        macro_result, micro_result = evaluate_faster_RCNN(model, val_loader, macro_metric, micro_metric, device)
    elif EVAL_TYPE in [Eval_Type.YOLO_MY, Eval_Type.YOLO_BASE, Eval_Type.YOLO_MAPILLARY]:
        macro_result, micro_result = evaluate_YOLO(model, val_loader, macro_metric, micro_metric, device)
    
    for k in macro_result.keys():
        print(f"{k}: {macro_result[k]}")
    print_metrics(macro_result, micro_result, mapping_dict)
    #print(num_of_appearences)
    print(f'Signs that are not in dict, but were predicted: {predicted_signs}')
