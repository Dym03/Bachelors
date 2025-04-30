import torch
import os

# from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_iou
from traffic_sign_dataset import TrafficSignDataset
from torcheval.metrics import MulticlassAccuracy
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, date
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision

NUM_CLASSES = 43
NUM_EPOCHS = 20
ACT_DATE = date.today().isoformat()
BASE_DATASET_DIR = "datasets"
DATASET_NAME = "100_000_n2"
OUTPUT_MODEL_DIR = f"torch_runs/run_{ACT_DATE}_{NUM_EPOCHS}_{DATASET_NAME}/models"
OUTPUT_RUN_DIR = f"torch_runs/run_{ACT_DATE}_{NUM_EPOCHS}_{DATASET_NAME}/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(dir=OUTPUT_RUN_DIR)


def save_model(epoch_loss, model, optimizer, lr_scheduler):
    """
    Saves model

    Args:
        epoch_loss (float): Curr epoch loss.
        model: Model to be saved.
        optimized: Optimizer to be saved.
        lr_scheduler: Learning rate scheduler to be saved.
    """
    output_path = os.path.join(OUTPUT_MODEL_DIR, f"{str(epoch_loss)}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        output_path,
    )


def plot_graph(scores, xlabel, ylabel, title):
    """
    Plots a metric

    Args:
        scores (list): List of metrics to be plotted
        xlabel (str): X axis label.
        ylabel (str): Y axis label.
        title (str): Title for the plot.
    """
    plt.clf()
    epochs = range(1, NUM_EPOCHS + 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(epochs, scores)
    plt.savefig(os.path.join(OUTPUT_RUN_DIR, title))


def evaluate(model, data_loader):
    """
    Evaluates the model

    Args:
        model: To be evaluated model.
        data_loader (DataLoader): Validation loader.

    Returns:
        mean_iou (float): List of matched predicted labels.
        label_accuracy (float): List of matched true labels.
        mAP50-95 (float): mAP50-95 metric.
        mAP50 (float): mAP50 metric.
    """
    total_iou = 0.0
    total_samples = 0
    metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
    mAP = MeanAveragePrecision(iou_type="bbox", class_metrics=True, average="micro")
    mAP.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (img, targets) in enumerate(tqdm((data_loader), desc="Evaluating")):
            # img = torch.stack(img).to(device)
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
            mAP.update(predictions, targets)

            matched_pred_labels, matched_true_labels, avg_iou = (
                match_boxes_and_calculate_iou(
                    predictions[0]["boxes"],
                    predictions[0]["labels"],
                    targets[0]["boxes"],
                    targets[0]["labels"],
                )
            )
            total_iou += avg_iou
            total_samples += 1
            if len(matched_pred_labels) > 0:
                metric.update(matched_pred_labels, matched_true_labels)

    mean_iou = total_iou / total_samples if total_samples > 0 else 0.0

    # Finalize label accuracy
    label_accuracy = metric.compute().item() if total_samples > 0 else 0.0
    mAP_result = mAP.compute()

    return mean_iou, label_accuracy, mAP_result["map"], mAP_result["map_50"]


def match_boxes_and_calculate_iou(
    pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold=0.5
):
    """
    Match predicted boxes and labels with ground truth and calculate IoU.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (N x 4).
        pred_labels (Tensor): Predicted labels (N).
        true_boxes (Tensor): Ground truth bounding boxes (M x 4).
        true_labels (Tensor): Ground truth labels (M).
        iou_threshold (float): IoU threshold for matching.

    Returns:
        matched_pred_labels (list): List of matched predicted labels.
        matched_true_labels (list): List of matched true labels.
        avg_iou (float): Average IoU for matched boxes.
    """
    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
        # No predictions or ground truth
        return [], [], 0.0

    # Compute IoU matrix
    iou_matrix = box_iou(pred_boxes, true_boxes)

    # Match predictions to ground truth using IoU threshold
    matched_pred_indices = []
    matched_true_indices = []
    iou_scores = []

    for i, row in enumerate(iou_matrix):
        max_iou, max_idx = row.max(0)
        if max_iou >= iou_threshold:
            matched_pred_indices.append(i)
            matched_true_indices.append(max_idx.item())
            iou_scores.append(max_iou.item())

    # Get matched labels
    matched_pred_labels = (
        pred_labels[matched_pred_indices] if matched_pred_indices else []
    )
    matched_true_labels = (
        true_labels[matched_true_indices] if matched_true_indices else []
    )

    # Calculate average IoU
    avg_iou = sum(iou_scores) / len(iou_scores) if len(iou_scores) > 0 else 0.0

    return matched_pred_labels, matched_true_labels, avg_iou


def train(model, train_data_loader, val_data_loader):
    """
    Train the model for NUM_EPOCHS defined

    Args:
        model: To be trained model.
        train_data_loader (DataLoader): Train loader.
        val_data_loader (DataLoader): Validation loader.

    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model.train()
    epoch_losses = []
    iou_losses = []
    label_accuracy_list = []
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        model.train()

        for idx, (img, targets) in enumerate(
            tqdm((train_data_loader), desc="Training epoch")
        ):
            img = torch.stack(img).to(device)
            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in targets
            ]
            optimizer.zero_grad()
            losses = model(img.to(device), targets)

            loss = sum([loss for loss in losses.values()])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if idx % 10000 == 0 and idx != 0:
                print(
                    f"Epoch : {epoch} img number {idx} out of {len(train_data_loader)} avg epoch_loss this epoch : {epoch_loss / (idx + 1)}"
                )
                wandb.log({"train_loss": loss})
        # update the learning rate
        mean_iou, label_accuracy, mAP50_95, mAP_50 = evaluate(model, val_data_loader)
        iou_losses.append(mean_iou)
        label_accuracy_list.append(label_accuracy)
        print(f"Mean IoU: {mean_iou}, Label Accuracy: {label_accuracy}")
        wandb.log(
            {
                "Mean_IoU": mean_iou,
                "label_accuracy": label_accuracy,
                "mAP50-95": mAP50_95,
                "mAP50": mAP_50,
            }
        )

        lr_scheduler.step()
        epoch_loss_avg = epoch_loss / len(train_data_loader)
        print(f"Avg Epoch loss {epoch_loss_avg}")
        if len(epoch_losses) == 0 or epoch_losses[-1] >= epoch_loss_avg:
            save_model(epoch_loss_avg, model, optimizer, lr_scheduler)

        epoch_losses.append(epoch_loss_avg)

    plot_graph(iou_losses, "Epochs", "IoU loss", "IoU_Train")
    plot_graph(label_accuracy_list, "Epochs", "Label Accuracy", "Label_Accuracy")
    plot_graph(epoch_losses, "Epochs", "Avg Epoch loss", "Avg_Epoch_loss")


def load_model(model_name, device, box_score_thresh=0.6):
    """
    Train the model for NUM_EPOCHS defined

    Args:
        model: To be loaded model.
        device (str): Device where the model should be placed.
        box_score_thresh (float): Model param for detections.
    """
    model = fasterrcnn_resnet50_fpn_v2(box_score_thresh=box_score_thresh)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES).to(
        device
    )
    weights_dict = torch.load(model_name, device, weights_only=True)
    model.load_state_dict(weights_dict["model"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    # optimizer.load_state_dict(weights_dict["optimizer"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler.load_state_dict(weights_dict["scheduler"])
    return (model, optimizer, scheduler)


def create_mapping_dict(sign_dir):
    """
    Creates a mapping dictionary for translating ids to sign name.

    Args:
        sign_dir (str): Directory where signs are saved with format id_name.

    Returns:
        sign_dict (dict): Returns dict key being id value being name of the sign.
    """
    sign_dict = {0: "background"}
    for filename in os.listdir(sign_dir):
        idx = int(filename[: filename.find("_")]) + 1
        sign_dict[idx] = filename[filename.find("_") + 1 : filename.find(".")]

    return sign_dict

def init_dirs():
    """
    Prepare dirs
    """
    os.mkdir(OUTPUT_RUN_DIR)
    os.mkdir(OUTPUT_MODEL_DIR)


if __name__ == "__main__":
    print(device)
    init_dirs()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7).train()
    model.to(device)
    wandb.watch(model, log_freq=100)
    train_dataset = TrafficSignDataset(
        os.path.join(BASE_DATASET_DIR, DATASET_NAME),
        "train/images",
        "train/labels",
        #        ToTensor(),
        transform=weights.transforms(),
    )
    val_dataset = TrafficSignDataset(
        os.path.join(BASE_DATASET_DIR, DATASET_NAME),
        "val/images",
        "val/labels",
        #        ToTensor(),
        transform=weights.transforms(),
    )
    print(len(train_dataset))
    mapping_dict = create_mapping_dict("data/signs")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES).to(
        device
    )

    training_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=4,
        collate_fn=lambda batch: tuple(
            zip(*batch)
        ),  # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=4,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    train(model, training_loader, val_loader)
