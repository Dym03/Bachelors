import torch
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from traffic_sign_dataset import TrafficSignDataset

NUM_CLASSES = 43
NUM_EPOCHS = 4
OUTPUT_MODEL_DICT = "models/"
BASE_DATASET_DIR = "datasets"
DATASET_NAME = "10_000"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(epoch_loss, model, optimizer, lr_scheduler):
    output_path = os.path.join(OUTPUT_MODEL_DICT, str(epoch_loss) + ".pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        output_path,
    )


def train(model, data_loader):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model.train()
    epoch_losses = [0]
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0

        for idx, (img, targets) in enumerate(data_loader):
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
            if idx % 500 == 0:
                print(
                    f"Epoch : {epoch} img number {idx} out of {len(data_loader)} avg epoch_loss this epoch : {epoch_loss / (idx + 1)}"
                )
        # update the learning rate
        lr_scheduler.step()
        epoch_loss_avg = epoch_loss / len(data_loader)
        print(f"Avg Epoch loss {epoch_loss_avg}")
        if len(epoch_losses) == 0 or epoch_losses[-1] >= epoch_loss_avg:
            save_model(epoch_loss_avg, model, optimizer, lr_scheduler)

        epoch_losses.append(epoch_loss_avg)


def load_model(model_name, device, box_score_thresh=0.6):
    model = fasterrcnn_resnet50_fpn_v2(box_score_thresh = box_score_thresh)
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
    sign_dict = {0: "background"}
    for filename in os.listdir(sign_dir):
        idx = int(filename[: filename.find("_")]) + 1
        sign_dict[idx] = filename[filename.find("_") + 1 : filename.find(".")]

    return sign_dict


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    # Optionally stack images into a batch
    images = torch.stack(images)
    targets = list(targets)
    # No need to stack target['boxes'] and 'labels' since you may want them as lists
    return images, targets[0]


if __name__ == "__main__":
    print(device)
    train_dataset = TrafficSignDataset(
        os.path.join(BASE_DATASET_DIR, DATASET_NAME),
        "annotation.csv",
        "img",
        "labels",
        ToTensor(),
    )
    # test_dataset = TrafficSignDataset(
    #     "datasets/test_dataset/test", "annotation.csv", "img", "labels", ToTensor()
    # )
    print(len(train_dataset))
    mapping_dict = create_mapping_dict("data/signs")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.6).train()
    model.to(device)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES).to(
        device
    )

    training_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        collate_fn=lambda batch: tuple(
            zip(*batch)
        ),  # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
    )

    train(model, training_loader)

    # model.eval()
    # testing_loader = DataLoader(test_dataset, collate_fn=custom_collate_fn)
    # data_iter = iter(testing_loader)
    # image, label = next(data_iter)
    # # print(label)
    # device = torch.device("cpu")
    # model = model.to(device)
    # image = image.to(device)

    # preprocess = weights.transforms()
    # batch = [preprocess(image)]

    # im = to_pil_image(image[0])
    # im.show()
    # predictions = model(image)
    # print(predictions)
    # print(type(predictions[0]["labels"]))
    # labels = [mapping_dict[int(id)] for id in predictions[0]["labels"]]
    # print(predictions)
    # print(labels)
    # boxes = torch.tensor(
    #     [
    #         [134.4126, 152.6398, 186.7712, 200.0036],
    #         [302.9673, 175.2862, 380.8156, 236.9193],
    #         [297.1067, 311.1100, 333.6919, 344.6310],
    #     ]
    # )
    # labels = [7, 7, 21]
    # labels = [mapping_dict[id + 1] for id in labels]
    # box = draw_bounding_boxes(
    #     image[0],
    #     boxes=boxes,
    #     labels=labels,
    #     colors="red",
    #     width=4,
    #     font_size=30,
    # )
    # im = to_pil_image(box.detach())
    # im.show()

    # dataiter = iter(training_loader)
    # images, labels = next(dataiter)

    # print(len(images))

    # # print(images)
    # images = images[0]

    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    # model.eval()

    # preprocess = weights.transforms()

    # batch = [preprocess(images)]
    # prediction = model(batch)[0]
    # labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    # box = draw_bounding_boxes(
    #     images,
    #     boxes=prediction["boxes"],
    #     labels=labels,
    #     colors="red",
    #     width=4,
    #     font_size=30,
    # )
    # im = to_pil_image(box.detach())
    # im.show()
