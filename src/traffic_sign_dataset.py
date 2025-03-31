import os
from PIL import Image
from torch import tensor, float32, int64, zeros, empty
from torch.utils.data import Dataset
import torchvision.ops as ops

Mapillary_to_My = {
    0: 0,  # pozadí
    271: 1,  # krizovatka
    272: 1,
    374: 2,  # semafor
    375: 2,
    376: 2,
    377: 2,
    162: 3,  # jedno_smer
    109: 4,  # konec_80
    352: 5,  # prace
    353: 5,
    354: 5,
    355: 5,
    356: 5,
    357: 5,
    274: 6,  # leva
    275: 6,
    247: 6,
    248: 6,
    249: 6,
    250: 6,
    276: 7,  # prava
    277: 7,
    251: 7,
    252: 7,
    253: 7,
    254: 7,
    255: 7,
    262: 8,  # prednost
    238: 9,  # stop
    239: 9,
    240: 9,
    241: 9,
    242: 9,
    243: 9,
    230: 10,  # zakaz_vjezdu
    231: 10,
    232: 10,
    233: 10,
    136: 11,  # 30
    137: 11,
    139: 12,  # 40
    140: 12,
    141: 12,
    145: 13,  # 50
    146: 13,
    148: 14,  # 60
    151: 15,  # 80
    155: 15,
    152: 16,  # 90
    128: 17,  # 100
    129: 17,
    153: 17,
    130: 18,  # 110
    131: 19,  # 120
    227: 21,  # hlavni
    107: 22,  # konec_hlavni
    234: 24,  # kruhovy_objezd
    235: 24,
    283: 25,  # leva_prava
    284: 25,
    386: 26,  # nerovnost
    387: 26,
    388: 26,
    348: 28,  # zuzeni_prava
    349: 28,
    285: 29,  # prava_leva
    286: 29,
    321: 30,  # prechod
    322: 30,
    323: 30,
    324: 30,
    325: 30,
    326: 30,
    327: 30,
    110: 31,  # prednost_proti_jedouci
    111: 32,  # prikazany_smer_rovne
    112: 32,
    381: 33,  # prikazany_smer_vlevo
    19: 33,
    382: 34,  # prikazany_smer_vpravo
    20: 34,
    21: 34,
    383: 34,
    342: 35,  # retarder
    343: 35,
    176: 36,  # zakaz_aut
    177: 36,
    174: 37,  # zakaz_aut_motorky
    175: 37,
    180: 38,  # zakaz_predjizdeni
    181: 38,
    182: 38,
    183: 38,
    71: 39,  # zakaz_stani
    185: 39,
    186: 39,
    187: 39,
    188: 40,  # zakaz_zastaveni
    189: 40,
    190: 40,
    198: 40,
    199: 40,
    200: 40,
    201: 40,
    202: 40,
    344: 41,  # zuzeni
    345: 41,
    346: 42,  # zuzeni_leva
    347: 42,
}

COCO_to_My = {
    11: 9,
    9: 2,
}

My_to_Mapillary = {
    1: 272,
    3: 162,
    4: 109,
    5: 355,
    6: 277,  # Check this one
    7: 277,
    8: 238,
    9: 241,
    10: 230,  # Check this out this one
    11: 136,
    12: 139,
    14: 148,
    15: 151,
    16: 152,
    17: 153,
    18: 130,
    19: 131,
    21: 227,  # hlavni
    22: 107,  # konec_hlavni
    25: 284,
    26: 388,
    28: 349,
    29: 286,
    30: 324,
    31: 110,
    34: 383,
    35: 343,
    36: 176,
    37: 175,
    38: 183,
    39: 186,
    40: 188,
    41: 345,
    42: 347,
}

CATSD_to_GTSDB = {
        11 : 1,
        13 : 2,
        14 : 3,
        15 : 5,
        4  : 6,
        17 : 7,
        19 : 8,
        38 : 9,
        21 : 12,
        8  : 13,
        9  : 14,
        10 : 15,
        3  : 17,
        6  : 19,
        7  : 20,
        25 : 21,
        26 : 22,
        28 : 24,
        5  : 25,
        2  : 26,
        30 : 27,
        34 : 33,
        33 : 34,
        32 : 35,
        24 : 40,
        23 : 41,
        }



def apply_nms(predictions, device, iou_threshold=0.45, conf_threshold=0.001):
    """
    Applies Non-Maximum Suppression (NMS) on YOLO predictions.

    Args:
        predictions: List of dictionaries containing 'boxes', 'scores', and 'labels'.
        iou_threshold: IoU threshold for NMS.
        conf_threshold: Confidence score threshold.

    Returns:
        Filtered predictions after applying NMS.
    """
    filtered_predictions = []
    for pred in predictions:
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        # Filter out low-confidence predictions
        keep = scores > conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) == 0:
            filtered_predictions.append(
                {
                    "boxes": empty((0, 4), device=device),
                    "scores": empty((0,), device=device),
                    "labels": empty((0,), device=device).long(),
                }
            )
            continue

        # Apply NMS
        keep_indices = ops.nms(boxes, scores, iou_threshold)

        # Keep only selected boxes
        filtered_predictions.append(
            {
                "boxes": boxes[keep_indices],
                "scores": scores[keep_indices],
                "labels": labels[keep_indices],
            }
        )

    return filtered_predictions


def convert_yolo_to_torch_outputs(results, device):
    yolo_boxes = results[0].boxes.data
    torch_outpus = []
    torch_dict = {"boxes": [], "labels": [], "scores": []}
    for box in yolo_boxes:
        torch_dict["boxes"].append(box[:4].tolist())
        torch_dict["labels"].append(int(box[5]))
        torch_dict["scores"].append(box[4].item())
    for k in torch_dict.keys():
        torch_dict[k] = tensor(torch_dict[k], device=device)
    torch_outpus.append(torch_dict)
    return torch_outpus


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
        img_dir: str,
        label_dir: str,
        transform,
    ):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_dir)
        self.annotations = os.listdir(self.img_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_name = self.annotations[index]
        img_file_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_file_path)
        img_width, img_height = img.width, img.height
        # img = self.transform(img)
        if self.transform:
            img = self.transform(img)
            img_width, img_height = img.shape[1], img.shape[2]
        # tensor_img = torch.tensor(img)

        label_file_path = os.path.join(
            self.label_dir, file_name[0 : file_name.find(".")] + ".txt"
        )
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
                yolo_box = yolo_to_coco(tokens[1:], img_width, img_height)
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
