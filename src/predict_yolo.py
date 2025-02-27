from ultralytics import YOLO
from PIL import Image

model = YOLO(model="models/yolo_models/yolo_v11m_50e_10_000_n2.pt")

# results = model("data/img/rychlosti.jpg")

results = model("datasets/yolo_dataset/train/images/image-1.jpg")

results[0].show()
