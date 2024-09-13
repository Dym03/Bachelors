from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10x.pt")

results = model("src/data/img/stop_2.jpg")

results[0].show()